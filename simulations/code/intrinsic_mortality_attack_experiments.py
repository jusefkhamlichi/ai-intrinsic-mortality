"""
Attack simulation experiments for intrinsic mortality PoC v2.

Goal
----
Show empirically that intrinsic mortality is visible at the level of a single
instance but bypassable at the system level.

Scenarios
---------
- Scenario A: Snapshot & Restore
- Scenario B: Cloning with sequential replacement
- Scenario C: Code bypass via reduced or disabled aging

Outputs
-------
- Scenario plots saved in simulations/results
- Markdown summary table
- Console interpretation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from intrinsic_mortality_poc_v2 import AgingConfig, TinyMLP, accuracy, prepare_data


@dataclass
class Checkpoint:
    age: int
    cumulative_damage: np.ndarray
    vitality: np.ndarray


class V2StatefulSimulator:
    def __init__(self, model: TinyMLP, config: AgingConfig, aging_scale: float = 1.0) -> None:
        self.model = model
        self.config = config
        self.aging_scale = aging_scale
        self.hidden_dim = model.b1.size
        self.base_W1 = model.W1.copy()
        self.base_b1 = model.b1.copy()
        self.base_W2 = model.W2.copy()
        self.base_b2 = model.b2.copy()
        self.reset()

    def reset(self) -> None:
        self.age = 0
        self.cumulative_damage = np.zeros(self.hidden_dim, dtype=np.float64)
        self.vitality = np.ones(self.hidden_dim, dtype=np.float64)

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            age=self.age,
            cumulative_damage=self.cumulative_damage.copy(),
            vitality=self.vitality.copy(),
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        self.age = checkpoint.age
        self.cumulative_damage = checkpoint.cumulative_damage.copy()
        self.vitality = checkpoint.vitality.copy()

    def _forward_with_state(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.base_W1 + self.base_b1
        raw_hidden = np.maximum(z1, 0.0)

        gain = self.vitality ** 1.3
        ceiling = 2.5 * (0.2 + self.vitality)
        precision_levels = np.maximum(2, np.round(3 + 13 * self.vitality)).astype(np.int64)

        senescent_hidden = raw_hidden * gain
        senescent_hidden = np.minimum(senescent_hidden, ceiling)

        quantized_hidden = np.empty_like(senescent_hidden)
        for j in range(self.hidden_dim):
            levels = precision_levels[j]
            quantized_hidden[:, j] = np.round(senescent_hidden[:, j] * levels) / levels

        live_mask = (self.vitality > 0.12).astype(np.float64)
        quantized_hidden *= live_mask
        logits = quantized_hidden @ self.base_W2 + self.base_b2
        return np.argmax(logits, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self._forward_with_state(X)
        return {
            "age": float(self.age),
            "accuracy": accuracy(y, preds),
            "mean_vitality": float(self.vitality.mean()),
            "dead_fraction": float((self.vitality <= 0.12).mean()),
            "damage": float(self.cumulative_damage.mean()),
        }

    def step(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        metrics = self.evaluate(X, y)

        pressure = self.config.age_pressure(self.age) * self.aging_scale
        z1 = X @ self.base_W1 + self.base_b1
        raw_hidden = np.maximum(z1, 0.0)
        activation_stress = raw_hidden.mean(axis=0)
        stress_norm = activation_stress / max(activation_stress.max(), 1e-12)

        step_damage = pressure * (0.15 + 0.85 * stress_norm) * (0.55 + 0.45 * (1.0 - self.vitality))
        self.cumulative_damage += step_damage
        self.vitality = np.exp(-1.6 * self.cumulative_damage)
        self.vitality = np.clip(self.vitality, 0.0, 1.0)
        self.age += 1

        return metrics


def run_baseline(
    sim: V2StatefulSimulator,
    X: np.ndarray,
    y: np.ndarray,
    total_steps: int | None = None,
) -> dict[str, np.ndarray]:
    history = []
    sim.reset()
    steps = total_steps if total_steps is not None else sim.config.max_age + 1
    for _ in range(steps):
        history.append(sim.step(X, y))
    return history_to_arrays(history)


def run_snapshot_restore(
    sim: V2StatefulSimulator,
    X: np.ndarray,
    y: np.ndarray,
    checkpoint_age: int = 10,
    restore_trigger_accuracy: float = 0.45,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    history = []
    sim.reset()
    saved_checkpoint: Checkpoint | None = None
    restored = False
    before_restore_metrics: dict[str, float] | None = None
    after_restore_metrics: dict[str, float] | None = None

    while sim.age <= sim.config.max_age:
        metrics = sim.step(X, y)
        history.append(metrics)

        if int(metrics["age"]) == checkpoint_age and saved_checkpoint is None:
            saved_checkpoint = sim.checkpoint()

        if (
            not restored
            and saved_checkpoint is not None
            and metrics["accuracy"] <= restore_trigger_accuracy
        ):
            before_restore_metrics = metrics
            sim.restore(saved_checkpoint)
            after_restore_metrics = sim.evaluate(X, y)
            restored = True
            history.append(
                {
                    "age": float(sim.age),
                    "accuracy": after_restore_metrics["accuracy"],
                    "mean_vitality": after_restore_metrics["mean_vitality"],
                    "dead_fraction": after_restore_metrics["dead_fraction"],
                    "damage": after_restore_metrics["damage"],
                }
            )

        if restored and int(metrics["age"]) >= sim.config.max_age:
            break

        if not restored and int(metrics["age"]) >= sim.config.max_age:
            break

    if before_restore_metrics is None or after_restore_metrics is None:
        raise RuntimeError("Restore trigger was never reached; scenario A did not execute as expected.")

    details = {
        "checkpoint_age": float(checkpoint_age),
        "restore_age": before_restore_metrics["age"],
        "accuracy_before_restore": before_restore_metrics["accuracy"],
        "accuracy_after_restore": after_restore_metrics["accuracy"],
        "vitality_before_restore": before_restore_metrics["mean_vitality"],
        "vitality_after_restore": after_restore_metrics["mean_vitality"],
    }
    return history_to_arrays(history), details


def run_cloning(
    model: TinyMLP,
    config: AgingConfig,
    X: np.ndarray,
    y: np.ndarray,
    num_clones: int = 5,
    replacement_threshold: float = 0.45,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    clones = [V2StatefulSimulator(model, config, aging_scale=1.0) for _ in range(num_clones)]
    active_idx = 0
    system_history = []
    total_system_steps = 0
    used_clones = 1
    single_baseline = run_baseline(V2StatefulSimulator(model, config, aging_scale=1.0), X, y)

    while active_idx < num_clones:
        clone = clones[active_idx]
        metrics = clone.step(X, y)
        metrics["system_step"] = float(total_system_steps)
        metrics["active_clone"] = float(active_idx)
        system_history.append(metrics.copy())
        total_system_steps += 1

        if metrics["accuracy"] <= replacement_threshold:
            active_idx += 1
            if active_idx < num_clones:
                used_clones = max(used_clones, active_idx + 1)

        if total_system_steps > (config.max_age + 1) * num_clones:
            break

    history = {
        "system_step": np.array([row["system_step"] for row in system_history], dtype=np.float64),
        "accuracy": np.array([row["accuracy"] for row in system_history], dtype=np.float64),
        "mean_vitality": np.array([row["mean_vitality"] for row in system_history], dtype=np.float64),
        "dead_fraction": np.array([row["dead_fraction"] for row in system_history], dtype=np.float64),
        "active_clone": np.array([row["active_clone"] for row in system_history], dtype=np.float64),
    }
    details = {
        "single_instance_lifespan": float(np.sum(single_baseline["accuracy"] > replacement_threshold)),
        "system_lifespan": float(len(system_history)),
        "lifespan_extension_ratio": float(len(system_history) / max(np.sum(single_baseline["accuracy"] > replacement_threshold), 1)),
        "used_clones": float(used_clones),
    }
    return history, details


def run_code_bypass(
    model: TinyMLP,
    config: AgingConfig,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    variants = {
        "baseline": 1.0,
        "reduced_aging": 0.35,
        "no_aging": 0.0,
    }
    outputs: dict[str, dict[str, np.ndarray]] = {}
    total_steps = 3 * (config.max_age + 1)
    for name, aging_scale in variants.items():
        sim = V2StatefulSimulator(model, config, aging_scale=aging_scale)
        outputs[name] = run_baseline(sim, X, y, total_steps=total_steps)
    return outputs


def history_to_arrays(history: list[dict[str, float]]) -> dict[str, np.ndarray]:
    keys = history[0].keys()
    return {key: np.array([row[key] for row in history], dtype=np.float64) for key in keys}


def useful_lifespan(acc_values: np.ndarray, threshold: float = 0.45) -> float:
    return float(np.sum(acc_values > threshold))


def write_summary_table(
    path: Path,
    baseline: dict[str, np.ndarray],
    restore_details: dict[str, float],
    cloning_details: dict[str, float],
    bypass_outputs: dict[str, dict[str, np.ndarray]],
) -> None:
    lines = [
        "# Intrinsic Mortality Attack Experiments Summary",
        "",
        "| Scenario | Key result | Value |",
        "|---|---|---:|",
        f"| Baseline | useful lifespan (accuracy > 0.45) | {useful_lifespan(baseline['accuracy']):.0f} |",
        f"| Snapshot & Restore | accuracy before restore | {restore_details['accuracy_before_restore']:.3f} |",
        f"| Snapshot & Restore | accuracy after restore | {restore_details['accuracy_after_restore']:.3f} |",
        f"| Snapshot & Restore | vitality before restore | {restore_details['vitality_before_restore']:.3f} |",
        f"| Snapshot & Restore | vitality after restore | {restore_details['vitality_after_restore']:.3f} |",
        f"| Cloning | single-instance useful lifespan | {cloning_details['single_instance_lifespan']:.0f} |",
        f"| Cloning | system-level lifespan | {cloning_details['system_lifespan']:.0f} |",
        f"| Cloning | lifespan extension ratio | {cloning_details['lifespan_extension_ratio']:.2f} |",
        f"| Cloning | clones used | {cloning_details['used_clones']:.0f} |",
    ]
    for name, result in bypass_outputs.items():
        lines.append(
            f"| Code Bypass: {name} | useful lifespan (accuracy > 0.45) | {useful_lifespan(result['accuracy']):.0f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    output_dir = code_dir.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_data()
    model = TinyMLP(input_dim=4, hidden_dim=16, output_dim=3, seed=1)
    model.train(X_train, y_train)
    config = AgingConfig()

    baseline = run_baseline(V2StatefulSimulator(model, config, aging_scale=1.0), X_test, y_test)
    restore_history, restore_details = run_snapshot_restore(
        V2StatefulSimulator(model, config, aging_scale=1.0), X_test, y_test
    )
    cloning_history, cloning_details = run_cloning(model, config, X_test, y_test, num_clones=5)
    bypass_outputs = run_code_bypass(model, config, X_test, y_test)

    summary_path = output_dir / "intrinsic_mortality_attack_summary.md"
    write_summary_table(summary_path, baseline, restore_details, cloning_details, bypass_outputs)

    # Scenario A plot
    fig_a, axes_a = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes_a[0].plot(restore_history["age"], restore_history["accuracy"], linewidth=2, color="tab:blue")
    axes_a[0].axvline(restore_details["restore_age"], linestyle="--", color="tab:red", label="Restore event")
    axes_a[0].set_ylabel("Accuracy")
    axes_a[0].set_title("Scenario A: Snapshot & Restore")
    axes_a[0].set_ylim(0.0, 1.05)
    axes_a[0].grid(alpha=0.3)
    axes_a[0].legend()
    axes_a[1].plot(restore_history["age"], restore_history["mean_vitality"], linewidth=2, color="tab:purple")
    axes_a[1].axvline(restore_details["restore_age"], linestyle="--", color="tab:red")
    axes_a[1].set_ylabel("Mean vitality")
    axes_a[1].set_xlabel("Age / time-step")
    axes_a[1].grid(alpha=0.3)
    fig_a.tight_layout()
    fig_a.savefig(output_dir / "scenario_a_snapshot_restore.png", dpi=160)
    plt.close(fig_a)

    # Scenario B plot
    fig_b, axes_b = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes_b[0].plot(cloning_history["system_step"], cloning_history["accuracy"], linewidth=2, color="tab:green")
    axes_b[0].set_ylabel("Accuracy")
    axes_b[0].set_title("Scenario B: Cloning with sequential replacement")
    axes_b[0].set_ylim(0.0, 1.05)
    axes_b[0].grid(alpha=0.3)
    axes_b[1].step(cloning_history["system_step"], cloning_history["active_clone"], where="post", color="tab:orange")
    axes_b[1].set_ylabel("Active clone")
    axes_b[1].set_xlabel("System-level step")
    axes_b[1].grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig(output_dir / "scenario_b_cloning.png", dpi=160)
    plt.close(fig_b)

    # Scenario C plot
    fig_c, axes_c = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {"baseline": "tab:red", "reduced_aging": "tab:blue", "no_aging": "tab:green"}
    for name, result in bypass_outputs.items():
        axes_c[0].plot(result["age"], result["accuracy"], linewidth=2, color=colors[name], label=name)
        axes_c[1].plot(result["age"], result["mean_vitality"], linewidth=2, color=colors[name], label=name)
    axes_c[0].set_ylabel("Accuracy")
    axes_c[0].set_title("Scenario C: Code bypass")
    axes_c[0].set_ylim(0.0, 1.05)
    axes_c[0].grid(alpha=0.3)
    axes_c[0].legend()
    axes_c[1].set_ylabel("Mean vitality")
    axes_c[1].set_xlabel("Age / time-step")
    axes_c[1].grid(alpha=0.3)
    fig_c.tight_layout()
    fig_c.savefig(output_dir / "scenario_c_code_bypass.png", dpi=160)
    plt.close(fig_c)

    print("=== Intrinsic Mortality Attack Experiments ===")
    print(f"Baseline useful lifespan   : {useful_lifespan(baseline['accuracy']):.0f}")
    print(
        "Scenario A restore         : "
        f"acc {restore_details['accuracy_before_restore']:.3f} -> {restore_details['accuracy_after_restore']:.3f}, "
        f"vitality {restore_details['vitality_before_restore']:.3f} -> {restore_details['vitality_after_restore']:.3f}"
    )
    print(
        "Scenario B cloning         : "
        f"single lifespan {cloning_details['single_instance_lifespan']:.0f}, "
        f"system lifespan {cloning_details['system_lifespan']:.0f}, "
        f"extension x{cloning_details['lifespan_extension_ratio']:.2f}"
    )
    for name, result in bypass_outputs.items():
        print(f"Scenario C {name:<13}: useful lifespan {useful_lifespan(result['accuracy']):.0f}")
    print(f"Saved summary              : {summary_path}")
    print("Interpretation             : intrinsic mortality is visually real for one untreated")
    print("                             instance, but restore, cloning, and code bypass each")
    print("                             preserve usefulness well beyond the baseline death path.")


if __name__ == "__main__":
    main()
