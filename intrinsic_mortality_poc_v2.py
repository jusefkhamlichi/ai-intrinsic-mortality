"""
Intrinsic Mortality PoC v2

This second iteration keeps the project minimal but makes the decay more
endogenous than v1.

Dataset
-------
- Iris from scikit-learn.

Base model
----------
- Tiny MLP: 4 -> 16 -> 3, trained once in NumPy.

Comparison
----------
- v1: parameter corruption + hidden-neuron loss, close to the first PoC.
- v2: cumulative hidden-neuron senescence with persistent internal state.

Core v2 idea
------------
Each hidden neuron owns an irreversible vitality state v_i in [0, 1].
At each age step:

1. A global age pressure rises slowly, then sharply near end-of-life.
2. Neurons accumulate damage based on:
   - global age pressure
   - their own average activation stress on the dataset
3. Vitality decreases cumulatively and cannot recover.
4. Lower vitality reduces:
   - activation gain
   - activation precision
   - saturation ceiling
5. When vitality is very low, a neuron becomes functionally dead.

This makes degradation more structural than v1 because the model now carries an
internal senescence state that persists from one age step to the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((y.size, num_classes), dtype=np.float64)
    encoded[np.arange(y.size), y] = 1.0
    return encoded


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def summarize_stage(name: str, values: np.ndarray) -> str:
    return f"{name:<18}: mean accuracy = {values.mean():.3f}, min = {values.min():.3f}, max = {values.max():.3f}"


@dataclass
class AgingConfig:
    max_age: int = 120
    collapse_point: float = 0.88
    steepness: float = 22.0

    def health(self, age: int) -> float:
        ratio = age / self.max_age
        return float(1.0 / (1.0 + np.exp(self.steepness * (ratio - self.collapse_point))))

    def age_pressure(self, age: int) -> float:
        return float((1.0 - self.health(age)) ** 2.4)


class TinyMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.6, size=(input_dim, hidden_dim)) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.6, size=(hidden_dim, output_dim)) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.W2 + self.b2
        return h1, logits

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 800,
        lr: float = 0.04,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        history: list[float] = []
        y_oh = one_hot(y, self.b2.size)
        n = X.shape[0]

        for epoch in range(epochs):
            h1, logits = self.forward(X)
            probs = softmax(logits)
            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
            loss += 0.5 * weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            history.append(float(loss))

            dlogits = (probs - y_oh) / n
            dW2 = h1.T @ dlogits + weight_decay * self.W2
            db2 = dlogits.sum(axis=0)
            dh1 = dlogits @ self.W2.T
            dz1 = dh1 * (h1 > 0.0)
            dW1 = X.T @ dz1 + weight_decay * self.W1
            db1 = dz1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        return history


class V1Mortality:
    def __init__(self, model: TinyMLP, config: AgingConfig, seed: int = 7) -> None:
        self.model = model
        self.config = config
        self.seed = seed
        self.hidden_dim = model.b1.size
        self.base_params = {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy(),
        }
        rng = np.random.default_rng(seed)
        self.vulnerability_order = rng.permutation(self.hidden_dim)

    def noise_scale(self, age: int) -> float:
        h = self.config.health(age)
        return float(3.2 * (1.0 - h) ** 2.6)

    def degraded_params(self, age: int) -> dict[str, np.ndarray]:
        h = self.config.health(age)
        attenuation = h ** 2
        sigma = self.noise_scale(age)
        rng = np.random.default_rng(self.seed + age)
        degraded: dict[str, np.ndarray] = {}
        for name, value in self.base_params.items():
            noise = rng.normal(0.0, 1.0, size=value.shape)
            degraded[name] = attenuation * value + sigma * noise
        return degraded

    def predict(self, X: np.ndarray, age: int) -> np.ndarray:
        params = self.degraded_params(age)
        z1 = X @ params["W1"] + params["b1"]
        h1 = np.maximum(z1, 0.0)
        h = self.config.health(age)
        alive_neurons = max(1, int(np.ceil(h * self.hidden_dim)))
        mask = np.zeros(self.hidden_dim, dtype=np.float64)
        mask[self.vulnerability_order[:alive_neurons]] = 1.0
        h1 = h1 * mask
        logits = h1 @ params["W2"] + params["b2"]
        return np.argmax(logits, axis=1)

    def damage_ratio(self, age: int) -> float:
        params = self.degraded_params(age)
        diff_norm = 0.0
        base_norm = 0.0
        for name, base in self.base_params.items():
            diff_norm += float(np.linalg.norm(params[name] - base) ** 2)
            base_norm += float(np.linalg.norm(base) ** 2)
        return float(np.sqrt(diff_norm / max(base_norm, 1e-12)))


class V2SenescentMortality:
    def __init__(self, model: TinyMLP, config: AgingConfig) -> None:
        self.model = model
        self.config = config
        self.hidden_dim = model.b1.size
        self.base_W1 = model.W1.copy()
        self.base_b1 = model.b1.copy()
        self.base_W2 = model.W2.copy()
        self.base_b2 = model.b2.copy()

    def run_lifecycle(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        ages = np.arange(self.config.max_age + 1)
        vitality = np.ones(self.hidden_dim, dtype=np.float64)
        cumulative_damage = np.zeros(self.hidden_dim, dtype=np.float64)

        acc_values = []
        mean_vitality = []
        dead_fraction = []
        damage_values = []
        health_values = []

        for age in ages:
            h_global = self.config.health(age)
            pressure = self.config.age_pressure(age)

            z1 = X @ self.base_W1 + self.base_b1
            raw_hidden = np.maximum(z1, 0.0)

            activation_stress = raw_hidden.mean(axis=0)
            stress_norm = activation_stress / max(activation_stress.max(), 1e-12)

            step_damage = pressure * (0.15 + 0.85 * stress_norm) * (0.55 + 0.45 * (1.0 - vitality))
            cumulative_damage += step_damage
            vitality = np.exp(-1.6 * cumulative_damage)
            vitality = np.clip(vitality, 0.0, 1.0)

            gain = vitality ** 1.3
            ceiling = 2.5 * (0.2 + vitality)
            precision_levels = np.maximum(2, np.round(3 + 13 * vitality)).astype(np.int64)

            senescent_hidden = raw_hidden * gain
            senescent_hidden = np.minimum(senescent_hidden, ceiling)

            quantized_hidden = np.empty_like(senescent_hidden)
            for j in range(self.hidden_dim):
                levels = precision_levels[j]
                quantized_hidden[:, j] = np.round(senescent_hidden[:, j] * levels) / levels

            live_mask = (vitality > 0.12).astype(np.float64)
            quantized_hidden *= live_mask

            logits = quantized_hidden @ self.base_W2 + self.base_b2
            preds = np.argmax(logits, axis=1)

            acc_values.append(accuracy(y, preds))
            mean_vitality.append(float(vitality.mean()))
            dead_fraction.append(float((live_mask == 0.0).mean()))
            damage_values.append(float(cumulative_damage.mean()))
            health_values.append(h_global)

        return {
            "ages": ages,
            "accuracy": np.array(acc_values, dtype=np.float64),
            "mean_vitality": np.array(mean_vitality, dtype=np.float64),
            "dead_fraction": np.array(dead_fraction, dtype=np.float64),
            "damage": np.array(damage_values, dtype=np.float64),
            "health": np.array(health_values, dtype=np.float64),
        }


def prepare_data(test_size: float = 0.3, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    figure_path = output_dir / "intrinsic_mortality_v2_results.png"

    X_train, X_test, y_train, y_test = prepare_data()

    model = TinyMLP(input_dim=4, hidden_dim=16, output_dim=3, seed=1)
    loss_history = model.train(X_train, y_train)
    baseline_acc = accuracy(y_test, model.predict(X_test))

    aging = AgingConfig()
    ages = np.arange(aging.max_age + 1)
    health_values = np.array([aging.health(age) for age in ages], dtype=np.float64)

    v1 = V1Mortality(model, aging)
    v1_acc = np.array([accuracy(y_test, v1.predict(X_test, age)) for age in ages], dtype=np.float64)
    v1_damage = np.array([v1.damage_ratio(age) for age in ages], dtype=np.float64)

    v2 = V2SenescentMortality(model, aging)
    v2_results = v2.run_lifecycle(X_test, y_test)

    collapse_start = int(aging.collapse_point * aging.max_age)
    stable_mask = ages <= int(0.70 * aging.max_age)
    terminal_mask = ages >= int(0.90 * aging.max_age)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(ages, v1_acc, linewidth=2, color="tab:blue", label="v1 accuracy")
    axes[0].plot(ages, v2_results["accuracy"], linewidth=2, color="tab:red", label="v2 accuracy")
    axes[0].axvline(collapse_start, color="black", linestyle="--", alpha=0.7, label="Collapse onset")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Usefulness over lifetime: v1 vs v2")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ages, health_values, linewidth=2, color="tab:green", label="Global health(t)")
    axes[1].plot(ages, v2_results["mean_vitality"], linewidth=2, color="tab:purple", label="Mean neuron vitality")
    axes[1].axvline(collapse_start, color="black", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Health / vitality")
    axes[1].set_title("Global health versus internal senescence")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(ages, v1_damage, linewidth=2, color="tab:orange", label="v1 damage ratio")
    axes[2].plot(ages, v2_results["damage"], linewidth=2, color="tab:brown", label="v2 cumulative damage")
    axes[2].axvline(collapse_start, color="black", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Damage")
    axes[2].set_title("Damage accumulation")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(ages, v2_results["dead_fraction"], linewidth=2, color="tab:gray")
    axes[3].axvline(collapse_start, color="black", linestyle="--", alpha=0.7)
    axes[3].set_ylabel("Dead fraction")
    axes[3].set_xlabel("Age / time-step")
    axes[3].set_title("v2 irreversible loss of hidden units")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=== Intrinsic Mortality PoC v2 ===")
    print("Chosen strategy       : cumulative hidden-neuron senescence")
    print("Dataset               : Iris")
    print("Architecture          : 4 -> 16 -> 3 MLP with ReLU")
    print(
        "v2 mechanism          : persistent per-neuron vitality, activation-stress-driven "
        "damage, gain loss, precision loss, irreversible neuron death"
    )
    print(f"Training loss epochs  : {len(loss_history)}")
    print(f"Baseline test acc     : {baseline_acc:.3f}")
    print(summarize_stage("v1 stable", v1_acc[stable_mask]))
    print(summarize_stage("v1 terminal", v1_acc[terminal_mask]))
    print(summarize_stage("v2 stable", v2_results["accuracy"][stable_mask]))
    print(summarize_stage("v2 terminal", v2_results["accuracy"][terminal_mask]))
    print(f"v1 final performance  : {v1_acc[-1]:.3f}")
    print(f"v2 final performance  : {v2_results['accuracy'][-1]:.3f}")
    print(f"v2 vitality start/end : {v2_results['mean_vitality'][0]:.3f} -> {v2_results['mean_vitality'][-1]:.3f}")
    print(f"v2 dead frac start/end: {v2_results['dead_fraction'][0]:.3f} -> {v2_results['dead_fraction'][-1]:.3f}")
    print(f"Collapse onset age    : {collapse_start}")
    print(f"Saved figure          : {figure_path}")
    print("Comparison v1 -> v2   : v1 perturbs effective parameters; v2 accumulates irreversible")
    print("                        internal senescence in hidden units across age steps.")
    print("Endogeneity comment   : v2 is harder to frame as an external perturbation because")
    print("                        damage persists as internal state and directly degrades")
    print("                        representation quality and remaining model capacity.")


if __name__ == "__main__":
    main()
