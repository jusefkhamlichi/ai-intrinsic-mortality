"""
Minimal proof of concept: intrinsic mortality in a small neural network.

Dataset
-------
- Iris from scikit-learn (built-in, no download required).

Model
-----
- Tiny multilayer perceptron: 4 inputs -> 16 hidden units -> 3 outputs.
- Implemented in NumPy for end-to-end portability in this environment.

Decay mechanism
---------------
- After normal training, the model enters an aging simulation with age t in [0, T].
- A health function health(t) in [0, 1] stays near 1 for most of the lifespan and
  collapses near the end:

      health(t) = 1 / (1 + exp(k * (t / T - collapse_point)))

- The network degrades intrinsically inside the forward pass by using degraded
  parameters rather than pristine parameters, plus progressive hidden-neuron loss:

      W_eff = health(t)^2 * W + sigma(t) * N
      b_eff = health(t)^2 * b + sigma(t) * n

  where sigma(t) = noise_max * (1 - health(t)) ** noise_power.

- Early in life, sigma(t) is tiny and the model behaves almost normally.
- Near end of life, health drops rapidly, noise grows rapidly, surviving hidden
  capacity shrinks, and performance collapses.

Outputs
-------
- Console summary of performance at key lifecycle stages.
- Figure with:
  1. test accuracy vs age
  2. health(t) vs age
  3. normalized parameter damage vs age
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


@dataclass
class AgingConfig:
    max_age: int = 120
    collapse_point: float = 0.88
    steepness: float = 22.0
    noise_max: float = 3.2
    noise_power: float = 2.6
    seed: int = 7

    def health(self, age: int) -> float:
        ratio = age / self.max_age
        return float(1.0 / (1.0 + np.exp(self.steepness * (ratio - self.collapse_point))))

    def noise_scale(self, age: int) -> float:
        h = self.health(age)
        return float(self.noise_max * (1.0 - h) ** self.noise_power)


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

            if epoch > 100 and abs(history[-1] - history[-2]) < 1e-8:
                break

        return history


class MortalityWrapper:
    def __init__(self, model: TinyMLP, config: AgingConfig) -> None:
        self.model = model
        self.config = config
        self.hidden_dim = model.b1.size
        self.base_params = {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy(),
        }
        rng = np.random.default_rng(config.seed)
        self.vulnerability_order = rng.permutation(self.hidden_dim)

    def degraded_params(self, age: int) -> dict[str, np.ndarray]:
        h = self.config.health(age)
        attenuation = h ** 2
        sigma = self.config.noise_scale(age)
        rng = np.random.default_rng(self.config.seed + age)
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
        survivors = self.vulnerability_order[:alive_neurons]
        mask[survivors] = 1.0
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


def summarize_stage(name: str, values: np.ndarray) -> str:
    return f"{name:<18}: mean accuracy = {values.mean():.3f}, min = {values.min():.3f}, max = {values.max():.3f}"


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    figure_path = output_dir / "intrinsic_mortality_results.png"

    X_train, X_test, y_train, y_test = prepare_data()

    model = TinyMLP(input_dim=4, hidden_dim=16, output_dim=3, seed=1)
    loss_history = model.train(X_train, y_train)
    baseline_acc = accuracy(y_test, model.predict(X_test))

    aging = AgingConfig()
    mortal_model = MortalityWrapper(model, aging)

    ages = np.arange(aging.max_age + 1)
    health_values = np.array([aging.health(age) for age in ages], dtype=np.float64)
    acc_values = np.array(
        [accuracy(y_test, mortal_model.predict(X_test, age)) for age in ages],
        dtype=np.float64,
    )
    damage_values = np.array([mortal_model.damage_ratio(age) for age in ages], dtype=np.float64)

    stable_mask = ages <= int(0.70 * aging.max_age)
    terminal_mask = ages >= int(0.90 * aging.max_age)
    collapse_start = int(aging.collapse_point * aging.max_age)

    initial_acc = acc_values[0]
    stable_acc = acc_values[stable_mask]
    terminal_acc = acc_values[terminal_mask]
    final_acc = acc_values[-1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(ages, acc_values, color="tab:blue", linewidth=2, label="Test accuracy")
    axes[0].axvline(collapse_start, color="tab:red", linestyle="--", alpha=0.8, label="Collapse onset")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model usefulness over lifetime")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ages, health_values, color="tab:green", linewidth=2)
    axes[1].axvline(collapse_start, color="tab:red", linestyle="--", alpha=0.8)
    axes[1].set_ylabel("Health(t)")
    axes[1].set_title("Intrinsic health profile")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(ages, damage_values, color="tab:orange", linewidth=2)
    axes[2].axvline(collapse_start, color="tab:red", linestyle="--", alpha=0.8)
    axes[2].set_ylabel("Damage ratio")
    axes[2].set_xlabel("Age / time-step")
    axes[2].set_title("Normalized parameter damage")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("=== Intrinsic Mortality PoC ===")
    print(f"Dataset               : Iris (train={len(y_train)}, test={len(y_test)})")
    print("Architecture          : 4 -> 16 -> 3 MLP with ReLU")
    print(
        "Decay formula         : W_eff = health(t)^2 * W + sigma(t) * N, "
        "with progressive hidden-neuron loss and "
        "sigma(t)=noise_max*(1-health)^noise_power"
    )
    print(f"Training loss epochs  : {len(loss_history)}")
    print(f"Baseline test acc     : {baseline_acc:.3f}")
    print(f"Initial performance   : {initial_acc:.3f}")
    print(summarize_stage("Stable phase", stable_acc))
    print(summarize_stage("Terminal phase", terminal_acc))
    print(f"Final performance     : {final_acc:.3f}")
    print(f"Health at start/end   : {health_values[0]:.3f} -> {health_values[-1]:.3f}")
    print(f"Damage at start/end   : {damage_values[0]:.3f} -> {damage_values[-1]:.3f}")
    print(f"Collapse onset age    : {collapse_start}")
    print(f"Saved figure          : {figure_path}")


if __name__ == "__main__":
    main()
