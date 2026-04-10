# Intrinsic Mortality PoC

Minimal proof of concept for "intrinsic mortality" in a small neural network.

The repository contains two iterations:

- `v1`: weight corruption plus progressive loss of hidden neurons.
- `v2`: cumulative internal senescence with per-neuron vitality and irreversible damage.

The goal is not industrial performance. The goal is to show a clear lifecycle pattern:

- high usefulness for most of the model lifetime,
- sharp late-stage decline,
- final functional death.

## Repository contents

- [`intrinsic_mortality_poc.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc.py): first PoC version.
- [`intrinsic_mortality_poc_v2.py`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/intrinsic_mortality_poc_v2.py): second PoC version with cumulative neuronal senescence.
- [`project_summary.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/project_summary.md): project summary for v1.
- [`v2_summary.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/v2_summary.md): technical summary for v2.
- [`v2_executive_summary.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/v2_executive_summary.md): executive-style recap.
- [`model_definition_v2.md`](c:/Users/Utente/OneDrive%20-%20jusef%20khamlichi/LAVORO/AI_security/model_definition_v2.md): compact formal definition of the v2 model.
- `intrinsic_mortality_results.png`: v1 plots.
- `intrinsic_mortality_v2_results.png`: v2 plots.

## Technical choices

- Dataset: `Iris` from `scikit-learn`.
- Model: small MLP `4 -> 16 -> 3`.
- Framework: NumPy, chosen for portability and readability in the current environment.
- Health profile: high and nearly flat for most of the lifetime, then rapid collapse near end-of-life.

### v1 mechanism

The first version degrades effective parameters during inference:

```text
W_eff = health(t)^2 * W + sigma(t) * N
b_eff = health(t)^2 * b + sigma(t) * n
sigma(t) = noise_max * (1 - health(t)) ^ noise_power
```

It also progressively removes hidden neurons according to a fixed vulnerability order.

### v2 mechanism

The second version moves from external-looking perturbation toward internal senescence:

- each hidden neuron has persistent vitality `v_i(t)`,
- damage accumulates irreversibly over age,
- vitality reduces activation gain and precision,
- low-vitality neurons eventually become functionally dead.

This makes collapse more endogenous because the model carries its own internal aging state.

## Installation

Python 3.11+ is sufficient. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Run

Run v1:

```powershell
python intrinsic_mortality_poc.py
```

Run v2:

```powershell
python intrinsic_mortality_poc_v2.py
```

## Expected outputs

Each script prints a lifecycle summary to console and saves a figure:

- `intrinsic_mortality_results.png` for v1
- `intrinsic_mortality_v2_results.png` for v2

The plots show the stable phase, late-stage collapse, and final death of the model.
