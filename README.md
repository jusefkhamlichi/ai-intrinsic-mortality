# Meaningful AI Mortality

Minimal public repository for the intrinsic mortality proof of concept and the associated conceptual framework.

## Scope

This repository contains:
- PoC v1: intrinsic mortality via effective parameter corruption and progressive loss of hidden neurons.
- PoC v2: intrinsic mortality via cumulative neuronal senescence with persistent per-neuron vitality.
- Attack simulations showing that intrinsic mortality is bypassable at system level via restore, cloning, and code bypass.
- Core theory notes on model degradation, identity, and meaningful mortality.
- LaTeX source and PDF for the paper.

## Repository structure

- `simulations/code`: executable PoC scripts and attack experiments.
- `simulations/results`: figures and summary outputs.
- `theory/core`: formal model definition and mortality framework.
- `theory/identity`: identity-focused note.
- `theory/security`: attack-surface note.
- `paper/tex`: LaTeX source and compiled PDF.

## Technical setup

- Dataset: Iris (`scikit-learn`)
- Architecture: MLP `4 -> 16 -> 3`
- Implementation: NumPy + matplotlib + scikit-learn

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the main scripts:

```powershell
python simulations/code/intrinsic_mortality_poc.py
python simulations/code/intrinsic_mortality_poc_v2.py
python simulations/code/intrinsic_mortality_attack_experiments.py
```

## Main empirical result

The project shows that intrinsic mortality can be made endogenous at the model-instance level, but remains local. Restore, cloning, and implementation-level bypass preserve continuity beyond the death of any single execution trajectory.

## Citation

Paper archived on Zenodo:

Jusef Khamlichi (2026). *Toward Meaningful Mortality in AI Systems: Intrinsic Degradation, Identity, and Succession*. Zenodo. https://doi.org/10.5281/zenodo.19556905

## Related paper files

- `paper/tex/meaningful_ai_mortality_v4.tex`
- `paper/tex/meaningful_ai_mortality_v4.pdf`
