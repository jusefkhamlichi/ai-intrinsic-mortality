# Model Definition: Intrinsic Mortality via Cumulative Neuronal Senescence

## 1. Overview

Consider a feed-forward neural classifier with fixed trained parameters and a hidden layer of `m` neurons. After training, the model enters an aging process indexed by discrete time `t = 0, 1, ..., T`. The key assumption is that each hidden neuron carries an internal senescence state that degrades irreversibly over time.

The model is therefore described not only by its trained parameters, but also by an internal state of cumulative damage and vitality. This makes deterioration endogenous: degradation is part of the model state, not only an external perturbation applied at inference time.

## 2. State Variables

Let:

- `x in R^d` be an input.
- `W1, b1, W2, b2` be fixed trained parameters.
- `t` be the age or lifecycle step.
- `h(t) in [0,1]` be a global health schedule, high for most of the lifespan and sharply decreasing near end-of-life.
- `p(t)` be the global aging pressure, derived from `h(t)`.
- `D_i(t) >= 0` be the cumulative damage of hidden neuron `i`.
- `v_i(t) in [0,1]` be the vitality of hidden neuron `i`.

For the hidden layer, `i = 1, ..., m`.

The model state at time `t` is:

```text
S(t) = (W1, b1, W2, b2, D(t), v(t))
```

where `D(t) = (D_1(t), ..., D_m(t))` and `v(t) = (v_1(t), ..., v_m(t))`.

## 3. Base Network

Before aging, the hidden representation is:

```text
z_i(x) = (W1[:, i])^T x + b1_i
r_i(x) = ReLU(z_i(x))
```

and the output logits are:

```text
ell(x) = W2^T r(x) + b2
```

with prediction obtained from `argmax ell(x)`.

## 4. Global Aging Schedule

The global health schedule is chosen to remain close to 1 for most of the lifetime and to drop rapidly near the end:

```text
h(t) = 1 / (1 + exp(k * (t/T - c)))
```

where:

- `T` is the maximum age,
- `c in (0,1)` is the collapse onset,
- `k > 0` controls steepness.

Define the aging pressure as an increasing function of lost health:

```text
p(t) = (1 - h(t))^gamma
```

with `gamma > 1`, so that aging pressure remains small for a long time and grows sharply near the end.

## 5. Damage Accumulation

Each neuron accumulates irreversible damage based on:

- global aging pressure,
- its own usage or activation stress,
- its current fragility.

Let `s_i(t)` be the normalized activation stress of neuron `i` at time `t`, with `s_i(t) in [0,1]`.

Let the per-step damage increment be:

```text
Delta D_i(t) = p(t) * (a + b s_i(t)) * (u + (1-u)(1 - v_i(t)))
```

with constants `a, b > 0`, `a + b = 1`, and `u in (0,1)`.

Then cumulative damage evolves as:

```text
D_i(t+1) = D_i(t) + Delta D_i(t)
```

with initialization:

```text
D_i(0) = 0
```

This update is monotone: damage never decreases.

## 6. Vitality Update

Vitality is a decreasing function of cumulative damage. A simple form is:

```text
v_i(t) = exp(-lambda D_i(t))
```

with `lambda > 0`.

Hence:

- `v_i(0) = 1`,
- `v_i(t)` decreases as damage grows,
- `v_i(t)` cannot increase unless damage is reduced,
- in the current model, damage is never reduced.

Therefore vitality loss is irreversible.

## 7. Impact on Internal Representations

Vitality affects the hidden representation in three ways.

### 7.1 Gain loss

The effective activation amplitude of neuron `i` is reduced:

```text
g_i(t) = v_i(t)^alpha
```

with `alpha > 0`, and

```text
r_i^g(x,t) = g_i(t) r_i(x)
```

### 7.2 Saturation

Low-vitality neurons have reduced dynamic range. Let:

```text
c_i(t) = c0 (beta + v_i(t))
```

with constants `c0 > 0` and `beta > 0`. Then:

```text
r_i^s(x,t) = min(r_i^g(x,t), c_i(t))
```

### 7.3 Precision loss

Low-vitality neurons transmit less precise internal values. Let `q_i(t)` be the number of usable quantization levels:

```text
q_i(t) = q_min + round((q_max - q_min) v_i(t))
```

Then the effective hidden activity becomes:

```text
r_i^*(x,t) = Q(r_i^s(x,t); q_i(t))
```

where `Q(.; q)` is a scalar quantization operator with `q` levels.

### 7.4 Functional death

If vitality drops below a threshold `theta_dead`, the neuron is considered dead:

```text
if v_i(t) <= theta_dead, then r_i^*(x,t) = 0
```

Thus the effective hidden representation shrinks over time.

## 8. Aged Model

The aged hidden vector is:

```text
r^*(x,t) = (r_1^*(x,t), ..., r_m^*(x,t))
```

and the aged logits are:

```text
ell(x,t) = W2^T r^*(x,t) + b2
```

The classifier at age `t` is therefore:

```text
f_t(x) = argmax ell(x,t)
```

## 9. Alive, Degraded, Dead

The model is **alive** when:

- most neurons retain high vitality,
- the hidden representation preserves useful amplitude and precision,
- task performance remains close to baseline.

The model is **degraded** when:

- a nontrivial fraction of neurons has reduced vitality,
- effective gain and precision are lower,
- representational capacity is reduced,
- performance is lower but still above unusable levels.

The model is **dead** when:

- most of the hidden layer has either very low vitality or is functionally dead,
- the hidden representation has collapsed in amplitude, precision, or dimensionality,
- predictive performance is near chance or otherwise operationally unusable.

## 10. Why Collapse Is Inevitable

The collapse follows from the structure of the dynamics:

1. `p(t)` is nonnegative and becomes large near end-of-life.
2. `Delta D_i(t) >= 0` for all neurons and times.
3. Therefore `D_i(t)` is nondecreasing.
4. Since `v_i(t) = exp(-lambda D_i(t))`, vitality is nonincreasing.
5. Lower vitality reduces gain, reduces precision, lowers saturation ceiling, and eventually kills neurons.
6. These effects reduce the usable hidden representation and therefore the information available to the output layer.

Thus, absent an explicit repair mechanism, the system is structurally driven toward collapse.

## 11. Why the Model Cannot Self-Recover

In the current formulation there is no recovery term:

- damage only accumulates,
- vitality is only a decreasing function of damage,
- dead neurons do not reactivate,
- no plasticity or retraining is allowed after aging begins.

Therefore the system has no endogenous route back toward a healthier state. Its dynamics are one-directional.

## 12. Why the Model Produces Long Stability Plus Rapid Collapse

The empirical pattern follows directly from the equations.

For most of the lifetime:

- `h(t)` is close to 1,
- `p(t)` is very small,
- damage increments `Delta D_i(t)` remain small,
- vitality stays near 1,
- gain, precision, and active dimensionality remain almost unchanged.

Near the collapse onset:

- `h(t)` drops rapidly,
- `p(t)` grows nonlinearly,
- damage increments increase sharply,
- vitality decays faster,
- more neurons become weak or dead,
- internal representations lose both quality and capacity.

This creates a long stable regime followed by a sharp terminal decline.

## 13. Limitations of the Model

This formulation is intentionally minimal. It does not yet solve several deeper issues:

- It does not prove ineliminability: the mechanism is defined by construction and could still be removed by redesigning the model.
- It does not include repair, adaptation, or competitive compensation between neurons.
- It uses a simple hidden-layer view and does not address deeper architectures.
- It treats aging as an imposed internal law, not as something derived from first principles.
- It does not model uncertainty, multi-seed variability, or statistical confidence.
- It does not yet establish whether the mechanism remains robust under retraining, fine-tuning, or adversarial circumvention.

Even with these limits, the model gives a compact formal definition of intrinsic mortality as irreversible, cumulative, internal loss of representational vitality leading to functional collapse.
