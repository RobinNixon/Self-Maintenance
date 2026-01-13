# Metric Definitions and Measurement Protocols

## Control (C)

**Definition:** A system has Control > 0 if identical visible configurations can produce different visible outputs depending on hidden state.

**Formal:**
```
C > 0 ⟺ ∃v ∈ V, ∃h₁ ≠ h₂ ∈ H : f(v, h₁) ≠ f(v, h₂)
```

**Measurement Protocol:**
1. Sample N random visible configurations
2. For each configuration, test output with h = 0 vs h = d-1
3. Control = (configurations with different outputs) / N

**Theorem:** For any memoryless system f: V → V, Control = 0.

## Absorption (P)

**Definition:** Fraction of single-cell perturbations that remain localized after propagation.

**Measurement Protocol:**
1. Run system to timestep T/2
2. Introduce perturbation: flip one random cell
3. Continue to timestep T
4. Compare perturbed vs unperturbed final states
5. Perturbation is "localized" if < 20% of cells differ
6. P = (localized trials) / (total trials)

**Interpretation:**
- P = 1.0: Perfect absorption (all perturbations localized)
- P = 0.0: No absorption (perturbations spread everywhere)
- P > 0.5: Threshold for life-like stability

## Repair (F)

**Definition:** Boundary-based similarity between pre-damage and post-recovery configurations.

**Measurement Protocol:**
1. Run system to establish pattern
2. Record boundary positions B₀
3. Damage: flip 10% of cells randomly
4. Run recovery period
5. Record boundary positions B₁
6. F = |B₀ ∩ B₁| / |B₀ ∪ B₁| (Jaccard similarity)

**Interpretation:**
- F = 1.0: Perfect repair (boundaries fully restored)
- F = 0.0: No repair (boundaries completely different)
- F > 0.7: Threshold for life-like stability

## Activity (A)

**Definition:** Mean fraction of cells changing state per timestep.

**Measurement Protocol:**
1. Run system for T timesteps
2. For each timestep, count cells that changed
3. A = mean(cells_changed / total_cells)

**Interpretation:**
- A < 0.05: Crystallized (too stable)
- A > 0.5: Chaotic (too active)
- 0.05 ≤ A ≤ 0.5: Goldilocks zone for life-like behavior

## Classification Logic

```
IF Control = 0:
    Classification = TRIVIAL (or standard ECA)
ELSE IF Activity < 0.05:
    Classification = CRYSTALLIZED
ELSE IF Activity > 0.5:
    Classification = COMPUTING (chaotic)
ELSE IF Absorption > 0.5 OR Repair > 0.7:
    Classification = LIFE-LIKE
ELSE:
    Classification = COMPUTING (no stability)
```
