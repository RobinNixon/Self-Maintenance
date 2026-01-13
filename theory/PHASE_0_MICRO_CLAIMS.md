# Phase 0: Calibration Micro-Claims

## What Do We Mean By "Maintains Itself"?

Before defining "aliveness," we must ground the concept in testable empirical claims. These micro-claims capture different aspects of what self-maintenance might mean in discrete dynamical systems.

---

## Micro-Claim 1: Structural Persistence Beyond Random Decay

**Claim:** A self-maintaining system exhibits pattern structures that persist longer than equivalent structures would in a random or purely dissipative system.

**Testable prediction:** Measure the half-life of recognizable patterns (boundaries, gliders, stable regions) in sticky ECAs vs. a null model where hidden state is randomized each step. Self-maintaining systems should show significantly longer pattern lifetimes.

**Operationalization:** Pattern lifetime ratio = T_sticky / T_random > 1

---

## Micro-Claim 2: Perturbation Recovery

**Claim:** A self-maintaining system returns to a recognizable attractor state after localized perturbation.

**Testable prediction:** Introduce single-cell perturbations to a running sticky ECA. Measure the fraction of perturbations that are "absorbed" (system returns to similar trajectory) vs. "catastrophic" (system diverges permanently). Self-maintaining systems should have high absorption rates.

**Operationalization:** Recovery rate = (absorbed perturbations) / (total perturbations) > threshold

---

## Micro-Claim 3: Active vs. Passive Stability

**Claim:** Self-maintenance requires ongoing computational activity, not mere passive stability.

**Testable prediction:** Static configurations (all 0s, all 1s, checkerboard) are passively stable but not "self-maintaining." Self-maintaining systems should show continuous internal dynamics even when the macroscopic pattern appears stable.

**Operationalization:** Internal activity = (cells with hidden state changes) / (total cells) > 0 even when visible pattern is stable.

---

## Micro-Claim 4: Boundary Integrity Under Stress

**Claim:** A self-maintaining system preserves its boundary structures when subjected to external perturbation.

**Testable prediction:** Introduce noise along boundaries of sticky ECA regions. Measure whether boundaries reform at approximately the same locations vs. dissolving or migrating randomly. Self-maintaining systems should show boundary repair.

**Operationalization:** Boundary persistence index = (post-perturbation boundary overlap with pre-perturbation) / (expected by chance)

---

## Micro-Claim 5: Information Preservation

**Claim:** A self-maintaining system preserves information about its initial conditions longer than a non-maintaining system.

**Testable prediction:** Start two sticky ECAs with slightly different initial conditions. Measure how long the systems remain distinguishable. Self-maintaining systems should preserve distinguishability longer than purely chaotic systems.

**Operationalization:** Information decay time = timesteps until mutual information drops below threshold

---

## Micro-Claim 6: Metabolic Turnover

**Claim:** Self-maintenance involves continuous replacement of components while preserving overall structure ("Ship of Theseus" property).

**Testable prediction:** Track individual cell states over time. In a self-maintaining system, cells should frequently change state (turnover) while the overall pattern (e.g., boundary locations, region sizes) remains recognizable.

**Operationalization:** Turnover rate = (cell state changes per timestep) / (total cells), with pattern similarity remaining high.

---

## Micro-Claim 7: Homeostatic Regulation

**Claim:** A self-maintaining system resists changes to key observables, keeping them within bounds.

**Testable prediction:** Define observables like "fraction of active cells" or "number of boundaries." Perturb the system and measure how quickly these observables return to their pre-perturbation values. Self-maintaining systems should show faster return times.

**Operationalization:** Homeostatic strength = 1 / (relaxation time to equilibrium)

---

## Micro-Claim 8: Negentropy Maintenance

**Claim:** A self-maintaining system maintains or increases local order (negative entropy) against a background tendency toward disorder.

**Testable prediction:** Measure spatial entropy of the visible configuration over time. In self-maintaining systems, entropy should remain bounded or decrease in localized regions, even as global dynamics continue.

**Operationalization:** Local negentropy = H_max - H_observed in boundary-adjacent regions

---

## Micro-Claim 9: Context-Dependent Response

**Claim:** A self-maintaining system responds differently to the same perturbation depending on its current state (context-dependence enabled by Control).

**Testable prediction:** Apply identical perturbations at different times/locations. Self-maintaining systems should show variable responses depending on hidden state, while non-maintaining systems respond identically.

**Operationalization:** Response variance = std(outcomes for identical perturbations) > 0

---

## Micro-Claim 10: Causal Closure

**Claim:** A self-maintaining system's dynamics are causally influenced by the structure it is maintaining (circular causation).

**Testable prediction:** Measure whether boundary presence affects hidden state evolution, and whether hidden state affects boundary formation. Self-maintaining systems should show bidirectional causation.

**Operationalization:** Granger causality test: boundaries -> hidden state AND hidden state -> boundaries

---

## Summary Table

| # | Micro-Claim | Key Observable | Threshold Type |
|---|-------------|----------------|----------------|
| 1 | Structural Persistence | Pattern lifetime ratio | Ratio > 1 |
| 2 | Perturbation Recovery | Recovery rate | Rate > threshold |
| 3 | Active Stability | Internal activity | Activity > 0 |
| 4 | Boundary Integrity | Boundary persistence index | Index > chance |
| 5 | Information Preservation | Information decay time | Time > null model |
| 6 | Metabolic Turnover | Turnover rate + pattern similarity | Both high |
| 7 | Homeostatic Regulation | Relaxation time | Time < threshold |
| 8 | Negentropy Maintenance | Local entropy | Entropy bounded |
| 9 | Context-Dependent Response | Response variance | Variance > 0 |
| 10 | Causal Closure | Bidirectional Granger causality | Both directions significant |

---

## Design Principles for Phase 1

These micro-claims suggest that "aliveness" in our framework likely requires:
1. **Active dynamics** (not passive stability)
2. **Hidden state** (enables context-dependence)
3. **Boundaries** (loci of Control and self-maintenance)
4. **Circular causation** (structure maintains itself through feedback)

The question becomes: which subset of these micro-claims is necessary, which is sufficient, and what is the minimal combination that captures "life-like" behavior?
