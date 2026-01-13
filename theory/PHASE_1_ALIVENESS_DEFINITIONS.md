# Phase 1: Candidate Operational Definitions of "Aliveness"

## The Challenge

We seek operational definitions that distinguish "life-like" behavior from "merely computing" in discrete dynamical systems. Each definition must be:
1. **Measurable** in sticky ECAs
2. **Discriminative** (not trivially satisfied by all systems)
3. **Grounded** in the micro-claims from Phase 0

---

## Definition A1: Autopoietic Index

**Inspiration:** Maturana & Varela's autopoiesis - a system that produces the components that produce itself.

**Operational Definition:**
A system S has autopoietic index A(S) = min(R_boundary, R_hidden, C_feedback) where:
- R_boundary = rate at which boundary structures are regenerated after perturbation
- R_hidden = rate at which hidden state patterns are restored after perturbation
- C_feedback = bidirectional Granger causality strength between boundaries and hidden state

**Threshold for "alive":** A(S) > 0.5

**Rationale:** Autopoiesis requires that boundaries create hidden state patterns AND hidden state patterns create boundaries (circular causation). The minimum captures the weakest link in the self-production loop.

**Measurement Protocol:**
1. Run sticky ECA for 100 steps to establish baseline
2. Introduce localized perturbation (flip 5% of cells + reset hidden state)
3. Measure timesteps until boundary pattern similarity exceeds 80% of pre-perturbation
4. Measure timesteps until hidden state distribution matches pre-perturbation
5. Compute Granger causality in both directions
6. Report A(S) = min of the three normalized scores

---

## Definition A2: Integrated Information Index (Phi-like)

**Inspiration:** Tononi's Integrated Information Theory - a system is more "alive" if its parts are more informationally integrated.

**Operational Definition:**
A system S has integration index I(S) = MI(left_half; right_half | boundary) where:
- MI = mutual information
- Conditioning on boundary presence tests whether information flows THROUGH boundaries

**Threshold for "alive":** I(S) > I_random (integration exceeds shuffled null model)

**Rationale:** Living systems are integrated wholes, not collections of independent parts. Boundaries should serve as information channels, not barriers.

**Measurement Protocol:**
1. Run sticky ECA for 100 steps
2. Partition lattice into left and right halves
3. Compute mutual information between half-states at each timestep
4. Condition on whether a boundary crosses the midpoint
5. Compare to null model where hidden states are shuffled
6. Report I(S) = (observed MI) / (shuffled MI)

---

## Definition A3: Metabolic Persistence Ratio

**Inspiration:** Schrodinger's "negentropy" - life maintains order through continuous energy expenditure.

**Operational Definition:**
A system S has metabolic persistence M(S) = (pattern_similarity over time) * (cell_turnover_rate)

**Threshold for "alive":** M(S) in "Goldilocks zone" - not too stable (dead), not too chaotic (dissolving)

**Rationale:** A truly self-maintaining system has high turnover (metabolism) while maintaining pattern (persistence). Low turnover + high persistence = dead crystal. High turnover + low persistence = dissipating chaos.

**Measurement Protocol:**
1. Run sticky ECA for 200 steps
2. Compute pattern similarity between t and t+50 (using boundary positions and region sizes)
3. Compute average cell state changes per timestep
4. Report M(S) = similarity * turnover_rate
5. "Alive" zone: 0.1 < M(S) < 0.9 (neither crystallized nor chaotic)

---

## Definition A4: Perturbation Absorption Capacity

**Inspiration:** Homeostasis - living systems maintain internal stability despite external perturbation.

**Operational Definition:**
A system S has absorption capacity P(S) = fraction of perturbations that are "absorbed" rather than propagating indefinitely or causing system collapse.

**Threshold for "alive":** P(S) > 0.7 (most perturbations are absorbed)

**Rationale:** Living systems are robust to small perturbations. They neither ignore perturbations (no response = dead) nor amplify them without bound (chaos).

**Measurement Protocol:**
1. Run sticky ECA for 50 steps to reach steady-state dynamics
2. Introduce 20 single-cell perturbations at random locations/times
3. For each perturbation, track whether:
   - Perturbation effect remains localized (absorbed)
   - Perturbation effect spreads without bound (not absorbed)
4. Report P(S) = (absorbed count) / 20

---

## Definition A5: Control-Boundary Coupling Strength

**Inspiration:** Our own Stickiness-Control framework - Control concentrates at boundaries.

**Operational Definition:**
A system S has coupling strength K(S) = correlation(boundary_presence, control_magnitude) * mean(control_at_boundaries)

**Threshold for "alive":** K(S) > K_threshold where threshold separates "merely computing" from "life-like"

**Rationale:** In "merely computing" systems, Control (if any) is distributed randomly. In "life-like" systems, Control is concentrated at boundaries because boundaries ARE the self-maintaining structures.

**Measurement Protocol:**
1. Run sticky ECA for 100 steps
2. At each cell and timestep, compute:
   - Is this cell at a boundary? (binary)
   - What is the counterfactual Control at this cell?
3. Compute Pearson correlation between boundary presence and Control
4. Compute mean Control at boundary cells
5. Report K(S) = correlation * mean_boundary_control

---

## Definition A6: Temporal Coherence Index

**Inspiration:** Bergson's "duration" - living systems have a characteristic temporal structure, not just spatial structure.

**Operational Definition:**
A system S has temporal coherence T(S) = autocorrelation_decay_time of boundary positions.

**Threshold for "alive":** T(S) > T_random (boundaries persist longer than random structures would)

**Rationale:** Living systems maintain identity over time. Their structures don't just exist at an instant but persist coherently. Boundary positions should be autocorrelated over time (same boundaries, moving predictably) rather than appearing randomly each timestep.

**Measurement Protocol:**
1. Run sticky ECA for 200 steps
2. At each timestep, record boundary positions as a binary vector
3. Compute autocorrelation of boundary positions at lags 1, 2, 5, 10, 20, 50
4. Fit exponential decay to get characteristic time T(S)
5. Compare to null model where boundary positions are shuffled temporally

---

## Definition A7: Self-Repair Fidelity

**Inspiration:** Biological regeneration - living systems can repair damage.

**Operational Definition:**
A system S has repair fidelity F(S) = similarity(pre-damage_pattern, post-recovery_pattern) after localized damage.

**Threshold for "alive":** F(S) > 0.8 (system returns to >80% similar state)

**Rationale:** A merely computing system damaged at time t will diverge to a completely different trajectory. A life-like system will return to a recognizably similar pattern because its attractors are robust.

**Measurement Protocol:**
1. Run sticky ECA for 100 steps, save state S0
2. Introduce damage: flip 10% of cells, reset hidden state in damaged region
3. Run for 50 more steps, save state S1
4. Compute similarity(S0[t=100], S1[t=150]) using boundary positions
5. Report F(S) = pattern similarity

---

## Summary: The Seven Candidates

| Definition | Symbol | Core Concept | Measurement Complexity |
|------------|--------|--------------|------------------------|
| Autopoietic Index | A(S) | Circular self-production | High |
| Integration Index | I(S) | Informational wholeness | Medium |
| Metabolic Persistence | M(S) | Turnover + stability | Low |
| Absorption Capacity | P(S) | Perturbation robustness | Medium |
| Control-Boundary Coupling | K(S) | Control at boundaries | Low |
| Temporal Coherence | T(S) | Persistence over time | Medium |
| Self-Repair Fidelity | F(S) | Recovery from damage | Medium |

---

## Hypothesis: Hierarchy of Aliveness

These definitions may form a hierarchy:

**Level 0 (Inert):** Standard ECA - no hidden state, no Control
- A = I = M = P = K = T = F = 0

**Level 1 (Computing):** Sticky ECA with dynamics but no self-maintenance
- K > 0, but A, P, F low (has Control but doesn't maintain itself)

**Level 2 (Proto-alive):** Sticky ECA with some self-maintenance
- K, P, T > threshold (maintains boundaries, absorbs perturbations)

**Level 3 (Alive):** Sticky ECA with full autopoiesis
- All metrics above threshold (circular causation, repair, integration)

The experiments in Phase 3 will test whether this hierarchy exists empirically.

---

## Selection for Phase 2

For measurement protocols, we will implement all seven definitions but prioritize:
1. **K(S)** - Control-Boundary Coupling (most directly connected to our theory)
2. **P(S)** - Absorption Capacity (captures homeostasis)
3. **M(S)** - Metabolic Persistence (captures the living vs. crystallized distinction)
4. **F(S)** - Self-Repair Fidelity (most directly tests "maintains itself")

These four capture complementary aspects: K is structural, P is dynamic, M is energetic, F is functional.
