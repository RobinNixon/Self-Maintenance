# Self-Maintenance as a Default Outcome in Hidden-State Discrete Dynamical Systems

## Abstract

We present a classification result for elementary cellular automata (ECA) augmented with hidden state: 83.7% of non-trivial rules exhibit life-like behavior characterized by active self-maintenance. This inverts the standard assumption that self-maintenance is rare or requires special conditions. We show that hidden state is necessary for Control (counterfactual context-dependence), but Control alone is not sufficient for life-like behavior. Life-like behavior additionally requires a stability mechanism (perturbation absorption or self-repair) and dynamics within an activity window. These three conditions—hidden state, stability substrate, and activity balance—are jointly necessary and sufficient within the class of deterministic, discrete-time, local dynamical systems studied here. Notably, computational universality (Rule 110) and maximal chaos (Rule 30) both fail to produce life-like behavior, demonstrating that self-maintenance is orthogonal to computational power and dynamical complexity.

---

## 1. Introduction

### 1.1 The Standard Narrative

The emergence of self-maintaining systems from non-living substrates is typically treated as an improbable event requiring special conditions. This view shapes research programs in artificial life, origins-of-life studies, and complex systems theory. The implicit assumption is that self-maintenance is rare: most dynamical systems do not maintain themselves.

We challenge this assumption with a systematic classification of elementary cellular automata augmented with hidden state. Our central finding: **83.7% of non-trivial rules exhibit life-like behavior** when hidden state is introduced through a stickiness mechanism.

### 1.2 Why Cellular Automata with Hidden State

Elementary cellular automata (ECA) provide an ideal testbed for several reasons:

1. **Completeness:** All 256 rules can be exhaustively enumerated
2. **Simplicity:** Rules are fully specified by 8 bits
3. **Hidden state:** The stickiness mechanism adds internal state in a controlled way
4. **Measurement:** Life-like properties can be operationally defined and computed

Standard ECAs are memoryless: the next state depends only on the current visible configuration. By adding hidden state through stickiness (confirmation counters that must reach threshold before changes take effect), we create systems where the same visible configuration can produce different outcomes depending on internal history.

### 1.3 What This Paper Resolves

We establish three results:

**Result 1 (Necessity):** Hidden state is necessary for Control. Without hidden state, a deterministic local rule produces identical outputs for identical visible inputs. Control = 0 for all standard ECAs.

**Result 2 (Insufficiency):** Hidden state is not sufficient for life-like behavior. Of 190 non-trivial sticky ECAs, 28 (14.7%) have Control > 0 but are not life-like because they lack stability mechanisms.

**Result 3 (Classification):** Life-like behavior = Control + Stability + Activity. Specifically:
- Control > 0 (hidden state influences output)
- Absorption > 0.5 OR Repair > 0.7 (stability mechanism)
- 0.05 ≤ Activity ≤ 0.5 (Goldilocks dynamics)

This classification achieves 100% accuracy on the census: all rules satisfying these conditions are life-like; no rule failing any condition is life-like.

---

## 2. Definitions and Metrics

### 2.1 Elementary Cellular Automata

An elementary cellular automaton is a triple (S, N, f) where:
- S = {0, 1} is the state space
- N = {-1, 0, 1} defines the neighborhood (left, center, right)
- f: S³ → S is the local update rule

Rules are indexed by Wolfram numbering: rule r applies f(l, c, r) = (r >> (4l + 2c + r)) & 1.

### 2.2 Stickiness Mechanism

The stickiness mechanism adds hidden state H = {0, 1, ..., d-1} (confirmation counter) to each cell. The augmented update rule is:

```
If f(l, c, r) ≠ c:          # Rule requests change
    h' = h + 1
    If h' ≥ d:              # Threshold reached
        v' = f(l, c, r)     # Apply change
        h' = 0              # Reset counter
    Else:
        v' = c              # Keep current value
Else:                       # Rule doesn't request change
    v' = c
    h' = 0                  # Reset counter
```

This creates context-dependence: the same visible neighborhood can produce different visible outputs depending on hidden state h.

### 2.3 Control

**Definition (Control):** A system has Control > 0 if there exist visible configurations v and hidden states h₁ ≠ h₂ such that the update produces different visible outputs: f(v, h₁) ≠ f(v, h₂).

**Measurement:** Sample random visible configurations and positions. For each, test whether hidden state h = 0 vs h = d-1 produces different outcomes. Control = fraction where outcomes differ.

**Theorem (Necessity):** For any memoryless deterministic system f: V → V, Control = 0.

*Proof:* If f depends only on visible state v, then f(v) is uniquely determined regardless of any putative hidden state. Thus f(v, h₁) = f(v, h₂) = f(v) for all h₁, h₂. ∎

### 2.4 Perturbation Absorption

**Definition:** Absorption P(S) = fraction of single-cell perturbations that remain localized (affect < 20% of cells) after propagation time T.

**Measurement:**
1. Run system to step T/2
2. Introduce perturbation: flip one random cell
3. Continue to step T
4. Compare perturbed vs unperturbed final states
5. P = (localized outcomes) / (total trials)

### 2.5 Self-Repair

**Definition:** Repair F(S) = boundary-based similarity between pre-damage and post-recovery configurations.

**Measurement:**
1. Run system to establish pattern
2. Record boundary positions B₀
3. Damage: flip 10% of cells
4. Run recovery period
5. Record boundary positions B₁
6. F = |B₀ ∩ B₁| / |B₀ ∪ B₁| (Jaccard similarity)

### 2.6 Activity

**Definition:** Activity A(S) = mean fraction of cells changing state per timestep.

**Measurement:** A = mean over timesteps of (cells changed / total cells).

### 2.7 Classification Logic

**Definition (Life-Like):** A system is life-like if and only if:
1. Control > 0
2. Absorption > 0.5 OR Repair > 0.7
3. 0.05 ≤ Activity ≤ 0.5

**Definition (Computing):** Control > 0 but fails stability or activity criteria.

**Definition (Crystallized):** Control > 0 and stability criteria met, but Activity < 0.05.

**Definition (Trivial):** Standard dynamics produce static, nilpotent, or uniform behavior.

---

## 3. Mechanism

### 3.1 The Causal Chain

```
Stickiness → Hidden State → Control → [Stability Substrate] → Life-Like
```

**Stage 1:** Stickiness introduces hidden state (confirmation counters).

**Stage 2:** Hidden state creates Control: the same visible configuration can produce different outputs depending on whether a cell is pending (h > 0) or stable (h = 0).

**Stage 3:** Control enables but does not guarantee life-like behavior. The underlying rule dynamics must provide a stability substrate that hidden state can leverage.

### 3.2 Why Control Is Not Sufficient

Control provides context-dependence: the system can respond differently to identical visible inputs. But this flexibility may be deployed for chaos (amplifying perturbations) rather than self-maintenance (absorbing perturbations).

**Example:** Rule 30 has Control = 0.52 under stickiness but Absorption = 0.03. The hidden state exists but the rule's chaotic dynamics spread perturbations regardless.

### 3.3 Stability Substrates

Two mechanisms enable life-like behavior:

**Absorption:** Some rules (especially linear/XOR-based) naturally cancel perturbations through superposition-like effects. Rule 90 has P = 1.0.

**Repair:** Some rules (especially particle-conserving) have strong attractors that pull the system back to recognizable states. Rule 184 has F = 0.93.

Rules lacking either mechanism cannot achieve life-like behavior regardless of Control.

### 3.4 The Activity Window

Life-like behavior requires dynamics in a Goldilocks range:

- **A < 0.05:** Crystallized. The stickiness mechanism overdamps dynamics. The system is stable but "dead."
- **A > 0.5:** Chaotic. Dynamics are too turbulent for stable patterns. Hidden state cannot anchor structure.
- **0.05 ≤ A ≤ 0.5:** Life-like zone. Active dynamics with structural persistence.

---

## 4. Census Results

### 4.1 Classification Distribution

| Classification | Count | % of All | % of Non-Trivial |
|---------------|-------|----------|------------------|
| **LIFE-LIKE** | **159** | **62.1%** | **83.7%** |
| COMPUTING | 28 | 10.9% | 14.7% |
| CRYSTALLIZED | 3 | 1.2% | 1.6% |
| TRIVIAL | 66 | 25.8% | — |
| **Total** | **256** | **100%** | **100%** |

### 4.2 Key Findings

**Finding 1:** Life-like behavior is the majority outcome among non-trivial rules (83.7%).

**Finding 2:** Only 28 rules (14.7%) have hidden state but fail to achieve life-like behavior due to lacking stability mechanisms.

**Finding 3:** Only 3 rules crystallize (become too stable). Overdamping is rare.

### 4.3 Famous Rules

| Rule | Classification | C | P | F | A |
|------|---------------|-----|-----|-----|-----|
| 30 | COMPUTING | 0.52 | 0.03 | 0.62 | 0.25 |
| 90 | LIFE-LIKE | 0.49 | 1.00 | 0.67 | 0.25 |
| 110 | COMPUTING | 0.39 | 0.30 | 0.47 | 0.20 |
| 184 | LIFE-LIKE | 0.53 | 0.77 | 0.93 | 0.44 |

**Rule 30** (Wolfram's chaotic example): Not life-like. Perturbations spread despite hidden state.

**Rule 90** (XOR/Sierpinski): Life-like. Linear dynamics enable perfect absorption.

**Rule 110** (Turing-complete): Not life-like. Computational universality does not confer self-maintenance.

**Rule 184** (traffic rule): Life-like. Particle conservation enables repair.

---

## 5. Falsification and Anomalies

### 5.1 The Rule 150 Anomaly

Rule 150 is XOR-based like Rule 90, yet:
- Rule 90: P = 1.00 (LIFE-LIKE)
- Rule 150: P = 0.00 (COMPUTING)

**Resolution:** Linearity is necessary but not sufficient for absorption. The specific bit pattern determines whether perturbations cancel or propagate. Rule 90's configuration enables cancellation; Rule 150's does not.

### 5.2 High-Control Failures

Rules 161, 151, 107, 97 have Control > 0.5 but are not life-like. They demonstrate that Control magnitude does not determine classification—only that Control > 0.

### 5.3 Borderline Cases

Six rules fall within 0.15 of threshold boundaries. These represent measurement uncertainty, not theoretical failure. Classification of borderline rules has ±1 category uncertainty.

### 5.4 Why Anomalies Do Not Refute

No rule satisfying all three conditions (Control > 0, Stability criterion, Activity window) fails to be life-like. No rule failing any condition achieves life-like status. The classification is complete within scope.

---

## 6. Discussion

### 6.1 Life-Like ≠ Computation

Rule 110 is Turing-complete yet not life-like. Computational universality is orthogonal to self-maintenance. A system can compute anything without maintaining itself; a system can maintain itself without universal computation.

### 6.2 Life-Like ≠ Chaos

Rule 30 exhibits maximal chaos yet is not life-like. Dynamical complexity does not confer self-maintenance. Chaotic systems may have rich dynamics but cannot stabilize against perturbation.

### 6.3 Life-Like ≠ Stability

Rules 108, 201, 216 have excellent stability metrics (P = 1.0) yet are not life-like because Activity < 0.05. Excessive stability is crystallization, not life. Self-maintenance requires active dynamics, not mere persistence.

### 6.4 Why Life-Like Is Common

Given the three-condition framework, the question is: why do 83.7% of non-trivial rules satisfy all three?

**Answer:**
1. Control > 0 is guaranteed by stickiness for any rule with dynamics
2. Activity window (0.05-0.5) captures most non-trivial dynamics
3. Stability mechanisms (absorption OR repair) are common because:
   - Many rules have linear or quasi-linear components enabling absorption
   - Many rules have attractor structure enabling repair
   - Only chaotic rules without either structure fail

The 14.7% failure rate represents the fraction of non-trivial rules that are chaotic *and* lack attractor structure—a specific and uncommon combination.

---

## 7. Implications

### 7.1 Artificial Life

Self-maintaining artificial systems do not require careful engineering of specific rules. Adding hidden state to almost any non-trivial dynamics produces life-like behavior. The design problem shifts from "how to achieve self-maintenance" to "how to avoid the 15% failure modes."

### 7.2 Origins of Life

If self-maintenance is generic once hidden state exists, origins-of-life scenarios need not explain why self-maintenance arose—only why hidden state arose. This reframes the question from "how did life emerge" to "how did internal state emerge."

### 7.3 Limits of Universality-Centered Narratives

Computational universality (Rule 110) does not confer biological-like properties. Research programs seeking life-like behavior through computational universality are fundamentally misdirected. Self-maintenance is an organizational property, not a computational one.

### 7.4 Self-Maintaining Artificial Systems

Engineering self-maintaining systems requires:
1. Internal state (any mechanism that creates context-dependence)
2. Stability substrate (avoiding purely chaotic rules)
3. Activity balance (avoiding overdamped or underdamped regimes)

These conditions are generic and achievable without precise parameter tuning.

---

## 8. Conclusion

We have demonstrated that self-maintenance is the default outcome when internal state is introduced to discrete dynamical systems with non-pathological dynamics. The central result—83.7% of non-trivial rules become life-like under stickiness—inverts the standard assumption that self-maintenance is rare.

The classification logic is complete: hidden state + stability mechanism + activity balance are jointly necessary and sufficient for life-like behavior within the scope studied. This does not redefine life; it constrains it. Life-like behavior is organizational, not miraculous.

**The final question:** When internal state becomes causally available, why does self-maintenance become the default outcome instead of a rare exception?

**The answer:** Hidden state provides the flexibility for context-dependent response. Most dynamical systems have either linear structure (enabling absorption) or attractor structure (enabling repair). When flexibility meets structure, self-maintenance follows. The 15% of rules that fail lack both structures—they are chaotic without compensating organization. Self-maintenance is not the emergence of something from nothing; it is the generic consequence of context-dependence meeting structural regularities already present in most dynamical systems.

---

## References

1. Wolfram, S. (1983). Statistical mechanics of cellular automata. *Reviews of Modern Physics*, 55(3), 601.
2. Cook, M. (2004). Universality in elementary cellular automata. *Complex Systems*, 15(1), 1-40.
3. Langton, C. G. (1990). Computation at the edge of chaos. *Physica D*, 42(1-3), 12-37.
4. Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition*. D. Reidel.
5. Schrodinger, E. (1944). *What is Life?* Cambridge University Press.

---

## Supplementary Materials

- **Table S1:** Complete census of all 256 rules with metrics (CSV)
- **Figure S1:** Depth-dependent classification heatmap
- **Table S2:** Anomaly analysis details
