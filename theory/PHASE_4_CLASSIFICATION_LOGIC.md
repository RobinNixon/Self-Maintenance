# Phase 4: Classification Logic for Life-Like vs Merely Computing

## Summary of Experimental Findings

From Phase 3 experiments on Rules 30, 54, 90, 110, and 184:

| Rule | Std Control | Stk Control | Std Alive | Stk Alive |
|------|-------------|-------------|-----------|-----------|
| 30   | 0.000       | 0.590       | NO        | NO        |
| 54   | 0.000       | 0.700       | NO        | **YES**   |
| 90   | 0.000       | 0.510       | NO        | **YES**   |
| 110  | 0.000       | 0.320       | NO        | NO        |
| 184  | 0.000       | 0.580       | NO        | **YES**   |

---

## Key Observation: Control is Necessary but Not Sufficient

**Theorem (Informal):** A dynamical system is "merely computing" if Control = 0. However, Control > 0 does not guarantee life-like behavior.

**Evidence:**
- All standard ECAs have Control = 0 and are classified as "merely computing"
- Sticky ECAs with Control > 0 are split: some are life-like (54, 90, 184), others are not (30, 110)

---

## The Life-Like Classification Criteria

A system is classified as **LIFE-LIKE** if and only if:

```
LIFE-LIKE = Control > 0 AND (Stability_Criteria >= 2)
```

Where Stability_Criteria counts the number of TRUE conditions among:
1. **Perturbation Absorption** > 0.5 (most perturbations stay localized)
2. **Self-Repair** > 0.7 (recovers from damage)
3. **Temporal Coherence** > 0.3 (patterns persist over time)
4. **Metabolic Activity** in range (0.05, 0.5) (active but not chaotic)

---

## Detailed Classification of Each Rule

### Rule 30 (Chaotic)
- **Standard:** Control=0, merely computing
- **Sticky:** Control=0.59, but:
  - Absorption=0.00 (FAIL) - perturbations spread everywhere
  - Repair=0.59 (FAIL) - doesn't recover well
  - Coherence=0.05 (FAIL) - patterns don't persist
  - Activity=0.26 (PASS)
  - **Stability criteria met: 1/4 -> NOT life-like**

**Interpretation:** Rule 30 is too chaotic. Even with hidden state (Control), it cannot maintain structure against perturbation.

### Rule 54 (Edge of Chaos)
- **Standard:** Control=0, merely computing
- **Sticky:** Control=0.70, and:
  - Absorption=0.60 (PASS) - contains perturbations
  - Repair=0.59 (FAIL)
  - Coherence=0.10 (FAIL)
  - Activity=0.35 (PASS)
  - **Stability criteria met: 2/4 -> LIFE-LIKE**

**Interpretation:** Rule 54 sits at the "edge of chaos" - enough structure to absorb perturbations, enough dynamism to be active.

### Rule 90 (XOR)
- **Standard:** Control=0, merely computing
- **Sticky:** Control=0.51, and:
  - Absorption=1.00 (PASS) - perfect absorption!
  - Repair=0.68 (near-PASS)
  - Coherence=-0.001 (FAIL)
  - Activity=0.25 (PASS)
  - **Stability criteria met: 2/4 -> LIFE-LIKE**

**Interpretation:** Rule 90's linear (XOR) dynamics mean perturbations cancel out rather than spreading - remarkably absorptive.

### Rule 110 (Computationally Universal)
- **Standard:** Control=0, merely computing
- **Sticky:** Control=0.32, but:
  - Absorption=0.50 (borderline FAIL)
  - Repair=0.58 (FAIL)
  - Coherence=-0.12 (FAIL)
  - Activity=0.21 (PASS)
  - **Stability criteria met: 1/4 -> NOT life-like**

**Interpretation:** Rule 110's computational universality comes at the cost of stability. It can compute anything, but it doesn't maintain itself.

### Rule 184 (Traffic Rule)
- **Standard:** Control=0, merely computing
- **Sticky:** Control=0.58, and:
  - Absorption=0.45 (borderline FAIL)
  - Repair=1.00 (PASS) - perfect self-repair!
  - Coherence=-0.06 (FAIL)
  - Activity=0.45 (PASS)
  - **Stability criteria met: 2/4 -> LIFE-LIKE**

**Interpretation:** Rule 184 models traffic flow - particles conserve and naturally return to equilibrium. Perfect self-repair.

---

## The Three Types of Sticky ECAs

### Type 1: Merely Computing with Control
- Has hidden state and Control (context-dependence)
- But too chaotic or too unstable to maintain structure
- Examples: Rules 30, 110 (sticky versions)
- **Analogy:** A computer program that executes but doesn't preserve any state

### Type 2: Life-Like
- Has hidden state, Control, AND stability mechanisms
- Maintains structure despite perturbation
- Shows "metabolism" - active dynamics while preserving form
- Examples: Rules 54, 90, 184 (sticky versions)
- **Analogy:** A self-maintaining system, like a biological cell

### Type 3: Dead/Crystallized (not observed)
- Would have no activity (Activity < 0.05)
- Static structures that don't change
- Not tested here, but trivial ECA rules would fall here
- **Analogy:** A crystal or frozen system

---

## The Classification Decision Tree

```
                         Is Control > 0?
                        /              \
                       NO               YES
                       |                 |
               MERELY COMPUTING    Count stability criteria
                                        |
                               >= 2 criteria met?
                              /                  \
                             NO                   YES
                             |                     |
                      MERELY COMPUTING        LIFE-LIKE
                      (with potential)
```

---

## Formal Definition

**Definition (Life-Like System):**
A discrete dynamical system (V, H, f) is *life-like* if:
1. **Control > 0:** There exist visible states v and hidden states h1 != h2 such that f(v, h1) != f(v, h2)
2. **Stability >= 2:** At least two of the following hold:
   - P(S) > 0.5: Perturbations remain localized
   - F(S) > 0.7: Damage is repaired
   - T(S) > 0.3: Patterns persist temporally
   - 0.05 < A(S) < 0.5: System is metabolically active

**Theorem (Classification):**
- Standard ECAs are never life-like (Control = 0 violates condition 1)
- Sticky ECAs may or may not be life-like (depends on rule dynamics)

---

## Implications

1. **Computation is not life:** Rule 110 is Turing-complete but not life-like. Computational power and self-maintenance are orthogonal.

2. **Stickiness enables but doesn't guarantee life:** Adding hidden state is necessary but not sufficient. The dynamics must also support stability.

3. **Life-like behavior requires balance:** Too chaotic (Rule 30) or too computational (Rule 110) fails. The sweet spot is edge-of-chaos (Rule 54) or particle-conserving (Rule 184).

4. **Self-repair is the strongest indicator:** Rule 184 shows perfect self-repair (F=1.0), making it the most robustly life-like.

---

## Open Questions for Phase 5

1. **Why do some rules become life-like and others don't?**
   - What property of Rules 54, 90, 184 enables stability?
   - Hypothesis: Conservation laws or attractor structure

2. **Is there a minimal combination of criteria for life-like behavior?**
   - Can we have life with just Control + Repair?
   - Can we have life with just Control + Absorption?

3. **What is the mechanism linking stickiness to life-like behavior?**
   - Stickiness -> Hidden State -> Control -> ???  -> Life-like
   - What fills the ???
