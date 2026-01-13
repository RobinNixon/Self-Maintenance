# Phase 5: Mechanism Hypothesis - From Stickiness to Aliveness

## The Causal Chain

We have established:
```
Stickiness -> Hidden State -> Control -> ??? -> Life-like
```

This document proposes what fills the "???" gap.

---

## The Stability-Mediated Control Hypothesis

### Core Claim

**Hypothesis:** Stickiness enables life-like behavior through a two-stage mechanism:

**Stage 1: Control Emergence**
Stickiness -> Hidden State -> Control (context-dependence)

**Stage 2: Stability Selection**
Control + (Rule-Intrinsic Stability Properties) -> Life-like

The key insight is that **Control alone is not enough**. The underlying rule dynamics must provide a "stability substrate" that the hidden state can leverage for self-maintenance.

---

## The Three Stability Mechanisms

Analysis of the life-like rules (54, 90, 184) reveals three mechanisms through which Control can support life-like behavior:

### Mechanism 1: Perturbation Absorption via Hidden State Memory

**Observation:** In sticky ECAs, hidden state acts as a "buffer" that can absorb perturbations before they affect visible state.

**How it works:**
1. A perturbation changes a visible cell
2. The confirmation mechanism requires multiple consecutive "votes" to change
3. A single perturbation is unlikely to be confirmed (probability = 1/depth)
4. Most perturbations "decay" without propagating

**Which rules benefit:** Rules where local neighborhoods naturally decorrelate over time (e.g., Rule 90's XOR dynamics)

**Prediction:** Life-like rules should show higher perturbation absorption in sticky mode vs. standard mode.

**Evidence:** Rule 90 shows P=1.0 in sticky mode - perfect absorption.

### Mechanism 2: Self-Repair via Attractor Convergence

**Observation:** Some rules have strong attractors that "pull" the system back to recognizable patterns after damage.

**How it works:**
1. Damage disrupts visible and hidden state
2. But the rule's dynamics have basin of attraction B
3. If damaged state is still in B, dynamics return to attractor
4. Hidden state (stickiness) widens the basin by slowing divergence

**Which rules benefit:** Rules with conserved quantities or stable attractor structures (e.g., Rule 184's particle conservation)

**Prediction:** Life-like rules should show improved self-repair in sticky mode.

**Evidence:** Rule 184 shows F=1.0 in sticky mode - perfect self-repair (vs. F=0.88 standard).

### Mechanism 3: Temporal Coherence via Hidden State Persistence

**Observation:** Hidden state provides "memory" that maintains pattern identity over time.

**How it works:**
1. Without hidden state, patterns can flicker (change then immediately change back)
2. Hidden state "anchors" patterns by requiring confirmation
3. Boundaries become stable because they represent confirmed transitions
4. Temporal autocorrelation increases

**Which rules benefit:** Rules that tend toward oscillation or flickering

**Prediction:** Sticky mode should increase temporal coherence (T metric).

**Evidence:** Rule 54 shows T=0.102 (sticky) vs. T=-0.118 (standard) - from anti-correlated to correlated.

---

## The Selection Principle

Not all rules benefit equally from stickiness. The **Selection Principle** states:

> A rule becomes life-like under stickiness if and only if its intrinsic dynamics provide at least two stability mechanisms that hidden state can amplify.

### Why Some Rules Fail

**Rule 30 (Chaotic):**
- Intrinsic dynamics are maximally chaotic
- No conservation laws, no stable attractors
- Perturbations spread regardless of hidden state
- Hidden state cannot "anchor" anything in chaos
- Result: Control but no stability -> NOT life-like

**Rule 110 (Universal but Unstable):**
- Intrinsic dynamics support computation via gliders
- But gliders are sensitive to initial conditions
- Perturbations disrupt computational structures
- Hidden state slows but cannot prevent disruption
- Result: Control but marginal stability -> NOT life-like

### Why Some Rules Succeed

**Rule 54 (Edge of Chaos):**
- Intrinsic dynamics at phase transition
- Enough structure for absorption, enough chaos for activity
- Hidden state tips balance toward stability
- Result: Control + Absorption + Activity -> LIFE-LIKE

**Rule 90 (Linear/XOR):**
- Intrinsic dynamics are linear (perturbations XOR not AND)
- Perturbations cancel rather than amplify
- Hidden state provides perfect absorption buffer
- Result: Control + Perfect Absorption + Activity -> LIFE-LIKE

**Rule 184 (Conservative):**
- Intrinsic dynamics conserve particle count
- System has natural equilibrium states
- Hidden state strengthens convergence to equilibrium
- Result: Control + Perfect Repair + Activity -> LIFE-LIKE

---

## The Complete Mechanism

```
                    STICKINESS
                        |
                        v
                   HIDDEN STATE
                        |
                        v
           +--------CONTROL--------+
           |                       |
           v                       v
     Context-Dependence    Temporal Buffering
           |                       |
           v                       v
    RULE DYNAMICS          STABILITY AMPLIFICATION
    (Intrinsic)            (from hidden state)
           |                       |
           +-----+     +-----------+
                 |     |
                 v     v
             STABILITY CRITERIA
             (Absorption, Repair,
              Coherence, Activity)
                    |
                    v
           >= 2 criteria met?
                   / \
                  /   \
                 v     v
            LIFE-LIKE  MERELY COMPUTING
```

---

## Testable Predictions

### Prediction 1: Stability Substrate Hypothesis
**Claim:** Rules with intrinsic conservation laws or linear dynamics should be more likely to become life-like under stickiness.

**Test:** Classify all 256 ECA rules by:
- Presence of conservation laws (particle count, density)
- Linearity (XOR-like vs. AND-like)
- Predict which become life-like under stickiness

### Prediction 2: Stickiness Depth Modulation
**Claim:** Increasing stickiness depth should improve stability metrics up to a point, then decrease them (too much stickiness = crystallization).

**Test:** Measure P, F, T, A for Rule 54 at depths 1, 2, 3, 4, 5. Expect optimal life-like behavior at intermediate depth.

### Prediction 3: Mechanism Independence
**Claim:** The three stability mechanisms (absorption, repair, coherence) are partially independent - rules can be life-like with different combinations.

**Test:** Find rules that are life-like with:
- High absorption, low repair
- High repair, low absorption
- Neither high, but both moderate

### Prediction 4: Minimum Control Threshold
**Claim:** There should be a minimum Control threshold below which no rule can be life-like, regardless of stability.

**Test:** Measure Control vs. life-like classification across all 168 non-trivial rules. Identify the minimum Control for life-like classification.

---

## Summary: The Mechanism in One Paragraph

**Stickiness enables life-like behavior by adding hidden state to a dynamical system, creating Control (context-dependence). However, Control alone is insufficient - the underlying rule dynamics must provide a "stability substrate" that hidden state can amplify. Rules with conservation laws (184), linear dynamics (90), or edge-of-chaos behavior (54) have intrinsic stability mechanisms that hidden state enhances, leading to perturbation absorption, self-repair, and/or temporal coherence. Rules without such substrates (30, 110) gain Control but remain merely computing. Thus, life-like behavior requires both the context-dependence from hidden state AND the stability amplification from compatible rule dynamics.**

---

## The Big Question Revisited

**Q: What distinguishes life-like from merely computing?**

**A:** Control enables context-dependence, but life-like behavior requires that context-dependence be *deployed in service of stability*. A system that can respond differently to the same input (Control) becomes life-like when those different responses include:
- Absorbing perturbations rather than amplifying them
- Repairing damage rather than diverging
- Persisting patterns rather than flickering

The hidden state must be *used for* self-maintenance, not just *exist as* potential.

---

## Next Steps: Phase 6 Falsification

To test this hypothesis, we need to:
1. Find counterexamples: rules that should be life-like by this hypothesis but aren't
2. Find alternative explanations: other mechanisms that could explain the data
3. Test edge cases: rules at the boundary of classification
