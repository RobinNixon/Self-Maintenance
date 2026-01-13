# Frequently Asked Questions

## Basic Concepts

### What does "life-like" mean?
Life-like behavior is characterized by active self-maintenance: the system responds to perturbations in a way that preserves its structure while remaining dynamically active. It's defined operationally by three conditions: Control > 0, stability mechanism (Absorption > 0.5 OR Repair > 0.7), and Activity in range 0.05-0.5.

### What is "stickiness"?
Stickiness is a mechanism that adds hidden state to cellular automata. It requires multiple consecutive "requests" for a change before the change takes effect. This creates context-dependence: the same visible configuration can produce different outcomes depending on how long the change has been requested.

### What is "Control"?
Control is context-dependent divergence: the same visible input can produce different visible outputs depending on hidden state. Formally, Control > 0 if there exist visible state v and hidden states h₁ ≠ h₂ such that f(v, h₁) ≠ f(v, h₂).

### Why do standard ECAs have Control = 0?
Standard ECAs are memoryless: output depends only on visible input. Without hidden state, the same input always produces the same output. The definition of Control requires different hidden states to produce different outputs, which is impossible when there's only one hidden state.

## Main Results

### What is the central finding?
83.7% of non-trivial elementary cellular automata become life-like when augmented with hidden state through stickiness. This inverts the standard assumption that self-maintenance is rare.

### Why is Rule 110 not life-like?
Rule 110 is Turing-complete (can compute anything) but lacks stability mechanisms. Its complex dynamics spread perturbations rather than absorbing them. Absorption = 0.30, Repair = 0.47—both below thresholds.

### Why is Rule 30 not life-like?
Rule 30 is maximally chaotic. Despite having Control = 0.52 under stickiness, its chaotic dynamics spread perturbations everywhere. Absorption = 0.03, far below the 0.5 threshold.

### Why is Rule 90 life-like?
Rule 90 has linear (XOR-based) dynamics that naturally cancel perturbations through superposition. Absorption = 1.00 (perfect). Combined with moderate activity, this produces life-like behavior.

### Why is Rule 184 life-like?
Rule 184 (traffic rule) conserves particles, creating strong attractors that pull the system back to recognizable states. Repair = 0.93 (excellent). This conservation law enables self-repair.

## Classification Logic

### What are the three conditions?
1. **Control > 0** — Hidden state must influence visible output
2. **Absorption > 0.5 OR Repair > 0.7** — Stability mechanism must exist
3. **0.05 ≤ Activity ≤ 0.5** — Dynamics must be in Goldilocks zone

### Why are all three necessary?
- Without Control: No context-dependence, can't respond adaptively
- Without Stability: Perturbations destroy structure
- Without Activity: System is frozen (crystallized), not alive

### What happens if Activity < 0.05?
The system crystallizes—it's stable but "dead." Rules 108, 201, 216 are examples: perfect stability but no dynamics.

### What happens if Activity > 0.5?
The system is too chaotic for hidden state to anchor structure. Stability mechanisms can't overcome the noise.

## Implications

### Does this explain the origin of life?
Not directly. But it shifts the question: instead of explaining how self-maintenance emerged, we need to explain how hidden state emerged. Hidden state arises from common physical phenomena (activation barriers, hysteresis), so the "hard" question may be easier than assumed.

### Does computation enable life?
No. Rule 110 is computationally universal but not life-like. Computational power and self-maintenance are orthogonal properties.

### Does chaos enable life?
No. Rule 30 is maximally chaotic but not life-like. Chaos spreads perturbations; self-maintenance absorbs them. These are opposing tendencies.

## Scope and Limitations

### Does this apply to real biological life?
The framework applies to abstract "life-like" behavior, not biological life specifically. The connection to biology is conjectural and requires validation.

### Does this apply to continuous systems?
The current framework is for discrete systems. Extension to continuous systems is an open problem.

### How robust are the thresholds (0.5 for Absorption, 0.7 for Repair)?
The thresholds are empirically determined and may vary with context. The qualitative finding (three conditions necessary) is more robust than specific threshold values.
