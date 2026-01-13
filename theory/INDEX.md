# Theory Files Index

This folder contains the theoretical development of the Self-Maintenance framework.

## Phase Documents

| File | Description |
|------|-------------|
| [PHASE_0_MICRO_CLAIMS.md](PHASE_0_MICRO_CLAIMS.md) | Initial calibration and micro-claims |
| [PHASE_1_ALIVENESS_DEFINITIONS.md](PHASE_1_ALIVENESS_DEFINITIONS.md) | Operational definitions of life-like metrics |
| [PHASE_4_CLASSIFICATION_LOGIC.md](PHASE_4_CLASSIFICATION_LOGIC.md) | Three-condition classification framework |
| [PHASE_5_MECHANISM_HYPOTHESIS.md](PHASE_5_MECHANISM_HYPOTHESIS.md) | Causal mechanism: stickiness → hidden state → Control |
| [PHASE_6_FALSIFICATION_RESULTS.md](PHASE_6_FALSIFICATION_RESULTS.md) | Falsification tests and anomaly analysis |

## Key Theoretical Results

### The Three Conditions for Life-Like Behavior

A system is **life-like** if and only if:
1. **Control > 0** — Hidden state influences visible output
2. **Absorption > 0.5 OR Repair > 0.7** — Stability mechanism exists
3. **0.05 ≤ Activity ≤ 0.5** — Dynamics in Goldilocks zone

### The Causal Chain

```
Stickiness → Hidden State → Control → [+ Stability] → Life-Like
```

### Central Finding

**83.7% of non-trivial ECA rules become life-like under stickiness.**

This inverts the standard assumption that self-maintenance is rare.
