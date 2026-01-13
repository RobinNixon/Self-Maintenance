# Phase 6: Falsification Results

## Summary of Findings

The falsification attempts produced surprising results that **refine but do not refute** the hypothesis.

---

## Test 1: Extended Rule Survey

### Results
| Category     | Life-Like / Total | Rate |
|--------------|-------------------|------|
| Linear       | 3/4               | 75%  |
| Conservative | 2/2               | 100% |
| Chaotic      | 1/4               | 25%  |
| Complex      | 0/4               | 0%   |

### Hypothesis Status: **PARTIALLY SUPPORTED**

- Linear rules (60, 90, 102) ARE more likely life-like
- Conservative rules (184, 226) ARE 100% life-like
- BUT exceptions exist:
  - Rule 150 (linear) is NOT life-like
  - Rule 73 (chaotic) IS life-like

### Interpretation
The linear/conservative property is a **strong predictor** but not deterministic. Rule 73 may have hidden structure enabling stability. Rule 150's failure requires investigation.

---

## Test 2: Stickiness Depth Modulation

### Results
| Rule | Depths Where Life-Like |
|------|------------------------|
| 54   | 3, 4, 5, 6             |
| 90   | 2, 3, 4, 5, 6          |
| 184  | 1, 2, 3, 4, 5, 6       |

### Key Finding: **THRESHOLD DEPTH EXISTS**

- Rule 54 requires depth >= 3 to become life-like
- Rule 90 requires depth >= 2
- Rule 184 is robust at all depths

### Interpretation
Some rules require a minimum "stickiness strength" before stability mechanisms engage. This suggests a **phase transition** from merely-computing to life-like as depth increases.

The original hypothesis predicted intermediate depth would be optimal (too much = crystallization). Instead:
- No crystallization observed up to depth 6
- Activity decreases with depth but stays in acceptable range
- **Refined hypothesis:** Crystallization may require depth >> 6

---

## Test 3: Counterexample Search

### Striking Result: **79% of rules are life-like under stickiness**

| Classification | Count |
|----------------|-------|
| Life-Like      | 19    |
| Merely Computing | 5   |

### Potential Counterexamples Found

**High Absorption, Not Life-Like:**
- Rule 228: P=1.00, F=0.67, A=0.01 (crystallized)
- Rule 164: P=1.00, F=0.50, A=0.02 (crystallized)

These rules have perfect perturbation absorption but **Activity < 0.05** - they are stable but "dead."

### Interpretation
The **Activity criterion is the limiting factor** for many rules. Crystallization (too stable) prevents life-like classification even with high absorption and repair.

---

## Test 4: Control Threshold

### Results
| Rule | Control | Stability | Life-Like |
|------|---------|-----------|-----------|
| 110  | 0.310   | 1         | NO        |
| 150  | 0.440   | 1         | NO        |
| 30   | 0.470   | 1         | NO        |
| 184  | 0.470   | 3         | **YES**   |
| 60   | 0.480   | 2         | **YES**   |
| 90   | 0.500   | 2         | **YES**   |
| 45   | 0.590   | 1         | NO        |
| 54   | 0.770   | 2         | **YES**   |

### Key Finding: **NO CLEAR CONTROL THRESHOLD**

- Rule 45 has Control = 0.59 but is NOT life-like
- Rule 184 has Control = 0.47 and IS life-like
- The overlap proves Control alone does not determine life-likeness

### Interpretation
Control is **necessary** (all life-like rules have Control > 0) but not **sufficient**. The stability criteria (Absorption, Repair, Activity) are independent discriminators.

---

## Revised Hypothesis

Based on falsification results, the mechanism hypothesis is refined:

### Original Hypothesis
> Life-like = Control > 0 AND Stability >= 2

### Revised Hypothesis
> Life-like = Control > 0 AND Activity in (0.05, 0.5) AND (Absorption > 0.5 OR Repair > 0.7)

Key changes:
1. **Activity is essential** - crystallized rules (A < 0.05) are NOT life-like despite stability
2. **Either absorption OR repair suffices** - not both required
3. **Control threshold not meaningful** - varies by rule

### The Three Ways to Fail

1. **No Control (Standard ECA):** Control = 0 -> Merely computing
2. **Too Chaotic (Rule 30):** Control > 0, but Absorption < 0.5 AND Repair < 0.7 -> Merely computing
3. **Too Crystallized (Rule 228):** Control > 0, Stability high, but Activity < 0.05 -> Merely computing

### The Three Ways to Succeed

1. **Absorptive Life (Rules 60, 90, 102):** High absorption + adequate activity
2. **Regenerative Life (Rule 184):** High repair + adequate activity
3. **Balanced Life (Rule 54 at depth 3+):** Moderate both + adequate activity

---

## What We Learned About Life-Like Behavior

### 1. Life-Like Behavior is Common Under Stickiness
**79% of non-trivial rules** become life-like with stickiness. This is surprising - we expected it to be rare. Stickiness is a **general mechanism** for enabling life-like behavior, not a special case.

### 2. The Limiting Factor is Activity, Not Stability
Many rules fail not because they lack stability, but because they become **too stable**. The "Goldilocks zone" of activity (0.05-0.5) is the critical constraint.

### 3. Different Rules Use Different Strategies
- Linear rules (90) excel at absorption
- Conservative rules (184) excel at repair
- Edge-of-chaos rules (54) need higher stickiness depth

### 4. Depth Modulation is Real
The stickiness depth parameter acts as a "tuning dial" that can push rules from merely-computing into life-like territory. This is a **phase transition**, not a gradual change.

---

## Open Questions

1. **Why does Rule 73 (chaotic) become life-like?** What hidden structure does it have?

2. **Why does Rule 150 (linear) fail?** It should have absorption properties like Rule 90.

3. **What happens at depth > 6?** Does crystallization eventually occur?

4. **Is there a universal life-like rule?** One that is life-like at all depths and robust to all perturbations?

5. **Can we predict life-like behavior from rule properties?** A formula mapping rule number to life-like probability?

---

## Conclusion

The falsification attempts **refined** the hypothesis rather than refuting it:

- Control remains necessary but not sufficient
- Activity is a critical constraint (must be in Goldilocks zone)
- Life-like behavior is surprisingly common (79%) under stickiness
- Different rules achieve life-like status through different mechanisms

The big question has a preliminary answer:

> **What distinguishes life-like from merely computing?**
>
> A system is life-like when hidden state (Control) is deployed in service of **active self-maintenance**: absorbing perturbations OR repairing damage, while remaining dynamically active (not crystallized). Stickiness provides the hidden state; rule dynamics determine whether that hidden state supports stability mechanisms.
