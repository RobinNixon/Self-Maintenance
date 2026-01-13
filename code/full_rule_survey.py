"""
Full Rule Survey: Test ALL 256 ECA rules for life-like behavior under stickiness
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from collections import defaultdict

# =============================================================================
# Core ECA Infrastructure
# =============================================================================

def apply_eca_rule(left: int, center: int, right: int, rule: int) -> int:
    """Apply an ECA rule to a neighborhood."""
    index = int(left) << 2 | int(center) << 1 | int(right)
    return (rule >> index) & 1

def run_standard_eca(rule: int, initial: np.ndarray, steps: int) -> np.ndarray:
    """Run standard ECA without stickiness."""
    width = len(initial)
    history = np.zeros((steps + 1, width), dtype=np.int8)
    history[0] = initial.copy()

    for t in range(steps):
        for i in range(width):
            left = history[t, (i - 1) % width]
            center = history[t, i]
            right = history[t, (i + 1) % width]
            history[t + 1, i] = apply_eca_rule(left, center, right, rule)

    return history

def run_sticky_eca(rule: int, initial: np.ndarray, steps: int, depth: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Run sticky ECA with confirmation mechanism."""
    width = len(initial)
    visible = np.zeros((steps + 1, width), dtype=np.int8)
    hidden = np.zeros((steps + 1, width), dtype=np.int8)

    visible[0] = initial.copy()

    for t in range(steps):
        for i in range(width):
            left = visible[t, (i - 1) % width]
            center = visible[t, i]
            right = visible[t, (i + 1) % width]

            proposed = apply_eca_rule(left, center, right, rule)

            if proposed != center:
                new_pending = hidden[t, i] + 1
                if new_pending >= depth:
                    visible[t + 1, i] = proposed
                    hidden[t + 1, i] = 0
                else:
                    visible[t + 1, i] = center
                    hidden[t + 1, i] = new_pending
            else:
                visible[t + 1, i] = center
                hidden[t + 1, i] = 0

    return visible, hidden

# =============================================================================
# Metrics
# =============================================================================

def is_trivial_rule(rule: int, width: int = 31, steps: int = 50) -> bool:
    """Check if rule produces trivial behavior (static, nilpotent, or uniform)."""
    initial = np.random.randint(0, 2, width, dtype=np.int8)
    history = run_standard_eca(rule, initial, steps)

    # Check for static (no change after first few steps)
    changes = np.sum(np.abs(np.diff(history[10:].astype(int), axis=0)))
    if changes == 0:
        return True

    # Check for nilpotent (converges to all 0 or all 1)
    final = history[-1]
    if np.all(final == 0) or np.all(final == 1):
        return True

    return False

def measure_activity(rule: int, width: int, steps: int, depth: int) -> float:
    """Measure activity level (fraction of cells changing per step)."""
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca(rule, initial, steps, depth)

    changes = np.abs(np.diff(history.astype(int), axis=0))
    return float(np.mean(changes))

def measure_absorption(rule: int, width: int, steps: int, depth: int, trials: int = 15) -> float:
    """Measure perturbation absorption capacity."""
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        baseline = run_standard_eca(rule, initial, steps)
    else:
        baseline, _ = run_sticky_eca(rule, initial, steps, depth)

    absorbed = 0

    for _ in range(trials):
        perturb_time = steps // 2
        perturb_loc = np.random.randint(width)

        perturbed_init = baseline[perturb_time].copy()
        perturbed_init[perturb_loc] = 1 - perturbed_init[perturb_loc]

        if depth == 0:
            baseline_cont = run_standard_eca(rule, baseline[perturb_time], steps // 2)
            perturbed_cont = run_standard_eca(rule, perturbed_init, steps // 2)
        else:
            baseline_cont, _ = run_sticky_eca(rule, baseline[perturb_time], steps // 2, depth)
            perturbed_cont, _ = run_sticky_eca(rule, perturbed_init, steps // 2, depth)

        final_diff = np.sum(np.abs(baseline_cont[-1].astype(int) - perturbed_cont[-1].astype(int)))
        if final_diff < width * 0.2:
            absorbed += 1

    return absorbed / trials

def measure_repair(rule: int, width: int, steps: int, depth: int) -> float:
    """Measure self-repair fidelity."""
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca(rule, initial, steps, depth)

    pre_damage = history[-1].copy()

    # Damage 10%
    damaged = pre_damage.copy()
    num_damage = max(1, width // 10)
    damage_locs = np.random.choice(width, num_damage, replace=False)
    for loc in damage_locs:
        damaged[loc] = 1 - damaged[loc]

    recovery_steps = steps // 2
    if depth == 0:
        recovered = run_standard_eca(rule, damaged, recovery_steps)
    else:
        recovered, _ = run_sticky_eca(rule, damaged, recovery_steps, depth)

    # Boundary-based similarity
    def get_boundaries(state):
        b = np.zeros_like(state)
        for i in range(len(state)):
            if state[i] != state[(i-1) % len(state)] or state[i] != state[(i+1) % len(state)]:
                b[i] = 1
        return b

    b1 = get_boundaries(pre_damage)
    b2 = get_boundaries(recovered[-1])

    if np.sum(b1) == 0 and np.sum(b2) == 0:
        return 1.0

    intersection = np.sum(b1 & b2)
    union = np.sum(b1 | b2)

    return intersection / union if union > 0 else 1.0

def measure_control(rule: int, width: int, depth: int) -> float:
    """Measure counterfactual Control."""
    if depth == 0:
        return 0.0

    control_count = 0
    samples = 50

    for _ in range(samples):
        visible = np.random.randint(0, 2, width, dtype=np.int8)
        pos = np.random.randint(width)

        left = visible[(pos - 1) % width]
        center = visible[pos]
        right = visible[(pos + 1) % width]
        proposed = apply_eca_rule(left, center, right, rule)

        if proposed != center:
            control_count += 1

    return control_count / samples

# =============================================================================
# Classification
# =============================================================================

def classify_rule(P: float, F: float, A: float, C: float) -> Tuple[str, Dict]:
    """Classify a rule based on metrics."""
    criteria = {
        'has_control': C > 0.01,
        'absorbs': P > 0.5,
        'repairs': F > 0.7,
        'active': 0.05 < A < 0.5,
        'crystallized': A < 0.05,
        'chaotic': A > 0.5
    }

    # Classification logic
    if not criteria['has_control']:
        return 'NO_CONTROL', criteria

    if criteria['crystallized']:
        return 'CRYSTALLIZED', criteria

    stability_met = criteria['absorbs'] or criteria['repairs']
    activity_ok = criteria['active']

    if stability_met and activity_ok:
        return 'LIFE-LIKE', criteria
    elif stability_met and criteria['chaotic']:
        return 'CHAOTIC_STABLE', criteria
    elif not stability_met and activity_ok:
        return 'COMPUTING', criteria
    else:
        return 'UNSTABLE', criteria

# =============================================================================
# Main Survey
# =============================================================================

def run_full_survey():
    """Test all 256 rules."""
    print("="*70)
    print("FULL ECA RULE SURVEY: Testing all 256 rules for life-like behavior")
    print("="*70)

    width = 41
    steps = 80
    depth = 2

    results = {}
    classifications = defaultdict(list)

    # First pass: identify trivial rules
    print("\nPhase 1: Identifying trivial rules...")
    trivial_rules = set()
    for rule in range(256):
        if is_trivial_rule(rule):
            trivial_rules.add(rule)
    print(f"  Found {len(trivial_rules)} trivial rules")

    # Second pass: test non-trivial rules
    print(f"\nPhase 2: Testing {256 - len(trivial_rules)} non-trivial rules...")
    print("-"*70)

    for rule in range(256):
        if rule in trivial_rules:
            results[rule] = {
                'trivial': True,
                'classification': 'TRIVIAL'
            }
            classifications['TRIVIAL'].append(rule)
            continue

        try:
            # Measure metrics
            A = measure_activity(rule, width, steps, depth)
            P = measure_absorption(rule, width, steps, depth)
            F = measure_repair(rule, width, steps, depth)
            C = measure_control(rule, width, depth)

            # Classify
            classification, criteria = classify_rule(P, F, A, C)

            results[rule] = {
                'trivial': False,
                'P': float(P),
                'F': float(F),
                'A': float(A),
                'C': float(C),
                'classification': classification,
                'criteria': {k: bool(v) for k, v in criteria.items()}
            }
            classifications[classification].append(rule)

            # Progress indicator
            if rule % 16 == 0:
                print(f"  Rule {rule:3d}: P={P:.2f} F={F:.2f} A={A:.2f} C={C:.2f} -> {classification}")

        except Exception as e:
            results[rule] = {'error': str(e)}
            classifications['ERROR'].append(rule)

    return results, classifications, trivial_rules

def print_summary(results: Dict, classifications: Dict, trivial_rules: set):
    """Print summary of results."""
    print("\n" + "="*70)
    print("SURVEY RESULTS SUMMARY")
    print("="*70)

    total = 256
    trivial = len(trivial_rules)
    non_trivial = total - trivial

    print(f"\nTotal rules: {total}")
    print(f"Trivial rules: {trivial}")
    print(f"Non-trivial rules: {non_trivial}")

    print("\n" + "-"*70)
    print("CLASSIFICATION BREAKDOWN")
    print("-"*70)

    for cls in ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED', 'CHAOTIC_STABLE', 'UNSTABLE', 'TRIVIAL', 'ERROR']:
        rules = classifications.get(cls, [])
        count = len(rules)
        pct = 100 * count / total
        pct_nontrivial = 100 * count / non_trivial if cls != 'TRIVIAL' and non_trivial > 0 else 0

        print(f"\n{cls}:")
        print(f"  Count: {count} ({pct:.1f}% of all, {pct_nontrivial:.1f}% of non-trivial)")
        if rules and cls != 'TRIVIAL':
            print(f"  Rules: {sorted(rules)[:20]}{'...' if len(rules) > 20 else ''}")

    # Life-like rules detail
    life_like_rules = classifications.get('LIFE-LIKE', [])
    if life_like_rules:
        print("\n" + "="*70)
        print(f"LIFE-LIKE RULES ({len(life_like_rules)} total)")
        print("="*70)
        print(f"{'Rule':<6} {'P(Abs)':<8} {'F(Rep)':<8} {'A(Act)':<8} {'C(Ctrl)':<8}")
        print("-"*40)

        for rule in sorted(life_like_rules)[:30]:
            r = results[rule]
            print(f"{rule:<6} {r['P']:<8.2f} {r['F']:<8.2f} {r['A']:<8.2f} {r['C']:<8.2f}")

        if len(life_like_rules) > 30:
            print(f"... and {len(life_like_rules) - 30} more")

    # Famous rules status
    print("\n" + "="*70)
    print("FAMOUS RULES STATUS")
    print("="*70)
    famous = {
        30: "Chaotic (Wolfram's favorite)",
        54: "Edge of chaos",
        90: "XOR / Sierpinski",
        110: "Turing complete",
        184: "Traffic rule",
        150: "XOR variant"
    }

    for rule, desc in famous.items():
        if rule in results:
            r = results[rule]
            if r.get('trivial'):
                status = "TRIVIAL"
            else:
                status = r['classification']
            print(f"  Rule {rule} ({desc}): {status}")

def main():
    results, classifications, trivial_rules = run_full_survey()
    print_summary(results, classifications, trivial_rules)

    # Save results
    output_dir = Path("C:/Github/TBQ/output")
    output_dir.mkdir(exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, set)):
            return [convert(v) for v in obj]
        return obj

    output = {
        'results': results,
        'classifications': {k: list(v) for k, v in classifications.items()},
        'trivial_rules': list(trivial_rules)
    }

    with open(output_dir / "full_rule_survey.json", 'w') as f:
        json.dump(convert(output), f, indent=2)

    print(f"\nResults saved to {output_dir / 'full_rule_survey.json'}")

    # Final stats
    life_like = len(classifications.get('LIFE-LIKE', []))
    non_trivial = 256 - len(trivial_rules)
    print(f"\n{'='*70}")
    print(f"FINAL ANSWER: {life_like}/{non_trivial} non-trivial rules are LIFE-LIKE ({100*life_like/non_trivial:.1f}%)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
