"""
Phase 3: Comparative Analysis - Standard ECA vs Sticky ECA Aliveness

This extends the Phase 2 measurements to compare sticky ECAs against
standard ECAs as a baseline, revealing what stickiness adds.
"""

import numpy as np
from typing import Dict
import json
from pathlib import Path
from aliveness_metrics import (
    run_standard_eca, run_sticky_eca_confirmation, detect_boundaries,
    compute_pattern_similarity, apply_eca_rule
)
from scipy import stats

# =============================================================================
# Simplified Metrics for Standard vs Sticky Comparison
# =============================================================================

def measure_perturbation_absorption(rule: int, width: int, steps: int,
                                     depth: int = 0, num_trials: int = 20) -> float:
    """
    Measure how well a system absorbs perturbations.
    depth=0 means standard ECA, depth>0 means sticky.
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        # Standard ECA
        baseline = run_standard_eca(rule, initial, steps)
    else:
        baseline, _ = run_sticky_eca_confirmation(rule, initial, steps, depth)

    absorbed = 0

    for _ in range(num_trials):
        perturb_time = steps // 2
        perturb_loc = np.random.randint(width)

        # Create perturbed copy
        perturbed_initial = baseline[perturb_time].copy()
        perturbed_initial[perturb_loc] = 1 - perturbed_initial[perturb_loc]

        # Run both from this point
        if depth == 0:
            baseline_continued = run_standard_eca(rule, baseline[perturb_time], steps // 2)
            perturbed_continued = run_standard_eca(rule, perturbed_initial, steps // 2)
        else:
            baseline_continued, _ = run_sticky_eca_confirmation(rule, baseline[perturb_time], steps // 2, depth)
            perturbed_continued, _ = run_sticky_eca_confirmation(rule, perturbed_initial, steps // 2, depth)

        # Check if perturbation stayed localized
        final_diff = np.sum(np.abs(baseline_continued[-1].astype(int) - perturbed_continued[-1].astype(int)))
        if final_diff < width * 0.2:
            absorbed += 1

    return absorbed / num_trials

def measure_self_repair(rule: int, width: int, steps: int, depth: int = 0) -> float:
    """
    Measure self-repair: similarity between pre-damage and post-recovery states.
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca_confirmation(rule, initial, steps, depth)

    pre_damage = history[-1].copy()

    # Damage 10% of cells
    damaged = pre_damage.copy()
    damage_locs = np.random.choice(width, width // 10, replace=False)
    for loc in damage_locs:
        damaged[loc] = 1 - damaged[loc]

    # Run recovery
    recovery_steps = steps // 2
    if depth == 0:
        recovered = run_standard_eca(rule, damaged, recovery_steps)
    else:
        recovered, _ = run_sticky_eca_confirmation(rule, damaged, recovery_steps, depth)

    return compute_pattern_similarity(pre_damage, recovered[-1])

def measure_temporal_persistence(rule: int, width: int, steps: int, depth: int = 0) -> float:
    """
    Measure how long patterns persist (autocorrelation of boundary positions).
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca_confirmation(rule, initial, steps, depth)

    boundaries = detect_boundaries(history)

    # Autocorrelation at lag 10
    lag = min(10, steps // 4)
    correlations = []
    for t in range(steps - lag):
        b1 = boundaries[t].astype(float)
        b2 = boundaries[t + lag].astype(float)
        if np.std(b1) > 0 and np.std(b2) > 0:
            r, _ = stats.pearsonr(b1, b2)
            correlations.append(r)

    return np.mean(correlations) if correlations else 0.0

def measure_activity_level(rule: int, width: int, steps: int, depth: int = 0) -> float:
    """
    Measure how active the system is (fraction of cells changing per step).
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca_confirmation(rule, initial, steps, depth)

    changes = np.abs(np.diff(history.astype(int), axis=0))
    return np.mean(changes)

def measure_counterfactual_control_simple(rule: int, width: int = 41, depth: int = 2) -> float:
    """
    Simplified counterfactual Control measurement.
    For standard ECA (depth=0), Control is exactly 0.
    For sticky ECA, measure fraction of configurations where hidden state matters.
    """
    if depth == 0:
        return 0.0  # By theorem, standard ECAs have zero Control

    # For sticky: sample configurations and test
    control_instances = 0
    samples = 100

    for _ in range(samples):
        visible = np.random.randint(0, 2, width, dtype=np.int8)
        pos = np.random.randint(width)

        left = visible[(pos - 1) % width]
        center = visible[pos]
        right = visible[(pos + 1) % width]
        proposed = apply_eca_rule(left, center, right, rule)

        # If rule wants to change this cell, hidden state matters
        if proposed != center:
            control_instances += 1

    return control_instances / samples

# =============================================================================
# Main Comparison
# =============================================================================

def compare_standard_vs_sticky(rules: list, width: int = 41, steps: int = 100) -> Dict:
    """
    Compare all metrics between standard ECAs and sticky ECAs.
    """
    results = {}

    for rule in rules:
        print(f"\n{'='*60}")
        print(f"Rule {rule}: Standard vs Sticky Comparison")
        print(f"{'='*60}")

        results[rule] = {
            'standard': {},
            'sticky': {}
        }

        # Standard ECA (depth=0)
        print("\nStandard ECA:")
        std = results[rule]['standard']
        std['perturbation_absorption'] = measure_perturbation_absorption(rule, width, steps, depth=0)
        std['self_repair'] = measure_self_repair(rule, width, steps, depth=0)
        std['temporal_persistence'] = measure_temporal_persistence(rule, width, steps, depth=0)
        std['activity'] = measure_activity_level(rule, width, steps, depth=0)
        std['control'] = measure_counterfactual_control_simple(rule, width, depth=0)

        print(f"  Perturbation Absorption: {std['perturbation_absorption']:.3f}")
        print(f"  Self-Repair: {std['self_repair']:.3f}")
        print(f"  Temporal Persistence: {std['temporal_persistence']:.3f}")
        print(f"  Activity: {std['activity']:.3f}")
        print(f"  Control: {std['control']:.3f}")

        # Sticky ECA (depth=2)
        print("\nSticky ECA (depth=2):")
        stk = results[rule]['sticky']
        stk['perturbation_absorption'] = measure_perturbation_absorption(rule, width, steps, depth=2)
        stk['self_repair'] = measure_self_repair(rule, width, steps, depth=2)
        stk['temporal_persistence'] = measure_temporal_persistence(rule, width, steps, depth=2)
        stk['activity'] = measure_activity_level(rule, width, steps, depth=2)
        stk['control'] = measure_counterfactual_control_simple(rule, width, depth=2)

        print(f"  Perturbation Absorption: {stk['perturbation_absorption']:.3f}")
        print(f"  Self-Repair: {stk['self_repair']:.3f}")
        print(f"  Temporal Persistence: {stk['temporal_persistence']:.3f}")
        print(f"  Activity: {stk['activity']:.3f}")
        print(f"  Control: {stk['control']:.3f}")

        # Compute deltas
        results[rule]['delta'] = {
            k: stk[k] - std[k] for k in std.keys()
        }

        print("\nDelta (Sticky - Standard):")
        for k, v in results[rule]['delta'].items():
            sign = '+' if v > 0 else ''
            print(f"  {k}: {sign}{v:.3f}")

    return results

# =============================================================================
# Aliveness Classification
# =============================================================================

def classify_aliveness(standard_metrics: Dict, sticky_metrics: Dict) -> Dict:
    """
    Classify a system as 'life-like' vs 'merely computing' based on metrics.
    """
    classification = {
        'standard': {},
        'sticky': {}
    }

    # Standard ECA
    std = standard_metrics
    classification['standard']['has_control'] = std['control'] > 0.01
    classification['standard']['absorbs_perturbations'] = std['perturbation_absorption'] > 0.5
    classification['standard']['self_repairs'] = std['self_repair'] > 0.7
    classification['standard']['temporally_coherent'] = std['temporal_persistence'] > 0.3
    classification['standard']['metabolically_active'] = 0.05 < std['activity'] < 0.5

    # Life-like requires at least 3 of 5 criteria + Control
    std_criteria_met = sum([
        classification['standard']['absorbs_perturbations'],
        classification['standard']['self_repairs'],
        classification['standard']['temporally_coherent'],
        classification['standard']['metabolically_active']
    ])
    classification['standard']['is_life_like'] = (
        classification['standard']['has_control'] and std_criteria_met >= 2
    )

    # Sticky ECA
    stk = sticky_metrics
    classification['sticky']['has_control'] = stk['control'] > 0.01
    classification['sticky']['absorbs_perturbations'] = stk['perturbation_absorption'] > 0.5
    classification['sticky']['self_repairs'] = stk['self_repair'] > 0.7
    classification['sticky']['temporally_coherent'] = stk['temporal_persistence'] > 0.3
    classification['sticky']['metabolically_active'] = 0.05 < stk['activity'] < 0.5

    stk_criteria_met = sum([
        classification['sticky']['absorbs_perturbations'],
        classification['sticky']['self_repairs'],
        classification['sticky']['temporally_coherent'],
        classification['sticky']['metabolically_active']
    ])
    classification['sticky']['is_life_like'] = (
        classification['sticky']['has_control'] and stk_criteria_met >= 2
    )

    return classification

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("C:/Github/TBQ/output")
    output_dir.mkdir(exist_ok=True)

    # Test rules
    test_rules = [30, 54, 90, 110, 184]

    print("="*70)
    print("PHASE 3: STANDARD vs STICKY ECA COMPARISON")
    print("="*70)

    results = compare_standard_vs_sticky(test_rules, width=41, steps=100)

    # Classification
    print("\n" + "="*70)
    print("ALIVENESS CLASSIFICATION")
    print("="*70)

    all_classifications = {}

    for rule in test_rules:
        cls = classify_aliveness(results[rule]['standard'], results[rule]['sticky'])
        all_classifications[rule] = cls

        print(f"\nRule {rule}:")
        print(f"  Standard: {'LIFE-LIKE' if cls['standard']['is_life_like'] else 'merely computing'}")
        print(f"    Criteria: Control={cls['standard']['has_control']}, "
              f"Absorb={cls['standard']['absorbs_perturbations']}, "
              f"Repair={cls['standard']['self_repairs']}, "
              f"Coherent={cls['standard']['temporally_coherent']}, "
              f"Active={cls['standard']['metabolically_active']}")
        print(f"  Sticky: {'LIFE-LIKE' if cls['sticky']['is_life_like'] else 'merely computing'}")
        print(f"    Criteria: Control={cls['sticky']['has_control']}, "
              f"Absorb={cls['sticky']['absorbs_perturbations']}, "
              f"Repair={cls['sticky']['self_repairs']}, "
              f"Coherent={cls['sticky']['temporally_coherent']}, "
              f"Active={cls['sticky']['metabolically_active']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Rule':<8} {'Std Control':<12} {'Stk Control':<12} {'Std Alive':<12} {'Stk Alive':<12}")
    print("-"*56)
    for rule in test_rules:
        std_ctrl = results[rule]['standard']['control']
        stk_ctrl = results[rule]['sticky']['control']
        std_alive = 'YES' if all_classifications[rule]['standard']['is_life_like'] else 'no'
        stk_alive = 'YES' if all_classifications[rule]['sticky']['is_life_like'] else 'no'
        print(f"{rule:<8} {std_ctrl:<12.3f} {stk_ctrl:<12.3f} {std_alive:<12} {stk_alive:<12}")

    # Save results
    def convert_types(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    output_file = output_dir / "phase3_comparison.json"
    combined = {'metrics': results, 'classifications': all_classifications}
    with open(output_file, 'w') as f:
        json.dump(convert_types(combined), f, indent=2)

    print(f"\nResults saved to {output_file}")
