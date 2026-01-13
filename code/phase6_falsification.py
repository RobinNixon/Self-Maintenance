"""
Phase 6: Falsification Attempts

Testing the Stability-Mediated Control Hypothesis:
1. Do more rules follow the pattern?
2. Is there an optimal stickiness depth?
3. Are there counterexamples?
"""

import numpy as np
from typing import Dict, List
import json
from pathlib import Path
from phase3_analysis import (
    compare_standard_vs_sticky, classify_aliveness,
    measure_perturbation_absorption, measure_self_repair,
    measure_activity_level, measure_counterfactual_control_simple
)
from aliveness_metrics import run_sticky_eca_confirmation, detect_boundaries

# =============================================================================
# Test 1: Extended Rule Survey
# =============================================================================

def test_extended_rules():
    """
    Test more rules to see if the pattern holds.
    Hypothesis: Linear/conservative rules -> life-like, chaotic rules -> not life-like
    """
    print("="*70)
    print("TEST 1: Extended Rule Survey")
    print("="*70)

    # Additional rules to test
    # Linear: 60, 90, 102, 150 (XOR-based)
    # Conservative: 184, 226 (particle-conserving)
    # Chaotic: 30, 45, 73, 105 (known chaotic)
    # Complex: 110, 54, 124, 137 (complex dynamics)

    test_rules = {
        'linear': [60, 90, 102, 150],
        'conservative': [184, 226],
        'chaotic': [30, 45, 73, 105],
        'complex': [110, 54, 124, 137]
    }

    results = {}
    width, steps = 41, 80

    for category, rules in test_rules.items():
        print(f"\n{category.upper()} RULES:")
        results[category] = {}

        for rule in rules:
            try:
                # Measure sticky metrics
                P = measure_perturbation_absorption(rule, width, steps, depth=2)
                F = measure_self_repair(rule, width, steps, depth=2)
                A = measure_activity_level(rule, width, steps, depth=2)
                C = measure_counterfactual_control_simple(rule, width, depth=2)

                # Classification
                stability_count = sum([
                    P > 0.5,
                    F > 0.7,
                    0.05 < A < 0.5
                ])
                is_life_like = (C > 0.01) and (stability_count >= 2)

                results[category][rule] = {
                    'P': P, 'F': F, 'A': A, 'C': C,
                    'stability_count': stability_count,
                    'is_life_like': is_life_like
                }

                status = "LIFE-LIKE" if is_life_like else "merely computing"
                print(f"  Rule {rule}: P={P:.2f}, F={F:.2f}, A={A:.2f}, C={C:.2f} -> {status}")

            except Exception as e:
                print(f"  Rule {rule}: ERROR - {e}")
                results[category][rule] = {'error': str(e)}

    # Summary
    print("\n" + "-"*70)
    print("HYPOTHESIS TEST: Linear/Conservative -> Life-Like")
    print("-"*70)

    for category, rules_data in results.items():
        life_like_count = sum(1 for r in rules_data.values() if r.get('is_life_like', False))
        total = len([r for r in rules_data.values() if 'error' not in r])
        print(f"  {category}: {life_like_count}/{total} life-like")

    return results

# =============================================================================
# Test 2: Stickiness Depth Modulation
# =============================================================================

def test_depth_modulation():
    """
    Test if there's an optimal stickiness depth.
    Hypothesis: Intermediate depth is optimal; too much = crystallization.
    """
    print("\n" + "="*70)
    print("TEST 2: Stickiness Depth Modulation")
    print("="*70)

    test_rules = [54, 90, 184]  # Known life-like rules
    depths = [1, 2, 3, 4, 5, 6]
    width, steps = 41, 80

    results = {}

    for rule in test_rules:
        print(f"\nRule {rule}:")
        results[rule] = {}

        for depth in depths:
            try:
                P = measure_perturbation_absorption(rule, width, steps, depth=depth)
                F = measure_self_repair(rule, width, steps, depth=depth)
                A = measure_activity_level(rule, width, steps, depth=depth)
                C = measure_counterfactual_control_simple(rule, width, depth=depth)

                stability_count = sum([
                    P > 0.5,
                    F > 0.7,
                    0.05 < A < 0.5
                ])
                is_life_like = (C > 0.01) and (stability_count >= 2)

                results[rule][depth] = {
                    'P': P, 'F': F, 'A': A, 'C': C,
                    'stability_count': stability_count,
                    'is_life_like': is_life_like
                }

                status = "ALIVE" if is_life_like else "not"
                print(f"  depth={depth}: P={P:.2f}, F={F:.2f}, A={A:.2f}, C={C:.2f} [{status}]")

            except Exception as e:
                print(f"  depth={depth}: ERROR - {e}")

    # Analyze optimal depth
    print("\n" + "-"*70)
    print("OPTIMAL DEPTH ANALYSIS")
    print("-"*70)

    for rule, depth_data in results.items():
        life_like_depths = [d for d, v in depth_data.items() if v.get('is_life_like', False)]
        if life_like_depths:
            print(f"  Rule {rule}: Life-like at depths {life_like_depths}")
        else:
            print(f"  Rule {rule}: Never life-like")

    return results

# =============================================================================
# Test 3: Search for Counterexamples
# =============================================================================

def test_counterexamples():
    """
    Search for counterexamples to the hypothesis:
    - Life-like rules that are NOT linear/conservative
    - Non-life-like rules that ARE linear/conservative
    """
    print("\n" + "="*70)
    print("TEST 3: Search for Counterexamples")
    print("="*70)

    # Sample more rules randomly
    np.random.seed(42)
    sample_rules = np.random.choice(range(256), size=30, replace=False)
    sample_rules = [int(r) for r in sample_rules]

    width, steps = 41, 80
    results = []

    print("\nTesting 30 random rules...")

    for rule in sample_rules:
        try:
            # Standard metrics
            std_A = measure_activity_level(rule, width, steps, depth=0)

            # Skip trivial rules (too static or uniform)
            if std_A < 0.01:
                continue

            # Sticky metrics
            P = measure_perturbation_absorption(rule, width, steps, depth=2)
            F = measure_self_repair(rule, width, steps, depth=2)
            A = measure_activity_level(rule, width, steps, depth=2)
            C = measure_counterfactual_control_simple(rule, width, depth=2)

            stability_count = sum([
                P > 0.5,
                F > 0.7,
                0.05 < A < 0.5
            ])
            is_life_like = (C > 0.01) and (stability_count >= 2)

            results.append({
                'rule': rule,
                'P': P, 'F': F, 'A': A, 'C': C,
                'stability_count': stability_count,
                'is_life_like': is_life_like
            })

        except Exception as e:
            pass

    # Analyze results
    life_like_rules = [r for r in results if r['is_life_like']]
    computing_rules = [r for r in results if not r['is_life_like']]

    print(f"\nOut of {len(results)} non-trivial rules tested:")
    print(f"  Life-like: {len(life_like_rules)}")
    print(f"  Merely computing: {len(computing_rules)}")

    if life_like_rules:
        print("\nLife-like rules found:")
        for r in life_like_rules:
            print(f"  Rule {r['rule']}: P={r['P']:.2f}, F={r['F']:.2f}, A={r['A']:.2f}")

    # Check for unexpected patterns
    print("\n" + "-"*70)
    print("COUNTEREXAMPLE ANALYSIS")
    print("-"*70)

    # High-absorption non-life-like (unexpected?)
    high_absorption_not_alive = [r for r in computing_rules if r['P'] > 0.7]
    if high_absorption_not_alive:
        print("\nPotential counterexample: High absorption but not life-like:")
        for r in high_absorption_not_alive:
            print(f"  Rule {r['rule']}: P={r['P']:.2f}, F={r['F']:.2f}, A={r['A']:.2f}")
    else:
        print("\nNo high-absorption non-life-like rules found.")

    # Low-absorption life-like (unexpected?)
    low_absorption_alive = [r for r in life_like_rules if r['P'] < 0.3]
    if low_absorption_alive:
        print("\nPotential counterexample: Low absorption but life-like:")
        for r in low_absorption_alive:
            print(f"  Rule {r['rule']}: P={r['P']:.2f}, F={r['F']:.2f}, A={r['A']:.2f}")
    else:
        print("\nNo low-absorption life-like rules found.")

    return results

# =============================================================================
# Test 4: Control Threshold
# =============================================================================

def test_control_threshold():
    """
    Is there a minimum Control threshold for life-like behavior?
    """
    print("\n" + "="*70)
    print("TEST 4: Control Threshold Analysis")
    print("="*70)

    # Test rules with varying Control levels
    test_rules = [30, 54, 90, 110, 184, 60, 150, 45]
    width, steps = 41, 80

    control_values = []

    for rule in test_rules:
        try:
            P = measure_perturbation_absorption(rule, width, steps, depth=2)
            F = measure_self_repair(rule, width, steps, depth=2)
            A = measure_activity_level(rule, width, steps, depth=2)
            C = measure_counterfactual_control_simple(rule, width, depth=2)

            stability_count = sum([P > 0.5, F > 0.7, 0.05 < A < 0.5])
            is_life_like = (C > 0.01) and (stability_count >= 2)

            control_values.append({
                'rule': rule,
                'control': C,
                'is_life_like': is_life_like,
                'stability_count': stability_count
            })

        except:
            pass

    # Sort by Control
    control_values.sort(key=lambda x: x['control'])

    print("\nRules sorted by Control:")
    print(f"{'Rule':<8} {'Control':<10} {'Stability':<10} {'Life-Like'}")
    print("-"*40)

    for r in control_values:
        status = "YES" if r['is_life_like'] else "no"
        print(f"{r['rule']:<8} {r['control']:<10.3f} {r['stability_count']:<10} {status}")

    # Find threshold
    life_like_controls = [r['control'] for r in control_values if r['is_life_like']]
    not_life_like_controls = [r['control'] for r in control_values if not r['is_life_like']]

    if life_like_controls and not_life_like_controls:
        min_life_like = min(life_like_controls)
        max_not_life_like = max(not_life_like_controls)

        print(f"\nMinimum Control for life-like: {min_life_like:.3f}")
        print(f"Maximum Control for not life-like: {max_not_life_like:.3f}")

        if min_life_like > max_not_life_like:
            print("-> Clear threshold exists!")
        else:
            print("-> No clear threshold (overlap)")

    return control_values

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("C:/Github/TBQ/output")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("PHASE 6: FALSIFICATION ATTEMPTS")
    print("="*70)

    # Run all tests
    results = {}

    results['extended_rules'] = test_extended_rules()
    results['depth_modulation'] = test_depth_modulation()
    results['counterexamples'] = test_counterexamples()
    results['control_threshold'] = test_control_threshold()

    # Final summary
    print("\n" + "="*70)
    print("FALSIFICATION SUMMARY")
    print("="*70)

    print("""
    TEST 1 (Extended Rules):
      - Checks if linear/conservative rules are more likely life-like
      - Result: [see output above]

    TEST 2 (Depth Modulation):
      - Checks if intermediate depth is optimal
      - Result: [see output above]

    TEST 3 (Counterexamples):
      - Searches for rules that violate the hypothesis
      - Result: [see output above]

    TEST 4 (Control Threshold):
      - Checks if there's a minimum Control for life-like behavior
      - Result: [see output above]
    """)

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

    output_file = output_dir / "phase6_falsification.json"
    with open(output_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"\nResults saved to {output_file}")
