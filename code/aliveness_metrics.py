"""
Phase 2: Measurement Protocols for Aliveness Metrics in Sticky ECAs

This module implements the seven candidate operational definitions of "aliveness"
from Phase 1, designed to measure self-maintenance properties in discrete dynamical systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from scipy import stats
from collections import defaultdict

# =============================================================================
# Core ECA Infrastructure (from previous experiments)
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

def run_sticky_eca_confirmation(rule: int, initial: np.ndarray, steps: int,
                                 depth: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run sticky ECA with confirmation mechanism.
    Returns (visible_history, hidden_history).
    """
    width = len(initial)
    visible = np.zeros((steps + 1, width), dtype=np.int8)
    hidden = np.zeros((steps + 1, width), dtype=np.int8)  # pending counters

    visible[0] = initial.copy()
    hidden[0] = np.zeros(width, dtype=np.int8)

    for t in range(steps):
        for i in range(width):
            left = visible[t, (i - 1) % width]
            center = visible[t, i]
            right = visible[t, (i + 1) % width]

            proposed = apply_eca_rule(left, center, right, rule)

            if proposed != center:
                # Rule wants to change - increment pending counter
                new_pending = hidden[t, i] + 1
                if new_pending >= depth:
                    # Confirmed! Apply change
                    visible[t + 1, i] = proposed
                    hidden[t + 1, i] = 0
                else:
                    # Still pending
                    visible[t + 1, i] = center
                    hidden[t + 1, i] = new_pending
            else:
                # Rule doesn't want change - reset pending
                visible[t + 1, i] = center
                hidden[t + 1, i] = 0

    return visible, hidden

# =============================================================================
# Boundary Detection
# =============================================================================

def detect_boundaries(visible: np.ndarray) -> np.ndarray:
    """
    Detect boundaries in a spacetime diagram.
    A boundary exists where adjacent cells differ.
    Returns binary array of same shape.
    """
    steps, width = visible.shape
    boundaries = np.zeros_like(visible)

    for t in range(steps):
        for i in range(width):
            left = visible[t, (i - 1) % width]
            right = visible[t, (i + 1) % width]
            center = visible[t, i]

            # Boundary if different from either neighbor
            if center != left or center != right:
                boundaries[t, i] = 1

    return boundaries

# =============================================================================
# Metric K: Control-Boundary Coupling Strength
# =============================================================================

def measure_counterfactual_control(rule: int, visible_state: np.ndarray,
                                    depth: int = 2, samples: int = 100) -> float:
    """
    Measure counterfactual Control: same visible state, different hidden states,
    different outcomes.
    """
    width = len(visible_state)
    control_count = 0

    for _ in range(samples):
        i = np.random.randint(width)

        # Hidden state 1: no pending
        hidden1 = np.zeros(width, dtype=np.int8)

        # Hidden state 2: pending at position i
        hidden2 = np.zeros(width, dtype=np.int8)
        hidden2[i] = depth - 1

        # Compute outcomes
        left = visible_state[(i - 1) % width]
        center = visible_state[i]
        right = visible_state[(i + 1) % width]
        proposed = apply_eca_rule(left, center, right, rule)

        # With hidden1 (no pending), what happens?
        if proposed != center:
            # Would start pending, stay at center
            outcome1 = center
        else:
            outcome1 = center

        # With hidden2 (pending at depth-1), what happens?
        if proposed != center:
            # Would confirm change
            outcome2 = proposed
        else:
            # No change requested, reset pending
            outcome2 = center

        if outcome1 != outcome2:
            control_count += 1

    return control_count / samples

def metric_K_control_boundary_coupling(rule: int, width: int = 61, steps: int = 100,
                                        depth: int = 2) -> Dict:
    """
    Metric K: Control-Boundary Coupling Strength

    K(S) = correlation(boundary_presence, control_magnitude) * mean(control_at_boundaries)
    """
    # Generate random initial condition
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    # Run sticky ECA
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)

    # Detect boundaries
    boundaries = detect_boundaries(visible)

    # Measure Control at each cell (steps+1 to match boundaries shape)
    control_map = np.zeros((steps + 1, width))

    for t in range(1, steps + 1):
        for i in range(width):
            # Measure counterfactual Control at this position
            control_map[t, i] = measure_counterfactual_control(
                rule, visible[t], depth, samples=20
            )

    # Flatten for correlation (skip first row, match shapes)
    boundary_flat = boundaries[1:].flatten().astype(float)
    control_flat = control_map[1:].flatten()

    # Compute correlation
    if np.std(boundary_flat) > 0 and np.std(control_flat) > 0:
        correlation, _ = stats.pearsonr(boundary_flat, control_flat)
    else:
        correlation = 0.0

    # Mean Control at boundaries
    boundary_mask = boundary_flat > 0
    if np.sum(boundary_mask) > 0:
        mean_control_at_boundaries = np.mean(control_flat[boundary_mask])
    else:
        mean_control_at_boundaries = 0.0

    K = correlation * mean_control_at_boundaries

    return {
        'K': float(K),
        'correlation': float(correlation),
        'mean_control_at_boundaries': float(mean_control_at_boundaries),
        'mean_control_overall': float(np.mean(control_flat)),
        'boundary_fraction': float(np.mean(boundary_flat))
    }

# =============================================================================
# Metric P: Perturbation Absorption Capacity
# =============================================================================

def metric_P_absorption_capacity(rule: int, width: int = 61, steps: int = 100,
                                  depth: int = 2, num_perturbations: int = 20) -> Dict:
    """
    Metric P: Perturbation Absorption Capacity

    P(S) = fraction of perturbations absorbed (localized) vs propagating
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    # Run baseline
    visible_baseline, hidden_baseline = run_sticky_eca_confirmation(rule, initial, steps, depth)

    absorbed_count = 0
    propagation_distances = []

    for _ in range(num_perturbations):
        # Choose perturbation time and location
        perturb_time = np.random.randint(steps // 4, steps // 2)
        perturb_loc = np.random.randint(width)

        # Run until perturbation time
        v_copy = visible_baseline[perturb_time].copy()
        h_copy = hidden_baseline[perturb_time].copy()

        # Apply perturbation: flip cell and reset hidden state
        v_copy[perturb_loc] = 1 - v_copy[perturb_loc]
        h_copy[perturb_loc] = 0

        # Continue running from perturbed state
        v_perturbed, h_perturbed = run_sticky_eca_confirmation(
            rule, v_copy, steps - perturb_time, depth
        )

        # Also run baseline from same point (without perturbation)
        v_control, h_control = run_sticky_eca_confirmation(
            rule, visible_baseline[perturb_time], steps - perturb_time, depth
        )

        # Measure how far the perturbation spread
        final_diff = np.abs(v_perturbed[-1].astype(int) - v_control[-1].astype(int))
        num_affected = np.sum(final_diff)

        # Perturbation is "absorbed" if it affects < 20% of cells
        if num_affected < width * 0.2:
            absorbed_count += 1

        propagation_distances.append(num_affected / width)

    P = absorbed_count / num_perturbations

    return {
        'P': float(P),
        'absorbed_count': absorbed_count,
        'total_perturbations': num_perturbations,
        'mean_propagation': float(np.mean(propagation_distances)),
        'max_propagation': float(np.max(propagation_distances))
    }

# =============================================================================
# Metric M: Metabolic Persistence Ratio
# =============================================================================

def compute_pattern_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute similarity between two visible states based on boundary positions."""
    b1 = detect_boundaries(v1.reshape(1, -1))[0]
    b2 = detect_boundaries(v2.reshape(1, -1))[0]

    if np.sum(b1) == 0 and np.sum(b2) == 0:
        return 1.0  # Both uniform - perfectly similar

    # Jaccard similarity of boundary positions
    intersection = np.sum(b1 & b2)
    union = np.sum(b1 | b2)

    if union == 0:
        return 1.0

    return intersection / union

def metric_M_metabolic_persistence(rule: int, width: int = 61, steps: int = 200,
                                    depth: int = 2, window: int = 50) -> Dict:
    """
    Metric M: Metabolic Persistence Ratio

    M(S) = pattern_similarity * cell_turnover_rate

    "Goldilocks zone" for aliveness: 0.1 < M < 0.9
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)

    # Compute pattern similarity between t and t+window
    similarities = []
    for t in range(steps - window):
        sim = compute_pattern_similarity(visible[t], visible[t + window])
        similarities.append(sim)

    mean_similarity = np.mean(similarities)

    # Compute cell turnover rate
    changes = np.abs(np.diff(visible.astype(int), axis=0))
    turnover_rate = np.mean(changes)

    M = mean_similarity * turnover_rate

    # Determine zone
    if M < 0.1:
        zone = 'crystallized'  # Too stable, not alive
    elif M > 0.9:
        zone = 'chaotic'  # Too unstable, not alive
    else:
        zone = 'alive'  # Goldilocks zone

    return {
        'M': float(M),
        'pattern_similarity': float(mean_similarity),
        'turnover_rate': float(turnover_rate),
        'zone': zone
    }

# =============================================================================
# Metric F: Self-Repair Fidelity
# =============================================================================

def metric_F_self_repair_fidelity(rule: int, width: int = 61, steps: int = 100,
                                   depth: int = 2, damage_fraction: float = 0.1,
                                   recovery_time: int = 50) -> Dict:
    """
    Metric F: Self-Repair Fidelity

    F(S) = similarity(pre-damage, post-recovery)

    Threshold for alive: F > 0.8
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    # Run to stable dynamics
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)

    # Save pre-damage state
    pre_damage_visible = visible[-1].copy()
    pre_damage_boundaries = detect_boundaries(pre_damage_visible.reshape(1, -1))[0]

    # Apply damage
    damaged_visible = pre_damage_visible.copy()
    num_damaged = int(width * damage_fraction)
    damage_locs = np.random.choice(width, num_damaged, replace=False)

    for loc in damage_locs:
        damaged_visible[loc] = 1 - damaged_visible[loc]

    # Reset hidden state in damaged region
    damaged_hidden = np.zeros(width, dtype=np.int8)

    # Run recovery
    recovered_visible, _ = run_sticky_eca_confirmation(rule, damaged_visible, recovery_time, depth)

    # Compute similarity
    post_recovery = recovered_visible[-1]
    F = compute_pattern_similarity(pre_damage_visible, post_recovery)

    return {
        'F': float(F),
        'damage_fraction': damage_fraction,
        'recovery_time': recovery_time,
        'is_alive': F > 0.8
    }

# =============================================================================
# Metric T: Temporal Coherence Index
# =============================================================================

def metric_T_temporal_coherence(rule: int, width: int = 61, steps: int = 200,
                                 depth: int = 2) -> Dict:
    """
    Metric T: Temporal Coherence Index

    T(S) = autocorrelation decay time of boundary positions
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)

    boundaries = detect_boundaries(visible)

    # Compute autocorrelation at various lags
    lags = [1, 2, 5, 10, 20, 50]
    autocorrs = []

    for lag in lags:
        if lag >= steps:
            continue

        corrs = []
        for t in range(steps - lag):
            b1 = boundaries[t].flatten()
            b2 = boundaries[t + lag].flatten()

            if np.std(b1) > 0 and np.std(b2) > 0:
                corr, _ = stats.pearsonr(b1, b2)
                corrs.append(corr)

        if corrs:
            autocorrs.append(np.mean(corrs))
        else:
            autocorrs.append(0)

    # Fit exponential decay to estimate characteristic time
    if len(autocorrs) >= 3 and autocorrs[0] > 0:
        # Simple estimate: time to drop to 1/e
        try:
            decay_threshold = autocorrs[0] / np.e
            T = lags[0]  # Default
            for i, (lag, ac) in enumerate(zip(lags, autocorrs)):
                if ac < decay_threshold:
                    T = lag
                    break
            else:
                T = lags[-1]  # Hasn't decayed yet
        except:
            T = 1
    else:
        T = 1

    return {
        'T': float(T),
        'autocorrelations': {str(lag): float(ac) for lag, ac in zip(lags[:len(autocorrs)], autocorrs)},
        'initial_autocorr': float(autocorrs[0]) if autocorrs else 0.0
    }

# =============================================================================
# Metric I: Integrated Information Index
# =============================================================================

def metric_I_integration(rule: int, width: int = 61, steps: int = 100,
                          depth: int = 2) -> Dict:
    """
    Metric I: Integrated Information Index (Phi-like)

    I(S) = mutual information between left and right halves
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)

    # Split into halves
    mid = width // 2

    # Compute joint distribution of half-states
    # Use coarse-graining: count active cells in each half
    left_counts = np.sum(visible[:, :mid], axis=1)
    right_counts = np.sum(visible[:, mid:], axis=1)

    # Bin counts into categories
    left_bins = np.digitize(left_counts, np.linspace(0, mid, 5))
    right_bins = np.digitize(right_counts, np.linspace(0, width - mid, 5))

    # Compute mutual information
    def compute_mi(x, y):
        joint = np.zeros((5, 5))
        for i, j in zip(x, y):
            if i < 5 and j < 5:
                joint[i, j] += 1
        joint /= np.sum(joint) + 1e-10

        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        mi = 0
        for i in range(5):
            for j in range(5):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

        return mi

    MI_observed = compute_mi(left_bins, right_bins)

    # Null model: shuffled
    MI_shuffled_list = []
    for _ in range(10):
        shuffled_right = np.random.permutation(right_bins)
        MI_shuffled_list.append(compute_mi(left_bins, shuffled_right))

    MI_shuffled = np.mean(MI_shuffled_list)

    I = MI_observed / (MI_shuffled + 1e-10)

    return {
        'I': float(I),
        'MI_observed': float(MI_observed),
        'MI_shuffled': float(MI_shuffled),
        'is_integrated': I > 1.5
    }

# =============================================================================
# Metric A: Autopoietic Index
# =============================================================================

def metric_A_autopoietic(rule: int, width: int = 61, steps: int = 100,
                          depth: int = 2) -> Dict:
    """
    Metric A: Autopoietic Index

    A(S) = min(boundary_recovery, hidden_recovery, bidirectional_causality)

    This is the most complex metric, measuring circular self-production.
    """
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    # Run baseline
    visible, hidden = run_sticky_eca_confirmation(rule, initial, steps, depth)
    boundaries = detect_boundaries(visible)

    # Measure boundary recovery after perturbation
    pre_perturb_boundaries = boundaries[steps // 2]

    # Perturb
    perturbed_visible = visible[steps // 2].copy()
    perturbed_hidden = hidden[steps // 2].copy()

    # Damage 20% of cells
    damage_locs = np.random.choice(width, width // 5, replace=False)
    for loc in damage_locs:
        perturbed_visible[loc] = 1 - perturbed_visible[loc]
        perturbed_hidden[loc] = 0

    # Run recovery
    recovered_v, recovered_h = run_sticky_eca_confirmation(rule, perturbed_visible, steps // 2, depth)
    recovered_boundaries = detect_boundaries(recovered_v)[-1]

    # Boundary recovery rate
    boundary_similarity = compute_pattern_similarity(
        visible[steps // 2], recovered_v[-1]
    )
    R_boundary = boundary_similarity

    # Hidden state recovery (measure distribution similarity)
    pre_hidden_dist = np.bincount(hidden[steps // 2].astype(int), minlength=depth)
    post_hidden_dist = np.bincount(recovered_h[-1].astype(int), minlength=depth)

    pre_hidden_dist = pre_hidden_dist / (np.sum(pre_hidden_dist) + 1e-10)
    post_hidden_dist = post_hidden_dist / (np.sum(post_hidden_dist) + 1e-10)

    R_hidden = 1 - 0.5 * np.sum(np.abs(pre_hidden_dist - post_hidden_dist))

    # Bidirectional causality (simplified: correlation between boundary changes and hidden changes)
    boundary_changes = np.abs(np.diff(boundaries.astype(int), axis=0))
    hidden_changes = np.abs(np.diff(hidden.astype(int), axis=0))

    boundary_flat = boundary_changes.flatten()
    hidden_flat = hidden_changes.flatten()

    if np.std(boundary_flat) > 0 and np.std(hidden_flat) > 0:
        C_feedback, _ = stats.pearsonr(boundary_flat, hidden_flat)
        C_feedback = max(0, C_feedback)  # Only positive correlation counts
    else:
        C_feedback = 0

    A = min(R_boundary, R_hidden, C_feedback)

    return {
        'A': float(A),
        'R_boundary': float(R_boundary),
        'R_hidden': float(R_hidden),
        'C_feedback': float(C_feedback),
        'is_autopoietic': A > 0.5
    }

# =============================================================================
# Comprehensive Aliveness Assessment
# =============================================================================

def assess_aliveness(rule: int, width: int = 61, steps: int = 100,
                     depth: int = 2, verbose: bool = True) -> Dict:
    """
    Run all aliveness metrics on a sticky ECA rule.
    Returns comprehensive assessment.
    """
    results = {
        'rule': rule,
        'width': width,
        'steps': steps,
        'depth': depth,
        'metrics': {}
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Aliveness Assessment: Rule {rule} (depth={depth})")
        print(f"{'='*60}")

    # Run each metric
    metrics = [
        ('K', 'Control-Boundary Coupling', metric_K_control_boundary_coupling),
        ('P', 'Absorption Capacity', metric_P_absorption_capacity),
        ('M', 'Metabolic Persistence', metric_M_metabolic_persistence),
        ('F', 'Self-Repair Fidelity', metric_F_self_repair_fidelity),
        ('T', 'Temporal Coherence', metric_T_temporal_coherence),
        ('I', 'Integration', metric_I_integration),
        ('A', 'Autopoietic Index', metric_A_autopoietic),
    ]

    for symbol, name, func in metrics:
        if verbose:
            print(f"\nMeasuring {name} ({symbol})...")

        try:
            result = func(rule, width, steps, depth)
            results['metrics'][symbol] = result

            if verbose:
                print(f"  {symbol} = {result[symbol]:.3f}")
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results['metrics'][symbol] = {'error': str(e)}

    # Compute overall aliveness score
    valid_scores = []
    for symbol in ['K', 'P', 'M', 'F', 'T', 'I', 'A']:
        if symbol in results['metrics'] and symbol in results['metrics'][symbol]:
            valid_scores.append(results['metrics'][symbol][symbol])

    if valid_scores:
        results['overall_score'] = float(np.mean(valid_scores))
    else:
        results['overall_score'] = 0.0

    # Classification
    if results['overall_score'] > 0.5:
        results['classification'] = 'life-like'
    elif results['overall_score'] > 0.2:
        results['classification'] = 'proto-alive'
    else:
        results['classification'] = 'merely computing'

    if verbose:
        print(f"\n{'='*60}")
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Classification: {results['classification']}")
        print(f"{'='*60}")

    return results

# =============================================================================
# Main: Run Assessment on Representative Rules
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("C:/Github/TBQ/output")
    output_dir.mkdir(exist_ok=True)

    # Test rules: complex (30, 110), simple (184), edge-of-chaos (54)
    test_rules = [30, 54, 90, 110, 184]

    print("="*70)
    print("PHASE 2: ALIVENESS METRICS MEASUREMENT")
    print("="*70)

    all_results = {}

    for rule in test_rules:
        results = assess_aliveness(rule, width=41, steps=80, depth=2, verbose=True)
        all_results[rule] = results

    # Save results
    output_file = output_dir / "aliveness_metrics.json"

    # Convert numpy types
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
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: ALIVENESS METRICS BY RULE")
    print("="*70)
    print(f"{'Rule':<8} {'K':<8} {'P':<8} {'M':<8} {'F':<8} {'T':<8} {'I':<8} {'A':<8} {'Score':<8} {'Class'}")
    print("-"*70)

    for rule, results in all_results.items():
        metrics = results['metrics']
        row = f"{rule:<8}"

        for symbol in ['K', 'P', 'M', 'F', 'T', 'I', 'A']:
            if symbol in metrics and symbol in metrics[symbol]:
                row += f"{metrics[symbol][symbol]:<8.3f}"
            else:
                row += f"{'ERR':<8}"

        row += f"{results['overall_score']:<8.3f} {results['classification']}"
        print(row)

    print("="*70)
