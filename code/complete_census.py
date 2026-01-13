"""
Complete Rule Census: Compute all metrics for all 256 ECA rules
Output: CSV file and summary statistics for paper
"""

import numpy as np
import csv
from pathlib import Path
from typing import Dict, Tuple, List

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

def run_sticky_eca(rule: int, initial: np.ndarray, steps: int, depth: int) -> Tuple[np.ndarray, np.ndarray]:
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
# Metrics (with multiple trials for stability)
# =============================================================================

def is_trivial(rule: int, width: int = 41, steps: int = 60, trials: int = 3) -> bool:
    """Check if rule produces trivial behavior."""
    for _ in range(trials):
        initial = np.random.randint(0, 2, width, dtype=np.int8)
        history = run_standard_eca(rule, initial, steps)

        # Check for activity after settling
        changes = np.sum(np.abs(np.diff(history[20:].astype(int), axis=0)))
        if changes > 0:
            # Check for nilpotent
            final = history[-1]
            if not (np.all(final == 0) or np.all(final == 1)):
                return False
    return True

def measure_control(rule: int, width: int = 41, depth: int = 2, samples: int = 100) -> float:
    """
    Measure counterfactual Control: fraction of configurations where
    hidden state affects output.
    """
    if depth == 0:
        return 0.0

    control_count = 0

    for _ in range(samples):
        visible = np.random.randint(0, 2, width, dtype=np.int8)
        pos = np.random.randint(width)

        left = visible[(pos - 1) % width]
        center = visible[pos]
        right = visible[(pos + 1) % width]
        proposed = apply_eca_rule(left, center, right, rule)

        # If rule wants change, hidden state matters
        if proposed != center:
            control_count += 1

    return control_count / samples

def measure_absorption(rule: int, width: int = 41, steps: int = 80,
                       depth: int = 2, trials: int = 20) -> float:
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

def measure_repair(rule: int, width: int = 41, steps: int = 80, depth: int = 2) -> float:
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

    return float(intersection / union) if union > 0 else 1.0

def measure_activity(rule: int, width: int = 41, steps: int = 80, depth: int = 2) -> float:
    """Measure activity level."""
    initial = np.random.randint(0, 2, width, dtype=np.int8)

    if depth == 0:
        history = run_standard_eca(rule, initial, steps)
    else:
        history, _ = run_sticky_eca(rule, initial, steps, depth)

    changes = np.abs(np.diff(history.astype(int), axis=0))
    return float(np.mean(changes))

def classify_rule(C: float, P: float, F: float, A: float) -> str:
    """Classify based on metrics."""
    # Crystallized: very low activity
    if A < 0.05:
        return "CRYSTALLIZED"

    # Check stability criteria
    has_absorption = P > 0.5
    has_repair = F > 0.7
    activity_ok = 0.05 <= A <= 0.5

    if (has_absorption or has_repair) and activity_ok:
        return "LIFE-LIKE"
    else:
        return "COMPUTING"

# =============================================================================
# Main Census
# =============================================================================

def run_complete_census(depth: int = 2, num_trials: int = 3):
    """Run complete census with multiple trials for stability."""

    print("="*70)
    print(f"COMPLETE RULE CENSUS (depth={depth}, trials={num_trials})")
    print("="*70)

    width = 41
    steps = 80

    results = []

    # First identify trivial rules
    print("\nPhase 1: Identifying trivial rules...")
    trivial_rules = set()
    for rule in range(256):
        if is_trivial(rule, width, steps):
            trivial_rules.add(rule)
    print(f"  Found {len(trivial_rules)} trivial rules")

    # Compute metrics for all rules
    print(f"\nPhase 2: Computing metrics for all 256 rules...")

    for rule in range(256):
        if rule % 32 == 0:
            print(f"  Processing rules {rule}-{min(rule+31, 255)}...")

        if rule in trivial_rules:
            results.append({
                'rule': rule,
                'trivial': True,
                'C': 0.0,
                'P': 0.0,
                'F': 0.0,
                'A': 0.0,
                'classification': 'TRIVIAL',
                'computed': True
            })
            continue

        # Multiple trials for stability
        C_vals, P_vals, F_vals, A_vals = [], [], [], []

        for trial in range(num_trials):
            np.random.seed(rule * 1000 + trial)  # Reproducible

            C_vals.append(measure_control(rule, width, depth))
            P_vals.append(measure_absorption(rule, width, steps, depth))
            F_vals.append(measure_repair(rule, width, steps, depth))
            A_vals.append(measure_activity(rule, width, steps, depth))

        C = np.mean(C_vals)
        P = np.mean(P_vals)
        F = np.mean(F_vals)
        A = np.mean(A_vals)

        classification = classify_rule(C, P, F, A)

        results.append({
            'rule': rule,
            'trivial': False,
            'C': round(C, 3),
            'P': round(P, 3),
            'F': round(F, 3),
            'A': round(A, 3),
            'C_std': round(np.std(C_vals), 3),
            'P_std': round(np.std(P_vals), 3),
            'F_std': round(np.std(F_vals), 3),
            'A_std': round(np.std(A_vals), 3),
            'classification': classification,
            'computed': True
        })

    return results, trivial_rules

def save_csv(results: List[Dict], output_path: Path):
    """Save results to CSV."""
    fieldnames = ['rule', 'trivial', 'C', 'P', 'F', 'A',
                  'C_std', 'P_std', 'F_std', 'A_std',
                  'classification', 'computed']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Fill missing std fields for trivial rules
            row_copy = row.copy()
            for field in ['C_std', 'P_std', 'F_std', 'A_std']:
                if field not in row_copy:
                    row_copy[field] = 0.0
            writer.writerow(row_copy)

    print(f"Saved CSV to {output_path}")

def print_summary(results: List[Dict]):
    """Print summary statistics."""

    classifications = {}
    for r in results:
        cls = r['classification']
        if cls not in classifications:
            classifications[cls] = []
        classifications[cls].append(r['rule'])

    total = len(results)
    trivial = len(classifications.get('TRIVIAL', []))
    non_trivial = total - trivial

    print("\n" + "="*70)
    print("CENSUS SUMMARY")
    print("="*70)

    print(f"\nTotal rules: {total}")
    print(f"Trivial rules: {trivial}")
    print(f"Non-trivial rules: {non_trivial}")

    print("\nClassification breakdown:")
    for cls in ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED', 'TRIVIAL']:
        rules = classifications.get(cls, [])
        count = len(rules)
        pct_all = 100 * count / total
        pct_nt = 100 * count / non_trivial if cls != 'TRIVIAL' and non_trivial > 0 else 0
        print(f"  {cls}: {count} ({pct_all:.1f}% of all, {pct_nt:.1f}% of non-trivial)")

    # List non-life-like non-trivial rules
    computing = classifications.get('COMPUTING', [])
    crystallized = classifications.get('CRYSTALLIZED', [])

    print(f"\nCOMPUTING rules ({len(computing)}): {sorted(computing)}")
    print(f"CRYSTALLIZED rules ({len(crystallized)}): {sorted(crystallized)}")

    return classifications

def generate_latex_table(results: List[Dict], output_path: Path):
    """Generate LaTeX table for paper."""

    # Summary table
    classifications = {}
    for r in results:
        cls = r['classification']
        if cls not in classifications:
            classifications[cls] = []
        classifications[cls].append(r)

    trivial_count = len(classifications.get('TRIVIAL', []))
    non_trivial = 256 - trivial_count

    latex = r"""\begin{table}[h]
\centering
\caption{Classification of all 256 ECA rules under stickiness (depth=2)}
\label{tab:census}
\begin{tabular}{lrrr}
\toprule
Classification & Count & \% of All & \% of Non-Trivial \\
\midrule
"""

    for cls in ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED', 'TRIVIAL']:
        count = len(classifications.get(cls, []))
        pct_all = 100 * count / 256
        pct_nt = 100 * count / non_trivial if cls != 'TRIVIAL' else 0

        if cls == 'LIFE-LIKE':
            latex += f"\\textbf{{{cls}}} & \\textbf{{{count}}} & \\textbf{{{pct_all:.1f}\\%}} & \\textbf{{{pct_nt:.1f}\\%}} \\\\\n"
        else:
            latex += f"{cls} & {count} & {pct_all:.1f}\\% & {pct_nt:.1f}\\% \\\\\n"

    latex += r"""\midrule
Total & 256 & 100\% & -- \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Detailed table for non-trivial rules
    latex += r"""

\begin{table}[h]
\centering
\caption{Metrics for non-life-like rules (COMPUTING and CRYSTALLIZED)}
\label{tab:failures}
\begin{tabular}{rccccl}
\toprule
Rule & Control & Absorption & Repair & Activity & Failure Mode \\
\midrule
"""

    failures = classifications.get('COMPUTING', []) + classifications.get('CRYSTALLIZED', [])
    failures.sort(key=lambda x: x['rule'])

    for r in failures:
        # Determine failure mode
        if r['A'] < 0.05:
            mode = "Crystallized"
        elif r['P'] <= 0.5 and r['F'] <= 0.7:
            mode = "No stability"
        elif r['A'] > 0.5:
            mode = "Too chaotic"
        else:
            mode = "Borderline"

        latex += f"{r['rule']} & {r['C']:.2f} & {r['P']:.2f} & {r['F']:.2f} & {r['A']:.2f} & {mode} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Saved LaTeX tables to {output_path}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("C:/Github/TBQ/paper")
    output_dir.mkdir(exist_ok=True)

    # Run census
    results, trivial_rules = run_complete_census(depth=2, num_trials=3)

    # Save outputs
    save_csv(results, output_dir / "rule_census.csv")
    classifications = print_summary(results)
    generate_latex_table(results, output_dir / "census_tables.tex")

    print("\n" + "="*70)
    print("CENSUS COMPLETE")
    print("="*70)
