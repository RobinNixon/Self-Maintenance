"""
Publication-Quality Figures for Life-Like Classification Paper

Generates only figures that directly support claims.
No decorative plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# Load Census Data
# =============================================================================

def load_census(csv_path: str) -> list:
    """Load census data from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['rule'] = int(row['rule'])
            row['trivial'] = row['trivial'] == 'True'
            for key in ['C', 'P', 'F', 'A']:
                row[key] = float(row[key])
            results.append(row)
    return results

# =============================================================================
# Figure 1: Classification Distribution
# =============================================================================

def figure_1_classification_distribution(results: list, output_dir: Path):
    """
    Bar chart showing life-like vs computing vs crystallized.
    Claims supported: Life-like is the majority outcome (83.7%).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Count classifications
    counts = {'LIFE-LIKE': 0, 'COMPUTING': 0, 'CRYSTALLIZED': 0, 'TRIVIAL': 0}
    for r in results:
        counts[r['classification']] += 1

    # Left panel: All 256 rules
    categories = ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED', 'TRIVIAL']
    values = [counts[c] for c in categories]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']

    bars1 = ax1.bar(range(len(categories)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(['Life-like', 'Computing', 'Crystal.', 'Trivial'], rotation=0)
    ax1.set_ylabel('Number of rules')
    ax1.set_title('(a) All 256 ECA rules')

    # Add count labels
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontsize=9)

    # Right panel: Non-trivial only (190)
    non_trivial_counts = [counts['LIFE-LIKE'], counts['COMPUTING'], counts['CRYSTALLIZED']]
    non_trivial_total = sum(non_trivial_counts)
    percentages = [100 * c / non_trivial_total for c in non_trivial_counts]

    bars2 = ax2.bar(range(3), non_trivial_counts, color=colors[:3], edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Life-like', 'Computing', 'Crystal.'], rotation=0)
    ax2.set_ylabel('Number of rules')
    ax2.set_title(f'(b) Non-trivial rules (n={non_trivial_total})')

    # Add percentage labels
    for bar, val, pct in zip(bars2, non_trivial_counts, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_classification.pdf')
    plt.savefig(output_dir / 'fig1_classification.png')
    plt.close()

    print("Generated Figure 1: Classification distribution")

# =============================================================================
# Figure 2: Control vs Classification
# =============================================================================

def figure_2_control_vs_class(results: list, output_dir: Path):
    """
    Scatter plot demonstrating Control is necessary but not sufficient.
    Claims supported: All life-like have C > 0, but some C > 0 are not life-like.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Separate by classification
    life_like = [(r['C'], r['A']) for r in results if r['classification'] == 'LIFE-LIKE']
    computing = [(r['C'], r['A']) for r in results if r['classification'] == 'COMPUTING']
    crystallized = [(r['C'], r['A']) for r in results if r['classification'] == 'CRYSTALLIZED']

    # Plot
    if life_like:
        ax.scatter([x[0] for x in life_like], [x[1] for x in life_like],
                   c='#2ecc71', label=f'Life-like (n={len(life_like)})',
                   alpha=0.7, s=30, edgecolors='black', linewidths=0.3)

    if computing:
        ax.scatter([x[0] for x in computing], [x[1] for x in computing],
                   c='#e74c3c', label=f'Computing (n={len(computing)})',
                   alpha=0.7, s=30, marker='s', edgecolors='black', linewidths=0.3)

    if crystallized:
        ax.scatter([x[0] for x in crystallized], [x[1] for x in crystallized],
                   c='#3498db', label=f'Crystallized (n={len(crystallized)})',
                   alpha=0.7, s=50, marker='^', edgecolors='black', linewidths=0.3)

    # Mark famous rules
    famous = {30: 'R30', 110: 'R110', 90: 'R90', 184: 'R184'}
    for r in results:
        if r['rule'] in famous:
            ax.annotate(famous[r['rule']], (r['C'], r['A']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Goldilocks zone
    ax.axhline(0.05, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.fill_between([0, 1], 0.05, 0.5, alpha=0.1, color='green')

    ax.set_xlabel('Control (C)')
    ax.set_ylabel('Activity (A)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.7)
    ax.legend(loc='upper right', framealpha=0.9)

    # Add annotation for Goldilocks zone
    ax.text(0.02, 0.27, 'Activity\nwindow', fontsize=8, color='gray', alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_control_activity.pdf')
    plt.savefig(output_dir / 'fig2_control_activity.png')
    plt.close()

    print("Generated Figure 2: Control vs Activity scatter")

# =============================================================================
# Figure 3: Absorption vs Repair
# =============================================================================

def figure_3_stability_mechanisms(results: list, output_dir: Path):
    """
    Scatter plot showing two paths to life-like: absorption OR repair.
    Claims supported: Life-like requires stability mechanism but not both.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Non-trivial only
    non_trivial = [r for r in results if not r['trivial']]

    life_like = [(r['P'], r['F']) for r in non_trivial if r['classification'] == 'LIFE-LIKE']
    computing = [(r['P'], r['F']) for r in non_trivial if r['classification'] == 'COMPUTING']
    crystallized = [(r['P'], r['F']) for r in non_trivial if r['classification'] == 'CRYSTALLIZED']

    if life_like:
        ax.scatter([x[0] for x in life_like], [x[1] for x in life_like],
                   c='#2ecc71', label='Life-like', alpha=0.7, s=30,
                   edgecolors='black', linewidths=0.3)

    if computing:
        ax.scatter([x[0] for x in computing], [x[1] for x in computing],
                   c='#e74c3c', label='Computing', alpha=0.7, s=30, marker='s',
                   edgecolors='black', linewidths=0.3)

    if crystallized:
        ax.scatter([x[0] for x in crystallized], [x[1] for x in crystallized],
                   c='#3498db', label='Crystallized', alpha=0.7, s=50, marker='^',
                   edgecolors='black', linewidths=0.3)

    # Threshold lines
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(0.7, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Shade the "either" region
    ax.fill_between([0.5, 1.0], 0, 1, alpha=0.1, color='green')
    ax.fill_between([0, 1.0], 0.7, 1.0, alpha=0.1, color='green')

    ax.set_xlabel('Perturbation Absorption (P)')
    ax.set_ylabel('Self-Repair (F)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', framealpha=0.9)

    ax.text(0.75, 0.3, 'P > 0.5', fontsize=8, color='gray')
    ax.text(0.1, 0.85, 'F > 0.7', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_stability.pdf')
    plt.savefig(output_dir / 'fig3_stability.png')
    plt.close()

    print("Generated Figure 3: Stability mechanisms scatter")

# =============================================================================
# Figure 4: Exemplar Spacetime Diagrams
# =============================================================================

def apply_eca_rule(left, center, right, rule):
    index = int(left) << 2 | int(center) << 1 | int(right)
    return (rule >> index) & 1

def run_sticky_eca(rule, initial, steps, depth):
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

def figure_4_spacetime_exemplars(output_dir: Path):
    """
    Three spacetime diagrams: life-like (Rule 90), computing (Rule 30), crystallized (Rule 108).
    Claims supported: Visual distinction between categories.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))

    np.random.seed(42)
    width = 61
    steps = 60
    depth = 2

    exemplars = [
        (90, 'LIFE-LIKE', 'Rule 90 (Life-like)'),
        (30, 'COMPUTING', 'Rule 30 (Computing)'),
        (108, 'CRYSTALLIZED', 'Rule 108 (Crystallized)')
    ]

    for ax, (rule, cls, title) in zip(axes, exemplars):
        initial = np.random.randint(0, 2, width, dtype=np.int8)
        visible, _ = run_sticky_eca(rule, initial, steps, depth)

        ax.imshow(visible, cmap='binary', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Cell position')
        ax.set_ylabel('Time')
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_spacetime.pdf')
    plt.savefig(output_dir / 'fig4_spacetime.png')
    plt.close()

    print("Generated Figure 4: Spacetime exemplars")

# =============================================================================
# Figure S1 (Supplementary): Depth Heatmap
# =============================================================================

def figure_s1_depth_heatmap(output_dir: Path):
    """
    Heatmap showing classification at different stickiness depths.
    Supplementary figure showing phase transition.
    """
    # Test subset of rules at multiple depths
    test_rules = [30, 54, 60, 90, 110, 150, 184]
    depths = [1, 2, 3, 4, 5]

    width = 41
    steps = 80

    # Classification matrix
    matrix = np.zeros((len(test_rules), len(depths)))

    for i, rule in enumerate(test_rules):
        for j, depth in enumerate(depths):
            np.random.seed(rule)
            initial = np.random.randint(0, 2, width, dtype=np.int8)
            visible, _ = run_sticky_eca(rule, initial, steps, depth)

            # Quick classification
            changes = np.abs(np.diff(visible.astype(int), axis=0))
            A = np.mean(changes)

            # Simplified absorption test
            perturbed = initial.copy()
            perturbed[width//2] = 1 - perturbed[width//2]
            v_orig, _ = run_sticky_eca(rule, initial, steps, depth)
            v_pert, _ = run_sticky_eca(rule, perturbed, steps, depth)
            diff = np.sum(np.abs(v_orig[-1].astype(int) - v_pert[-1].astype(int)))
            P = 1.0 if diff < width * 0.2 else 0.0

            if A < 0.05:
                matrix[i, j] = 0  # Crystallized
            elif P > 0.5 and 0.05 <= A <= 0.5:
                matrix[i, j] = 2  # Life-like
            else:
                matrix[i, j] = 1  # Computing

    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = plt.cm.colors.ListedColormap(['#3498db', '#e74c3c', '#2ecc71'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(test_rules)))
    ax.set_yticklabels([f'R{r}' for r in test_rules])

    ax.set_xlabel('Stickiness Depth')
    ax.set_ylabel('Rule')
    ax.set_title('Classification by Depth')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='Crystallized'),
        mpatches.Patch(facecolor='#e74c3c', label='Computing'),
        mpatches.Patch(facecolor='#2ecc71', label='Life-like'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(output_dir / 'figS1_depth_heatmap.pdf')
    plt.savefig(output_dir / 'figS1_depth_heatmap.png')
    plt.close()

    print("Generated Figure S1: Depth heatmap (supplementary)")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("C:/Github/TBQ/paper/figures")
    output_dir.mkdir(exist_ok=True)

    # Load census
    results = load_census("C:/Github/TBQ/paper/rule_census.csv")

    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    figure_1_classification_distribution(results, output_dir)
    figure_2_control_vs_class(results, output_dir)
    figure_3_stability_mechanisms(results, output_dir)
    figure_4_spacetime_exemplars(output_dir)
    figure_s1_depth_heatmap(output_dir)

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}")
