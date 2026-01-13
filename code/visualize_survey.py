"""
Visualize the full rule survey results
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_rule_grid():
    """Create a 16x16 grid visualization of all 256 rules."""

    # Load results
    with open("C:/Github/TBQ/output/full_rule_survey.json", 'r') as f:
        data = json.load(f)

    results = data['results']
    classifications = data['classifications']

    # Color map
    colors = {
        'LIFE-LIKE': '#2ecc71',      # Green
        'COMPUTING': '#e74c3c',       # Red
        'CRYSTALLIZED': '#3498db',    # Blue
        'TRIVIAL': '#95a5a6',         # Gray
        'UNSTABLE': '#f39c12',        # Orange
        'CHAOTIC_STABLE': '#9b59b6',  # Purple
        'ERROR': '#000000'            # Black
    }

    # Create grid
    fig, ax = plt.subplots(figsize=(14, 14))

    grid = np.zeros((16, 16, 3))

    for rule in range(256):
        row = rule // 16
        col = rule % 16

        r = results.get(str(rule), {})
        cls = r.get('classification', 'ERROR')

        # Convert hex to RGB
        hex_color = colors.get(cls, '#000000')
        rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5))
        grid[row, col] = rgb

    ax.imshow(grid, aspect='equal')

    # Add rule numbers
    for rule in range(256):
        row = rule // 16
        col = rule % 16

        r = results.get(str(rule), {})
        cls = r.get('classification', 'ERROR')

        # White text on dark, black on light
        text_color = 'white' if cls in ['COMPUTING', 'TRIVIAL', 'ERROR'] else 'black'

        ax.text(col, row, str(rule), ha='center', va='center',
                fontsize=7, color=text_color, fontweight='bold')

    # Highlight famous rules
    famous = [30, 54, 90, 110, 184, 150]
    for rule in famous:
        row = rule // 16
        col = rule % 16
        rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False,
                             edgecolor='yellow', linewidth=3)
        ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('All 256 ECA Rules: Life-Like Classification Under Stickiness\n'
                 '(Yellow border = famous rules)', fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=colors['LIFE-LIKE'], label=f"LIFE-LIKE ({len(classifications['LIFE-LIKE'])})"),
        plt.Rectangle((0,0), 1, 1, facecolor=colors['COMPUTING'], label=f"COMPUTING ({len(classifications['COMPUTING'])})"),
        plt.Rectangle((0,0), 1, 1, facecolor=colors['CRYSTALLIZED'], label=f"CRYSTALLIZED ({len(classifications['CRYSTALLIZED'])})"),
        plt.Rectangle((0,0), 1, 1, facecolor=colors['TRIVIAL'], label=f"TRIVIAL ({len(classifications['TRIVIAL'])})"),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig('C:/Github/TBQ/output/rule_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved rule_grid.png")

def create_metric_scatter():
    """Create scatter plots of metrics."""

    with open("C:/Github/TBQ/output/full_rule_survey.json", 'r') as f:
        data = json.load(f)

    results = data['results']

    # Extract non-trivial rules
    P_vals, F_vals, A_vals, C_vals = [], [], [], []
    colors_list = []
    rules_list = []

    color_map = {
        'LIFE-LIKE': '#2ecc71',
        'COMPUTING': '#e74c3c',
        'CRYSTALLIZED': '#3498db',
        'UNSTABLE': '#f39c12'
    }

    for rule in range(256):
        r = results.get(str(rule), {})
        if r.get('trivial', True):
            continue

        cls = r.get('classification', '')
        if cls not in color_map:
            continue

        P_vals.append(r.get('P', 0))
        F_vals.append(r.get('F', 0))
        A_vals.append(r.get('A', 0))
        C_vals.append(r.get('C', 0))
        colors_list.append(color_map[cls])
        rules_list.append(rule)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # P vs A (Absorption vs Activity)
    ax = axes[0, 0]
    ax.scatter(A_vals, P_vals, c=colors_list, alpha=0.7, s=50)
    ax.set_xlabel('Activity (A)')
    ax.set_ylabel('Perturbation Absorption (P)')
    ax.set_title('Absorption vs Activity')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='P threshold')
    ax.axvline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.fill_betweenx([0.5, 1.0], 0.05, 0.5, alpha=0.1, color='green', label='Life-like zone')

    # F vs A (Repair vs Activity)
    ax = axes[0, 1]
    ax.scatter(A_vals, F_vals, c=colors_list, alpha=0.7, s=50)
    ax.set_xlabel('Activity (A)')
    ax.set_ylabel('Self-Repair (F)')
    ax.set_title('Repair vs Activity')
    ax.axhline(0.7, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # P vs F (Absorption vs Repair)
    ax = axes[1, 0]
    ax.scatter(F_vals, P_vals, c=colors_list, alpha=0.7, s=50)
    ax.set_xlabel('Self-Repair (F)')
    ax.set_ylabel('Perturbation Absorption (P)')
    ax.set_title('Absorption vs Repair')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.7, color='gray', linestyle='--', alpha=0.5)

    # C vs A (Control vs Activity)
    ax = axes[1, 1]
    ax.scatter(A_vals, C_vals, c=colors_list, alpha=0.7, s=50)
    ax.set_xlabel('Activity (A)')
    ax.set_ylabel('Control (C)')
    ax.set_title('Control vs Activity')
    ax.axvline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark famous rules
    famous = {30: 'R30', 54: 'R54', 90: 'R90', 110: 'R110', 184: 'R184'}
    for rule, label in famous.items():
        if rule in rules_list:
            idx = rules_list.index(rule)
            for ax in axes.flat:
                pass  # Could add annotations here

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='LIFE-LIKE'),
        Patch(facecolor='#e74c3c', label='COMPUTING'),
        Patch(facecolor='#3498db', label='CRYSTALLIZED'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Metric Space of Non-Trivial ECA Rules', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Github/TBQ/output/metric_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved metric_scatter.png")

def create_summary_chart():
    """Create pie chart summary."""

    with open("C:/Github/TBQ/output/full_rule_survey.json", 'r') as f:
        data = json.load(f)

    classifications = data['classifications']

    # Pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # All 256 rules
    labels1 = ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED', 'TRIVIAL']
    sizes1 = [len(classifications.get(c, [])) for c in labels1]
    colors1 = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']

    ax1.pie(sizes1, labels=[f'{l}\n({s})' for l, s in zip(labels1, sizes1)],
            colors=colors1, autopct='%1.1f%%', startangle=90)
    ax1.set_title('All 256 ECA Rules', fontsize=12, fontweight='bold')

    # Non-trivial only
    labels2 = ['LIFE-LIKE', 'COMPUTING', 'CRYSTALLIZED']
    sizes2 = [len(classifications.get(c, [])) for c in labels2]
    colors2 = ['#2ecc71', '#e74c3c', '#3498db']

    ax2.pie(sizes2, labels=[f'{l}\n({s})' for l, s in zip(labels2, sizes2)],
            colors=colors2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Non-Trivial Rules Only (184 total)', fontsize=12, fontweight='bold')

    plt.suptitle('Life-Like Classification Under Stickiness (depth=2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Github/TBQ/output/summary_pie.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved summary_pie.png")

def print_detailed_stats():
    """Print detailed statistics."""

    with open("C:/Github/TBQ/output/full_rule_survey.json", 'r') as f:
        data = json.load(f)

    results = data['results']
    classifications = data['classifications']

    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)

    # Life-like rules breakdown
    life_like = classifications.get('LIFE-LIKE', [])
    computing = classifications.get('COMPUTING', [])

    print(f"\nLIFE-LIKE rules ({len(life_like)}):")

    # Analyze what makes them life-like
    absorption_only = []
    repair_only = []
    both = []

    for rule in life_like:
        r = results[str(rule)]
        has_absorption = r['P'] > 0.5
        has_repair = r['F'] > 0.7

        if has_absorption and has_repair:
            both.append(rule)
        elif has_absorption:
            absorption_only.append(rule)
        elif has_repair:
            repair_only.append(rule)

    print(f"  - Via absorption only: {len(absorption_only)} rules")
    print(f"  - Via repair only: {len(repair_only)} rules")
    print(f"  - Via both: {len(both)} rules")

    print(f"\nCOMPUTING rules ({len(computing)}):")
    print(f"  Rules: {sorted(computing)}")

    # Why do computing rules fail?
    print("\n  Why they fail:")
    for rule in computing:
        r = results[str(rule)]
        reasons = []
        if r['P'] <= 0.5:
            reasons.append(f"low absorption ({r['P']:.2f})")
        if r['F'] <= 0.7:
            reasons.append(f"low repair ({r['F']:.2f})")
        if r['A'] <= 0.05:
            reasons.append(f"crystallized ({r['A']:.2f})")
        if r['A'] >= 0.5:
            reasons.append(f"chaotic ({r['A']:.2f})")

        print(f"    Rule {rule}: {', '.join(reasons)}")

if __name__ == "__main__":
    print("Creating visualizations...")
    create_rule_grid()
    create_metric_scatter()
    create_summary_chart()
    print_detailed_stats()
    print("\nDone!")
