# Figure Captions

## Figure 1: Classification Distribution

**File:** `fig1_classification.png/pdf`

**Caption:** Classification Distribution of All 256 ECA Rules. Pie chart showing the breakdown of rule classifications under stickiness (depth=2). Life-like rules (green, 62.1%) dominate the non-trivial population, with Computing rules (orange, 10.9%) and Crystallized rules (blue, 1.2%) representing failure modes. Trivial rules (gray, 25.8%) have insufficient dynamics to generate meaningful hidden state variation. Among non-trivial rules, 83.7% become life-like—inverting the assumption that self-maintenance is rare.

---

## Figure 2: Control vs Activity

**File:** `fig2_control_activity.png/pdf`

**Caption:** Control vs Activity Distribution. Scatter plot showing the relationship between Control and Activity across all 256 ECA rules under stickiness. Life-like rules (green) cluster in the region with Control > 0 and Activity in the Goldilocks zone (0.05–0.5). Computing rules (orange) have Control but lack stability mechanisms. Crystallized rules (blue) have excessive stability with Activity < 0.05. The distribution demonstrates that Control is necessary but not sufficient for life-like behavior.

---

## Figure 3: Stability Mechanisms

**File:** `fig3_stability.png/pdf`

**Caption:** Stability Mechanisms: Absorption vs Repair. Scatter plot of Absorption (P) vs Repair (F) for all non-trivial ECA rules. Rules in the upper-right region satisfy stability criteria and become life-like when combined with appropriate activity levels. The two stability mechanisms are partially independent: some rules achieve life-like status through high absorption (linear rules like 90), others through high repair (conservative rules like 184), and some through both.

---

## Figure 4: Spacetime Evolution

**File:** `fig4_spacetime.png/pdf`

**Caption:** Spacetime Evolution of Representative Rules. Comparison of spacetime diagrams for four key rules under stickiness. (a) Rule 90 (Life-Like): XOR-based dynamics with perfect absorption—perturbations cancel through linear superposition. (b) Rule 184 (Life-Like): Traffic rule with particle conservation—strong repair through attractor dynamics. (c) Rule 30 (Computing): Chaotic dynamics spread perturbations despite hidden state. (d) Rule 110 (Computing): Turing-complete but unstable—computational universality does not confer self-maintenance.

---

## Figure S1: Stickiness Depth Heatmap (Supplementary)

**File:** `figS1_depth_heatmap.png/pdf`

**Caption:** Effect of Stickiness Depth on Classification. Heatmap showing how rule classifications change with increasing stickiness depth (confirmation threshold). Most rules achieve stable classification by depth 2. Some rules (e.g., Rule 54) require higher depths to transition to life-like behavior, while others crystallize at higher depths. The depth parameter acts as a phase transition control, with different rules having different critical depths for life-like behavior.

---

## Summary Table

| Figure | Title | Key Message |
|--------|-------|-------------|
| 1 | Classification Distribution | 83.7% of non-trivial rules are life-like |
| 2 | Control vs Activity | Control necessary but not sufficient |
| 3 | Stability Mechanisms | Two routes: absorption and repair |
| 4 | Spacetime Evolution | Visual comparison of rule behaviors |
| S1 | Depth Heatmap | Stickiness depth effects |
