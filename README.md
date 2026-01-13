# Self-Maintenance as a Default Outcome in Hidden-State Discrete Dynamical Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We present a classification result for elementary cellular automata (ECA) augmented with hidden state: **83.7% of non-trivial rules exhibit life-like behavior** characterized by active self-maintenance. This inverts the standard assumption that self-maintenance is rare or requires special conditions. We show that hidden state is necessary for Control (counterfactual context-dependence), but Control alone is not sufficient for life-like behavior. Life-like behavior additionally requires a stability mechanism (perturbation absorption or self-repair) and dynamics within an activity window.

## Key Results

### Central Finding
**83.7% of non-trivial ECA rules become life-like under stickiness.**

This inverts the standard assumption: self-maintenance is the default outcome, not a rare exception.

### The Three Conditions

A system is **life-like** if and only if:
1. **Control > 0** — Hidden state influences visible output
2. **Absorption > 0.5 OR Repair > 0.7** — Stability mechanism exists
3. **0.05 ≤ Activity ≤ 0.5** — Dynamics in Goldilocks zone

### Classification Distribution

| Category | Count | % of Non-Trivial |
|----------|-------|------------------|
| **LIFE-LIKE** | **159** | **83.7%** |
| COMPUTING | 28 | 14.7% |
| CRYSTALLIZED | 3 | 1.6% |

### Famous Rules

| Rule | Classification | Why |
|------|----------------|-----|
| 30 | COMPUTING | Chaotic—perturbations spread |
| 90 | LIFE-LIKE | Linear—perfect absorption |
| 110 | COMPUTING | Universal but unstable |
| 184 | LIFE-LIKE | Conservative—strong repair |

## Repository Structure

```
Self/
├── paper/
│   ├── self-maintenance.tex            # Full paper (LaTeX) with figures
│   ├── self-maintenance.md             # Full paper (Markdown)
│   └── references.bib                  # Bibliography
├── figures/
│   ├── fig1_classification.png/pdf     # Classification pie chart
│   ├── fig2_control_activity.png/pdf   # Control vs Activity scatter
│   ├── fig3_stability.png/pdf          # Absorption vs Repair scatter
│   ├── fig4_spacetime.png/pdf          # Spacetime comparisons
│   ├── figS1_depth_heatmap.png/pdf     # Stickiness depth effects
│   └── FIGURE_CAPTIONS.md              # Detailed figure captions
├── code/
│   ├── aliveness_metrics.py            # Core metric implementations
│   ├── full_rule_survey.py             # Complete 256-rule survey
│   ├── complete_census.py              # Census generation
│   ├── generate_figures.py             # Figure generation
│   └── phase*.py                       # Experimental phases
├── data/
│   ├── rule_census.csv                 # Complete metrics for all 256 rules
│   ├── full_rule_survey.json           # Detailed survey results
│   └── *.json                          # Additional data files
├── theory/
│   ├── INDEX.md                        # Theory file index
│   ├── PHASE_0_MICRO_CLAIMS.md         # Initial calibration
│   ├── PHASE_1_ALIVENESS_DEFINITIONS.md # Metric definitions
│   ├── PHASE_4_CLASSIFICATION_LOGIC.md  # Classification framework
│   ├── PHASE_5_MECHANISM_HYPOTHESIS.md  # Causal mechanism
│   └── PHASE_6_FALSIFICATION_RESULTS.md # Falsification tests
├── supplementary/
│   ├── census_analysis.md              # Detailed census analysis
│   └── metric_definitions.md           # Metric definitions and protocols
├── discussion/
│   ├── INDEX.md                        # Discussion index
│   ├── IMPLICATIONS.md                 # Implications for AL, origins, complexity
│   ├── FUTURE_WORK.md                  # Open problems
│   └── FAQ.md                          # Common questions
├── LICENSE
└── README.md                           # This file
```

## Quick Start

### Requirements
- Python 3.10+
- NumPy, Matplotlib, SciPy

### Run Experiments
```bash
cd code
python aliveness_metrics.py      # Core metrics
python full_rule_survey.py       # Complete survey
```

### Generate Figures
```bash
cd code
python generate_figures.py
```

### Build Paper (LaTeX)
```bash
cd paper
pdflatex self-maintenance.tex
bibtex self-maintenance
pdflatex self-maintenance.tex
pdflatex self-maintenance.tex
```

## Key Insights

### Life-Like ≠ Computation
Rule 110 is Turing-complete but NOT life-like. Computational universality is orthogonal to self-maintenance.

### Life-Like ≠ Chaos
Rule 30 is maximally chaotic but NOT life-like. Complex dynamics do not confer self-maintenance.

### Life-Like ≠ Stability
Rules 108, 201, 216 have perfect stability but are NOT life-like (crystallized). Excessive stability is death, not life.

### Why Life-Like Is Common
- Control > 0 is guaranteed by stickiness
- Activity window captures most dynamics
- Stability mechanisms are generic (linear or conservative rules)

The 15% failure rate represents rules that are chaotic AND lack attractor structure.

## The Mechanism

```
Stickiness → Hidden State → Control → [+ Stability] → Life-Like
```

Hidden state provides **flexibility** (context-dependent response).
Rule dynamics provide **structure** (absorption or repair mechanisms).
When flexibility meets structure, self-maintenance follows.

## Citation

```bibtex
@article{self_maintenance_2026,
  title={Self-Maintenance as a Default Outcome in Hidden-State Discrete Dynamical Systems},
  author={Nixon, Robin},
  journal={[Journal]},
  year={2026},
  note={Preprint}
}
```

## References

1. Wolfram, S. (1983). "Statistical mechanics of cellular automata." *Rev. Mod. Phys.*
2. Cook, M. (2004). "Universality in Elementary Cellular Automata." *Complex Systems*
3. Langton, C.G. (1990). "Computation at the Edge of Chaos." *Physica D*
4. Maturana, H.R. & Varela, F.J. (1980). *Autopoiesis and Cognition*. D. Reidel.
5. Schrödinger, E. (1944). *What is Life?* Cambridge University Press.

## Research Series

This paper is part of an ongoing multi-paper research project exploring the foundations of computation, self-organization, and life-like behavior in discrete dynamical systems.

### Papers in This Series

| # | Paper | Repository | Key Contribution |
|---|-------|------------|------------------|
| 1 | The Five-Bit Threshold (UCT) | [RobinNixon/UCT](https://github.com/RobinNixon/UCT) | Proves minimum complexity for universal computation |
| 2 | Stickiness Control | [RobinNixon/Stickiness](https://github.com/RobinNixon/Stickiness) | Temporal filtering mechanism for self-maintenance |
| **3** | **Self-Maintenance** | [This repo](https://github.com/RobinNixon/Self-Maintenance) | Complete framework for engineering life-like behavior |
| 4 | Substrate Leakiness | [RobinNixon/Leakiness](https://github.com/RobinNixon/Leakiness) | Predictive model for life-like potential |

### Recommended Reading Order

1. **UCT** - Establishes the theoretical foundation: why 5 bits of complexity are necessary and sufficient for universal computation
2. **Stickiness** - Introduces temporal filtering as a mechanism for controlling chaotic dynamics
3. **Self-Maintenance (this paper)** - Builds on stickiness to define and measure life-like behavior in cellular automata
4. **Leakiness** - Synthesizes the framework into a predictive two-axis model for engineering self-maintenance

This paper synthesizes UCT's complexity threshold and stickiness control into a complete framework for self-maintenance. It defines what "life-like" means operationally and demonstrates how to engineer it in Rule 110 and similar systems.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions, particularly:
- Extensions to 2D cellular automata
- Continuous system formulations
- Physical substrate implementations
- Evolutionary dynamics studies

Please open an issue or submit a pull request.
