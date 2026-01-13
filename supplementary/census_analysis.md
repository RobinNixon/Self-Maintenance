# Complete Census Analysis

## Classification Distribution

| Category | Count | % of All | % of Non-Trivial |
|----------|-------|----------|------------------|
| **LIFE-LIKE** | **159** | **62.1%** | **83.7%** |
| COMPUTING | 28 | 10.9% | 14.7% |
| CRYSTALLIZED | 3 | 1.2% | 1.6% |
| TRIVIAL | 66 | 25.8% | — |
| **Total** | **256** | **100%** | — |

## Failure Mode Analysis

### Computing Rules (28 total)

Rules with Control > 0 but failing stability criteria:

| Rule | Control | Absorption | Repair | Activity | Failure Mode |
|------|---------|------------|--------|----------|--------------|
| 30 | 0.52 | 0.03 | 0.62 | 0.25 | Chaotic |
| 110 | 0.39 | 0.30 | 0.47 | 0.20 | No stability |
| 150 | 0.54 | 0.00 | 0.62 | 0.25 | No absorption |
| ... | ... | ... | ... | ... | ... |

### Crystallized Rules (3 total)

Rules with stability but Activity < 0.05:

| Rule | Control | Absorption | Repair | Activity |
|------|---------|------------|--------|----------|
| 108 | 0.27 | 1.00 | 0.79 | 0.05 |
| 201 | 0.22 | 1.00 | 0.76 | 0.02 |
| 216 | 0.24 | 1.00 | 0.52 | 0.01 |

## Famous Rules Analysis

| Rule | Type | Classification | Key Property |
|------|------|----------------|--------------|
| 30 | Chaotic | COMPUTING | Maximum chaos spreads perturbations |
| 90 | Linear (XOR) | LIFE-LIKE | Perfect absorption through linearity |
| 110 | Universal | COMPUTING | Computational power ≠ self-maintenance |
| 184 | Traffic | LIFE-LIKE | Particle conservation enables repair |

## The Rule 150 Anomaly

Both Rule 90 and Rule 150 are XOR-based (linear), yet:
- Rule 90: Absorption = 1.00 → LIFE-LIKE
- Rule 150: Absorption = 0.00 → COMPUTING

**Resolution:** Linearity is necessary but not sufficient. The specific bit pattern determines cancellation vs propagation behavior.

## Data Files

- `data/rule_census.csv` — Complete metrics for all 256 rules
- `data/full_rule_survey.json` — Detailed survey results
- `data/phase6_falsification.json` — Falsification test results
