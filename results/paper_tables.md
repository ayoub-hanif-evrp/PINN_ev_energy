# Paper tables

Generated from `results/tables/`. Do not edit the numbers by hand.

### Table 1. HELECAR-D data used in this study

| Characteristic | Value |
|---|---|
| Analysed trips | 17 |
| Trajectory distribution | T1: 4, T2: 3, T3: 10 |
| Trip duration | 15.9–143.0 min |
| Trip distance | 13.83–59.57 km |
| Mean trip distance | 29.94 km |
| Empirical SoC resolution | ~0.02 percentage points |
| Useful supervision windows | 57,671 |
| Nominal usable battery capacity used in the model | 6.0 kWh (modelling assumption, not a measured dataset field) |

### Table 2. Comparison of evaluated energy-estimation approaches

| Method | Learning level | Supervision | Analytical vehicle physics | Battery-conservation constraint | Primary output |
|---|---|---|---|---|---|
| Physics Model | No learning | Vehicle equations | Yes | No learned conservation loss | Trip energy from integrated analytical power |
| Ridge Regression | Trip-level supervised | SoC-derived trip energy | No | No | Trip energy |
| Data-Driven MLP | Sample-level latent power | Multi-window SoC energy | No | No | Integrated latent power |
| PINN | Sample-level latent power | Multi-window SoC energy + depletion constraint | Yes | Yes | Integrated latent power |

### Table 3. Leave-one-trip-out trip-energy estimation performance

| Method | MAE [kWh] | 95% CI | RMSE [kWh] | MAPE [%] | WAPE [%] | Bias [kWh] | R² |
|---|---:|---|---:|---:|---:|---:|---:|
| Physics Model | 0.780 | [0.614, 0.948] | 0.854 | 30.06 | 29.42 | -0.780 | 0.508 |
| Ridge Regression | 0.368 | [0.229, 0.537] | 0.491 | 16.45 | 13.87 | -0.014 | 0.837 |
| Data-Driven MLP | 0.475 | [0.312, 0.655] | 0.629 | 16.34 | 17.93 | +0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | 11.51 | 11.13 | -0.052 | 0.906 |

MAE and its confidence interval use one value per held-out trip: the mean, across training seeds, of that trip's absolute error. The interval is a trip-level bootstrap of those values. RMSE, MAPE, WAPE, bias, and R² are computed on the pooled seed-level predictions. Neural seeds are training replicates, not additional trips.

### Table 4. Performance under varying numbers of training trips

| Training trips | Ridge Regression | Data-Driven MLP | PINN |
|---:|---:|---:|---:|
| 3 | 2.247 ± 1.986 | 0.710 ± 0.150 | 0.439 ± 0.115 |
| 5 | 5.520 ± 3.031 | 0.697 ± 0.174 | 0.344 ± 0.048 |
| 8 | 2.031 ± 2.521 | 0.434 ± 0.080 | 0.297 ± 0.065 |
| 12 | 0.400 ± 0.061 | 0.459 ± 0.122 | 0.305 ± 0.055 |
| 16 | 0.368 ± 0.000 | 0.475 ± 0.026 | 0.295 ± 0.048 |

Values are mean held-out-trip MAE [kWh] across predefined training-subset replicates; uncertainty denotes the SD of replicate-level means. Neural replicates include initialization seeds. Ridge is deterministic conditional on its selected training subset, so identical full-set fits are not repeated.

### Table 5. Paired trip-level comparison of PINN absolute errors

| Comparison | n | Mean paired absolute-error difference [kWh] | 95% paired bootstrap CI | Wilcoxon p |
|---|---:|---:|---|---:|
| PINN - Physics Model | 17 | -0.485 | [-0.604, -0.375] | 1.53e-05 |
| PINN - Ridge Regression | 17 | -0.073 | [-0.264, 0.110] | 0.579 |
| PINN - Data-Driven MLP | 17 | -0.180 | [-0.316, -0.056] | 0.020 |

Difference = PINN absolute error − comparator absolute error, after averaging per-seed absolute errors within each held-out trip for neural methods. A negative value means lower PINN absolute error. The bootstrap interval is the primary paired summary. The Wilcoxon p-value is a secondary paired test, not a claim based on overlap of separate MAE intervals.

### Table 6. Battery-feasibility decision errors

| Reserve [%] | Physics false-safe [%] | PINN false-safe [%] | Physics conservative [%] | PINN conservative [%] |
|---:|---:|---:|---:|---:|
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 5.88 |
| 15 | 5.88 | 0.00 | 0.00 | 0.00 |
| 20 | 17.65 | 5.88 | 0.00 | 0.00 |

Rates are the fraction of held-out trips on which the energy estimate produces a false-safe or overly conservative battery-feasibility decision. This is not an EV routing algorithm.
