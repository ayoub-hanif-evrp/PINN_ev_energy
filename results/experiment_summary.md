# Experiment summary (paper-writing reference)

Numbers below are taken from `results/tables/` after the final export.
If a CSV changes, regenerate this file from that CSV.

## Dataset

- 17 analysed HELECAR-D trips (T1=4, T2=3, T3=10).
- Duration 952–8577 s.
- CAN-speed distance 13.83–59.57 km (mean 29.94 km).
- Empirical SoC resolution/quantization q ≈ 0.02 percentage points.
- Useful multi-window supervision windows: 57671.

## Main experiment (leave-one-trip-out)

Trip-level energy metrics. MAE confidence intervals are trip-level bootstrap intervals.

| Method | MAE_kWh | MAE_CI | RMSE_kWh | MAPE_pct | WAPE_pct | Bias_kWh | R2 |
|---|---:|---|---:|---:|---:|---:|---:|
| Physics Model | 0.780 | [0.614, 0.948] | 0.854 | 30.06 | 29.42 | -0.780 | 0.508 |
| Regularized Regression | 0.167 | [0.113, 0.232] | 0.210 | 8.72 | 6.31 | -0.006 | 0.970 |
| Data-Driven MLP | 0.475 | [0.312, 0.655] | 0.629 | 16.34 | 17.93 | 0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | 11.51 | 11.13 | -0.052 | 0.906 |

## Data scarcity

Held-out energy MAE as a function of the number of training trips.
Each complete (method, training size, seed) block is included.
For n=3,5,8,12 there are three predefined subset repeats; n=16 uses the full outer-training set.
MAE_kWh is the mean, across training replicates, of the mean absolute error over held-out trips.
uncertainty is the standard deviation of those replicate-level means (seed x subset repeat).
Seeds are training replicates, not additional trips. The statistical unit remains the held-out trip.

| n_train | method | MAE_kWh | uncertainty | number_of_runs |
|---:|---|---:|---:|---:|
| 3 | Regularized Regression | 1.325 | 1.439 | 9 |
| 3 | Data-Driven MLP | 0.710 | 0.150 | 9 |
| 3 | PINN | 0.439 | 0.115 | 9 |
| 5 | Regularized Regression | 1.712 | 2.837 | 9 |
| 5 | Data-Driven MLP | 0.697 | 0.174 | 9 |
| 5 | PINN | 0.344 | 0.048 | 9 |
| 8 | Regularized Regression | 1.171 | 1.932 | 9 |
| 8 | Data-Driven MLP | 0.434 | 0.080 | 9 |
| 8 | PINN | 0.297 | 0.065 | 9 |
| 12 | Regularized Regression | 0.221 | 0.032 | 9 |
| 12 | Data-Driven MLP | 0.459 | 0.122 | 9 |
| 12 | PINN | 0.305 | 0.055 | 9 |
| 16 | Regularized Regression | 0.167 | 0.000 | 3 |
| 16 | Data-Driven MLP | 0.475 | 0.026 | 3 |
| 16 | PINN | 0.295 | 0.048 | 3 |

## Routing-oriented battery-feasibility assessment

Reserve-margin decision errors. This is not an EVRP algorithm.

| reserve_soc_pct | method | false_safe_pct | overly_conservative_pct |
|---:|---|---:|---:|
| 5 | Physics Model | 0.0 | 0.0 |
| 5 | PINN | 0.0 | 0.0 |
| 10 | Physics Model | 0.0 | 0.0 |
| 10 | PINN | 0.0 | 5.9 |
| 15 | Physics Model | 5.9 | 0.0 |
| 15 | PINN | 0.0 | 0.0 |
| 20 | Physics Model | 17.6 | 0.0 |
| 20 | PINN | 5.9 | 0.0 |

## Scientific interpretation

What the full-data LOTO results support:

- Regularized Regression has the lowest full-data LOTO MAE (0.167 kWh).
- PINN MAE (0.295 kWh) is substantially lower than Data-Driven MLP (0.475 kWh).
- PINN MAE is substantially lower than the analytical Physics Model (0.780 kWh).
- The physics-informed structure is useful for the neural estimator under weak SoC supervision.
- The data-scarcity experiment is the evidence for behaviour when fewer training trips are available.

What the results do not support:

- A claim that PINN has lower error than every baseline on full-data LOTO.
- Validation of instantaneous battery power; there are no direct power labels.
- A new EV routing algorithm. The routing section is a battery-feasibility sensitivity.

Data-scarcity detail (3 seeds, predefined repeats):

- n=3: Regularized Regression 1.325 kWh; Data-Driven MLP 0.710 kWh; PINN 0.439 kWh.
- n=5: Regularized Regression 1.712 kWh; Data-Driven MLP 0.697 kWh; PINN 0.344 kWh.
- n=8: Regularized Regression 1.171 kWh; Data-Driven MLP 0.434 kWh; PINN 0.297 kWh.
- n=12: Regularized Regression 0.221 kWh; Data-Driven MLP 0.459 kWh; PINN 0.305 kWh.
- n=16: Regularized Regression 0.167 kWh; Data-Driven MLP 0.475 kWh; PINN 0.295 kWh.
- Compare PINN with Data-Driven MLP at each training size above; do not collapse the curve into a single ranking.
- Main-table PINN MAE averages LOTO seeds 0/1/2 on all 16 remaining trips. Scarcity n=16 reuses those same predictions, so the scarcity n=16 PINN entry is the mean of the three seed-wise trip MAEs.

Feasibility detail:

- At a 20% reserve, Physics Model false-safe rate is 17.6% vs PINN 5.9%.
- The Physics Model produces more false-safe decisions because it systematically under-predicts trip energy.

## Internal note (not a paper experiment)

An internal full-trip-only ablation had a lower LOTO trip-energy MAE than the multi-window PINN.
That finding is not used to retune window sizes or to introduce another research direction.
Multi-window SoC energy supervision remains the method promised by the abstract.
