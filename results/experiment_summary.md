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
| Physics | 0.780 | [0.614, 0.948] | 0.854 | 30.06 | 29.42 | -0.780 | 0.508 |
| ElasticNet | 0.167 | [0.113, 0.232] | 0.210 | 8.72 | 6.31 | -0.006 | 0.970 |
| WeakMLP | 0.475 | [0.312, 0.655] | 0.629 | 16.34 | 17.93 | 0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | 11.51 | 11.13 | -0.052 | 0.906 |

## Data scarcity

Held-out energy MAE as a function of the number of training trips.
Complete aggregation currently uses seed 0 and three subset repeats for n=3,5,8,12;
n=16 is leave-one-trip-out with 16 training trips (one seed).

| n_train | method | MAE_kWh | uncertainty | number_of_runs |
|---:|---|---:|---:|---:|
| 3 | ElasticNet | 0.374 | 0.068 | 3 |
| 3 | WeakMLP | 0.575 | 0.035 | 3 |
| 3 | PINN | 0.317 | 0.038 | 3 |
| 5 | ElasticNet | 0.314 | 0.030 | 3 |
| 5 | WeakMLP | 0.756 | 0.127 | 3 |
| 5 | PINN | 0.340 | 0.040 | 3 |
| 8 | ElasticNet | 1.006 | 1.052 | 3 |
| 8 | WeakMLP | 0.431 | 0.049 | 3 |
| 8 | PINN | 0.333 | 0.038 | 3 |
| 12 | ElasticNet | 0.225 | 0.032 | 3 |
| 12 | WeakMLP | 0.348 | 0.046 | 3 |
| 12 | PINN | 0.266 | 0.013 | 3 |
| 16 | ElasticNet | 0.167 | 0.000 | 1 |
| 16 | WeakMLP | 0.445 | 0.000 | 1 |
| 16 | PINN | 0.350 | 0.000 | 1 |

## Routing-oriented battery-feasibility assessment

Reserve-margin decision errors. This is not an EVRP algorithm.

| reserve_soc_pct | method | false_safe_pct | overly_conservative_pct |
|---:|---|---:|---:|
| 5 | Physics | 0.0 | 0.0 |
| 5 | PINN | 0.0 | 0.0 |
| 10 | Physics | 0.0 | 0.0 |
| 10 | PINN | 0.0 | 5.9 |
| 15 | Physics | 5.9 | 0.0 |
| 15 | PINN | 0.0 | 0.0 |
| 20 | Physics | 17.6 | 0.0 |
| 20 | PINN | 5.9 | 0.0 |

## Scientific interpretation

What the results support:

- ElasticNet is strongest on ordinary full-data LOTO (MAE 0.167 kWh).
- PINN improves substantially over the purely data-driven WeakMLP (0.295 vs 0.475 kWh).
- PINN improves substantially over the analytical physics model (0.295 vs 0.780 kWh).
- Physics-informed learning is a useful inductive bias when instantaneous battery-power labels are unavailable, particularly when training trips are scarce.

What the results do **not** support:

- PINN is not globally best. ElasticNet remains better overall on full-data LOTO.
- Instantaneous battery power has not been validated against ground truth; there are no direct power labels.
- The routing section is only a battery-feasibility sensitivity, not a new routing algorithm.

Data-scarcity detail:

- At 3 training trips, PINN MAE 0.317 kWh is lower than ElasticNet 0.374 and WeakMLP 0.575.
- At 8 training trips, ElasticNet is unstable (MAE 1.006 kWh).
- At 12 and 16 training trips, ElasticNet is again strongest (0.225 and 0.167 kWh).
- PINN remains better than WeakMLP at every reported training size.
- The main-table PINN MAE uses LOTO seeds 0/1/2. Scarcity n=16 uses seed 0 only, so those two PINN numbers need not match.

Feasibility detail:

- At a 20% reserve, Physics false-safe rate is 17.6% vs PINN 5.9%.
- Physics is more false-safe because it systematically under-predicts trip energy.

## Internal note (not a paper experiment)

An internal full-trip-only ablation had a lower LOTO trip-energy MAE than the multi-window PINN.
That finding is not used to retune window sizes or to introduce another research direction.
Multi-window SoC energy supervision remains the method promised by the abstract.
