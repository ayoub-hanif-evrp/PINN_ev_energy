# Cached HELECAR-D ablation analysis

Source: `outputs/loto/paper/per_trip_predictions.csv` (existing leave-one-trip-out cache).
No model was retrained for this table.

## Method definitions (from historical trainer)

- `pinn`: full discrete-time PINN (multi-window + conservation + state + residual prior).
- `physics`: analytical vehicle power only (no neural residual).
- `pinn_fulltrip`: same PINN architecture, but window supervision restricted to `full_trip` scale.
- `pinn_no_dynamics`: PINN with conservation/dynamics loss weight set to zero.
- `weak_mlp`: data-driven MLP with multi-window energy supervision only.
- `mlp_state`: legacy MLP with a depletion head (not a requested camera-ready ablation).

## Main trip-energy MAE

| method_id | method | n_trips | MAE_kWh | MAE_CI_low | MAE_CI_high | RMSE_kWh | Bias_kWh | R2 |
|---|---|---|---|---|---|---|---|---|
| physics | Physics Model (no residual) | 17 | 0.780 | 0.614 | 0.948 | 0.854 | -0.780 | 0.508 |
| ridge | Ridge Regression | 17 | 0.368 | 0.229 | 0.537 | 0.491 | -0.014 | 0.837 |
| weak_mlp | Data-Driven MLP | 17 | 0.475 | 0.312 | 0.655 | 0.629 | 0.191 | 0.734 |
| pinn | Full PINN (reference) | 17 | 0.295 | 0.209 | 0.392 | 0.373 | -0.052 | 0.906 |
| pinn_fulltrip | PINN full-trip supervision only | 17 | 0.243 | 0.182 | 0.306 | 0.301 | -0.099 | 0.939 |
| pinn_no_dynamics | PINN without conservation loss | 17 | 0.278 | 0.215 | 0.342 | 0.348 | -0.023 | 0.918 |
| mlp_state | MLP with depletion state (legacy) | 17 | 0.372 | 0.265 | 0.486 | 0.487 | 0.140 | 0.840 |

## Paired absolute-error difference: Full PINN minus comparator

Negative values mean lower PINN absolute error.

| comparison | method_id | n_trips | mean_absolute_error_difference_kWh | bootstrap_CI_low | bootstrap_CI_high | Wilcoxon_p |
|---|---|---|---|---|---|---|
| PINN - Physics Model (no residual) | physics | 17 | -0.485 | -0.604 | -0.375 | 0.000 |
| PINN - Ridge Regression | ridge | 17 | -0.073 | -0.264 | 0.110 | 0.579 |
| PINN - Data-Driven MLP | weak_mlp | 17 | -0.180 | -0.316 | -0.056 | 0.020 |
| PINN - PINN full-trip supervision only | pinn_fulltrip | 17 | 0.052 | -0.030 | 0.132 | 0.517 |
| PINN - PINN without conservation loss | pinn_no_dynamics | 17 | 0.017 | -0.055 | 0.089 | 0.963 |
| PINN - MLP with depletion state (legacy) | mlp_state | 17 | -0.077 | -0.167 | 0.013 | 0.145 |

## Scientific note

If `pinn_fulltrip` has a lower point-estimate MAE than full multi-window `pinn`, that finding is reported,
not concealed. Multi-window supervision remains the accepted-paper method.
