# Frozen paper protocol

This file records the experimental protocol **before** the paper LOTO is used for conclusions.
Do not tune these choices using outer-test paper metrics.

- commit SHA: `9bae16f6342a5c45a81f23b70a7b747a30d9d064` (last committed revision at freeze; the pipeline files in the working tree were frozen with this protocol and should be committed with the paper run)
- config: `configs/paper.yaml`
- Python: 3.12.10
- PyTorch: 2.14.0+cpu
- device: auto (resolved at runtime; smoke/quick used CPU)

## Features

- Power-branch telemetry: ['speed_mps', 'acc_mps2', 'theta_rad', 'temperature_c', 'humidity_pct', 'wind_speed_mps', 'traffic', 'speed_limit_mps']
- Causal progress features (state head only): ['elapsed_s', 'cumulative_distance_km']
- ElasticNet whitelist: ['distance_km', 'duration_s', 'mean_speed_mps', 'std_speed_mps', 'max_speed_mps', 'idle_fraction', 'mean_positive_acc', 'acc_std', 'positive_acc_time_fraction', 'braking_time_fraction', 'cumulative_elevation_gain_m', 'cumulative_elevation_loss_m', 'mean_abs_grade', 'mean_temperature_c', 'mean_humidity_pct', 'mean_wind_speed_mps', 'mean_traffic', 'traffic_frac_0', 'traffic_frac_1', 'traffic_frac_2', 'mean_speed_limit_kmh']
- Excluded from the power branch: SoC, LAT, LON, Date, absolute Time
- WindSpeedAv is m/s and is **not** used as aerodynamic headwind.

## Physics parameters (assumptions, not HELECAR measurements)

- `battery_capacity_kwh`: 6.0
- `mass_kg`: 549.0
- `Crr`: 0.012
- `cda_m2`: 0.64
- `cd`: None
- `frontal_area_m2`: None
- `rho_air`: 1.225
- `eta_drive`: 0.85
- `eta_regen`: 0.4
- `auxiliary_power_kw`: 0.2
- `maximum_wheel_power_kw`: 13.0
- `maximum_regen_wheel_power_kw`: 8.0
- `g`: 9.80665

## SoC windows

- event ΔSoC thresholds (pp): [0.1, 0.2, 0.5]
- event min duration (s): 60.0
- fixed times (s): [60, 120, 300, 600]
- full trip: True
- Observed window energy: `E_obs(a,b) = E_batt (SOC[a]-SOC[b])/100`
- Predicted window energy: trapezoidal `E(a,b) = prefix[b]-prefix[a]`

## Discrete-time PINN

- `P_hat = P_physics + residual_limit * tanh(head_delta(h))`
- `D_hat(t) = D_raw(t) - D_raw(0)` so `D_hat(0)=0`
- Interval residual: `r_i = D_hat[i+1]-D_hat[i] - 100 ΔE_i / E_batt`
- `ΔE_i = 0.5 (P_hat[i]+P_hat[i+1]) dt[i] / 3600`
- Loss: `L = λ_window L_window + λ_dynamics L_dynamics + λ_state L_state + λ_prior L_prior`
- λ_window=1.0 λ_dynamics=1.0 λ_state=0.1 λ_prior=0.1 λ_boundary=0.0 (diagnostic only)
- hidden=[64, 64, 64] activation=tanh residual_limit_kw=4.0
- optimizer=adam lr=0.001 weight_decay=1e-05
- max_epochs=1200 patience=100

## Validation / retraining

- Grouped inner validation: one trip per available T1/T2/T3, deterministic in fold ID + seed
- Validation score: mean over scales of window MAE (scale-balanced)
- After selection, discard the model; refit scalers and q on all outer-training trips; retrain selected epochs
- q_inner_train / q_outer_train from training trips only; q_test_posthoc is evaluation-only

- seeds: [0, 1, 2]
- Primary metric: trip energy MAE (kWh). Also RMSE, bias, MAPE (exclude |E_obs|<=1e-12), WAPE, R²
- Statistics: trip-level paired bootstrap (10,000), then optional Wilcoxon as secondary
- Exclusions: none planned; failed folds are recorded, not dropped

This is a discrete-time physics-informed neural network, not an automatic-differentiation PDE PINN.
