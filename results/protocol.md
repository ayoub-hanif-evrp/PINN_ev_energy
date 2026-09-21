# Frozen paper protocol

This document describes the code path that produced `results/`. It replaces older notes that said the pipeline had not been committed.

## Code revision

- Pre-cleanup snapshot: git tag `pre-final-cleanup` (commit `061dae6`).
- Main LOTO, ablations, cross-trajectory, and seed-0 scarcity were computed from that scientific pipeline (paper profile).
- The final cleanup renamed the window/state loss to Smooth L1 without changing numerics, added power/window diagnostics, and exported `results/`. YAML training keys were not renamed, so LOTO cache identity is unchanged.

## Software

- Python 3.12.10
- PyTorch 2.14.0+cpu
- Device: CPU (`experiment.device: auto`)
- Seeds for main LOTO: 0, 1, 2
- Profile: `configs/paper.yaml` extending `configs/base.yaml`
- Vehicle: `configs/vehicle_twizy.yaml`

## Features

Power-branch inputs: `speed_mps, acc_mps2, theta_rad, temperature_c, humidity_pct, wind_speed_mps, traffic, speed_limit_mps`.

Excluded: Date, Time, LAT, LON, measured SoC. Progress features `elapsed_s, cumulative_distance_km` are used only by the depletion head. ElasticNet whitelist excludes every SoC-derived column.

## Physics

See vehicle YAML. `apply_power_bounds: false` in the frozen protocol. 13 kW is wheel/motor power. Soft wheel clamp is a sensitivity only (`physics_bounded_wheel` physics MAE 0.783 kWh vs 0.780 kWh unbounded). PINN was not retrained under the clamp because trip energy did not change materially.

## Windows

SoC-event thresholds 0.1, 0.2, 0.5 percentage points, minimum duration 60 s. Fixed windows 60, 120, 300, 600 s with minimum |ΔSoC| 0.1 pp. Full-trip window included. Train windows overlap; evaluation windows do not. Training overlapping totals: 54522 event, 3132 fixed, 17 full-trip.

## Loss

Smooth L1 / normalized Huber as implemented in `src/training/losses.py` (`smooth_l1`). Discrete conservation residual as in the README. Optimizer Adam, lr 0.001, weight decay 1e-5, max 1200 epochs, patience 100, outer retrain on the selected epoch count.

## LOTO procedure

17 analysed trips. For each test trip and seed: estimate \(q\) on inner train; grouped validation; select epoch by scale-balanced window MAE; discard the selection model; retrain on all 16 outer-training trips; infer without measured SoC. 51 fold manifests, 0 recorded failures, 0 residual-head saturations.

## Metrics

Primary: trip energy MAE (kWh). Secondary: RMSE, MAPE (undefined if |E_obs|≤1e-12), WAPE, bias, \(R^2\). Uncertainty: 10 000-resample trip bootstrap, seed 20260921. Paired AE differences for PINN vs WeakMLP, ElasticNet, Physics. Wilcoxon is secondary.

## Sensitivity experiments

Battery capacity 5.5/6.0/6.5 kWh (relabel E_obs only). One-at-a-time physical parameters. Preprocessing smoother windows. Wheel-power bounds. Feasibility reserve SoC 5/10/15/20% for Constant, Physics, PINN.
