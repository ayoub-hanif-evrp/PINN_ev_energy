# Phase 1 — Original method audit

Audit date: 2026-11-09  
Baseline commit: `0a85b57`

## Architecture (`src/models/pinn.py`)

- Shared 64-64-64 tanh torso on SoC-free telemetry.
- Residual power: `δP = L tanh(head)`, `L = 4 kW`, `P_hat = P_physics + δP`.
- Depletion head: `D_hat(t) = D_raw(t) - D_raw(0)`.
- Not a PDE / autodiff-in-time PINN.

## Data-driven baseline (`src/models/weak_mlp.py`)

- Same torso depth; power = `5 + 30 tanh(raw)` kW.
- No physics branch, no depletion head.

## Physics (`src/physics/vehicle_model.py`)

- Longitudinal forces: inertia, rolling resistance, grade, aero (`CdA = 0.64`).
- Battery power from wheel power with drivetrain / regen efficiencies.
- Wind never used as headwind (no direction in HELECAR-D).

## Losses (`src/training/losses.py`)

`L = λ_window L_window + λ_dynamics L_dyn + λ_state L_state + λ_prior L_prior`

| Term | Role | Default λ |
|---|---|---:|
| Window | Scale-balanced Smooth-L1 on SoC-derived window energy | 1.0 |
| Dynamics | Mean squared conservation residual | 1.0 |
| State | Smooth-L1 on `D_hat` vs observed depletion | 0.1 |
| Prior | Mean squared `δP / p_scale` | 0.1 |

## Training / evaluation protocol

- Nested leave-one-trip-out; inner validation selects epoch / λ.
- Outer test trip never used for scaling, q, or early stopping.
- SoC is supervision only; never a power-branch input.
- Seeds `[0,1,2]` for neural methods.

## Ablation codes previously removed from `main`

Historical trainer supported:

| Method id | Meaning |
|---|---|
| `pinn` | Full multi-window PINN |
| `pinn_fulltrip` | Window filter = `{full_trip}` only |
| `pinn_no_dynamics` | `λ_dynamics = 0` |
| `mlp_state` | Legacy MLP + depletion head |

Cached predictions for these methods remain under `outputs/loto/paper/per_trip_predictions.csv`.

## Missing clean ablation for Reviewers 2–3

| Requested | Status |
|---|---|
| A Full PINN | Cached |
| B Physics-only | Cached (`physics`) |
| C Full-trip supervision only | Cached (`pinn_fulltrip`) |
| D No conservation loss | Cached (`pinn_no_dynamics`) |
| E No depletion-state supervision | **Not cached** → needs review retrain (`pinn_no_state`) |
| Optional no residual prior | **Not cached** → needs review retrain (`pinn_no_prior`) |
