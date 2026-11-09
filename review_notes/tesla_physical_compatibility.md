# Phase B — Tesla physical compatibility

## Residual bound strategy (predetermined; not test-tuned)

Twizy used `L = 4 kW`. Tesla Model Y battery power reaches tens to >100 kW on the audited drives (`RawBattCurrent132` up to ~500 A at ~350 V).

Copying `L = 4 kW` would effectively freeze the neural residual and is scientifically inappropriate.

**Predetermined selection protocol (training-only):**

1. On training sessions only, compute longitudinal physics power `P_physics`.
2. Compute a rough measured power proxy `P_meas = V * I / 1000` **only for bound calibration diagnostics on train**, never as a training target.
3. Choose `L` from a small fixed grid using the 95th percentile of `|P_meas - P_physics|` on training data, rounded up to a nearby grid value.
4. Candidate grid (kW): `{10, 20, 40, 60, 80}`.
5. Freeze `L` before touching held-out test metrics.

If train-only percentiles are unavailable at implementation time, use a conservative default of **40 kW** for both BEV1/BEV2 as an interim documented assumption, then refine with the train-only percentile rule.

## Grade / elevation

Drive CSVs lack elevation. Options ranked by fidelity:

1. **Baseline (default):** `θ = 0` (flat road). Explicit limitation.
2. **Optional later:** external DEM elevation from GPS lat/lon, predetermined smoothing, no test tuning.

## Feature-adapted inputs

Shared by Physics residual path / MLP / PINN:

`speed_mps, acc_mps2, theta_rad, temperature_c`

No humidity, wind, traffic, or speed limit.

## Conservation and SoC supervision

Unchanged in form:

- Window energy from SoC and `E_batt`
- `D_hat` depletion head
- Conservation residual linking `ΔD_hat` to integrated `P_hat`

Capacity assumptions differ by vehicle YAML.
