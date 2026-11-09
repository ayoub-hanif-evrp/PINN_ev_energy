# Additional Phase — Sensitivity plan (predetermined, not test-tuned)

All sensitivity runs write under `review_results/sensitivity/` and `outputs/review/sensitivity/`.
Original `results/` and `configs/paper.yaml` stay frozen.

## 1. Residual correction bound L

Reference: `L = 4.0` kW.

Predetermined grid (chosen before looking at review test ranks):

| L [kW] | Role |
|---:|---|
| 2.0 | Tighter residual |
| 4.0 | Original reference |
| 6.0 | Wider residual |
| 8.0 | Wide residual |

Method: PINN only, seeds `[0,1,2]`, same LOTO folds.

## 2. Loss weights (one-at-a-time)

Reference: `λ_window=1.0`, `λ_dynamics=1.0`, `λ_state=0.1`, `λ_prior=0.1`.

For each coefficient, test `{0.5×, 1×, 2×}` of the reference while others stay fixed.

Do not pick a new default from outer-test MAE.

## 3. Battery capacity

Reference assumption: 6.0 kWh.

Predetermined values: `{5.5, 6.0, 6.5}` kWh.

Report separately:

- change in SoC-derived observed energy;
- change in model MAE under the same assumption used for targets.

Never claim these are measured usable capacities.

## 4. SoC quantization / measurement noise

Predetermined protocol (training-side only):

- additive Gaussian noise on training SoC with σ ∈ `{0, 0.01, 0.02}` percentage points;
- or random ±1 quantization-bin flips at low rate.

Held-out test SoC for evaluation metrics uses the original series; training targets use the perturbed training SoC only.

## 5. Physical parameters

Predetermined one-at-a-time ranges (document as assumptions):

| Parameter | Low | Ref | High |
|---|---:|---:|---:|
| mass_kg | 524 | 549 | 589 |
| Crr | 0.008 | 0.012 | 0.016 |
| eta_drive | 0.80 | 0.85 | 0.90 |
| eta_regen | 0.20 | 0.40 | 0.60 |
| P_aux_kw | 0.0 | value in vehicle YAML | +0.2 |

## Execution order

1. Finish missing ablations (`pinn_no_state`, `pinn_no_prior`).
2. Residual-bound sensitivity.
3. Loss-weight sensitivity (subset if compute-limited: λ_dynamics and λ_prior first).
4. Capacity sensitivity.
5. Physics-parameter and SoC-noise sensitivities as compute allows before 15 Nov 2026.

Every table must include the original reference result.
