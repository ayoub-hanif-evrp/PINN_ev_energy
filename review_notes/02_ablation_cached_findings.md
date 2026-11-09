# Phase 2 — Ablation findings

Sources:
- Cached: `outputs/loto/paper/per_trip_predictions.csv`
- New review run: `outputs/review/ablations/loto/review_ablations/per_trip_predictions.csv`
- Summary: `review_results/ablations/table_ablation_complete.csv`

The new run finished successfully (`sanity_report.json`: ok=True). No paper artifacts were overwritten.

## Compact ablation table (trip-energy MAE)

| Variant | MAE [kWh] | 95% CI |
|---|---:|---|
| Physics Model (no residual) | 0.780 | [0.614, 0.948] |
| Data-Driven MLP | 0.475 | [0.312, 0.655] |
| Full PINN (reference) | 0.295 | [0.209, 0.392] |
| PINN full-trip supervision only | 0.243 | [0.182, 0.306] |
| PINN without state loss | 0.253 | [0.191, 0.315] |
| PINN without conservation loss | 0.278 | [0.215, 0.342] |
| PINN without residual prior | 0.321 | [0.226, 0.434] |

## Paired absolute-error difference vs full PINN

Difference = full PINN AE − comparator AE. Negative ⇒ full PINN better.

| Comparator | Mean Δ [kWh] | 95% paired CI | Wilcoxon p |
|---|---:|---|---:|
| Physics | −0.485 | [−0.604, −0.375] | 1.5e-05 |
| Data-Driven MLP | −0.180 | [−0.316, −0.056] | 0.020 |
| PINN full-trip only | +0.052 | [−0.030, +0.132] | 0.517 |
| PINN no state loss | +0.042 | [−0.028, +0.109] | 0.244 |
| PINN no conservation | +0.017 | [−0.055, +0.089] | 0.963 |
| PINN no residual prior | −0.026 | [−0.140, +0.058] | 0.963 |

## Scientific interpretation (honest)

1. **Full-trip-only PINN has the lowest point-estimate MAE** (0.243 vs 0.295 kWh for full multi-window PINN). The paired CI includes zero.
2. Removing the state loss also lowers the point MAE slightly; paired CI includes zero.
3. Removing the residual prior slightly **worsens** the point MAE (0.321); paired CI includes zero.
4. Removing conservation has little effect on this metric; paired CI includes zero.
5. These findings do **not** authorize retuning the accepted paper method. Multi-window supervision remains the locked abstract method. Report the ablation honestly for the camera-ready paper.

## Camera-ready suggestion

One compact table in Results/Discussion with the rows above, plus one short paragraph stating the full-trip finding without replacing the main method.
