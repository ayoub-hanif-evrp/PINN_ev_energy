# Ablation interpretation for INTIS 2026 camera-ready

Sources:
- Trip MAE: `review_results/ablations/table_ablation_complete.csv`
- Paired tests: `review_results/ablations/table_ablation_paired_complete.csv`
- Diagnostics: `outputs/loto/paper/per_trip_predictions.csv` + `outputs/review/ablations/loto/review_ablations/per_trip_predictions.csv`
- Window MAE: `outputs/loto/paper/window_eval/test_window_mae_by_scale.csv`

Original HELECAR-D paper numbers in `results/` were not overwritten.

## 1. Trip-energy MAE (leave-one-trip-out)

| Variant | MAE [kWh] | 95% CI |
|---|---:|---|
| Physics Model | 0.780 | [0.614, 0.948] |
| Data-Driven MLP | 0.475 | [0.312, 0.655] |
| Full PINN (reference) | 0.295 | [0.209, 0.392] |
| PINN full-trip only | 0.243 | [0.182, 0.306] |
| PINN no state loss | 0.253 | [0.191, 0.315] |
| PINN no conservation | 0.278 | [0.215, 0.342] |
| PINN no residual prior | 0.321 | [0.226, 0.434] |

## 2. Paired absolute-error differences (full PINN − comparator)

Negative ⇒ full PINN better on trip energy.

| Comparator | Mean Δ | 95% CI | Wilcoxon p |
|---|---:|---|---:|
| Physics | −0.485 | [−0.604, −0.375] | 1.5e-05 |
| MLP | −0.180 | [−0.316, −0.056] | 0.020 |
| Full-trip only | +0.052 | [−0.030, +0.132] | 0.517 |
| No state | +0.042 | [−0.028, +0.109] | 0.244 |
| No conservation | +0.017 | [−0.055, +0.089] | 0.963 |
| No residual prior | −0.026 | [−0.140, +0.058] | 0.963 |

## 3. Beyond trip MAE: windows and physical consistency

### Held-out window-energy MAE (paper LOTO scales; mean over trips/seeds)

| Method | full_trip | time_60s | time_300s | soc_event_0.1 |
|---|---:|---:|---:|---:|
| Full PINN | 0.295 | 0.017 | 0.065 | 0.017 |
| PINN full-trip only | 0.243 | 0.017 | (similar scale band) | 0.017 |
| MLP (state head table) | 0.372 | 0.023 | 0.090 | 0.024 |

Multi-window supervision does **not** inflate short-window errors relative to the full-trip-only variant on these aggregates; short-window MAE remains on the order of 0.017 kWh for both PINN variants.

### State / conservation diagnostics (mean over LOTO rows)

| Method | trip MAE | state_mae_pp | dynamics_residual_rmse |
|---|---:|---:|---:|
| Full PINN | 0.295 | 9.97 | 0.222 |
| Full-trip only | 0.243 | 12.13 | 0.225 |
| No conservation | 0.278 | 10.15 | **0.516** |
| No state loss | 0.253 | **14.04** | 0.019 |
| No residual prior | 0.321 | 11.63 | 0.237 |

Reading:
- Removing **conservation** roughly **doubles** dynamics residual RMSE while trip MAE barely changes (CI includes 0 vs full PINN).
- Removing **state loss** worsens SoC-state MAE (14.0 vs 10.0 pp) even though trip MAE point estimate improves slightly (CI includes 0).
- Full-trip-only wins trip MAE point estimate but has worse mean state MAE than the full multi-window PINN.

## 4. Honest conclusions for the manuscript

1. On HELECAR-D, the **full PINN significantly outperforms physics and the data-driven MLP** on trip-energy MAE (paired CIs exclude 0).
2. We **cannot claim** that each individual PINN constraint improves trip-energy MAE: paired CIs vs full-trip-only / no-state / no-conservation / no-prior all include zero.
3. Constraints still contribute to **physical consistency**: conservation loss reduces dynamics residuals; state loss improves depletion tracking. Multi-scale supervision is a methodological design choice supported by short-window accuracy, not by a statistically significant trip-MAE gain over full-trip-only.
4. Do **not** replace the accepted multi-window PINN with full-trip-only in the camera-ready method section. Report the ablation honestly.

## 5. Compact camera-ready table draft

Keep one table with Physics / MLP / Full PINN / Full-trip / No-conservation (optional: No-state), MAE + 95% CI, footnote: “Paired differences among PINN variants include zero; full PINN vs MLP/Physics do not.”
