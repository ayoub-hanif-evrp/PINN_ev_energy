# INTIS 2026 review validation — baseline protection

## Branch and baseline

| Item | Value |
|---|---|
| Review branch | `review/intis2026-validation` |
| Baseline commit | `0a85b57c3cad6d81a9dfb225f69445546860cc50` |
| Baseline message | Simplify the manuscript data table for the problem-formulation section. |
| `main` policy | Unchanged. Do not merge review into `main` until camera-ready is approved. |
| Remote status | Branch pushed to `origin/review/intis2026-validation`. Repo is currently public; change visibility separately if needed. |

Untracked local manuscript files (`.docx`, `.pdf`, `.pptx`, `intis submission.png`) were left untracked and are not part of this review work.

## Protected original artifacts

Do not overwrite:

- `results/` paper figures and tables
- `configs/paper.yaml`
- `configs/base.yaml`
- `configs/vehicle_twizy.yaml`
- Original PINN architecture, physics equations, training loss, and LOTO protocol

Review outputs go under:

- `review_notes/`
- `review_results/`
- `outputs/review/`
- `configs/review/`

## Original HELECAR-D reference results (from `results/tables/`)

| Method | MAE [kWh] | 95% CI | RMSE | Bias | R² |
|---|---:|---|---:|---:|---:|
| Physics Model | 0.780 | [0.614, 0.948] | 0.854 | −0.780 | 0.508 |
| Ridge Regression | 0.368 | [0.229, 0.537] | 0.491 | −0.014 | 0.837 |
| Data-Driven MLP | 0.475 | [0.312, 0.655] | 0.629 | +0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | −0.052 | 0.906 |

Paired absolute-error difference (PINN − comparator), n = 17:

| Comparison | Mean [kWh] | 95% paired CI | Wilcoxon p |
|---|---:|---|---:|
| PINN − Physics Model | −0.485 | [−0.604, −0.375] | 1.53e-05 |
| PINN − Ridge Regression | −0.073 | [−0.264, 0.110] | 0.579 |
| PINN − Data-Driven MLP | −0.180 | [−0.316, −0.056] | 0.020 |

## Original method snapshot

- Discrete-time residual PINN: `P_hat = P_physics + L * tanh(NN)`, `L = 4 kW`
- Shared torso + depletion head `D_hat` with `D_hat(0)=0`
- Multi-window SoC energy supervision (event 0.1/0.2/0.5 pp, fixed 60/120/300/600 s, full trip)
- Conservation residual linking `ΔD_hat` to integrated energy
- Residual prior and depletion-state Smooth-L1 terms
- Nested trip-level validation; outer test never used for selection
- Neural seeds `[0, 1, 2]`; LOTO on 17 HELECAR-D trips
