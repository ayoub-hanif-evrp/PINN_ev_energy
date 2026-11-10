# Freeze manifest — corrected review evidence

## Git

| Item | Value |
|---|---|
| Branch | `review/intis2026-validation` |
| Remote | `origin` (GitHub; currently public) |
| Pre-Tesla baseline | `0a85b57` |
| First review push | `c301072` |
| `main` | Untouched; do not merge until camera-ready approved |

## Tesla corrected experiment freeze

| Item | Value |
|---|---|
| Profiles | `tesla_bev1_loto_corrected`, `tesla_bev2_loto_corrected` |
| Eval pools | `review_results/tesla/pools/bev{1,2}_pool.json` (seed 2026) |
| Calib pools | `review_results/tesla/pools/bev{1,2}_calib_pool.json` (seed 2027; disjoint) |
| L rule | `0.5×P95(|P_physics|)` → grid; **no P_meas** |
| L BEV1 / BEV2 | 40 kW / 20 kW |
| Seeds | 0, 1, 2 |
| dt | forward i→i+1, last 0; rebuilt after drops |
| Features | speed, acc, θ=0, temperature (feature-adapted) |
| Supersedes | `tesla_bev1_loto`, `tesla_bev2_loto` |

## HELECAR-D ablations

`review_results/ablations/` + interpretation `review_notes/02_ablation_interpretation.md`

## Twizy sensitivity

`review_results/sensitivity/` — residual L, λ_dynamics/λ_prior, and capacity grids complete (`05_sensitivity_findings.md`).

## Gitignored (not frozen in git)

- `BEV energy dynamics dataset/`
- `data/tesla/`
- checkpoints / training histories under review_results
