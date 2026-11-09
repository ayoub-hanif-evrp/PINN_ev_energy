# Review status and camera-ready recommendation

Date: 2026-11-09  
Branch: `review/intis2026-validation` (local only; not pushed)  
Baseline: `0a85b57`

## Completed

| Phase | Status |
|---|---|
| 0 Protect original research | Done — branch created; `main` untouched; baseline recorded |
| 1 Method audit | Done — `review_notes/01_method_audit.md` |
| 2 Ablations from cache | Done — table + paired tests in `review_results/ablations/` |
| 3 BMW feasibility | Done as **NO-GO** until Measurement Data are downloaded |
| Tesla dataset audit | Done — `review_notes/tesla_dataset_audit.md` |
| Tesla GO/NO-GO | **CONDITIONAL GO** — `review_notes/tesla_go_nogo.md` |
| Tesla primary LOTO | Done — BEV1 & BEV2, 17 trips each, seed 0 |
| Tesla validation report | Done — `review_notes/tesla_validation_report.md` |
| Sensitivity | Planned — `review_notes/03_sensitivity_plan.md` (Twizy residual/capacity) |

## Strong enough for camera-ready today

1. **Ablation table from cached HELECAR-D LOTO**, including the honest full-trip finding.
2. Keep original main results unchanged.
3. **Optional Tesla subsection (~½ page)** — feature-adapted external replication; PINN≈MLP; physics beaten; V×I eval-only. See draft in `tesla_validation_report.md` §10.

## Not yet / secondary

1. BMW i3 — still NO-GO (no local Measurement Data).
2. Twizy residual/capacity sensitivity grids — optional if time remains.
3. Tesla scarcity — partial BEV1 cells only (n=3/5/8); supportive but not definitive.
4. Tesla multi-seed — not run (CPU); single seed 0 documented.

## Manuscript integration (do not auto-edit the paper)

### A. Ablation table (recommended)

Physics / MLP / Full PINN / Full-trip PINN / No-conservation PINN  
with MAE and 95% CI; footnote that full-trip has lower point MAE but paired CI includes zero.

### B. External vehicle (Tesla, optional)

One short table + ~½ page. Frame as feature-adapted re-training on Model Y, not zero-shot transfer. Do not compare absolute Twizy/Tesla MAE as like-for-like. Do not expand into a multi-vehicle benchmark.

### C. Sensitivity

One compact sensitivity summary for residual bound and capacity, if compute finishes before camera-ready deadline.

## Original contribution remains protected

No change to `results/` paper artifacts, no overwrite of `configs/paper.yaml` / `base.yaml` / `vehicle_twizy.yaml`, no push of this branch or datasets.
