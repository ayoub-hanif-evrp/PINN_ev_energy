# INTIS 2026 camera-ready revision package (≤15 pages)

Apply manually to the Springer manuscript. Do **not** auto-merge into `main`.  
Original HELECAR-D tables in `results/` remain the primary result.

Branch `review/intis2026-validation` @ `3b53bc4` is on GitHub; the repo is currently public.  
**Exact section-by-section plan (replace / condense / add + page impact):** `review_notes/07_camera_ready_section_plan.md`.

---

## A. Ablation (Results / Discussion)

**Table (trip-energy MAE, LOTO):**

| Variant | MAE [kWh] | 95% CI |
|---|---:|---|
| Physics Model | 0.780 | [0.614, 0.948] |
| Data-Driven MLP | 0.475 | [0.312, 0.655] |
| Full PINN | 0.295 | [0.209, 0.392] |
| PINN full-trip only | 0.243 | [0.182, 0.306] |
| PINN no conservation | 0.278 | [0.215, 0.342] |

Footnote: Paired AE differences among PINN variants include zero; full PINN vs Physics/MLP do not.

**Paragraph (~120 words):** see `02_ablation_interpretation.md` §4. Emphasize consistency diagnostics (conservation residual doubles without dynamics loss; state MAE worsens without state loss).

---

## B. Feature-adapted Tesla subsection (~½ page)

**Table — trip-energy MAE vs measured V×I energy (17 sessions/config, seeds 0–1–2):**

| Method | Model Y BEV1 | Model Y BEV2 |
|---|---:|---:|
| Physics | 1.051 | 0.972 |
| Data-Driven MLP | 0.971 | 0.917 |
| PINN | 0.831 | 0.727 |

**Suggested text:**

> To address single-vehicle concerns, we re-applied the same physics-informed weak-supervision methodology to the KU Leuven Tesla Model Y dataset (SR/LR configurations), without using measured battery power as training labels. Missing covariates and elevation forced a documented four-feature, flat-road adaptation. The residual bound was fixed from a disjoint physics-only calibration corpus (BEV1: 40 kW; BEV2: 20 kW). On 17 held-out sessions per configuration with three neural seeds, learning-based estimators reduced error relative to the physics baseline; PINN mean MAE was lower than the matched MLP, with a significant paired advantage on BEV1 SoC-derived energy, while measured-energy paired intervals still included zero. These results support methodological applicability beyond HELECAR-D under explicit limitations, not zero-shot cross-vehicle transfer.

Limitations: flat grade; assumed mass/CdA/η; low-energy negative predictions retained as diagnostics.

---

## C. Sensitivity (Reviewer 3)

One sentence + optional mini-table:

> Predetermined sensitivity of the Twizy PINN residual bound L∈{2,4,6,8} kW and of λ_dynamics / λ_prior at {0.5×,1×,2×} shows MAE near the paper reference (0.295 kWh); we do not retune defaults from outer-test ranks. Capacity assumptions {5.5,6.0,6.5} kWh are reported in `05_sensitivity_findings.md`.

---

## D. Editorial checklist (R1–R3)

- [ ] Condense methodology (merge redundant loss prose).
- [ ] Clarify novelty: weak SoC multi-window + bounded residual under scarce power labels.
- [ ] Justify Eq. 4 with longitudinal EV energy references (add bibliography entries).
- [ ] Discuss 17-trip / single-region limits + Tesla feature-adapted external check.
- [ ] Justify 4 kW bound via sensitivity flat region (do not switch to L=6).
- [ ] Clarify train/val protocol (inner early-stop; outer retrain; SoC windows).
- [ ] Stay ≤ 15 pages; one Tesla table max.

## E. What not to change

- Main HELECAR-D result table.
- Architecture / accepted training formulation.
- Do not claim every PINN constraint improves trip MAE.
- Do not claim Tesla proves universal PINN>MLP.
