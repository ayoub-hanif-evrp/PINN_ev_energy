# INTIS 2026 camera-ready final scientific audit

**Paper:** Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity (Paper 62)  
**Deadline:** 15 November 2026  
**Branch:** `review/intis2026-validation` (ahead of `main` @ `0a85b57`; local corrections beyond pushed `c301072`)  
**Audit date:** 10 November 2026  

**Scope of this document:** Verify completed experiments, state what is publication-ready, and propose *minimal* manuscript additions. **The accepted manuscript was not edited.**

Status legend used below: **COMPLETED** · **VERIFIED** · **NEEDS CORRECTION** · **NOT NECESSARY**

---

## Executive verdict

| Workstream | Status | Camera-ready use? |
|---|---|---|
| HELECAR-D ablations | **COMPLETED** + **VERIFIED** | **Yes** — short honest table + paragraph |
| Twizy residual / λ / capacity sensitivity | **COMPLETED** + **VERIFIED** | **Yes** — 1–2 sentences (do not retune L=4) |
| Tesla external validation (corrected) | **COMPLETED** + **VERIFIED** (after prior leakage fix) | **Conditional yes** — ≤½ page, feature-adapted framing only |
| BMW / other vehicles | **NOT NECESSARY** / blocked | No |
| Automatic manuscript rewrite | **NOT NECESSARY** here | Manual paste from §6 |

Original HELECAR-D main results, PINN architecture, and paper configs remain untouched under `results/` and `configs/paper.yaml`.

---

## 1. Completed experiments (inventory)

### 1.1 HELECAR-D ablations

| Artifact | Path |
|---|---|
| Config | `configs/review/ablations.yaml` (extends paper; methods `pinn_no_state`, `pinn_no_prior`) |
| Cached full variants | `outputs/loto/paper/per_trip_predictions.csv` |
| New ablation folds | `outputs/review/ablations/loto/review_ablations/` |
| Summary table | `review_results/ablations/table_ablation_complete.csv` |
| Paired tests | `review_results/ablations/table_ablation_paired_complete.csv` |
| Interpretation | `review_notes/02_ablation_interpretation.md` |

### 1.2 Tesla Model Y external validation (corrected)

| Artifact | Path |
|---|---|
| Configs | `configs/review/tesla_bev{1,2}_loto.yaml` → profiles `*_corrected` |
| Physics-only L calib | `scripts/review/calibrate_tesla_residual.py` + `review_results/tesla/residual_calibration_bev{1,2}.json` |
| Disjoint calib pools | `review_results/tesla/pools/bev{1,2}_calib_pool.json` |
| Eval pools | `review_results/tesla/pools/bev{1,2}_pool.json` |
| Authoritative LOTO | `review_results/tesla/loto/tesla_bev{1,2}_loto_corrected/` |
| Paired PINN–MLP | `review_results/tesla/paired_pinn_vs_mlp_corrected.json` |
| Integrity tests | `scripts/review/test_tesla_integrity.py` (all PASS when data present) |
| **Superseded** | `tesla_bev1_loto`, `tesla_bev2_loto` (seed-0 / P_meas-on-eval-pool leakage) |

### 1.3 Twizy sensitivity (Reviewer 3)

| Artifact | Path |
|---|---|
| Runner | `scripts/review/run_twizy_sensitivity.py` |
| Summaries | `review_results/sensitivity/summary_*.csv` |
| Findings | `review_notes/05_sensitivity_findings.md` |

---

## 2. Task 1 — Ablation validation

### Status: **COMPLETED** · **VERIFIED** · retraining **NOT NECESSARY**

### Methodological verification

| Check | Result |
|---|---|
| Trip set | 17 HELECAR-D analysed trips |
| Seeds | `{0,1,2}` for all neural variants |
| Fold identity | 51 paper vs ablation manifests: **0 train/val mismatches** |
| Aggregation | Trip-level mean AE over seeds, then MAE + trip bootstrap CI |
| Recomputation | Independent recompute matches saved MAE to 4 decimals; CI differs only by bootstrap RNG (~0.002–0.005) |
| Paper artifacts | Not overwritten |

Variants `physics`, `weak_mlp`, `pinn`, `pinn_fulltrip`, `pinn_no_dynamics` come from the paper LOTO cache; `pinn_no_state` and `pinn_no_prior` from the review ablation run under identical LOTO folds (`configs/review/ablations.yaml`).

### Final ablation table (use this)

| Variant | MAE [kWh] | 95% CI |
|---|---:|---|
| Physics Model | 0.780 | [0.614, 0.948] |
| Data-Driven MLP | 0.475 | [0.312, 0.655] |
| **Full PINN (reference)** | **0.295** | **[0.209, 0.392]** |
| PINN full-trip supervision only | 0.243 | [0.182, 0.306] |
| PINN without conservation loss | 0.278 | [0.215, 0.342] |
| PINN without depletion-state loss | 0.253 | [0.191, 0.315] |
| PINN without residual prior | 0.321 | [0.226, 0.434] |

### Paired absolute-error difference (full PINN − comparator)

| Comparator | Mean Δ [kWh] | 95% CI | Wilcoxon p |
|---|---:|---|---:|
| Physics | −0.485 | [−0.604, −0.375] | 1.5×10⁻⁵ |
| MLP | −0.180 | [−0.316, −0.056] | 0.020 |
| Full-trip only | +0.052 | [−0.030, +0.132] | 0.517 |
| No state | +0.042 | [−0.028, +0.109] | 0.244 |
| No conservation | +0.017 | [−0.055, +0.089] | 0.963 |
| No residual prior | −0.026 | [−0.140, +0.058] | 0.963 |

### Honest scientific reading (required)

1. Full PINN **significantly** beats physics and MLP on trip MAE (paired CIs exclude 0).
2. Among PINN variants, **paired CIs all include zero** — we **cannot** claim every loss term improves trip-energy MAE.
3. Full-trip-only has the best *point* MAE (0.243) but is not statistically better than full PINN; diagnostics show worse mean state tracking (`02_ablation_interpretation.md`).
4. Removing conservation roughly doubles dynamics residual RMSE while trip MAE barely changes.

**Reviewer mapping:** primarily **R2 ablation**; supports honest discussion for **R3** (what constraints contribute).

---

## 3. Task 2 — Tesla external validation audit

### 3.1 Leakage issue — **CORRECTED** (was **NEEDS CORRECTION**, now **VERIFIED**)

| Item | Old (superseded) | Corrected |
|---|---|---|
| L selection | \|P_meas − P_physics\| on **all 17 eval-pool trips** | `0.5 × P95(\|P_physics\|)` on **disjoint calib pool**; `uses_measured_power: false` |
| Eval/calib overlap | Entire pool | **Empty** intersection (`bev{1,2}_calib_pool.json`) |
| L values | 40 / 40 kW (leaky) | BEV1 **40 kW**, BEV2 **20 kW** |
| Timestamps | Backward-style / not rebuilt after drops | Forward `dt[i]=t[i+1]−t[i]`, last 0; rebuild after cleaning |
| Seeds | 0 only | **0, 1, 2** |

Authoritative outputs: `review_results/tesla/loto/tesla_bev*_loto_corrected/`.  
Do **not** cite `tesla_bev1_loto` / `tesla_bev2_loto` in the paper.

### 3.2 Sanity diagnostics (corrected runs) — **VERIFIED**, trips **not dropped**

| Profile | Flags | Detail |
|---|---|---|
| BEV1 corrected | `negative_predicted_energy` (2 rows) | `BEV1_2024-09-01_drive3`: PINN seed0 −0.33 kWh; MLP seed2 −0.34 kWh (SoC obs 3.025 kWh). Peak-power flags: **0** at 400 kW threshold |
| BEV2 corrected | `negative_predicted_energy` (3 rows) | Ridge only on `BEV2_2024-11-13_drive2` (−0.53 kWh, all seeds). Peak-power flags: **0** |

Interpretation: low-energy / unstable extrapolation — retained. Not used to censor the evaluation set.

### 3.3 Corrected performance (17 trips/vehicle, seeds 0–1–2)

**Against independent V×I energy (evaluation only):**

| Method | BEV1 MAE | BEV2 MAE |
|---|---:|---:|
| Physics | 1.051 | 0.972 |
| Ridge | 1.646 | 1.236 |
| Weak-MLP | 0.971 | 0.917 |
| PINN | **0.831** | **0.727** |

**Against SoC-derived energy (paper-analogous target):**

| Method | BEV1 MAE | BEV2 MAE |
|---|---:|---:|
| Physics | 1.264 | 1.052 |
| Weak-MLP | 1.006 | 0.923 |
| PINN | **0.659** | **0.827** |

**Paired PINN − MLP (trip means over seeds):**

| Setting | Mean Δ | 95% CI | Notes |
|---|---:|---|---|
| BEV1 vs SoC | −0.347 | [−0.619, −0.114] | CI excludes 0 (PINN better) |
| BEV1 vs V×I | −0.141 | [−0.405, +0.072] | CI includes 0 |
| BEV2 vs SoC | −0.095 | [−0.383, +0.177] | CI includes 0 |
| BEV2 vs V×I | −0.190 | [−0.492, +0.092] | CI includes 0 |

### 3.4 Suitability for a camera-ready subsection

**Recommendation: INCLUDE as ≤½ page, with strict framing.**

**Strong enough if:**
- Called **feature-adapted** (4 inputs; θ=0), not identical 8-feature Twizy transfer.
- Training claimed as SoC weak supervision without power labels (true for corrected L protocol).
- V×I used only as independent evaluation.
- Mixed paired significance disclosed (V×I CIs include 0).

**Not strong enough if** reviewers require:
- Identical feature set / elevation fidelity, or
- Statistically significant PINN>MLP on measured energy on both vehicles.

**Limitations to state:** flat-road assumption; assumed mass/CdA/η/P_aux; HVAC not modelled; predetermined 17-trip pools; occasional negative predictions on low-energy trips; public GitHub branch contains review artifacts.

**Reviewer mapping:** **R1/R2/R3** additional-vehicle / generalizability concerns (partial).

---

## 4. Task 3 — Twizy sensitivity

### Status: **COMPLETED** + **VERIFIED** · retuning paper defaults **NOT NECESSARY**

Reference model preserved: L=4 kW, paper λ, LOTO, seeds `{0,1,2}`.  
`summary_L_4.csv` MAE = **0.294914…** identical to paper Full PINN MAE (traceable consistency check).

### Residual bound

| L [kW] | MAE [kWh] |
|---:|---:|
| 2 | 0.300 |
| **4 (ref)** | **0.295** |
| 6 | 0.263 |
| 8 | 0.339 |

L=6 has a lower *point* MAE but was **not** adopted (no outer-test model selection).

### Loss coefficients (one-at-a-time)

| Variant | MAE |
|---|---:|
| λ_dynamics ×0.5 / ×1 / ×2 | 0.303 / **0.295** / 0.294 |
| λ_prior ×0.5 / ×1 / ×2 | 0.303 / **0.295** / 0.286 |

Stable under ±2×; do not retune.

### Capacity {5.5, 6.0, 6.5} kWh

| Capacity | MAE |
|---:|---:|
| 5.5 | 0.202 |
| **6.0** | **0.295** |
| 6.5 | 0.330 |

**Caveat:** HELECAR-D has **no** independent battery-power ground truth; capacity changes rescale SoC-derived labels and absolute MAE. Report as assumption uncertainty, not measured-energy accuracy.

**Reviewer mapping:** **R3** residual bound, loss weights, capacity uncertainty.

---

## 5. Issues found and corrected

| Issue | Severity | Resolution |
|---|---|---|
| Tesla L chosen with P_meas on future test trips | Critical (leakage) | Physics-only disjoint calib; old runs superseded |
| Tesla `dt` not rebuilt after row drops / convention mismatch | High | Forward intervals + integrity tests |
| Twizy 200 kW “absurd” threshold on Tesla | Medium | Tesla-aware 400 kW diagnostic; peaks logged |
| Negative trip energies | Medium | Documented; trips retained |
| Outdated “branch not pushed” notes | Low | Updated in `00_baseline_protection.md` / status notes |
| Ablation mixed MAE effects | Scientific | Interpreted honestly; no cherry-picking |

**Incomplete / unsuitable for publication as primary claims:**
- Preliminary Tesla `tesla_bev*_loto` (leaky L, seed 0).
- Claiming PINN≻MLP on Tesla V×I with statistical certainty.
- Claiming every PINN loss term improves trip MAE.
- BMW i3 (no local data) — **NOT NECESSARY** for this deadline.

---

## 6. Reviewer comment → experiment map

| Reviewer request | Experiment / evidence | Status |
|---|---|---|
| R1 single-vehicle / strengthen validation | Corrected Tesla BEV1/BEV2 | **VERIFIED** conditional |
| R1 Eq. 4 / references | Manuscript only | **NOT** done in this audit (editorial) |
| R2 condense methodology / novelty | Manuscript only | Editorial |
| R2 ablation | HELECAR-D ablation table + paired CIs | **VERIFIED** |
| R2 additional vehicles | Tesla feature-adapted | **VERIFIED** conditional |
| R3 17-trip limitation | Discuss + Tesla external check | Partial (editorial + Tesla) |
| R3 4 kW bound | L ∈ {2,4,6,8} sensitivity | **VERIFIED** |
| R3 loss weights | λ_dynamics / λ_prior ×{0.5,1,2} | **VERIFIED** |
| R3 train/val protocol | Code + notes; manuscript clarification | Partial (editorial) |
| R3 capacity uncertainty | {5.5,6.0,6.5} + Tesla SoC–V×I gap | **VERIFIED** (HELECAR caveat) |
| R3 SoC measurement noise | Not run | **NOT NECESSARY** for deadline if space tight |
| R3 full physical-parameter grid | Not run | **NOT NECESSARY** (lower priority) |

---

## 7. Minimal suggested manuscript additions

Do **not** auto-edit the paper. Paste manually; keep ≤15 pages.

### 7.1 Ablation (~⅓–½ page)

Use the table in §2. Suggested wording:

> Ablation of multi-window supervision, conservation, state consistency, and residual-prior terms shows that the full PINN significantly outperforms the physics baseline and the data-driven MLP on trip-energy MAE. Paired differences among PINN variants, however, include zero: the full-trip-only variant even attains a slightly lower point MAE. Diagnostics indicate that conservation and state terms primarily improve physical consistency rather than trip MAE. We therefore retain the accepted multi-scale formulation without claiming that every constraint is MAE-optimal.

### 7.2 Tesla subsection (~½ page) — **optional but recommended**

Use the **corrected** V×I table in §3.3. Require “feature-adapted” language and SoC vs V×I distinction. Full draft: `review_notes/camera_ready_revisions.md` §B.

### 7.3 Sensitivity (2–4 sentences)

> Predetermined sensitivity of the residual bound L∈{2,4,6,8} kW and of λ_dynamics / λ_prior at {0.5×,1×,2×} yields trip MAE near the reference 0.295 kWh; we do not replace the accepted 4 kW / reference weights based on outer-test ranks. Assumed pack capacity {5.5,6.0,6.5} kWh rescales SoC-derived energies; 6.0 kWh remains the locked modelling assumption.

### 7.4 Editorial still required (no new experiments)

- Condense methodology; clarify novelty.
- Justify Equation 4 + add references.
- Explicit limitations: 17 trips, one region, capacity assumption, Tesla feature gaps.

---

## 8. Remaining risks before 15 Nov

1. Corrected Tesla code/results are largely **local uncommitted** relative to pushed `c301072` — commit/push review branch before relying on remote reproducibility.
2. Repository is **public** — confirm visibility policy for review artifacts.
3. Springer page budget: prefer ablation + sensitivity sentences; add Tesla only if space allows without cutting core contribution.
4. Capacity MAE shifts with label scale — easy to misread as “accuracy”; wording must stay careful.
5. Negative Tesla predictions may draw reviewer questions — answer with diagnostics, not censorship.

---

## 9. Decision checklist for the authors

| Decision | Recommendation |
|---|---|
| Include HELECAR ablation table? | **Yes** |
| Include Twizy L/λ sensitivity sentence? | **Yes** |
| Include Tesla subsection? | **Yes, if space** — corrected numbers only |
| Change accepted L=4 or λ defaults? | **No** |
| Retrain ablations? | **No** |
| Edit manuscript in this agent pass? | **No** (stopped after audit, as requested) |

---

## 10. Traceability anchors

- Ablation MAE/CI: `review_results/ablations/table_ablation_complete.csv`
- Ablation paired: `review_results/ablations/table_ablation_paired_complete.csv`
- Tesla corrected summaries: `review_results/tesla/loto/tesla_bev*_loto_corrected/summary_vs_{meas,soc}.csv`
- Tesla paired: `review_results/tesla/paired_pinn_vs_mlp_corrected.json`
- Sensitivity: `review_results/sensitivity/summary_*.csv`
- Freeze notes: `review_notes/06_freeze_manifest.md`
