# Tesla Model Y external validation — INTIS 2026 review

**Status:** Primary held-out-trip LOTO complete (BEV1 & BEV2). Scarcity run secondary (in progress / see §6).  
**Branch:** `review/intis2026-validation`  
**Baseline commit before Tesla implementation:** `0a85b57c3cad6d81a9dfb225f69445546860cc50`  
**Dataset:** KU Leuven BEV Energy Dynamics (DOI [10.48804/8KPDTW](https://doi.org/10.48804/8KPDTW)) — local, gitignored.  
**Decision:** CONDITIONAL GO executed under documented adaptations. Original Twizy paper configs/results were not modified.

---

## 1. Dataset structure and usable sessions

| Item | Value |
|---|---|
| Source | Zip under `BEV energy dynamics dataset/` |
| Vehicles | BEV1 (Model Y SR, LFP, 60.5 kWh nominal); BEV2 (Model Y LR, NCA, 78.8 kWh nominal) |
| Extracted QC trips | BEV1 214; BEV2 321 (`data/tesla/`, gitignored) |
| QC | duration ≥ 15 min, ΔSoC ≥ 2 pp, moving fraction ≥ 0.4 |
| Primary LOTO pool | **17 trips/vehicle**, stratified by ΔSoC×distance tertiles, `pool_seed=2026`, frozen in `review_results/tesla/pools/` before training |

Drive CSVs: ~1 Hz telemetry. No elevation/humidity/wind/traffic/speed-limit columns. Grade forced to θ = 0.

---

## 2. Available and missing features

| Required information | HELECAR-D | Tesla dataset | Adaptation |
|---|---|---|---|
| Vehicle speed | Available | `DI_uiSpeed` | Use as `speed_mps` |
| Acceleration | Derived | Derived (Savitzky–Golay) | Same derivation |
| Road grade | Derived from altitude | **Unavailable** | θ = 0 (flat-road) |
| Temperature | Available | Ambient (`VCFRONT_tempAmbient` / fallback) | Use ambient °C |
| Humidity | Available | **Missing** | Dropped (not invented) |
| Wind speed | Available | **Missing** | Dropped |
| Traffic | Available | **Missing** | Dropped |
| Speed limit | Available | **Missing** | Dropped |
| Battery SoC | Available | `SOCave292` | Weak supervision |
| Measured battery power | Unavailable | `V×I` (`BattVoltage132`, `RawBattCurrent132`) | **Evaluation only** |

**Neural inputs (4):** `speed_mps, acc_mps2, theta_rad, temperature_c`.

---

## 3. Vehicle parameters and sources

YAML: `configs/review/vehicle_tesla_bev{1,2}.yaml`.

| Parameter | BEV1 | BEV2 | Provenance |
|---|---|---|---|
| `battery_capacity_kwh` | 60.5 | 78.8 | Dataset paper Table 3 (nominal) |
| `mass_kg` | 1984 | 2054 | Public curb + 75 kg occupant (**assumption**) |
| `Crr` | 0.010 | 0.010 | Typical EV tyre (**assumption**) |
| `cda_m2` | 0.60 | 0.60 | Order-of-magnitude Model Y CdA (**assumption**) |
| `eta_drive` / `eta_regen` | 0.90 / 0.70 | same | Typical BEV (**assumption**) |
| `auxiliary_power_kw` | 0.5 | 0.5 | Constant placeholder; HVAC is a limitation |

**Residual bound:** train-pool P95 of `|P_meas − P_physics|` → grid `{10,20,40,60,80}` → **L = 40 kW** for both vehicles (`review_results/tesla/residual_calibration_bev{1,2}.json`). Not test-tuned. Twizy L = 4 kW is not reused.

---

## 4. Methodological adaptations (exact)

Unchanged core: longitudinal physics + bounded residual + SoC multi-window weak labels + discrete conservation + depletion state + residual prior.

Documented changes only:

1. Feature dimension 8 → 4 (missing covariates not filled with constants).  
2. Vehicle YAML for Tesla mass/capacity/drag/efficiency.  
3. Residual limit 4 → 40 kW (train-only calibration).  
4. Flat-road grade.  
5. Separate review configs/scripts/outputs; HELECAR-D `configs/paper.yaml` untouched.  
6. Methodology re-trained on Tesla pools (not zero-shot Twizy weights).  
7. Ridge uses kinematics+temperature trip summaries only (fair reduced whitelist).

**Training supervision:** SoC-derived window/trip energy only.  
**Forbidden in training/selection:** `batt_current_a`, `batt_voltage_v`, `p_meas_kw`.  
**Final evaluation:** both SoC-derived reference (paper-analogous) and independent V×I energy.

Current sign: positive `RawBattCurrent132` = discharge (audit + integrity test).

---

## 5. Training and validation protocol

- Leave-one-trip-out within the 17-trip predetermined pool (per vehicle).  
- Inner validation trip for early stopping; outer retrain on all non-test trips.  
- Training-only normalization.  
- Methods: Physics, Ridge, Weak-MLP, PINN.  
- Seed: 0 (CPU; multi-seed feasible later).  
- No measured energy in model selection.

Integrity script: `python scripts/review/test_tesla_integrity.py` (all checks passed).

---

## 6. Final performance results

### 6.1 Compact publication table (trip energy)

Values are leave-one-trip-out over **n = 17** independent sessions per vehicle, seed 0.

**Against SoC-derived reference (paper-analogous supervision target):**

| Vehicle | Method | MAE (kWh) | RMSE (kWh) | WAPE (%) | Bias (kWh) |
|---|---|---:|---:|---:|---:|
| BEV1 | Physics | 1.264 | 1.857 | 16.8 | −0.798 |
| BEV1 | Ridge | 1.729 | 2.975 | 23.0 | −0.910 |
| BEV1 | Weak-MLP | **0.584** | 0.711 | 7.8 | −0.114 |
| BEV1 | PINN | 0.675 | 0.875 | 9.0 | +0.026 |
| BEV2 | Physics | 1.050 | 1.368 | 11.2 | +0.095 |
| BEV2 | Ridge | 1.228 | 1.681 | 13.1 | +0.133 |
| BEV2 | Weak-MLP | **0.770** | 1.047 | 8.2 | −0.117 |
| BEV2 | PINN | 0.776 | 1.040 | 8.3 | +0.014 |

**Against independent measured battery energy (V×I):**

| Vehicle | Method | MAE (kWh) | RMSE (kWh) | WAPE (%) | Bias (kWh) |
|---|---|---:|---:|---:|---:|
| BEV1 | Physics | 1.051 | 1.374 | 15.0 | −0.301 |
| BEV1 | Ridge | 1.645 | 2.560 | 23.4 | −0.413 |
| BEV1 | Weak-MLP | **0.616** | 0.783 | 8.8 | +0.383 |
| BEV1 | PINN | 0.805 | 1.124 | 11.4 | +0.523 |
| BEV2 | Physics | 0.972 | 1.270 | 10.7 | +0.374 |
| BEV2 | Ridge | 1.237 | 1.663 | 13.6 | +0.412 |
| BEV2 | Weak-MLP | 0.810 | 1.030 | 8.9 | +0.162 |
| BEV2 | PINN | **0.640** | 0.845 | 7.1 | +0.293 |

### 6.2 Paired PINN − MLP absolute-error differences

Bootstrap 95% CIs include 0 in all four settings (`review_results/tesla/paired_pinn_vs_mlp.json`):

| Setting | Mean (PINN−MLP) | 95% CI | PINN better (count) |
|---|---:|---|---|
| BEV1 vs SoC | +0.092 | [−0.16, +0.35] | 9/17 |
| BEV1 vs meas | +0.188 | [−0.09, +0.46] | 8/17 |
| BEV2 vs SoC | +0.006 | [−0.42, +0.41] | 10/17 |
| BEV2 vs meas | −0.170 | [−0.53, +0.17] | 12/17 |

**Honest reading:** learning methods beat physics/ridge. PINN and MLP are statistically comparable on this pool; PINN is not uniformly superior. On BEV2 measured energy, PINN has lower mean MAE; on BEV1, MLP is slightly better. Do **not** claim a Tesla MAE win comparable to Twizy absolute numbers (different energy scale).

### 6.3 SoC-derived vs measured energy agreement

| Vehicle | MAE \|E_SoC − E_VI\| (kWh) | Correlation |
|---|---:|---:|
| BEV1 | 0.50 | 0.9997 |
| BEV2 | 0.33 | 0.997 |

References are highly correlated but not identical (~0.3–0.5 kWh typical absolute gap), supporting independent final evaluation.

### 6.4 Data-scarcity (secondary, BEV1 partial)

Protocol: fixed last-4 pool trips as tests; train sizes {3,5,8}; up to 3 subset repeats; seed 0.  
Run interrupted during `n=8` rep 2 / before `n=12`; aggregated completed cells only.

| n_train | Physics MAE | Ridge | Weak-MLP | PINN |
|---:|---:|---:|---:|---:|
| 3 | 2.083 | 2.518 | 0.630 | **0.511** |
| 5 | 2.083 | 1.756 | 0.846 | **0.805** |
| 8 | 2.083 | 1.319 | 0.806 | **0.231** |

(MAE vs SoC-derived kWh; `review_results/tesla/scarcity/bev1/scarcity_summary_soc_partial.csv`.)

Under few training trips, PINN mean MAE is lower than MLP in these completed cells, consistent with the paper’s scarcity narrative, but the sample is small and variance across repeats is high (treat as supportive, not definitive).

---

## 7. Comparison with original research conclusions

| Claim from Twizy paper | Tesla evidence |
|---|---|
| Physics alone is insufficient / improvable | Supported — neural methods reduce MAE vs physics on both vehicles |
| Weak SoC supervision can train without power labels | Supported — training never used V×I targets |
| PINN helps under data scarcity / physics constraints | Partially supported — PINN competitive with MLP; advantage not significant on this 17-trip protocol |
| Methodology transfers across EVs | Conditionally supported as **feature-adapted re-training**, not zero-shot transfer |

Absolute Twizy vs Tesla MAE must not be compared as like-for-like (pack size, trip energy, feature set differ).

---

## 8. Limitations and threats to validity

1. Flat-road grade (no elevation in drive CSVs).  
2. Reduced 4-feature inputs vs Twizy 8.  
3. Vehicle parameters partly assumed (CdA, η, Crr, P_aux).  
4. HVAC/aux not modelled dynamically.  
5. Predetermined 17-trip pool, not full QC corpus (compute / length constraint).  
6. Single neural seed (0) on CPU.  
7. Some PINN folds early-stopped at very low epochs (validation volatility).  
8. Nominal capacity ≠ exact usable energy (see SoC–VI gaps).  
9. Same-day session dependence may remain within the pool despite trip-level LOTO.

---

## 9. Reproduction commands

```bash
# Extract (gitignored data/tesla)
python scripts/review/prepare_tesla_trips.py --vehicle both

# Integrity
python scripts/review/test_tesla_integrity.py

# Train-only residual bound
python scripts/review/calibrate_tesla_residual.py --config configs/review/tesla_bev1_loto.yaml
python scripts/review/calibrate_tesla_residual.py --config configs/review/tesla_bev2_loto.yaml

# Primary LOTO
python scripts/review/run_tesla_loto.py --config configs/review/tesla_bev1_loto.yaml --seeds 0
python scripts/review/run_tesla_loto.py --config configs/review/tesla_bev2_loto.yaml --seeds 0

# Optional scarcity
python scripts/review/run_tesla_scarcity.py --config configs/review/tesla_bev1_loto.yaml --seed 0
```

Do **not** push datasets or experimental dumps to GitHub.

---

## 10. Suitability for INTIS 2026 camera-ready

**Recommendation: optional short subsection only — not a new contribution axis.**

Suitable as ~½ page in Section 6 if framed carefully:

- External **feature-adapted** replication on Tesla Model Y (BEV1/BEV2 separate).  
- Same weak-supervision question; V×I used only for independent evaluation.  
- One compact table (e.g. BEV2 vs measured energy, or both vehicles MAE columns).  
- Explicit limitations: flat grade, 4 features, assumed parameters, PINN≈MLP.

**Do not** expand into a multi-vehicle benchmark or change the accepted architecture.

### Suggested ~½-page subsection draft (for editor consideration)

> **External evaluation on Tesla Model Y.** To address single-vehicle generalizability concerns, we re-applied the same physics-informed weak-supervision methodology to the KU Leuven BEV Energy Dynamics dataset (Model Y SR/LR), without using measured battery power as training labels. Missing humidity/wind/traffic/speed-limit covariates and elevation forced a documented 4-feature, flat-road adaptation and a train-only residual bound of 40 kW. On 17 held-out driving sessions per configuration, learning-based estimators reduced trip-energy error relative to the physics baseline; PINN and a matched MLP were statistically comparable (paired bootstrap CIs of absolute-error differences included zero). Independent V×I energy agreed closely with SoC-derived references (MAE 0.3–0.5 kWh). These results support applicability of the proposed training scheme beyond the Twizy HELECAR-D setting under explicit feature and modelling limitations, without claiming zero-shot cross-vehicle transfer.

---

## Artifacts

| Path | Role |
|---|---|
| `review_notes/tesla_dataset_audit.md` | Phase A audit |
| `review_notes/tesla_physical_compatibility.md` | Phase B |
| `review_notes/tesla_go_nogo.md` | Phase C |
| `review_results/tesla/loto/tesla_bev{1,2}_loto/` | Primary metrics |
| `review_results/tesla/paired_pinn_vs_mlp.json` | Paired tests |
| `configs/review/tesla_bev{1,2}_loto.yaml` | Experiment configs |
| `scripts/review/*` | Pipeline |
