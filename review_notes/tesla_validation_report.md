# Tesla Model Y external validation — INTIS 2026 review (CORRECTED)

**Status:** Phase-1 protocol corrections applied. Prior `tesla_bev*_loto` (seed 0 only, P_meas-based L) are **superseded**.  
**Authoritative profiles:** `tesla_bev1_loto_corrected`, `tesla_bev2_loto_corrected`  
**Branch:** `review/intis2026-validation` (on `origin`; repo currently public)  
**Baseline commit before Tesla work:** `0a85b57`  
**Decision:** CONDITIONAL GO as **feature-adapted** methodology transfer (not identical 8-feature Twizy transfer).

---

## Protocol corrections (Problems A–D)

| Issue | Fix |
|---|---|
| A. L from \|P_meas−P_physics\| on eval pool | Replaced by physics-only rule on **disjoint calib pool**; no V/I/P_meas. BEV1 L=40 kW; BEV2 L=20 kW |
| B. BEV2 sanity | Negatives retained; Tesla-aware peak threshold 400 kW; documented in `tesla_bev2_sanity_diagnosis.md` |
| C. Timestamp integration | Forward `dt[i]=t[i+1]−t[i]`, last 0; rebuild after row drops; matches `units.energy_kwh_from_power` |
| D. Single seed | Neural seeds `[0,1,2]` |
| E. Feature set | Still 4 inputs; θ=0; **must be called feature-adapted** |

Integrity: `python scripts/review/test_tesla_integrity.py` — all PASS after correction.

---

## Compact results (n=17 trips/vehicle, seeds 0–1–2)

### Against independent measured battery energy (V×I)

| Method | BEV1 MAE | BEV2 MAE |
|---|---:|---:|
| Physics | 1.051 | 0.972 |
| Ridge | 1.646 | 1.236 |
| Weak-MLP | 0.971 | 0.917 |
| PINN | **0.831** | **0.727** |

### Against SoC-derived reference

| Method | BEV1 MAE | BEV2 MAE |
|---|---:|---:|
| Physics | 1.264 | 1.052 |
| Weak-MLP | 1.006 | 0.923 |
| PINN | **0.659** | **0.827** |

Full CSVs: `review_results/tesla/loto/tesla_bev*_loto_corrected/`.

### Paired PINN − MLP (trip-mean over seeds)

| Setting | Mean Δ | 95% CI | PINN better |
|---|---:|---|---|
| BEV1 vs SoC | −0.347 | [−0.619, −0.114] | 12/17 |
| BEV1 vs meas | −0.141 | [−0.405, +0.072] | 9/17 |
| BEV2 vs SoC | −0.095 | [−0.383, +0.177] | 11/17 |
| BEV2 vs meas | −0.190 | [−0.492, +0.092] | 11/17 |

**Reading:** After protocol correction and three seeds, PINN mean MAE is lower than MLP on both vehicles for both references. The BEV1 SoC paired CI excludes zero (PINN better). Measured-energy paired CIs still include zero — do not over-claim statistical superiority on V×I.

### Sanity (corrected BEV2)

- `ok=false` solely due to Ridge negative predictions on `BEV2_2024-11-13_drive2` (all three seeds). Kept.
- No peak-power flags at 400 kW threshold.
- BEV1 corrected sanity: check `sanity_report.json` in that folder.

### Superseded preliminary numbers

Do **not** cite seed-0-only / P_meas-calibrated L results from `tesla_bev1_loto` / `tesla_bev2_loto` in the paper.

---

## Camera-ready framing

Feature-adapted Tesla evaluation supports **applicability of the weak-supervision methodology** to another EV under documented limitations (flat grade, 4 features, assumed parameters). It does not prove universal PINN>MLP dominance; report paired CIs honestly.

Suggested subsection text: `review_notes/camera_ready_revisions.md` §B (fill table from this report).
