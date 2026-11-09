# Tesla Model Y — BEV Energy Dynamics Dataset Audit

**Paper context:** INTIS 2026 external-vehicle validation (reviewers’ generalizability concern)  
**Branch:** `review/intis2026-validation`  
**Baseline commit:** `0a85b57c3cad6d81a9dfb225f69445546860cc50`  
**Dataset DOI:** https://doi.org/10.48804/8KPDTW  
**Local path:** `BEV energy dynamics dataset/` (gitignored; not for commit)  
**Associated paper:** Yasko et al., *Scientific Data* (2025), “Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data”

---

## 1. Local structure

| Item | Content |
|---|---|
| `readme_V2.txt` | Dataset description, CC BY 4.0, session-type folders |
| `BEV energy dynamic data_V2.zip` (~162 MB) | Pre-processed CSVs + additional files |
| Inside zip | `driving sessions/` (1396), `parking sessions/` (780), `slow charging sessions/` (109), `fast charging sessions/` (25), `Additional files/` (variable list, DBC, example MF4/CSV, scripts) |

Driving sessions by vehicle (filename prefix):

| Vehicle | Driving CSV count |
|---|---:|
| BEV1 | 382 |
| BEV2 | 1014 |
| **Total** | **1396** |

---

## 2. Vehicle configurations (from Scientific Data Table 3)

| ID | Model | Battery | Chemistry | Motor | Nominal capacity |
|---|---|---|---|---|---:|
| BEV1 | Tesla Model Y Standard Range | LFP | LFP | RWD, 220 kW | **60.5 kWh** |
| BEV2 | Tesla Model Y Long Range | NCA | NCA | RWD, 324 kW | **78.8 kWh** |

These are **different vehicles**. They must not be pooled into one model without an explicit multi-vehicle protocol. Default plan: evaluate **separately**.

Mass is not tabulated in the dataset paper. Provisional literature/certification tare masses (assumptions, not measured on these fleets):

| Vehicle | Assumed mass | Source / note |
|---|---:|---|
| BEV1 (SR) | ~1909 kg curb (+ occupant assumption TBD) | Public certification reports for Model Y SR |
| BEV2 (LR) | ~1979 kg curb (+ occupant assumption TBD) | Public certification reports for Model Y LR |

---

## 3. What is actually in the driving CSVs

Inspected multiple BEV1/BEV2 drive files. Pre-processed driving tables are **narrow** (~26–34 columns), not the full 2040-variable dictionary.

### Present and useful for this study

| Signal | Column | Notes |
|---|---|---|
| Timestamp | `Timestamp` | ≈ 1 s resolution (UTC strings) |
| Speed | `DI_uiSpeed` | Appears in km/h (max ~130); UI speed, not `DI_vehicleSpeed` |
| SoC | `SOCave292` | Continuous-looking; usable for weak supervision |
| Battery voltage | `BattVoltage132` | V |
| Battery current | `RawBattCurrent132` | A |
| Ambient temperature | `VCFRONT_tempAmbient` / `VCRIGHT_tempAmbientRaw` | °C |
| GPS | `GPSLatitude04F`, `GPSLongitude04F` | Nearly complete |
| Cumulative discharge | `TotalDischargeKWh3D2`, `BMS_kwhDriveDischargeTotal` | Counters; useful diagnostics |
| Odometer | `Odometer3B6` | Often sparse / sometimes corrupt values |

### Absent from driving CSVs (critical)

| Signal | Status |
|---|---|
| `Elevation3D8` | **Not present** in audited drive files |
| Road grade | **Not available** without elevation or external DEM |
| Longitudinal accelerometer | **Not present**; acceleration must be derived from speed |
| Humidity | Variable exists in dictionary (`VCRIGHT_thsHumidity`) but **not in drive CSVs** |
| Wind speed | **Not available** as meteorology |
| Traffic | **Not available** |
| Speed limit | Dictionary has map speed-limit fields; **not in drive CSVs** |
| `DI_vehicleSpeed` | **Not in drive CSVs** (only `DI_uiSpeed`) |

---

## 4. Feature compatibility vs HELECAR-D

| Required information | HELECAR-D | Tesla drive CSVs | Adaptation needed |
|---|---|---|---|
| Vehicle speed | Available (CAN) | `DI_uiSpeed` (UI speed, km/h) | Convert to m/s; document UI vs CAN difference |
| Acceleration | Derived from speed | Must derive from `DI_uiSpeed` | Same derivation approach; noisier if UI speed is smoothed |
| Road grade | Derived from GPS altitude | **No elevation in drive CSVs** | **Flat-road assumption (`θ=0`)** or external DEM from lat/lon (extra uncertainty) |
| Temperature | Available | Ambient available | Use ambient; not cabin |
| Humidity | Available | Missing in drive CSVs | **Drop** from feature set |
| Wind speed | Available (not headwind) | Missing | **Drop** |
| Traffic | Available | Missing | **Drop** |
| Speed limit | Available | Missing in drive CSVs | **Drop** |
| Battery SoC | Available | `SOCave292` | Supervision only; never a power-branch input |
| Measured battery power | Unavailable | V×I available | **Evaluation only**; never training targets/inputs |

**Common-feature configuration for a fair PINN vs MLP comparison on Tesla data:**

`[speed_mps, acc_mps2, theta_rad, temperature_c]`

with `theta_rad = 0` under the flat-road baseline, unless a predetermined DEM grade protocol is added later.

This is a **feature-adapted replication** of the methodology, not an identical eight-input Twizy model.

---

## 5. Independent battery-energy reference and current sign

On 20 random usable drives:

- Correlation(`ΔSoC`, energy from `V * I`) ≈ **+0.978** when **positive current = discharge**
- Correlation with the opposite sign ≈ −0.978

Therefore the working convention is:

**Positive `RawBattCurrent132` = battery discharge (traction/aux draw).**  
**Measured power `P = V * I` (W), integrated with actual `Δt`.**

Implied usable energy from `100 * E_VI / ΔSoC` (sessions with ΔSoC ≥ 2 pp):

| Vehicle | Mean implied kWh | Dataset nominal capacity |
|---|---:|---:|
| BEV1 | ≈ 56.3 | 60.5 |
| BEV2 | ≈ 76.8 | 78.8 |

Interpretation: SoC-derived energy using nominal pack capacity will be **close but not identical** to V×I energy. Nominal ≠ exact usable capacity. For weak supervision use a documented capacity assumption; for final evaluation report both SoC-derived and V×I energy.

---

## 6. Usable driving sessions (predetermined QC)

Preliminary QC (not tuned on model test MAE):

1. Duration ≥ 5 min  
2. SoC drop ≥ 0.5 percentage points  
3. Moving fraction (speed > 1) ≥ 0.2  
4. Non-null coverage ≥ 0.9 for SoC, speed, V, I  

| Vehicle | Passing sessions |
|---|---:|
| BEV1 | 193 |
| BEV2 | 540 |
| Total | 733 / 1396 |

Stricter candidates for primary LOTO (duration ≥ 15 min, ΔSoC ≥ 2 pp, moving ≥ 0.4):

| Vehicle | Count |
|---|---:|
| BEV1 | 136 |
| BEV2 | 247 |

Charging and parking sessions are **excluded** from driving-energy estimation. Regenerative braking (negative current) is retained inside driving sessions.

Audit tables:

- `review_results/tesla/drive_session_audit.csv`
- `review_results/tesla/drive_session_usable_candidates.csv`

---

## 7. Physical / methodological compatibility

### What remains of the original PINN

Still applicable in principle:

1. Longitudinal dynamics (with documented grade assumption)  
2. Bounded neural residual  
3. SoC-derived multi-window energy supervision  
4. Discrete conservation via depletion head  
5. State consistency + residual prior  

### Required documented adaptations

| Item | Twizy | Tesla adaptation |
|---|---|---|
| Residual bound L | 4 kW | Must be re-justified (Tesla pack power is far larger). Predetermined from training-only power percentiles / physical ceiling; **not** test-tuned |
| Battery capacity | 6.0 kWh assumption | BEV1 60.5 / BEV2 78.8 nominal; usable ≈ 56 / 77 from V×I–SoC check |
| Feature dimension | 8 | 4 common features (speed, accel, grade≈0, ambient T) |
| Physics mass / CdA / Crr / η | Twizy YAML | New Tesla YAML with cited assumptions |
| Session definition | 17 analysed trips | Many short/long drives; use QC’d independent sessions; avoid leakage across same-day splits carefully |

---

## 8. GO / NO-GO assessment

| Question | Answer |
|---|---|
| Compatible with essential PINN physics formulation? | **Conditional yes** — yes with flat-road (or DEM) grade assumption and reduced features |
| Train with SoC-derived energy without power labels? | **Yes** — `SOCave292` available |
| Independent battery-energy reference available? | **Yes** — V×I with verified discharge-positive current |
| Vehicle parameters justifiable? | **Partially** — capacities documented; mass/CdA/Crr/η are assumptions that must be labelled |
| Independent held-out trips? | **Yes** — hundreds of candidate sessions per vehicle |
| Without changing main scientific contribution? | **Yes** — feature-adapted replication; not a new architecture |

### Decision: **CONDITIONAL GO**

Proceed under these hard constraints:

1. Evaluate **BEV1 and BEV2 separately**.  
2. Label the experiment as a **feature-adapted** replication (4 inputs; no humidity/wind/traffic/speed-limit).  
3. Default physics grade = 0 unless a predetermined external DEM protocol is added.  
4. Keep V, I, V×I **out of training and model selection**.  
5. Re-justify residual bound L without using held-out test ranks.  
6. Do not claim identical Twizy feature fidelity or universal EV generalization.  
7. Do not overwrite HELECAR-D paper results or `main`.

If reviewers require grade fidelity and refuse flat-road / DEM assumptions, revisit as NO-GO for physics-faithful transfer.

---

## 9. Reproduction of this audit

```bash
# From repo root, with the zip present under "BEV energy dynamics dataset/"
python -c "import pandas as pd; print(pd.read_csv('review_results/tesla/drive_session_usable_candidates.csv').groupby('vehicle').size())"
```

Dataset must remain gitignored.

---

## 10. Post-audit extraction note (implementation)

`scripts/review/prepare_tesla_trips.py` wrote gitignored CSVs under `data/tesla/`:

| Vehicle | QC-passing extracted trips |
|---|---|
| BEV1 | 214 |
| BEV2 | 321 |

Primary LOTO uses a **predetermined stratified pool of 17 trips/vehicle** (`pool_seed=2026`) documented in `review_results/tesla/pools/`, matching the Twizy fold count for a fair methodology check under review-time compute limits. This is not a claim that the unused QC trips are invalid.
