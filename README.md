# Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity

Research code for a conference paper investigating whether a **physics-informed neural network** can estimate Renault Twizy energy consumption from **sparse HELECAR-D trips** when **instantaneous battery-power labels are unavailable** and the available battery supervision is a **coarse/quantized State-of-Charge (SoC)** signal.

This is **not** an instantaneous power-prediction paper. There is no reliable decoded battery-power target in the analysed HELECAR-D files. The model is weakly supervised: integrated latent power must match SoC-derived energy over informative windows.

```
driving telemetry
        |
        v
longitudinal physics ------+
                           |
                           v
                     P_physics(t)
                           |
                           + <--- small NN correction (Phase 7)
                           |
                           v
                       P_hat(t)
                           |
                  numerical integration
                           |
                           v
                    E_hat[a:b]
                           |
                           v
          compare with SoC-derived energy
```

**Do not fabricate results.** All tables and figures are computed from the actual dataset once it is present. If the dataset is missing, the software structure and unit tests still run on synthetic fixtures; experiment scripts refuse to invent paper metrics.

## Status

| Phase | Content | Status |
|------:|---------|--------|
| 1 | Project structure, dataset discovery, audit | Implemented |
| 2 | SI preprocessing, acceleration, distance, grade | Implemented |
| 3 | Physics-only vehicle model | Implemented |
| 4 | Quantization-aware energy windows | Implemented |
| 5–13 | Baselines, MLP, PINN, LOTO, ablations, scarcity, figures | Stubs only |

## Dataset

Intended dataset: **HELECAR-D** (NaitMalek et al., Data in Brief, 2023). Place it anywhere inside this project directory. Discovery is recursive and does **not** assume a published folder layout.

The copy currently documented with the paper typically contains 17 analysed trips across trajectories T1, T2 and T3. Counts are **measured at runtime**, never hard-coded as a correctness requirement.

Expected analysed columns (names are normalised; aliases such as `Soc`/`SoC` are accepted):

`Date, Time, SoC, Speed, Mode, LAT, LON, ALT, GpsSpeed, Temperature, Humidity, Weather, WindSpeedAv, Traffic, SpeedLimit`

- Use the vehicle CAN **Speed** field as the primary speed (km/h → m/s internally).
- Use **GpsSpeed** mainly for quality checking.
- Do **not** use Date, absolute Time, LAT, LON, or measured SoC as main neural-network predictors (optional SoC-input ablation is a later phase).
- **WindSpeedAv is not treated as a headwind** in the analytical aerodynamic equation. The dataset does not provide wind direction. Vehicle speed is used for drag; wind speed may remain an ML covariate later.

Original CSVs are never modified.

## Setup

Python 3.10+.

```bash
python -m pip install -e ".[dev]"
# or
python -m pip install -r requirements.txt
```

## Commands

```bash
# Dataset audit (Phase 1–4 diagnostics)
python scripts/audit_data.py --config configs/base.yaml

# Unit tests (synthetic fixtures; no paper numbers)
python -m pytest -q
# or: make test

# Later phases (not implemented yet; will fail honestly)
python scripts/run_loto.py --config configs/quick.yaml
python scripts/run_loto.py --config configs/paper.yaml
make paper
```

## Anti-leakage (later experiments)

Primary evaluation is **leave-one-trip-out** (never a random row split). Normalisation, calibration, hyperparameter selection and early stopping use **training trips only**. Test-trip SoC is used only after prediction to compute metrics.

## Vehicle physics (nominal)

Implemented in `src/physics/vehicle_model.py` with parameters in `configs/vehicle_twizy.yaml`.

- `F = m a + m g Crr cos(θ) + m g sin(θ) + ½ ρ Cd A v²`
- `P_wheel = F v`
- Traction: `P_battery = P_wheel / η_drive + P_aux`
- Regen: `P_battery = η_regen P_wheel + P_aux` (`P_wheel < 0`)
- Energy: `E_kWh = Σ P_kW Δt_s / 3600`
- `ΔSOC_% = 100 E_kWh / E_battery_kWh`

**Main nominal battery capacity is 6 kWh** (HELECAR-D paper approximation). Manufacturer specification is 6.1 kWh. Driver mass, Crr, CdA, efficiencies and auxiliary power are **literature estimates or modelling assumptions, not HELECAR measurements**. See comments in `configs/vehicle_twizy.yaml`.

## Weak supervision

Do **not** form fake instantaneous labels `P_t = E_battery (SOC_t − SOC_{t+1})`. SoC is stepwise. Observed energy over a window `w = [a, b]` is:

`E_obs(w) = E_battery (SOC[a] − SOC[b]) / 100`

Predicted energy uses cumulative integration of latent power.

## Reproducibility

Every audit/experiment run writes a manifest (timestamp, config, file names, hashes, versions, git commit if available) under `outputs/`.

## License

Research code. HELECAR-D retains its own CC-BY-4.0 license (`HELECAR-D/LICENSE.md` if present).
