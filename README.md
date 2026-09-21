# Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity

Research code for a conference paper investigating whether a **physics-informed neural network** can estimate Renault Twizy energy consumption from **sparse HELECAR-D trips** when **instantaneous battery-power labels are unavailable** and the available battery supervision is a **quantized/noisy State-of-Charge (SoC)** signal.

This is **not** an instantaneous power-prediction paper. There is no reliable decoded battery-power target in the analysed HELECAR-D files. The model is weakly supervised: integrated latent power must match SoC-derived energy over informative windows.

Analysed SoC is **not** a 1% step. Empirical successive changes are typically multiples of about **0.02 percentage points** (~1.2 Wh at 6 kWh). The paper framing is therefore **quantized/noisy SoC measurements without instantaneous battery-power labels**, not “coarse 1% SoC.” Quantization \(q\) is estimated from the data, not hard-coded.

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
            trapezoidal interval integration
                           |
                           v
                    E_hat[a:b]
                           |
                           v
          compare with SoC-derived energy
```

**Do not fabricate results.** All tables and figures are computed from the actual dataset once it is present. If the dataset is missing, the software structure and unit tests still run on synthetic fixtures; experiment scripts refuse to invent paper metrics.

## Status

The reproducible pipeline is implemented: per-fold SoC quantization \(q\), causal PINN state head, grouped inner validation, outer-fold retrain, leakage tests, smoke/quick/paper LOTO, ablations, data scarcity, cross-trajectory, sensitivity, feasibility, figures and tables.

Smoke and quick numbers are **pipeline diagnostics**. Only `configs/paper.yaml` after `reports/protocol_frozen.md` is the paper experiment.

This is a **discrete-time physics-informed neural network**, not an automatic-differentiation PDE PINN.

## Dataset

Intended dataset: **HELECAR-D** (NaitMalek et al., Data in Brief, 2023; CC-BY-4.0). Download it locally and place it anywhere inside this project directory. Discovery is recursive and does **not** assume a published folder layout. The dataset directory `HELECAR-D/` is gitignored and should not be committed.

The copy currently documented with the paper typically contains 17 analysed trips across trajectories T1, T2 and T3. Counts are **measured at runtime**, never hard-coded as a correctness requirement.

Expected analysed columns (names are normalised; aliases such as `Soc`/`SoC` are accepted):

`Date, Time, SoC, Speed, Mode, LAT, LON, ALT, GpsSpeed, Temperature, Humidity, Weather, WindSpeedAv, Traffic, SpeedLimit`

- Use the vehicle CAN **Speed** field as the primary speed (km/h → m/s internally). Trip distance and Wh/km use integrated CAN speed.
- Use **GpsSpeed** mainly for quality checking. The audit also reports integrated GPS-speed distance and Haversine path length.
- GPS coordinates and altitude still provide spatial elevation for grade.
- Do **not** use Date, absolute Time, LAT, LON, or measured SoC as main neural-network predictors (optional SoC-input ablation is a later phase).
- Trip-level ElasticNet predictors are an **explicit whitelist** that excludes **all SoC-derived quantities** (`soc_start`, `soc_end`, `soc_delta`, …). Those columns are labels only; `ΔSOC = SOC_start − SOC_end` must never be recoverable from `X`.
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
python scripts/audit_data.py --config configs/base.yaml
python scripts/run_loto.py --config configs/smoke.yaml
python scripts/run_loto.py --config configs/quick.yaml
python scripts/run_loto.py --config configs/paper.yaml
python scripts/run_ablations.py --config configs/paper.yaml
python scripts/run_data_scarcity.py --config configs/paper.yaml
python scripts/run_cross_trajectory.py --config configs/paper.yaml
python scripts/run_sensitivity.py --config configs/paper.yaml
python scripts/run_feasibility_sensitivity.py --config configs/paper.yaml
python scripts/make_paper_figures.py --results outputs/
python scripts/make_paper_tables.py --config configs/paper.yaml

make test
make audit
make smoke
make quick
make paper
```

`make paper` runs the frozen pipeline in order and reuses cached identical fold/method/seed runs. Do not treat smoke/quick metrics as paper results.

## Anti-leakage

Primary evaluation is **leave-one-trip-out** (never a random row split). Normalisation, \(q\), windows, Huber scale, early stopping and ElasticNet fitting use **training trips only**. Inner validation is a deterministic grouped set (one trip per available T1/T2/T3). After model selection the neural model is **discarded** and retrained on all outer-training trips. Test-trip SoC is not a network input; `SOC_test[0]` is used only after power prediction for diagnostic SoC reconstruction. `q_test_posthoc` is evaluation-only.

## Vehicle physics (nominal)

Implemented in `src/physics/vehicle_model.py` with parameters in `configs/vehicle_twizy.yaml`.

- `F = m a + m g Crr cos(θ) + m g sin(θ) + ½ ρ (CdA) v²` with `CdA = 0.64 m²` (Renault SCx; not `Cd ×` a separate assumed frontal area)
- `P_wheel = F v`
- Traction: `P_battery = P_wheel / η_drive + P_aux`
- Regen: `P_battery = η_regen P_wheel + P_aux` (`P_wheel < 0`)
- 13 kW is the **motor/wheel** maximum, not a battery-power cap. Battery power can exceed 13 kW because of drivetrain losses.
- Power bounds are **off** for the physics baseline. If enabled, they apply only to wheel power and are approximately the identity inside the valid range.
- Energy uses trapezoidal interval integration: `Δt_i = t_{i+1} − t_i`, `P_{i+1/2} = (P_i + P_{i+1}) / 2`, `E(a,b) = Σ_{i=a}^{b-1} P_{i+1/2} Δt_i / 3600 = prefix[b] − prefix[a]`
- `ΔSOC_% = 100 E_kWh / E_battery_kWh`

**Main nominal battery capacity is 6 kWh** (HELECAR-D paper approximation). Manufacturer specification is 6.1 kWh. A 75 kg driver is a modelling assumption; Renault payload figures are around 110–115 kg. Crr, CdA, efficiencies and auxiliary power are **literature estimates or modelling assumptions, not HELECAR measurements**. See comments in `configs/vehicle_twizy.yaml`.

## Weak supervision

Do **not** form fake instantaneous labels `P_t = E_battery (SOC_t − SOC_{t+1})`. Observed energy over a window `w = [a, b]` is:

`E_obs(w) = E_battery (SOC[a] − SOC[b]) / 100`

Default SoC-event thresholds are **0.1, 0.2 and 0.5 percentage points** with a **60 s** minimum duration, plus fixed windows of **60, 120, 300 and 600 s**. Those scales are chosen so supervision does not collapse to a few seconds of near-instantaneous SoC difference (1q/2q/3q at `q ≈ 0.02` would do that).

Predicted energy uses trapezoidal prefix integration of latent power.

## Reproducibility

Every audit/experiment run writes a manifest (timestamp, config, file names, hashes, versions, git commit if available) under `outputs/`.

## License

Research code is released under the MIT License (`LICENSE`). HELECAR-D retains its own CC-BY-4.0 license and is not redistributed in this repository.
