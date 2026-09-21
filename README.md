# Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity

Research code for a conference paper on **weakly supervised EV trip-energy estimation** from HELECAR-D. Instantaneous battery-power labels are unavailable. Supervision comes from quantized/noisy State-of-Charge (SoC) through integrated energy windows. The model is a **discrete-time physics-informed neural network**, not an automatic-differentiation PDE PINN.

Paper-ready numbers, PNG figures, and CSV tables live in [`results/`](results/). Do not treat runtime dumps under `outputs/` as the archival paper record.

## 1. Overview

The practical question is trip energy: how much battery energy a Renault Twizy uses on a held-out real drive. The analysed HELECAR-D files do not provide a reliable instantaneous battery-power target. Sample-to-sample SoC differences are too quantized and noisy to treat as power. Differentiating SoC at 1 Hz would pretend that a ~0.02 percentage-point step is an energy label of about 1.2 Wh, which collapses supervision onto noise.

The estimator therefore predicts a latent battery-power trajectory \(\hat P(t)\), integrates it with the trapezoidal rule, and matches SoC-derived window energy. A longitudinal vehicle model supplies \(P_{\mathrm{physics}}(t)\). A small residual network may correct that prior. A causal depletion head \(\hat D(t)\) is tied to \(\hat P(t)\) by discrete battery-energy conservation.

## 2. Research question

Can physics-informed learning recover useful EV energy estimates from limited real-world trips when instantaneous battery-power measurements are unavailable and supervision is available only through quantized/noisy SoC?

## 3. Key contributions

1. A weakly supervised discrete-time PINN that combines longitudinal EV physics with SoC-derived energy windows.
2. Multi-scale energy supervision and a discrete conservation residual, without instantaneous battery-power labels.
3. Trip-level evaluation under leave-one-trip-out, data scarcity, trajectory shift, and a small routing-oriented battery-feasibility sensitivity.

This is not a claim of “first PINN for EV routing”, and it is not a new EVRP solver.

## 4. Dataset — HELECAR-D

HELECAR-D (NaitMalek et al., *Data in Brief*, 2023; CC-BY-4.0) records a Renault Twizy on roads in the Rabat–Salé–Kénitra region of Morocco (March–July 2021). Three trajectories are labelled T1 (urban + ring), T2 (urban), and T3 (ring).

The final analysed run uses **17 trips**: 4 T1, 3 T2, 10 T3. Counts are measured at runtime, not hard-coded as a correctness requirement. Duration ranges from 952 s to 8577 s. CAN-speed distance ranges from 13.8 km to 59.6 km (mean 29.9 km). Mean start SoC is 81.2%; mean end SoC is 37.1%.

Empirical successive SoC changes are typically multiples of about **0.02 percentage points**, not a 1% BMS step. Quantization \(q\) is estimated on training trips only.

Variables used as neural inputs: CAN speed, longitudinal acceleration, road grade angle, temperature, humidity, wind speed (covariate only; never a headwind), traffic ordinal, speed limit. Excluded from the power branch: Date, absolute Time, LAT, LON, and measured SoC. WindSpeedAv is not treated as a headwind because the analysed files have no wind direction.

Place the official dataset locally as described in [`data/README.md`](data/README.md). Do not commit HELECAR-D CSVs into this research-code repository.

## 5. Why this is weak supervision

There is no target \(P_{\mathrm{battery}}(t)\). Available battery information is \(\mathrm{SOC}(t)\). Window energy is

\[
E_{\mathrm{obs}}(a,b) = E_{\mathrm{battery}}\frac{\mathrm{SOC}[a]-\mathrm{SOC}[b]}{100}.
\]

Predicted window energy is the trapezoidal prefix difference \(E_{\mathrm{hat}}(a,b)=\mathrm{prefix}[b]-\mathrm{prefix}[a]\). Primary inference is the full-trip integral of \(\hat P\). Test-trip SoC is never a neural-network input. The held-out initial SoC is used only after prediction, to reconstruct a diagnostic SoC curve.

## 6. Physical model

Longitudinal force:

\[
F = ma + mgC_{rr}\cos\theta + mg\sin\theta + \tfrac12\rho\,C_dA\,v^2.
\]

Wheel power \(P_{\mathrm{wheel}}=Fv\). Battery mapping:

- traction: \(P_{\mathrm{batt}}=P_{\mathrm{wheel}}/\eta_{\mathrm{drive}}+P_{\mathrm{aux}}\)
- regen: \(P_{\mathrm{batt}}=\eta_{\mathrm{regen}}P_{\mathrm{wheel}}+P_{\mathrm{aux}}\)

Nominal modelling assumptions (not HELECAR measurements) are in [`configs/vehicle_twizy.yaml`](configs/vehicle_twizy.yaml): mass 549 kg (474 kg kerb + 75 kg assumed driver), \(C_{rr}=0.012\), \(C_dA=0.64\,\mathrm{m}^2\), \(\eta_{\mathrm{drive}}=0.85\), \(\eta_{\mathrm{regen}}=0.40\), \(P_{\mathrm{aux}}=0.20\,\mathrm{kW}\), usable capacity 6.0 kWh. Renault motor/wheel maximum is 13 kW; that is a **wheel** bound, not a battery cap. With the traction map, 13 kW wheel corresponds to about 15.5 kW battery. The frozen paper protocol leaves the wheel clamp **off**. A sensitivity with the clamp on changes physics trip-energy MAE from 0.780 kWh to 0.783 kWh, so the energy conclusions are insensitive. Instantaneous peaks still exceed the motor rating (see Figure 8).

## 7. PINN architecture

\[
\hat P(t)=P_{\mathrm{physics}}(t)+\delta P_{\mathrm{NN}}(t),\qquad \delta P=\ell\tanh(\cdot),\ \ell=4\,\mathrm{kW}.
\]

Shared torso: 8 telemetry features \(\to\) 64-64-64 \(\tanh\). The depletion head sees causal progress (elapsed time, cumulative CAN distance) and is shifted so \(\hat D(0)=0\). Primary energy is \(\int\hat P\,dt\). This is **not** an autodiff PDE PINN: we do not form \(\mathrm{SOC}=\mathrm{NN}(t,\ldots)\) and take \(\partial\mathrm{SOC}/\partial t\) while holding time-varying covariates constant.

## 8. Physics-informed loss

\[
L=\lambda_{\mathrm{window}}L_{\mathrm{window}}+\lambda_{\mathrm{dynamics}}L_{\mathrm{dynamics}}+\lambda_{\mathrm{state}}L_{\mathrm{state}}+\lambda_{\mathrm{prior}}L_{\mathrm{prior}}.
\]

Default weights: \(\lambda_{\mathrm{window}}=1\), \(\lambda_{\mathrm{dynamics}}=1\), \(\lambda_{\mathrm{state}}=0.1\), \(\lambda_{\mathrm{boundary}}=0\), \(\lambda_{\mathrm{prior}}=0.1\). Window and state terms use **Smooth L1** (normalized Huber / PyTorch SmoothL1 with \(\beta=\delta\)):

\[
0.5\,r^2/\delta\quad(|r|\le\delta),\qquad |r|-0.5\delta\quad\text{otherwise}.
\]

This is not the classical unscaled Huber definition. YAML keys still say `huber_*` so cached trainings keep the same identity. Conservation residual:

\[
r_i=\hat D[i+1]-\hat D[i]-\frac{100\,\Delta E_i}{E_{\mathrm{battery}}},\qquad
\Delta E_i=\tfrac12(\hat P_i+\hat P_{i+1})\Delta t_i/3600.
\]

Window Smooth-L1 \(\delta\) is \(E_{\mathrm{battery}}q/100\) from **training-trip** \(q\).

## 9. Anti-leakage protocol

- Leave-one-trip-out. Never a random 1 Hz row split.
- Scalers, \(q\), windows, and early stopping use training / grouped inner-validation trips only.
- Grouped inner validation prefers one trip from each of T1/T2/T3 when possible.
- Selected epoch is retrained on all outer-training trips (`retrain_outer: true`).
- Test SoC is not a predictor. Inference batches assert that measured SoC is absent.
- ElasticNet features are an explicit whitelist with **no SoC-derived columns**.
- The statistical unit is the **trip**. Neural multi-seed absolute errors are averaged per trip before pairing. Seed-averaged predictions in Figure 2 are a visualization, not a deployed ensemble.
- Paper numbers come only from `configs/paper.yaml` after the frozen protocol. Smoke/quick configs were engineering diagnostics and are not part of the final package.

## 10. Baselines and ablations

| Method | What it answers |
|---|---|
| Constant | Mean Wh/km on training trips. |
| Physics | Analytical \(P_{\mathrm{batt}}\) integral, no learning. |
| ElasticNet | Trip-level linear baseline on a SoC-free feature whitelist. |
| WeakMLP | Data-only latent power from the same windows; no physics. |
| MLP_STATE | WeakMLP plus causal depletion head, no physics residual. |
| PINN_NO_DYNAMICS | Physics residual without the conservation term. |
| PINN_FULLTRIP | Full PINN loss, but only the full-trip window. |
| PINN | Multi-window PINN with dynamics, state, and prior. |

## 11. Experiments

1. Dataset audit (17 analysed trips).
2. Main LOTO, seeds 0/1/2, 17 folds, no recorded failures.
3. Trip-level bootstrap and paired comparisons.
4. PINN ablations listed above.
5. Data scarcity, \(n_{\mathrm{train}}\in\{3,5,8,12,16\}\), 3 subset repeats. Complete paper table currently uses seed 0; seeds 1–2 were launched and can be aggregated when finished without retuning.
6. Cross-trajectory: hold out all of T1, or T2, or T3.
7. Sensitivity of physics energy to capacity, parameters, and preprocessing.
8. Physical wheel-power bound sensitivity.
9. Routing-oriented battery-feasibility sensitivity at 5/10/15/20% reserve SoC. Not an EVRP algorithm.

## 12. Main results

ElasticNet has the best ordinary LOTO trip-energy accuracy. PINN improves substantially on WeakMLP and on analytical physics. PINN_FULLTRIP has lower trip-energy MAE than the complete multi-window PINN. Those are kept as findings; the protocol was not retuned from test ranks.

| Method | MAE (kWh) | 95% trip bootstrap CI | RMSE | Bias |
|---|---:|---|---:|---:|
| Constant | 0.249 | [0.168, 0.337] | 0.306 | +0.009 |
| Physics | 0.780 | [0.614, 0.948] | 0.854 | −0.780 |
| ElasticNet | 0.167 | [0.113, 0.232] | 0.210 | −0.006 |
| WeakMLP | 0.475 | [0.312, 0.655] | 0.629 | +0.191 |
| MLP_STATE | 0.372 | [0.265, 0.486] | 0.487 | +0.140 |
| PINN_NO_DYNAMICS | 0.278 | [0.215, 0.342] | 0.348 | −0.023 |
| PINN_FULLTRIP | 0.243 | [0.182, 0.306] | 0.301 | −0.099 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | −0.052 |

Paired trip-level bootstrap of absolute-error differences (negative means the first method is better): PINN − WeakMLP = −0.180 kWh, interval [−0.316, −0.056]; PINN − ElasticNet = +0.128 [0.014, 0.242]; PINN − Physics = −0.485 [−0.604, −0.375]. Intervals are descriptive; they are not called “statistically significant”. Wilcoxon \(p\)-values are secondary.

Data scarcity (seed 0, 3 subset repeats): PINN has lower mean MAE than ElasticNet and WeakMLP at \(n=3\) (0.317 vs 0.374 vs 0.575 kWh). ElasticNet is unstable at \(n=8\) (mean MAE 1.006 kWh). ElasticNet wins \(n=12\) and \(n=16\). Cross-trajectory: PINN is strongest on held-out T1 and T3; ElasticNet is strongest on held-out T2 (3 trips). Do not overclaim from three trajectories.

## 13. Figures

All final figures are PNG, 300 dpi, under [`results/figures/`](results/figures/).

| File | Role |
|---|---|
| `fig01_method_overview.png` | Discrete-time PINN schematic. |
| `fig02_observed_vs_predicted.png` | Trip-level parity for Physics, ElasticNet, WeakMLP, PINN. |
| `fig03_trip_absolute_errors.png` | Paired trip absolute errors. |
| `fig04_ablation.png` | Ablation MAE with 95% bootstrap intervals, including PINN_FULLTRIP vs PINN. |
| `fig05_data_scarcity.png` | Sample-efficiency curve (log MAE). |
| `fig06_cross_trajectory.png` | Hold-out T1/T2/T3. |
| `fig07_soc_reconstruction.png` | Predetermined one-per-trajectory SoC reconstruction. |
| `fig08_power_plausibility.png` | Latent power vs traction/regen battery equivalents. |
| `fig09_feasibility.png` | Routing-oriented battery-feasibility sensitivity. |
| `fig10_training_diagnostics.png` | One representative-fold loss diagnostic. |

## 14. Repository structure

```
PINN_ev_energy/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/                 # paper.yaml, base.yaml, vehicle_twizy.yaml
├── data/README.md
├── src/                     # data, physics, models, training, evaluation, experiment, plotting
├── scripts/                 # audit + experiment runners + final export
└── results/                 # paper figures, tables, protocol, summary
```

Place HELECAR-D locally; it is gitignored. Runtime caches may appear under `outputs/` and are also gitignored.

## 15. Reproduction

```bash
python -m pip install -e .
python scripts/run_final_pipeline.py --config configs/paper.yaml
```

The wrapper audits the dataset, reuses existing paper LOTO outputs when present, runs scarcity / missing sensitivities, writes PNG figures, and writes CSV tables. It does not run smoke or quick diagnostics. Optional bounded-PINN retrain:

```bash
python scripts/run_power_bound_sensitivity.py --config configs/paper.yaml --retrain-pinn --seeds 0
```

That retrain is a sensitivity, not a replacement of the main table.

## 16. Expected output

[`results/tables/`](results/tables/) contains `table01`–`table09` CSV files. [`results/figures/`](results/figures/) contains PNG only. [`results/protocol.md`](results/protocol.md) describes the actual code path. [`results/experiment_summary.md`](results/experiment_summary.md) is the paper-writing cheat sheet.

## 17. Limitations

Seventeen trips, one vehicle, limited geography. No instantaneous battery-power ground truth, so latent \(P(t)\) cannot be validated directly. Usable capacity and vehicle parameters are modelling assumptions. SoC is quantized/noisy; one analysed trip (`T3_05_25_2021_04`) contains a large SoC jump. Instantaneous physics/PINN power can exceed the 13 kW motor rating even though trip energy is insensitive to a wheel clamp. The downstream experiment is only a battery-feasibility sensitivity, not a routing algorithm.

## 18. Citation / dataset attribution

NaitMalek, Y., Najib, M., Bakhouya, M., Essaaidi, M., 2023. HELECAR-D: A dataset for electric vehicle energy consumption and driving patterns. *Data in Brief*. License: CC-BY-4.0. See the official HELECAR-D release for the authoritative citation.

## 19. License

Research code: MIT (`LICENSE`). HELECAR-D: CC-BY-4.0, not redistributed here.
