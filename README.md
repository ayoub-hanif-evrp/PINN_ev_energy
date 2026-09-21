# Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity

Research code for a **normal conference paper**. The scientific scope is locked to the abstract below. This repository does not add extra research questions, architectures, or journal-scale experiments.

A discrete-time physics-informed neural network (PINN) estimates EV energy when only limited real-world driving data are available, instantaneous battery-power measurements are unavailable, and battery information arrives through quantized/noisy State-of-Charge (SoC) observations. The model combines longitudinal vehicle dynamics with neural learning, estimates latent battery power from driving and environmental variables, enforces battery-energy conservation, and uses multi-window energy constraints derived from SoC depletion instead of noisy instantaneous SoC differences.

Paper-ready numbers live in [`results/`](results/). Runtime dumps under `outputs/` are a cache, not the archival record.

## 1. Project Overview

The practical task is trip energy: how much battery energy a Renault Twizy uses on a held-out real drive. The analysed HELECAR-D files do not provide a reliable instantaneous battery-power target. The estimator therefore predicts a latent battery-power trajectory \(\hat P(t)\), integrates it, and matches SoC-derived window energy.

**Research question.** Can a physics-informed neural network provide useful EV energy estimates under severe data scarcity when instantaneous battery-power measurements are unavailable and supervision is available only through quantized/noisy SoC observations?

**Secondary question.** Does the resulting energy estimation affect battery-feasibility decisions relevant to EV routing?

The paper comparison contains four methods: Physics, ElasticNet, WeakMLP, and the proposed PINN. Evaluation is leave-one-trip-out on Moroccan HELECAR-D trips, plus a data-scarcity study and one small routing-oriented battery-feasibility assessment.

## 2. Scientific Motivation

Sample-to-sample SoC differences are too quantized and noisy to treat as instantaneous power. Differentiating SoC at 1 Hz would treat a ~0.02 percentage-point step as an energy label of about 1.2 Wh and collapse supervision onto noise.

There are **no direct instantaneous battery-power labels**. The model predicts latent battery power and is supervised indirectly through integrated energy. Physics supplies an inductive bias when few trips are available. The claim is not that PINN beats every baseline; the claim is that physics-informed learning is useful for weakly supervised neural EV energy estimation, particularly when training data are limited.

## 3. Dataset

HELECAR-D records a Renault Twizy on roads in the Rabat–Salé–Kénitra region of Morocco. Three trajectories are labelled T1 (urban + ring), T2 (urban), and T3 (ring). Place the official download locally as described in [`data/README.md`](data/README.md). HELECAR-D CSVs are gitignored and must not be committed.

From [`results/tables/table01_dataset.csv`](results/tables/table01_dataset.csv):

- 17 analysed trips: T1=4, T2=3, T3=10
- duration 952–8577 s
- CAN-speed distance 13.83–59.57 km (mean 29.94 km)
- empirical SoC resolution/quantization \(q \approx 0.02\) percentage points
- 57,671 useful multi-window supervision windows

SoC is quantized/noisy. Quantization \(q\) is estimated on **training trips only**.

Neural power-branch inputs: speed, acceleration, road grade, temperature, humidity, wind speed, traffic, speed limit. Excluded from the power branch: latitude, longitude, absolute date/time, and measured SoC. Wind speed is a covariate, not a headwind, because the analysed files have no wind direction.

## 4. Problem Formulation

Predict latent battery power \(\hat P(t)\) and the implied trip energy from driving/environment variables. Observed window energy is

\[
E_{\mathrm{obs}}(a,b)=E_{\mathrm{battery}}\frac{\mathrm{SOC}[a]-\mathrm{SOC}[b]}{100}.
\]

Predicted window energy is the trapezoidal integral of \(\hat P\) from \(a\) to \(b\). Primary inference is the full-trip integral. Test-trip SoC never enters the neural model. After prediction, \(\mathrm{SOC}_{\mathrm{test}}[0]\) may be used only to reconstruct a diagnostic SoC curve.

This is **not** validation of instantaneous \(\hat P(t)\) against power ground truth.

## 5. Longitudinal Physics Model

\[
F = ma + mgC_{rr}\cos\theta + mg\sin\theta + \tfrac12\rho\,C_dA\,v^2
\]

\[
P_{\mathrm{wheel}}=Fv
\]

Battery-power mapping:

- traction: \(P_{\mathrm{batt}}=P_{\mathrm{wheel}}/\eta_{\mathrm{drive}}+P_{\mathrm{aux}}\)
- regen: \(P_{\mathrm{batt}}=\eta_{\mathrm{regen}}P_{\mathrm{wheel}}+P_{\mathrm{aux}}\)

Nominal modelling assumptions (not HELECAR measurements) are in [`configs/vehicle_twizy.yaml`](configs/vehicle_twizy.yaml): mass 549 kg, \(C_{rr}=0.012\), \(C_dA=0.64\,\mathrm{m}^2\), \(\eta_{\mathrm{drive}}=0.85\), \(\eta_{\mathrm{regen}}=0.40\), \(P_{\mathrm{aux}}=0.20\,\mathrm{kW}\), usable capacity 6.0 kWh.

## 6. Weak SoC Energy Supervision

SoC is a label source, not a power-branch input. Tiny one-step SoC differences are not treated as instantaneous battery-power labels. Supervision uses integrated energy over windows. The PINN therefore learns \(\hat P(t)\) without a \(P_{\mathrm{batt}}(t)\) target.

## 7. Multi-Window Energy Constraints

Multi-window supervision is required by the abstract. For each window \((a,b)\),

\[
E_{\mathrm{pred}}(a,b)=\int_a^b \hat P(t)\,dt
\]

using trapezoidal integration, compared with \(E_{\mathrm{obs}}(a,b)\). Window definitions are fixed in the protocol and are **not** tuned on the test set. An internal full-trip-only ablation can have a lower trip-energy MAE; that finding is not used to remove multi-window supervision or to start another research direction.

## 8. Discrete-Time PINN

This is a **discrete-time physics-informed neural network**, not a PDE / autodiff PINN.

\[
\delta P(t)=\ell\tanh(\mathrm{NN}(x_t)),\qquad
\hat P(t)=P_{\mathrm{physics}}(t)+\delta P(t)
\]

with residual limit \(\ell=4\,\mathrm{kW}\).

Battery-energy conservation on interval \(i\to i+1\):

\[
\Delta E_i=\tfrac12\bigl(\hat P[i]+\hat P[i+1]\bigr)\,\Delta t_i/3600
\]

with a discrete SoC/depletion conservation residual tying \(\hat D\) to those energy increments. Default loss weights include window energy, conservation/dynamics, a light state term, and a residual prior. Window and state terms use Smooth L1. YAML keys may still say `huber_*` so cached trainings keep the same identity.

## 9. Baselines

| Method | Role in the paper |
|---|---|
| Physics | Conventional analytical longitudinal-vehicle model. No fitting. |
| ElasticNet | Regularized regression on a SoC-free feature whitelist. |
| WeakMLP | Purely data-driven neural network with the same weak window supervision and no physics. |
| PINN | Proposed discrete-time physics-informed model. |

A constant Wh/km baseline exists internally as a simple reference. It is **not** a major paper method. Extra ablations that may exist in cached LOTO files (MLP_STATE, PINN_NO_DYNAMICS, PINN_FULLTRIP) are **not** paper methods.

## 10. Trip-Level Cross-Validation

Primary evaluation is **leave-one-trip-out** (LOTO). No random row splitting.

For every fold: test = one full trip; training = all remaining trips. All of the following are training-only: normalization, SoC quantization estimate \(q\), model fitting, validation, early stopping, and ElasticNet fitting. Grouped inner validation prefers one trip from each of T1/T2/T3 when possible. The selected epoch is retrained on all outer-training trips. The statistical unit is the trip.

## 11. Data-Scarcity Experiment

Required by the abstract. Training-set sizes: 3, 5, 8, 12, 16 trips. Compared methods: ElasticNet, WeakMLP, PINN. Physics is a fixed reference line because it does not depend on training size. Models are **not** retuned per training-set size. Subsets are deterministic. The reported table uses seed 0 (three subset repeats for \(n=3,5,8,12\); \(n=16\) is LOTO with 16 training trips). This experiment answers only: how does each approach behave as fewer training trips are available?

## 12. Routing-Oriented Battery-Feasibility Experiment

This is a **small additional experiment**, not a new EVRP solver and not a routing optimization algorithm. Energy predictions are used only to evaluate battery-feasibility decisions at reserve levels 5%, 10%, 15%, and 20% SoC. Compared methods: Physics and PINN. Reported quantities: false-safe rate and overly conservative rate.

Call it a routing-oriented battery-feasibility assessment / sensitivity.

## 13. Main Results

Numbers below match [`results/tables/`](results/tables/). Do not rank methods with subjective labels.

**LOTO trip energy** ([`table02_main_results.csv`](results/tables/table02_main_results.csv)):

| Method | MAE (kWh) | 95% CI | RMSE | MAPE (%) | WAPE (%) | Bias | \(R^2\) |
|---|---:|---|---:|---:|---:|---:|---:|
| Physics | 0.780 | [0.614, 0.948] | 0.854 | 30.06 | 29.42 | −0.780 | 0.508 |
| ElasticNet | 0.167 | [0.113, 0.232] | 0.210 | 8.72 | 6.31 | −0.006 | 0.970 |
| WeakMLP | 0.475 | [0.312, 0.655] | 0.629 | 16.34 | 17.93 | +0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | 11.51 | 11.13 | −0.052 | 0.906 |

ElasticNet is strongest on ordinary full-data LOTO. PINN improves substantially over WeakMLP and over analytical Physics. PINN is **not** globally best.

**Data scarcity** ([`table03_data_scarcity.csv`](results/tables/table03_data_scarcity.csv)), held-out MAE (kWh):

| \(n_{\mathrm{train}}\) | ElasticNet | WeakMLP | PINN |
|---:|---:|---:|---:|
| 3 | 0.374 | 0.575 | 0.317 |
| 5 | 0.314 | 0.756 | 0.340 |
| 8 | 1.006 | 0.431 | 0.333 |
| 12 | 0.225 | 0.348 | 0.266 |
| 16 | 0.167 | 0.445 | 0.350 |

PINN is strongest at \(n=3\). ElasticNet is unstable at \(n=8\) and strongest at \(n=12\) and \(n=16\). PINN remains better than WeakMLP at every reported training size. Scarcity \(n=16\) PINN (0.350 kWh, seed 0) need not match the main-table PINN MAE (0.295 kWh, seeds 0/1/2).

**Feasibility** ([`table04_feasibility.csv`](results/tables/table04_feasibility.csv)): at a 20% reserve, Physics false-safe rate is 17.6% vs PINN 5.9%. Physics is more false-safe because it systematically under-predicts trip energy.

What the results support: physics-informed learning provides a useful inductive bias for weakly supervised neural EV energy estimation when direct power labels are unavailable, particularly when training data are limited.

What they do **not** support: PINN beats every model; instantaneous battery power is recovered; the method is a new EVRP algorithm.

## 14. Figures

All final figures are PNG, 300 dpi, under [`results/figures/`](results/figures/).

| File | Role |
|---|---|
| `fig01_method.png` | Discrete-time PINN: physics + neural correction + conservation + multi-window SoC energy supervision. No instantaneous power labels. |
| `fig02_observed_vs_predicted.png` | Observed vs predicted trip energy for Physics, ElasticNet, WeakMLP, PINN. |
| `fig03_trip_errors.png` | Absolute held-out trip-energy errors (17 trips visible). |
| `fig04_data_scarcity.png` | Held-out MAE vs number of training trips. |
| `fig05_feasibility.png` | Routing-oriented false-safe and overly conservative rates. |
| `fig06_soc_reconstruction.png` | Optional: observed quantized SoC vs SoC reconstructed by integrating PINN power on three predetermined trips (one per trajectory). |

## 15. Repository Structure

```
PINN_ev_energy/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── paper.yaml
│   └── vehicle_twizy.yaml
├── data/
│   └── README.md
├── src/
│   ├── data/
│   ├── physics/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── experiment/
├── scripts/
│   ├── audit_data.py
│   ├── run_loto.py
│   ├── run_data_scarcity.py
│   ├── run_feasibility_sensitivity.py
│   ├── make_final_figures.py
│   ├── make_final_tables.py
│   └── run_final_pipeline.py
└── results/
    ├── README.md
    ├── experiment_summary.md
    ├── figures/
    └── tables/
```

Place HELECAR-D locally; it is gitignored. `outputs/` is gitignored and used only as a temporary runtime workspace.

## 16. How to Reproduce

```bash
python -m pip install -e .
python scripts/run_final_pipeline.py --config configs/paper.yaml
```

The wrapper performs only the conference-paper pipeline:

1. data audit
2. main LOTO if results are missing
3. data-scarcity experiment if results are missing
4. feasibility experiment if results are missing
5. final tables
6. final PNG figures

Existing paper LOTO, scarcity, and feasibility caches are reused. The wrapper does **not** run cross-trajectory studies, ablation suites, or parameter-sensitivity sweeps, and it does not retrain merely to change PINN’s rank.

## 17. Limitations

Seventeen trips, one Renault Twizy, three Moroccan trajectories. No instantaneous battery-power ground truth, so latent \(\hat P(t)\) cannot be validated directly. Usable capacity and vehicle parameters are modelling assumptions. SoC is quantized/noisy; one analysed trip (`T3_05_25_2021_04`) contains a large SoC jump. The data-scarcity table uses seed 0; a three-seed neural scarcity study was not completed. ElasticNet can be unstable with very few trips. The routing section is only a battery-feasibility sensitivity. An internal full-trip-only ablation had a lower LOTO trip MAE than the multi-window PINN; windows were not retuned from that result.

## 18. Dataset Citation

NaitMalek, Y., Najib, M., Bakhouya, M., Gaber, J., 2023. HELECAR-D: A dataset for urban electro mobility in Moroccan context. *Data in Brief* 48, 109080. https://doi.org/10.1016/j.dib.2023.109080

Data: https://doi.org/10.5281/zenodo.7217707. License: CC-BY-4.0. See the official HELECAR-D release for the authoritative citation.

## 19. License

Research code: MIT (`LICENSE`). HELECAR-D: CC-BY-4.0, not redistributed here.
