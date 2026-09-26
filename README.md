# Physics-Informed Neural Network for Electric Vehicle Energy Estimation under Data Scarcity

Research code accompanying a study of physics-informed electric-vehicle energy estimation under data scarcity. The scientific scope is locked to the paper abstract. This repository does not add extra research questions, architectures, or a machine-learning benchmark.

A discrete-time physics-informed neural network (PINN) estimates EV energy when only limited real-world driving data are available, instantaneous battery-power measurements are unavailable, and battery information arrives through quantized/noisy State-of-Charge (SoC) observations. The model combines longitudinal vehicle dynamics with neural learning, estimates latent battery power from driving and environmental variables, enforces battery-energy conservation, and uses multi-window energy constraints derived from SoC depletion instead of noisy instantaneous SoC differences.

Paper-ready numbers live in [`results/`](results/). Runtime dumps under `outputs/` are a cache, not the archival record.

## 1. Project Overview

The practical task is trip energy: how much battery energy a Renault Twizy uses on a held-out real drive. The analysed HELECAR-D files do not provide a reliable instantaneous battery-power target. The estimator therefore predicts a latent battery-power trajectory \(\hat P(t)\), integrates it, and matches SoC-derived window energy.

**Research question.** Can a physics-informed neural network provide useful EV energy estimates under severe data scarcity when instantaneous battery-power measurements are unavailable and supervision is available only through quantized/noisy SoC observations?

**Secondary question.** Does the resulting energy estimation affect battery-feasibility decisions relevant to EV routing?

The paper comparison contains four methods: Physics Model, Ridge Regression, Data-Driven MLP, and the proposed PINN. Evaluation is leave-one-trip-out on 17 analysed Moroccan HELECAR-D trips, plus a data-scarcity study and one small routing-oriented battery-feasibility assessment.

## 2. Scientific Motivation

Sample-to-sample SoC differences are too quantized and noisy to treat as instantaneous power. Differentiating SoC at 1 Hz would treat a ~0.02 percentage-point step as an energy label of about 1.2 Wh and collapse supervision onto noise.

There are **no direct instantaneous battery-power labels**. The model predicts latent battery power and is supervised indirectly through integrated energy. SoC is label information, not a neural-network input. The physics-informed structure is a candidate inductive bias for that weakly supervised neural estimator. On the 17-trip leave-one-trip-out comparison, PINN has the lowest point-estimate MAE. The paired PINN-versus-Ridge absolute-error difference has a 95% interval that includes zero.

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

Paper display names, in logical order:

| Method | Role in the paper |
|---|---|
| Physics Model | Analytical non-learning longitudinal-vehicle reference. No fitting. |
| Ridge Regression | Regularized linear baseline required by the experimental design. It is trained only on SoC-free trip-level features. A predetermined alpha grid is selected by leave-one-trip-out on the outer-training trips only. |
| Data-Driven MLP | A purely data-driven multilayer perceptron trained using the same SoC-derived multi-window energy supervision as the PINN, but without the analytical physics branch and physics-informed battery-dynamics constraint. |
| PINN | Proposed discrete-time physics-informed model. |

Internal cache identifiers remain `physics`, `ridge`, and `weak_mlp`. Those strings are not the paper names. `weak_mlp` is the Data-Driven MLP. Cached extra ablations are not paper methods and are not trained by `configs/paper.yaml`.

## 10. Trip-Level Cross-Validation

Primary evaluation is **leave-one-trip-out** (LOTO). No random row splitting.

For every fold: test = one full trip; training = all remaining trips. All of the following are training-only: normalization, SoC quantization estimate \(q\), model fitting, validation, early stopping, and Ridge scaling and alpha selection. Grouped inner validation prefers one trip from each of T1/T2/T3 when possible. The selected epoch is retrained on all outer-training trips. The statistical unit is the trip.

## 11. Data-Scarcity Experiment

Required by the abstract. Training-set sizes: 3, 5, 8, 12, 16 trips. Compared methods: Ridge Regression, Data-Driven MLP, and PINN. The Physics Model is a fixed reference line because it does not depend on training size. Models are **not** retuned per training-set size. Subsets are the same predetermined trip subsets for every method. Neural seeds are 0, 1, and 2. For \(n=3,5,8,12\) each seed uses three predefined subset repeats. For \(n=16\) the outer-training set is the full leave-one-trip-out complement, and the neural predictions are the existing LOTO predictions. Ridge is deterministic, so identical full-set fits that differ only by a nominal seed are counted once. MAE is the mean of replicate-level held-out-trip MAEs. The reported uncertainty is the standard deviation of those replicate means. Seeds are training replicates, not extra trips. This experiment answers only: how does each approach behave as fewer training trips are available?

## 12. Routing-Oriented Battery-Feasibility Experiment

This is a **small additional experiment**, not a new EVRP solver and not a routing optimization algorithm. Energy predictions are used only to evaluate battery-feasibility decisions at reserve levels 5%, 10%, 15%, and 20% SoC. Compared methods: Physics Model and PINN. Reported quantities: false-safe rate and overly conservative rate.

Call it a routing-oriented battery-feasibility assessment / sensitivity.

## 13. Main Results

Numbers below match [`results/tables/`](results/tables/). Do not rank methods with subjective labels.

**LOTO trip energy** ([`table02_main_results.csv`](results/tables/table02_main_results.csv)):

| Method | MAE (kWh) | 95% CI | RMSE | MAPE (%) | WAPE (%) | Bias | \(R^2\) |
|---|---:|---|---:|---:|---:|---:|---:|
| Physics Model | 0.780 | [0.614, 0.948] | 0.854 | 30.06 | 29.42 | −0.780 | 0.508 |
| Ridge Regression | 0.368 | [0.229, 0.537] | 0.491 | 16.45 | 13.87 | −0.014 | 0.837 |
| Data-Driven MLP | 0.475 | [0.312, 0.655] | 0.629 | 16.34 | 17.93 | +0.191 | 0.734 |
| PINN | 0.295 | [0.209, 0.392] | 0.373 | 11.51 | 11.13 | −0.052 | 0.906 |

PINN has the lowest point-estimate MAE. The paired difference versus the Data-Driven MLP is −0.180 kWh (95% paired bootstrap [−0.316, −0.056], Wilcoxon p = 0.020). The paired difference versus Ridge Regression is −0.073 kWh (95% paired bootstrap [−0.264, 0.110], Wilcoxon p = 0.579). Ridge is close to unbiased; the Physics Model under-predicts every trip in this split (bias equals −MAE).

**Data scarcity** ([`table03_data_scarcity.csv`](results/tables/table03_data_scarcity.csv)), held-out MAE in kWh. Uncertainty in parentheses is the standard deviation of replicate-level means. Neural replicates are seed × subset repeat. Ridge counts each distinct training subset once, so \(n=16\) has one Ridge run:

| \(n_{\mathrm{train}}\) | Ridge Regression | Data-Driven MLP | PINN |
|---:|---:|---:|---:|
| 3 | 2.247 (1.986) | 0.710 (0.150) | 0.439 (0.115) |
| 5 | 5.520 (3.031) | 0.697 (0.174) | 0.344 (0.048) |
| 8 | 2.031 (2.521) | 0.434 (0.080) | 0.297 (0.065) |
| 12 | 0.400 (0.061) | 0.459 (0.122) | 0.305 (0.055) |
| 16 | 0.368 (0.000) | 0.475 (0.026) | 0.295 (0.048) |

At 3, 5, and 8 training trips, Ridge MAE is higher and much more variable than either neural method. PINN MAE is lower than Data-Driven MLP at every reported training size. Scarcity \(n=16\) reuses the three-seed LOTO predictions, so those PINN and Data-Driven MLP entries match the main-table MAEs. The Ridge \(n=16\) entry matches the Ridge main-table MAE.

**Feasibility** ([`table04_feasibility.csv`](results/tables/table04_feasibility.csv)): at a 20% reserve, the Physics Model false-safe rate is 17.6% and the PINN false-safe rate is 5.9%. The Physics Model produces more false-safe decisions because it systematically under-predicts trip energy.

What the results support: the physics-informed structure lowers neural trip-energy error relative to the comparable Data-Driven MLP under the same weak SoC supervision, including when few training trips are available. PINN also has a lower full-data point-estimate MAE than Ridge Regression; the paired interval for that difference includes zero. Ridge is unstable when only a few training trips are available.

What they do not support: recovery of instantaneous battery power; a new EVRP algorithm; treating the paired PINN–Ridge difference as resolved when its interval includes zero.

## 14. Figures

All final figures are PNG, 400 dpi, under [`results/figures/`](results/figures/). The recommended subset is listed in [`results/README.md`](results/README.md).

| File | Role |
|---|---|
| `fig01_method.png` | Discrete-time PINN: physics + neural correction + conservation + multi-window SoC energy supervision. No instantaneous power labels. |
| `fig02_observed_vs_predicted.png` | Observed vs predicted trip energy for Physics Model, Ridge Regression, Data-Driven MLP, and PINN. |
| `fig03_trip_errors.png` | Absolute held-out trip-energy errors (17 trips visible). |
| `fig04_mae_ci.png` | Trip-energy MAE with 95% trip-level bootstrap intervals. |
| `fig05_residuals.png` | Trip-energy residual versus observed energy. |
| `fig06_error_ecdf.png` | Empirical distribution of absolute trip-energy error. |
| `fig07_trip_error_heatmap.png` | Per-trip absolute error, ordered by trajectory. |
| `fig08_data_scarcity_learning_curve.png` | Held-out MAE versus number of training trips, with a logarithmic panel and a neural zoom. |
| `fig09_pinn_vs_mlp.png` | Paired absolute error, PINN versus Data-Driven MLP. |
| `fig10_soc_reconstruction.png` | Observed quantized SoC vs SoC reconstructed by integrating predicted power on three predetermined trips. |
| `fig11_window_consistency.png` | Held-out window-energy MAE by time scale for Data-Driven MLP and PINN. |
| `fig12_feasibility.png` | Routing-oriented false-safe and overly conservative rates. |
| `fig13_pinn_training_convergence.png` | Inner-validation multi-window MAE and weighted training contributions for one predetermined PINN fold and seed. |
| `fig14_validation_window_scales.png` | Inner-validation window-energy MAE by time scale for that same run. |
| `fig15_trip_energy_comparison.png` | Observed and predicted trip energy on every held-out trip, by trajectory. |
| `fig16_trajectory_mae.png` | Trip absolute errors summarized by trajectory, with the individual trips shown. |

Recommended main-paper figures are listed in [`results/README.md`](results/README.md).

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
├── docs/
│   ├── dataset_analysis.md
│   └── dataset_figures/
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
│   ├── run_ridge_baseline.py
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
2. main LOTO if neural and physics results are missing; otherwise it refits only Ridge Regression on the existing folds
3. data-scarcity experiment, reusing stored neural rows and fitting any missing Ridge rows
4. feasibility experiment if results are missing
5. final tables
6. final PNG figures

Existing neural LOTO, neural scarcity, and feasibility caches are reused. The wrapper does **not** retrain the PINN or the Data-Driven MLP merely because the regression baseline changed, and it does not run ablation suites or parameter-sensitivity sweeps.

## 17. Limitations

Seventeen trips, one Renault Twizy, three Moroccan trajectories. No instantaneous battery-power ground truth, so latent \(\hat P(t)\) cannot be validated directly. Usable capacity and vehicle parameters are modelling assumptions. SoC is quantized/noisy; one analysed trip (`T3_05_25_2021_04`) contains a large SoC jump. Ridge Regression is unstable at the smallest training sizes because the trip-level feature set is wide relative to three to eight trips. The routing section is only a battery-feasibility sensitivity. Multi-window supervision was not retuned from any internal full-trip-only comparison.

## 18. Dataset Citation

NaitMalek, Y., Najib, M., Bakhouya, M., Gaber, J., 2023. HELECAR-D: A dataset for urban electro mobility in Moroccan context. *Data in Brief* 48, 109080. https://doi.org/10.1016/j.dib.2023.109080

Data: https://doi.org/10.5281/zenodo.7217707. License: CC-BY-4.0. See the official HELECAR-D release for the authoritative citation.

## 19. License

Research code: MIT (`LICENSE`). HELECAR-D: CC-BY-4.0, not redistributed here.
