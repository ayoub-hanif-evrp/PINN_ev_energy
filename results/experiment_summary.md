# Experiment summary (paper-writing reference)

Numbers below are copied from `results/tables/` after the final export. If a CSV changes, this file must be regenerated from that CSV.

## 1. Dataset

17 analysed HELECAR-D trips (T1=4, T2=3, T3=10). Duration 952–8577 s. CAN distance 13.83–59.57 km, mean 29.94 km. Mean SoC 81.23% → 37.07%. Empirical \(q=0.02\) pp. Overlapping training-style windows: 57671 total (54522 SoC-event, 3132 fixed-time, 17 full-trip).

## 2–4. Main LOTO, bootstrap, paired tests

Trip-level MAE with 95% bootstrap CI (n=17):

- Constant 0.249 [0.168, 0.337]
- Physics 0.780 [0.614, 0.948]
- ElasticNet 0.167 [0.113, 0.232]
- WeakMLP 0.475 [0.312, 0.655]
- MLP_STATE 0.372 [0.265, 0.486]
- PINN_NO_DYNAMICS 0.278 [0.215, 0.342]
- PINN_FULLTRIP 0.243 [0.182, 0.306]
- PINN 0.295 [0.209, 0.392]

Paired AE difference (first minus second):

- PINN vs WeakMLP: −0.180 [−0.316, −0.056]
- PINN vs ElasticNet: +0.128 [0.014, 0.242]
- PINN vs Physics: −0.485 [−0.604, −0.375]

Wilcoxon two-sided p-values are secondary (0.020, 0.027, 1.5e-5). Do not call a bootstrap interval “statistically significant”.

## 5. Data scarcity

Complete aggregation currently uses seed 0 and 3 subset repeats (seeds 1–2 were started; do not mix incomplete seeds into the paper table). Mean MAE (kWh):

| n_train | ElasticNet | WeakMLP | PINN |
|---:|---:|---:|---:|
| 3 | 0.374 | 0.575 | 0.317 |
| 5 | 0.314 | 0.756 | 0.340 |
| 8 | 1.006 | 0.431 | 0.333 |
| 12 | 0.225 | 0.348 | 0.266 |
| 16 | 0.167 | 0.445 | 0.350 |

Physics reference 0.780 kWh. PINN is better at n=3. ElasticNet is unstable at n=8 and wins n=12 and n=16. n=16 PINN 0.350 is the seed-0 LOTO value, not the 3-seed mean 0.295.

## 6. Cross-trajectory

Hold-out MAE (kWh): T1 ElasticNet 0.263 / PINN 0.145; T2 ElasticNet 0.050 / PINN 0.216; T3 ElasticNet 0.761 / PINN 0.266. PINN is more stable when a whole trajectory is unseen, especially T3, but T2 has only 3 trips. Do not overclaim.

## 7. Ablation

PINN_FULLTRIP 0.243 < PINN_NO_DYNAMICS 0.278 < PINN 0.295 < MLP_STATE 0.372 < WeakMLP 0.475 << Physics 0.780. Multi-window PINN does **not** beat full-trip-only PINN on trip-energy MAE. Non-overlapping short-window test MAE is essentially the same for PINN and PINN_FULLTRIP (~0.017 kWh at SoC-event 0.1). Window weights were not retuned.

## 8. Physical plausibility

Physics peak battery power 34.2 kW; PINN 35.0 kW. Mean |δP| 0.96 kW, max |δP| 2.03 kW, residual saturation 0. About 0.43% of physics samples exceed the 15.5 kW battery equivalent of 13 kW wheel traction; 99th percentile |P| is 11.6 kW. Spikes exist and are shown in Figure 8. Clamping wheel power leaves trip-energy MAE unchanged at two decimal places (0.780 vs 0.783). The frozen protocol stays unbounded. This is a plausibility caveat, not a reason to rewrite the energy ranking.

## 9. Feasibility (routing-oriented sensitivity only)

At 20% reserve SoC, false-safe fraction: Constant 0, Physics 0.176, PINN 0.059. Physics is more false-safe because it systematically under-predicts energy. This is not an EVRP algorithm.

## 10. Limitations / caveats

17 trips; one Twizy; Morocco T1/T2/T3 only; no P_batt labels; assumed 6 kWh and vehicle parameters; quantized SoC; one large SoC jump on T3_05_25_2021_04; latent power unvalidated; 3-seed scarcity not fully finished at export time.

## 11. Claims the paper CAN make

- Physics-informed learning substantially improves upon a comparably weakly supervised data-only neural estimator (PINN vs WeakMLP).
- The physical prior improves neural learning when power labels are unavailable (PINN vs WeakMLP and vs Physics).
- PINN remains competitive under severe trip-level data scarcity (n=3).
- Cross-trajectory experiments indicate useful robustness under some trajectory shifts (T1, T3), with a small number of trajectories.

## 12. Claims the paper SHOULD NOT make

- PINN is the best model overall (ElasticNet wins ordinary LOTO).
- Multi-window supervision always improves trip-energy accuracy (PINN_FULLTRIP is better on that metric).
- Instantaneous battery power is accurately recovered.
- The method guarantees physically correct power (peaks exceed 13 kW wheel / 15.5 kW battery).
- The proposed method solves EVRP better.
- This is the first PINN ever used for EV energy estimation or routing.
