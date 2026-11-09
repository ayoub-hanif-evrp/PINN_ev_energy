# Phase C — GO / NO-GO (Tesla Model Y, KU Leuven)

**Decision: CONDITIONAL GO** (recorded before neural LOTO training).

| Question | Answer |
|---|---|
| Compatible with essential PINN physics formulation? | **Yes, with adaptations** — longitudinal dynamics + residual + SoC windows + conservation + state head remain. Grade forced to θ=0 (no elevation in drive CSVs). |
| Train with SoC-derived energy without direct power labels? | **Yes** — `SOCave292` available; V×I excluded from training targets/inputs. |
| Independent battery-energy reference available? | **Yes** — `BattVoltage132 × RawBattCurrent132` (positive current = discharge). Correlates with, but is not identical to, SoC-derived energy. |
| Vehicle parameters justifiable? | **Partially** — nominal capacities from dataset paper; mass/CdA/η/Crr/P_aux are labelled assumptions. |
| Independent held-out trips? | **Yes** — predetermined stratified pool of 17 sessions/vehicle (seed 2026), LOTO within pool. |
| Without changing main scientific contribution? | **Yes** — separate review configs/scripts; Twizy paper configs untouched; methodology transfer, not zero-shot weight reuse. |

## Blockers accepted as limitations (not NO-GO)

1. Flat-road grade assumption.
2. Reduced 4-feature input set (no humidity/wind/traffic/speed limit).
3. Residual bound recalibrated to 40 kW via train-pool P95 protocol (not Twizy 4 kW).
4. Evaluation pool is a predetermined subsample of QC-passing sessions (computational / paper-length constraint), not the full 214/321 QC set.

## Reproduction anchors

- Branch: `review/intis2026-validation`
- Baseline commit before Tesla work: `0a85b57c3cad6d81a9dfb225f69445546860cc50`
- Residual calibration: `review_results/tesla/residual_calibration_bev{1,2}.json` → L=40 kW
