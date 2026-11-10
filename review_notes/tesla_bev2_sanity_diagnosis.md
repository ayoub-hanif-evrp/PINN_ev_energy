# BEV2 sanity diagnosis (preliminary LOTO; do not drop trips)

Source: `review_results/tesla/loto/tesla_bev2_loto/sanity_report.json` (pre-correction run).

## Flags

1. **Negative predicted trip energy**
   - `weak_mlp` on `BEV2_2024-07-29_drive2`: −0.827 kWh (observed SoC energy ≈ 1.89 kWh).
   - `ridge` on `BEV2_2024-11-13_drive2`: −0.536 kWh (observed SoC energy ≈ 2.21 kWh).
2. **Peak predicted power > 200 kW** (Twizy-oriented heuristic) on two PINN folds.

## Trip profiles (kept in evaluation pool)

| Trip | Duration | Distance | ΔSoC | e_meas | Mean speed | Moving frac |
|---|---:|---:|---:|---:|---:|---:|
| BEV2_2024-07-29_drive2 | 36.7 min | 8.7 km | 2.4 pp | 1.77 kWh | 14.2 km/h | 0.50 |
| BEV2_2024-11-13_drive2 | 20.7 min | 8.9 km | 2.8 pp | 2.13 kWh | 25.8 km/h | 0.80 |

Both are **low-energy** sessions near the QC floor (ΔSoC ≈ 2–3 pp). MLP went negative on the slower urban-like trip; Ridge went negative on the other. Physics and PINN stayed positive on both in the preliminary run.

PINN peaks of 223–243 kW on other trips are high but not physically impossible for Model Y pack power under aggressive acceleration; they remain diagnostic, not automatic discard criteria.

## Interpretation

- Negative trip-energy predictions are **retained**. They indicate model instability / extrapolation on low-energy held-out sessions when trained on a mixed-energy pool, not a reason to delete trips.
- The 200 kW threshold is inappropriate for Model Y pack power. Corrected runs use a Tesla-aware diagnostic threshold of **400 kW** (`tesla.absurd_power_kw`), while still logging high peaks.
- These diagnostics do not justify changing the PINN architecture or discarding unfavorable sessions.

## Corrective follow-up — done

Re-ran under `tesla_bev2_loto_corrected` after forward `dt` rebuild, physics-only disjoint L=20 kW, seeds `[0,1,2]`.

**Corrected sanity:** only Ridge negatives on `BEV2_2024-11-13_drive2` (all seeds); **0** peak-power flags at 400 kW. All 17 eval-pool trips retained. See `05_camera_ready_final_audit.md`.
