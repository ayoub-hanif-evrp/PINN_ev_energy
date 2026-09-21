# Paper review artifacts

Lightweight copies of experimental summaries. Full run artifacts live under `outputs/`; the dataset copy lives under `HELECAR-D/`.

Profile: `paper` (frozen protocol). Last committed SHA at protocol freeze: `9bae16f`. The pipeline that produced these numbers is in the working tree and has not been committed yet.

Primary LOTO energy MAE (trip-level, neural errors averaged across seeds 0/1/2 first):

- ElasticNet 0.167 kWh
- PINN_FULLTRIP 0.243 kWh
- Constant 0.249 kWh
- PINN_NO_DYNAMICS 0.278 kWh
- PINN 0.295 kWh
- MLP_STATE 0.372 kWh
- WeakMLP 0.475 kWh
- Physics 0.780 kWh

These are computed results, not predetermined. ElasticNet wins the main LOTO table.

Regenerate with `make paper` after committing the frozen protocol.
