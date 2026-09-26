# Paper artifacts

This folder is the only archival location for conference-paper outputs.

- `figures/` — PNG only, 300 dpi
- `tables/` — CSV tables written from cached predictions
- `experiment_summary.md` — paper-writing reference copied from those CSVs

Runtime dumps under `outputs/` and the local HELECAR-D copy are not paper artifacts.

## Recommended main figures

Use these first if page space is limited:

- `fig01_method.png` — method overview
- `fig02_observed_vs_predicted.png` — observed versus predicted trip energy
- `fig04_mae_ci.png` — MAE with 95% trip-level bootstrap intervals
- `fig08_data_scarcity.png` — learning curve
- `fig09_pinn_vs_mlp.png` — paired PINN versus Data-Driven MLP
- `fig10_soc_reconstruction.png` — SoC rollout on three predetermined trips
- `fig12_feasibility.png` — battery-feasibility sensitivity

## Optional / secondary figures

- `fig03_trip_errors.png`
- `fig05_residuals.png`
- `fig06_error_ecdf.png`
- `fig07_trip_error_heatmap.png`
- `fig11_window_consistency.png` — Data-Driven MLP and PINN only; the stored window file has no Physics Model rows
- `fig13_training_loss.png` — predetermined fold `T1_03_29_2021`, PINN, seed 0, selection history. The file stores training loss terms, not a separate validation-loss series.

Panel annotations in `fig02` use the same MAE and \(R^2\) as `tables/table02_main_results.csv`. Scatter points for the neural methods are seed-averaged predictions.

Dataset exploration plots live under `docs/dataset_figures/` and are not paper result figures.

Regenerate from cached experiment outputs:

```bash
python scripts/make_final_tables.py --config configs/paper.yaml --out results
python scripts/make_final_figures.py --config configs/paper.yaml --out results
```

Or run the full conference pipeline, which reuses existing neural LOTO / scarcity / feasibility results and refits Ridge when needed:

```bash
python scripts/run_final_pipeline.py --config configs/paper.yaml
```
