# Paper artifacts

This folder is the only archival location for conference-paper outputs.

- `figures/` — PNG only, 400 dpi
- `tables/` — CSV tables written from cached predictions
- `paper_tables.md` — rounded tables for the manuscript
- `paper_figure_captions.md` — captions for the manuscript
- `paper_results.md` — results-section notes generated from the tables
- `experiment_summary.md` — research record copied from those CSVs

Runtime dumps under `outputs/` and the local HELECAR-D copy are not paper artifacts.

## Recommended main manuscript figures

The 10-page paper uses these four figures only:

1. `fig01_method.png` — method overview, including the depletion head $\hat{D}(t)$
2. `fig02_observed_vs_predicted.png` — observed versus predicted trip energy
3. `fig04_mae_ci.png` — trip-energy MAE and 95% bootstrap intervals
4. `fig08_data_scarcity_learning_curve.png` — learning curve under fewer training trips

## Secondary / optional figures

Keep these for slides, reviewer discussion, or a supplement. They are not part of the 10-page figure set.

- `fig03_trip_errors.png`
- `fig05_residuals.png`
- `fig06_error_ecdf.png`
- `fig07_trip_error_heatmap.png`
- `fig09_pinn_vs_mlp.png`
- `fig10_soc_reconstruction.png`
- `fig11_window_consistency.png` — Data-Driven MLP and PINN only
- `fig12_feasibility.png`
- `fig13_pinn_training_convergence.png`
- `fig14_validation_window_scales.png`
- `fig15_trip_energy_comparison.png`
- `fig16_trajectory_mae.png`

Regenerate from cached experiment outputs:

```bash
python scripts/make_final_tables.py --config configs/paper.yaml --out results
python scripts/make_final_figures.py --config configs/paper.yaml --out results
```

Or run the conference pipeline, which reuses existing neural results and refits Ridge only when needed:

```bash
python scripts/run_final_pipeline.py --config configs/paper.yaml
```
