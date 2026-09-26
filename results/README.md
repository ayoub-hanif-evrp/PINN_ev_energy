# Paper artifacts

This folder is the only archival location for conference-paper outputs.

- `figures/` — PNG only, 400 dpi
- `tables/` — CSV tables written from cached predictions
- `paper_tables.md` — rounded tables for the manuscript
- `paper_figure_captions.md` — captions for the manuscript
- `paper_results.md` — results-section notes generated from the tables
- `experiment_summary.md` — research record copied from those CSVs

Runtime dumps under `outputs/` and the local HELECAR-D copy are not paper artifacts.

## Recommended main figures

1. `fig01_method.png`
2. `fig02_observed_vs_predicted.png`
3. `fig04_mae_ci.png`
4. `fig08_data_scarcity_learning_curve.png`
5. `fig09_pinn_vs_mlp.png`
6. `fig10_soc_reconstruction.png`
7. `fig13_pinn_training_convergence.png`

If page space permits:

8. `fig12_feasibility.png`
9. `fig15_trip_energy_comparison.png`

## Secondary / repository figures

- `fig03_trip_errors.png`
- `fig05_residuals.png`
- `fig06_error_ecdf.png`
- `fig07_trip_error_heatmap.png`
- `fig11_window_consistency.png` — Data-Driven MLP and PINN only
- `fig14_validation_window_scales.png`
- `fig16_trajectory_mae.png`

Do not force every figure into the conference manuscript.

Regenerate from cached experiment outputs:

```bash
python scripts/make_final_tables.py --config configs/paper.yaml --out results
python scripts/make_final_figures.py --config configs/paper.yaml --out results
```

Or run the conference pipeline, which reuses existing neural results and refits Ridge only when needed:

```bash
python scripts/run_final_pipeline.py --config configs/paper.yaml
```
