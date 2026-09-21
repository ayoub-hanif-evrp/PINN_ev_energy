# Paper artifacts

This folder is the only archival location for conference-paper outputs.

- `figures/` — PNG only, 300 dpi
- `tables/` — four CSV tables required by the abstract
- `experiment_summary.md` — paper-writing reference copied from those CSVs

Runtime dumps under `outputs/` and the local HELECAR-D copy are not paper artifacts.

Regenerate from cached experiment outputs:

```bash
python scripts/make_final_tables.py --config configs/paper.yaml --out results
python scripts/make_final_figures.py --config configs/paper.yaml --out results
```

Or run the full conference pipeline, which reuses existing LOTO / scarcity / feasibility results when present:

```bash
python scripts/run_final_pipeline.py --config configs/paper.yaml
```
