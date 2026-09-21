# Paper artifacts

This folder is the only archival location for paper-ready outputs.

- `figures/` — PNG only, 300 dpi
- `tables/` — concise CSV summaries
- `protocol.md` — actual frozen method
- `experiment_summary.md` — claims the paper can and cannot make

Runtime dumps (`outputs/`), the local HELECAR-D copy, and engineering caches are not paper artifacts.

Regenerate:

```bash
python scripts/make_final_tables.py --config configs/paper.yaml --out results
python scripts/make_final_figures.py --config configs/paper.yaml --out results
```
