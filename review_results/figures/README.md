# Review-experiment figures (INTIS 2026)

**Location:** `review_results/figures/` only.  
**Does not modify** `results/figures/`, `results/` tables, or `configs/paper.yaml`.

Regenerate:

```bash
python scripts/review/make_review_figures.py
```

Implementation: `src/plotting/review_figures.py` (style/colors from `src/plotting/style.py` and `src/evaluation/names.py`).  
Verification dump: `figure_verification.json` (values recomputed before save).

Figure titles avoid internal review jargon (`corrected`, `superseded`, `leakage-fixed`). Methodological distinctions remain documented here and in `review_notes/`.

---

## Files

| Stem | Formats | Role |
|---|---|---|
| `figA_ablation_paired_diff` | PNG / PDF / SVG | HELECAR-D paired MAE differences |
| `figB_tesla_meas_mae` | PNG / PDF / SVG | Tesla Model Y MAE vs V×I |
| `figC_sensitivity_L` | PNG / PDF / SVG | Twizy residual-bound sensitivity |
| `figAB_ablation_tesla_two_panel` | PNG / PDF / SVG | Compact manuscript candidate (A+B) |

PNG export: **400 dpi**. PDF uses Type 42 fonts.

---

## Figure A — HELECAR-D ablation

**Sources**
- `review_results/ablations/table_ablation_paired_complete.csv`
- (context) `review_results/ablations/table_ablation_complete.csv`

**Quantity plotted**  
Paired difference \(\Delta_i = \mathrm{AE}^{\mathrm{Full\,PINN}}_i - \mathrm{AE}^{\mathrm{comparator}}_i\) with mean and **95% paired bootstrap CI** over 17 trips. Negative values favor the full PINN.

**Caption**  
> Paired differences in trip-energy absolute error between the full PINN and comparison methods on HELECAR-D (17 trips). Markers indicate mean differences and whiskers show 95% paired bootstrap confidence intervals. Negative values favor the full PINN.

---

## Figure B — Tesla Model Y validation

**Sources (authoritative only)**
- `review_results/tesla/loto/tesla_bev1_loto_corrected/`
- `review_results/tesla/loto/tesla_bev2_loto_corrected/`
- MAE matched to `summary_vs_meas.csv` after recomputation from `trip_energy_with_meas_ref.csv` (`abs_err_vs_meas_kwh`).

**Aggregation**  
Seed-mean absolute error per trip vs measured V×I, then mean over 17 trips. No uncertainty bars (not archived for this summary).

**Caption**  
> Trip-energy MAE for the Physics Model, Data-Driven MLP, and PINN on two Tesla Model Y configurations, evaluated against independently measured battery energy (V×I). Results cover 17 evaluation trips per configuration and three neural initialization seeds. The method uses a documented four-feature, flat-road adaptation. Differences between PINN and MLP against measured energy are not statistically conclusive.

---

## Figure C — Twizy residual-bound sensitivity

**Sources**
- `review_results/sensitivity/all_sensitivity_mae.csv` (`L_2`…`L_8`)
- Cross-checked against `summary_L_{2,4,6,8}.csv`

**Quantity**  
Point trip-energy MAE for \(L\in\{2,4,6,8\}\) kW. The blue marker identifies the accepted reference \(L=4\) kW. No CI in the sensitivity archive.

**Caption**  
> Trip-energy MAE of the Twizy PINN for predetermined residual bounds \(L\in\{2,4,6,8\}\) kW. The highlighted marker is the accepted reference \(L=4\) kW; the lower point estimate at \(L=6\) kW was not adopted from outer-test ranks.

---

## Combined Figure AB

**Caption**  
> (a) Paired differences in trip-energy absolute error between the full PINN and comparison methods on HELECAR-D (17 trips; markers: means; whiskers: 95% paired bootstrap CIs; negative values favor the full PINN). (b) Trip-energy MAE on two Tesla Model Y configurations against independently measured battery energy (V×I; 17 trips per configuration; three neural seeds; four-feature flat-road adaptation). Differences between PINN and MLP against measured energy are not statistically conclusive.

---

## Manuscript inclusion

See `review_notes/08_manuscript_figure_plan.md`.
