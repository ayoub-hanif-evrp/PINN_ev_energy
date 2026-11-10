# Review-experiment figures (INTIS 2026)

**Location:** `review_results/figures/` only.  
**Does not modify** `results/figures/`, `results/` tables, or `configs/paper.yaml`.

Regenerate:

```bash
python scripts/review/make_review_figures.py
```

Implementation: `src/plotting/review_figures.py` (style/colors from `src/plotting/style.py` and `src/evaluation/names.py`).

Verification dump: `figure_verification.json` (values recomputed before save).

---

## Files

| Stem | Formats | Role |
|---|---|---|
| `figA_ablation_paired_diff` | PNG / PDF / SVG | HELECAR-D paired AE differences |
| `figB_tesla_meas_mae` | PNG / PDF / SVG | Corrected Tesla MAE vs V×I |
| `figC_sensitivity_L` | PNG / PDF / SVG | Twizy residual-bound sensitivity |
| `figAB_ablation_tesla_two_panel` | PNG / PDF / SVG | Compact manuscript candidate (A+B) |

PNG export: **400 dpi**. PDF uses Type 42 fonts.

---

## Figure A — HELECAR-D ablation

**Sources**
- `review_results/ablations/table_ablation_paired_complete.csv`
- (context) `review_results/ablations/table_ablation_complete.csv`

**Quantity plotted**  
For each comparator, trip-level absolute-energy errors are aggregated as in the paper (seed-mean AE per trip for neural methods).  
Paired difference:

\[
\Delta_i = \mathrm{AE}^{\mathrm{Full\,PINN}}_i - \mathrm{AE}^{\mathrm{comparator}}_i
\]

Mean \(\bar\Delta\) with **95% paired bootstrap CI** over the 17 trips (`CI_low`, `CI_high` in the CSV).  
**Negative \(\bar\Delta\)** ⇒ Full PINN has lower trip AE.

**Scientific reading (required)**  
Full PINN vs Physics / MLP: CIs exclude 0.  
Within-PINN ablations: CIs **include** 0 — do **not** claim every constraint improves trip MAE.

**Proposed caption**  
> Paired trip-energy absolute-error differences (Full PINN − comparator) on HELECAR-D leave-one-trip-out evaluation (17 trips). Whiskers are 95% paired bootstrap confidence intervals. Negative values indicate lower Full-PINN error. Differences versus Physics and the data-driven MLP exclude zero; differences among PINN variants include zero.

---

## Figure B — Corrected Tesla external validation

**Sources (authoritative only)**
- `review_results/tesla/loto/tesla_bev1_loto_corrected/`
- `review_results/tesla/loto/tesla_bev2_loto_corrected/`
- Plotted MAE matched to `summary_vs_meas.csv` after recomputation from `trip_energy_with_meas_ref.csv` (`abs_err_vs_meas_kwh`).

**Aggregation**  
1. Per trip, mean absolute error vs measured V×I over seeds `{0,1,2}` (physics: single run).  
2. MAE = mean of those 17 trip values.

**Not plotted**  
Superseded `tesla_bev*_loto` profiles; SoC-reference MAE; Ridge; uncertainty bars (no MAE CI archived for this summary); SoC-state diagnostics as “accurate SoC tracking.”

**Proposed caption**  
> Feature-adapted Tesla Model Y external check (BEV1/BEV2): trip-energy MAE against independent battery V×I energy on 17 evaluation sessions per configuration (neural methods: three seeds). The residual bound was fixed from a disjoint physics-only calibration corpus. Point MAE favors the PINN over the matched MLP and the physics baseline; paired intervals versus measured energy are not shown and are not claimed to exclude zero. Flat-road / four-input adaptation applies.

---

## Figure C — Twizy residual-bound sensitivity

**Sources**
- `review_results/sensitivity/all_sensitivity_mae.csv` (`L_2`…`L_8`)
- Cross-checked against `summary_L_{2,4,6,8}.csv`

**Quantity**  
Trip-energy MAE (kWh) for the Twizy PINN under predetermined \(L\in\{2,4,6,8\}\) kW. **Point estimates only** (no CI in the sensitivity archive).  
\(L=4\) kW is the accepted paper reference; \(L=6\) has a lower point MAE and was **not** adopted (no outer-test retuning).

**Proposed caption**  
> Predetermined sensitivity of the Twizy PINN residual bound \(L\). Trip-energy MAE is shown as a point estimate. The accepted configuration remains \(L=4\,\mathrm{kW}\); the lower point MAE at \(L=6\,\mathrm{kW}\) was not selected from outer-test ranks.

---

## Combined Figure AB

Two-panel composition of A (full paired set) and B for a possible single manuscript figure if page budget allows one review graphic.

**Proposed caption**  
> (a) HELECAR-D paired absolute-error differences for the Full PINN versus baselines and ablated variants (95% paired bootstrap CIs; negative = Full PINN better). (b) Corrected, feature-adapted Tesla Model Y MAE versus measured V×I energy (17 trips per configuration; seeds 0–1–2). Panel (b) reports point MAE only.

---

## Manuscript inclusion recommendation

See `review_notes/08_manuscript_figure_plan.md`.
