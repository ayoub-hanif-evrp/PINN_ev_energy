# Manuscript figure / table plan (≤15 pages)

**Policy:** Keep original HELECAR-D results and the existing four manuscript figures. Add at most **one** new review figure and **one** compact ablation table. DOCX not edited in this pass.

**New assets:** `review_results/figures/` (PNG/PDF/SVG) + `README.md` captions.  
**Reproduce:** `python scripts/review/make_review_figures.py`

---

## Keep in the Springer paper (unchanged)

| Item | File / content | Role |
|---|---|---|
| Fig. 1 method | `results/figures/fig01_method.png` | Architecture |
| Fig. 2 parity | `results/figures/fig02_observed_vs_predicted.png` | Main HELECAR results |
| Fig. 3 MAE CI | `results/figures/fig04_mae_ci.png` | Primary accuracy + CI |
| Fig. 4 scarcity | `results/figures/fig08_data_scarcity_learning_curve.png` | Data-scarcity claim |
| Table 5 | Original trip-level metrics | Primary numbers |
| Table 6 | Scarcity MAE | Scarcity numbers |
| Tables 1–4 | Positioning / data / settings / physics params | Context |

Do **not** replace or regenerate these without a specific reason.

---

## Add to the camera-ready paper (minimal)

| Priority | Asset | Recommendation | Est. pages |
|---|---|---|---:|
| **1 (must)** | Ablation **table** (from `table_ablation_complete.csv`) + 1 honest paragraph | Addresses R2; text-efficient | +0.25…0.35 |
| **2 (must if space)** | **Either** `figAB_ablation_tesla_two_panel` **or** `figA` alone | Prefer **figAB** if Tesla subsection is included; else **figA** | +0.35…0.55 |
| **3 (text)** | Sensitivity: 2–4 sentences (optionally tiny inline numbers; **no** mandatory figure) | Addresses R3 L / λ | +0.08…0.12 |
| **4 (optional)** | Tesla numbers as a **small table** *or* panel (b) of figAB — not both | Avoid duplication | included in #2 |

### Preferred single new figure

**Use `figAB_ablation_tesla_two_panel`** as one Springer figure (e.g. Fig. 5) if the optional Tesla subsection is kept:

- Panel (a): HELECAR-D ablation (within-PINN CIs cross zero).  
- Panel (b): Tesla Model Y validation — V×I point MAE only; no significance claim.  
- Figure titles avoid internal terms such as “corrected”; use authoritative `*_loto_corrected` sources only.

If Tesla is dropped for page reasons: insert **`figA_ablation_paired_diff`** only.

---

## Keep as repository / supplementary only

| Asset | Reason |
|---|---|
| `figC_sensitivity_L` | Useful for R3; manuscript can state the numbers in prose; saves ~⅓ page |
| Separate `figA` + `figB` if figAB is used | Avoid duplicate graphics |
| λ-weight / capacity sensitivity plots (not generated) | Covered by prose + `05_sensitivity_findings.md` |
| SoC-reconstruction / state MAE diagnostic plots | Not verified as evidence of accurate SoC tracking; do not elevate |
| Superseded Tesla `tesla_bev*_loto` anything | Leakage; never cite |
| Remaining `results/figures/fig05`–`fig16` not already in the paper | Optional repo material (unchanged policy) |

---

## Tables: paper vs repo

| Table | Paper? | Notes |
|---|---|---|
| Ablation MAE (+ optional paired footnote) | **Yes** | Prefer MAE table; figAB/figA carries paired CIs visually |
| Full paired-difference numeric table | Repo / footnote | Redundant with figA/figAB if space tight |
| Tesla V×I MAE (3×2) | **Yes if Tesla kept** — either in figAB labels or a 3-row table, not both large | Corrected only |
| Tesla SoC-reference MAE | Repo only | Clarifies SoC–V×I gap; optional one sentence |
| Sensitivity L / λ / capacity | Prose in paper; full CSV in repo | Do not retune L=4 |

---

## Caption pack (for paste after review)

**Fig. AB (recommended)**  
> (a) Paired differences in trip-energy absolute error between the full PINN and comparison methods on HELECAR-D (17 trips; markers: means; whiskers: 95% paired bootstrap CIs; negative values favor the full PINN). (b) Trip-energy MAE on two Tesla Model Y configurations against independently measured battery energy (V×I; 17 trips per configuration; three neural seeds; four-feature flat-road adaptation). Differences between PINN and MLP against measured energy are not statistically conclusive.

**Fig. A (fallback)**  
> Paired differences in trip-energy absolute error between the full PINN and comparison methods on HELECAR-D (17 trips). Markers indicate mean differences and whiskers show 95% paired bootstrap confidence intervals. Negative values favor the full PINN.

**Fig. B (if used alone)**  
> Trip-energy MAE for the Physics Model, Data-Driven MLP, and PINN on two Tesla Model Y configurations, evaluated against independently measured battery energy (V×I). Results cover 17 evaluation trips per configuration and three neural initialization seeds. The method uses a documented four-feature, flat-road adaptation. Differences between PINN and MLP against measured energy are not statistically conclusive.

**Fig. C (repo)**  
> Trip-energy MAE of the Twizy PINN for predetermined residual bounds \(L\in\{2,4,6,8\}\) kW. The highlighted marker is the accepted reference \(L=4\) kW; the lower point estimate at \(L=6\) kW was not adopted from outer-test ranks.

---

## Page-budget decision tree

1. Insert ablation **table** + honest paragraph.  
2. Measure Springer page count after Eq. (17) / train–val / condensation edits.  
3. If ≥0.5 page free **and** Tesla subsection desired → add **figAB**.  
4. Else if ≥0.35 page free → add **figA** only.  
5. Keep **figC** in the repository; cite L∈{2,4,6,8} in text.  
6. Never displace Figs 1–4 or Tables 5–6.
