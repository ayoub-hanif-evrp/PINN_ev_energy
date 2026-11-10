# Camera-ready section-by-section modification plan

**Manuscript:** `Hanif_PINN_EV_Energy_INTIS2026.docx` (Springer LNCS; **do not overwrite until this plan is reviewed**)  
**Constraint:** ≤15 pages including references  
**Policy:** Keep accepted architecture and original HELECAR-D / paper configs / `results/` tables unchanged. Add only review-branch evidence that is already verified.  
**Authoritative Tesla:** `tesla_bev1_loto_corrected`, `tesla_bev2_loto_corrected` only. Preserve documented negative predictions.  
**Audit / commit:** `review_notes/05_camera_ready_final_audit.md` @ pushed `3b53bc4`

**Net page budget target:** ≈ **+0.0 to +0.3 pages** after condensation offsets new content (aim flat or slight shrink).

---

## Global priorities (what to protect vs cut)

| Keep (non-negotiable) | Condense / trim first | Add (minimal) |
|---|---|---|
| Core contribution + Fig. 1 | Repeated “three terms / complementarity” prose in §4.3 | Correct Eq. (17) → four implemented terms |
| Essential eqs (1)–(10), (13)–(17) | Long metric definitions in §5 if space tight | Compact ablation table + honest paragraph |
| Table 5 + Figs 2–4 primary HELECAR results | Redundant scarcity narrative overlapping §6.1 | 2–4 sensitivity sentences |
| Data-scarcity Table 6 / Fig. 4 | Soft routing paragraph if Tesla is included | Optional ≤½ p Tesla subsection |
| Novelty claim (weak SoC multi-window + bounded residual) | Duplicate limitation sentences across §6 | Stronger capacity / SoC / params / generalizability |

---

## Title / Abstract / Keywords

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Title, keywords, core claim | 0 |
| **Light edit** | One abstract sentence: estimation under scarce trips **without power labels**, using multi-window SoC supervision + bounded residual | ≈0 |
| **Do not** | Claim cross-vehicle superiority or MAE-optimal every loss term | — |

---

## §1 Introduction

| Action | What to replace / condense / add | Est. pages |
|---|---|---|
| **Condense** | Merge the two opening paragraphs’ overlapping motivation (range, BMS, routing) into one tighter block (~15–20% shorter) | **−0.05 to −0.10** |
| **Replace (1–2 sentences)** | Contribution list: state **four** training terms explicitly (window, conservation/dynamics, depletion/state, residual prior) instead of implying three | ≈0 |
| **Add (optional, 1 sentence)** | Scope: primary evidence is HELECAR-D Twizy; optional feature-adapted Tesla check appears in Results | ≈0 |
| **Do not** | Expand related-problem laundry lists | — |

---

## §2 Related Work

| Action | What to replace / condense / add | Est. pages |
|---|---|---|
| **Keep** | Table 1 positioning comparison | 0 |
| **Condense** | Shorten data-driven survey paragraph by ~25% (keep De Cauwer / PINN bridge citations) | **−0.05 to −0.10** |
| **Add (R1)** | 1–2 references next to Eq. (4) discussion **or** here: longitudinal EV force / power-based energy models (existing [2]–[4] already cite-adjacent; ensure Eq. 4 paragraph cites them explicitly) | ≈0 / **+0.02** if new refs |
| **Do not** | New taxonomy tables | — |

---

## §3 Data and Problem Formulation

### §3.1 HELECAR-D Dataset

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | 17 trips, inputs, Table 2, SoC resolution note | 0 |
| **Strengthen (2 sentences)** | SoC is quantized (~0.02 pp); interval energy is the supervised quantity; capacity 6.0 kWh is an **assumption** from HELECAR-D / docs, not measured usable energy | **+0.05** |
| **Condense** | Trim feature-derivation sentences that repeat §4.1 kinematics | **−0.05** |

### §3.2 Energy Estimation Without Power Measurements

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Eqs (1)–(3) and power-vs-energy distinction | 0 |
| **Clarify (1 sentence)** | Training never uses instantaneous SoC differences as power labels | ≈0 |

---

## §4 Proposed Physics-Informed Energy Estimator

### §4.1 Physics-Based Power Estimation — Eq. (4) justification

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Eqs (4)–(6), Fig. 1 | 0 |
| **Replace / expand (3–5 sentences)** | Justify longitudinal force sum (inertia, rolling, grade, aero) with citations to power-based EV models; state parameter sources (Table 4); note auxiliaries as lumped \(P_\mathrm{aux}\) | **+0.05 to +0.10** |
| **Add (1 sentence)** | Parameters are fixed a priori for Twizy; not fitted on outer-test trips | ≈0 |

### §4.2 Neural Correction and Energy Estimation

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Bounded residual \(L\), eqs (7)–(10) | 0 |
| **Condense** | Cut one of the two paragraphs that restate “network learns correction not full map” | **−0.05** |
| **Clarify** | \(L=4\,\mathrm{kW}\) is the accepted Twizy residual bound (sensitivity discussed in §6; not retuned) | ≈0 |

### §4.3 Multi-Window Supervision and Physical Consistency — **mandatory equation fix**

| Action | Detail | Est. pages |
|---|---|---|
| **REPLACE** para before Eq. (17) | Current: “combines the **three** complementary constraints” → “combines **four** implemented terms” | ≈0 |
| **REPLACE** Eq. (17) | Match code (`src/training/losses.py`): \(\mathcal{L}=\lambda_w\mathcal{L}_\mathrm{window}+\lambda_d\mathcal{L}_\mathrm{dynamics}+\lambda_s\mathcal{L}_\mathrm{state}+\lambda_p\mathcal{L}_\mathrm{prior}\) | ≈0 |
| **REPLACE** following “where λ…” sentence | Name **four** weights (window, dynamics/conservation, state/depletion, **prior / residual regularization**) | ≈0 |
| **REPLACE** “The three terms…” paragraph | Four-term roles in ≤4 short sentences; prior = \(\|\Delta P/L\|^2\)-style residual regularization (keep notation consistent with eqs 7–8) | ≈0 / **−0.05** if shorter than current |
| **Keep** | Eqs (11)–(16) as-is (window, conservation residual, state) | 0 |
| **Do not** | Change architecture, window construction, or accepted λ defaults | — |

---

## §5 Experimental Setup — train/val clarity

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | LOTO, metrics (19)–(25), scarcity protocol, Tables 3–4 | 0 |
| **Add (short paragraph, ~80–100 words)** | Explicit protocol: (i) leave-one-trip-out outer folds; (ii) **inner** validation split from remaining trips for early stopping / model selection only; (iii) final retrain on all non-test training trips with selected epoch; (iv) preprocessing/scaling fit on training only; (v) SoC multi-windows from **training** trips only; test-trip SoC not fed to the power estimator at inference | **+0.10 to +0.15** |
| **Condense if over page limit** | Merge metric eqs (22)–(24) MAPE/WAPE/\(R^2\) into one compact sentence + keep MAE/RMSE primary; or move secondary metrics to a footnote under Table 5 | **−0.10 to −0.15** |
| **Keep numbers** | Twizy \(L=4\,\mathrm{kW}\), capacity 6.0 kWh, seeds, λ from paper config | 0 |

---

## §6 Results and Discussion

### §6.1 Energy-Estimation Results

| Action | Detail | Est. pages |
|---|---|---|
| **Keep unchanged** | Figs 2–3, Table 5 numbers (0.295 / 0.368 / 0.475 / 0.780), paired PINN–MLP discussion | 0 |
| **Condense** | Trim Ridge “competitive but…” paragraph by ~20% (keep the honesty) | **−0.05** |
| **ADD — Ablation subsection or short block after Table 5** | Title suggestion: *Ablation of supervision and consistency terms* | **+0.25 to +0.35** |

**Ablation table to insert** (from `review_results/ablations/`; do not alter paper main table):

| Variant | MAE [kWh] | 95% CI |
|---|---:|---|
| Physics Model | 0.780 | [0.614, 0.948] |
| Data-Driven MLP | 0.475 | [0.312, 0.655] |
| Full PINN (reference) | 0.295 | [0.209, 0.392] |
| PINN full-trip supervision only | 0.243 | [0.182, 0.306] |
| PINN without conservation loss | 0.278 | [0.215, 0.342] |
| PINN without depletion-state loss | 0.253 | [0.191, 0.315] |
| PINN without residual prior | 0.321 | [0.226, 0.434] |

**Ablation discussion (required honesty, ~100–120 words):**
- Full PINN ≫ Physics and MLP (paired CIs exclude 0).
- Among PINN variants, paired CIs **include 0** — do **not** claim every term improves trip MAE.
- Full-trip-only has best *point* MAE but not statistically better; conservation/state mainly improve physical consistency (diagnostics).
- Retain accepted multi-scale formulation.

If page-critical: drop Physics/MLP rows from ablation table (already in Table 5) → save ~0.05–0.08 p.

### §6.2 Data-Scarcity Results and Discussion

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Fig. 4, Table 6, main scarcity claims | 0 |
| **Condense** | Merge the three interpretive paragraphs (PINN stable / Ridge brittle / “limited but useful conclusion”) into two | **−0.08 to −0.12** |
| **ADD — Residual & λ sensitivity (2–4 sentences)** | Predetermined \(L\in\{2,4,6,8\}\) and \(\lambda_\mathrm{dynamics},\lambda_\mathrm{prior}\in\{0.5\times,1\times,2\times\}\); MAE near 0.295; **do not** adopt L=6 or retune λ from outer-test ranks; capacity {5.5, 6.0, 6.5} rescales SoC-derived labels — report as assumption uncertainty | **+0.08 to +0.12** |
| **REPLACE / strengthen limitations paragraph** | Expand existing limitations: 17 trips, one region; **capacity assumption**; SoC quantization/noise; fixed physical parameters (mass, \(C_dA\), η, \(P_\mathrm{aux}\)); generalizability not claimed beyond feature-adapted checks | **+0.05 to +0.10** (offset by cutting routing detail if needed) |
| **Condense or drop if Tesla added** | Routing false-safe paragraph → 1 sentence or move to footnote | **−0.08 to −0.12** |

### **NEW optional §6.3 External validation on Tesla Model Y (feature-adapted)** — ≤½ page

| Action | Detail | Est. pages |
|---|---|---|
| **Add only if space** | Corrected results only; framing: same weak-supervision method, **not** zero-shot 8-feature transfer | **+0.35 to +0.50** |
| **Table** | MAE vs measured V×I (17 sessions/config, seeds 0–1–2): Physics / MLP / PINN for BEV1 & BEV2 (audit §3.3) | included above |
| **Qualifications (mandatory)** | 4 inputs, θ=0; physics-only disjoint L calib (40 / 20 kW); V×I evaluation-only; paired V×I CIs often include 0; SoC vs V×I gap; **negative predictions retained** on low-energy trips | — |
| **Do not** | Cite superseded `tesla_bev*_loto`; claim universal PINN≻MLP | — |

Draft wording: `review_notes/camera_ready_revisions.md` §B.

---

## §7 Conclusion

| Action | Detail | Est. pages |
|---|---|---|
| **Keep** | Core takeaway: PINN under scarce SoC-only supervision | 0 |
| **Replace 1–2 sentences** | Mention four-term objective; ablation honesty; sensitivity supports locked L/λ; limitations + optional Tesla as feature-adapted external check | ≈0 / **+0.03** |
| **Do not** | Overclaim generalizability | — |

---

## References / Disclosure

| Action | Detail | Est. pages |
|---|---|---|
| **Add** | Any missing longitudinal-force / validation citations for Eq. (4) (prefer reuse of [2]–[4]; add ≤2 new if reviewers require) | **+0.05 to +0.10** |
| **Keep** | Disclosure of Interests | 0 |
| **Condense if needed** | Prefer not cutting refs; cut body prose first | — |

---

## Page-impact ledger (planning estimate)

| Block | Impact |
|---|---:|
| Intro + Related Work condensation | −0.10 to −0.20 |
| §4.3 three→four term rewrite (net) | −0.05 to 0.00 |
| §4.1 Eq. (4) justification | +0.05 to +0.10 |
| §5 train/val clarification | +0.10 to +0.15 |
| §5 metric condensation (if needed) | −0.10 to −0.15 |
| Ablation table + honest paragraph | +0.25 to +0.35 |
| Sensitivity sentences | +0.08 to +0.12 |
| Stronger limitations (capacity/SoC/params) | +0.05 to +0.10 |
| Scarcity prose + routing trim | −0.15 to −0.25 |
| **Subtotal without Tesla** | **≈ −0.05 to +0.25** |
| Optional Tesla §6.3 | +0.35 to +0.50 |
| **With Tesla** | **≈ +0.30 to +0.75** → **must** cut more scarcity/routing/related-work if over 15 pp |

**Decision rule:** Implement non-Tesla edits first; measure Springer page count; add Tesla §6.3 only if ≤15 pages after cuts.

---

## What must not change

- Accepted PINN architecture and residual formulation.
- Original HELECAR-D main results in `results/` / Table 5 / scarcity Table 6.
- Paper configs / locked Twizy \(L=4\,\mathrm{kW}\), λ, capacity 6.0 kWh as defaults.
- Superseded Tesla runs (`tesla_bev1_loto`, `tesla_bev2_loto`).
- Censoring of negative Tesla predictions.

---

## Implementation order (after plan review)

1. Fix Eq. (17) + four-term prose (§4.3).  
2. Add train/val paragraph (§5).  
3. Insert ablation table + honest discussion (§6.1).  
4. Add sensitivity + strengthen limitations (§6.2); justify Eq. (4) (§4.1).  
5. Condense intro / related work / scarcity / routing to reclaim pages.  
6. Page-check; optionally insert Tesla §6.3.  
7. Sync abstract/conclusion wording.  

**Stop condition:** Manuscript file remains untouched until authors approve this plan.
