# Review status and camera-ready recommendation

Date: 2026-11-10  
Branch: `review/intis2026-validation` (on `origin`; repo currently public)  
Baseline: `0a85b57`

## Completed

| Phase | Status |
|---|---|
| Protect `main` | Done |
| HELECAR-D ablations + honest interpretation | Done — `02_ablation_interpretation.md` |
| Tesla audit / GO-NOGO | Done |
| Tesla protocol fixes (L leak, dt, sanity, 3 seeds) | Done — corrected LOTO |
| Twizy residual + λ sensitivity | Done — `05_sensitivity_findings.md` |
| Capacity sensitivity | Running / see `review_results/sensitivity/` |
| Camera-ready revision package | Draft — `camera_ready_revisions.md` |

## Camera-ready recommendation

Enough validated evidence exists for a precise ≤15-page revision:

1. Keep HELECAR-D main results unchanged.  
2. Add honest ablation table + consistency diagnostics paragraph.  
3. Add ~½-page feature-adapted Tesla subsection with corrected table and limitations.  
4. Add short L / λ sensitivity sentence (defaults not retuned).  
5. Condense methodology; clarify novelty; fix Eq. 4 references.

Do **not** invent stronger PINN claims than the CIs support.
