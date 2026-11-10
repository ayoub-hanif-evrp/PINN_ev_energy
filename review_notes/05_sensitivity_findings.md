# Twizy sensitivity findings (Reviewer 3)

Predetermined grids. **Do not** replace the paper default from these ranks.  
Outputs: `review_results/sensitivity/`. Paper `results/` untouched.

## 1. Residual bound L (PINN, seeds 0–1–2, LOTO)

| L [kW] | Trip MAE [kWh] | Role |
|---:|---:|---|
| 2 | 0.300 | Tighter |
| **4** | **0.295** | Paper reference |
| 6 | 0.263 | Wider (lower point MAE; not adopted) |
| 8 | 0.339 | Wide (worse) |

Interpretation: L=4 kW sits in a reasonably flat region. L=6 improves the point estimate slightly but was **not** selected for the camera-ready method. L=8 degrades accuracy.

## 2. Loss weights (one-at-a-time; seeds 0–1–2)

Reference: λ_dynamics=1.0, λ_prior=0.1.

| Variant | MAE [kWh] |
|---|---:|
| λ_dynamics ×0.5 | 0.303 |
| λ_dynamics ×1 (ref) | 0.295 |
| λ_dynamics ×2 | 0.294 |
| λ_prior ×0.5 | 0.303 |
| λ_prior ×1 (ref) | 0.295 |
| λ_prior ×2 | 0.286 |

Interpretation: MAE is stable under ±2× reweighting of dynamics and prior terms. Reference weights remain justified; small gains at ×2 are **not** used to retune the paper.

## 3. Battery capacity {5.5, 6.0, 6.5} kWh

Same capacity assumption used for SoC-derived targets and evaluation (not measured usable capacity).

| Assumed capacity [kWh] | PINN trip MAE [kWh] |
|---:|---:|
| 5.5 | 0.202 |
| **6.0** | **0.295** |
| 6.5 | 0.330 |

Interpretation: absolute MAE scales with the energy unit induced by the capacity assumption. The paper’s 6.0 kWh assumption remains the locked reference; report uncertainty rather than retuning capacity from test MAE.
