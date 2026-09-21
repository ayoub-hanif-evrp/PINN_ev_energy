# Quick-profile diagnostics

These files are **engineering checks**, not paper results.

Smoke (3 held-out trips, 25 epochs) completed with:

- no NaN/Inf
- residual-head saturation = 0
- `D_hat(0) = 0` by construction
- per-fold `q_inner_train` / `q_outer_train` / `q_test_posthoc`
- grouped T1/T2/T3 validation IDs
- outer retraining enabled

Do not cite these numbers in the paper.
