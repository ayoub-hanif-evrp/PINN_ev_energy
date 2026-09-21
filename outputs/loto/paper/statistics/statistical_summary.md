# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- constant: mean MAE 0.2488 kWh with a 95% bootstrap interval of [0.1682, 0.3369] kWh (n=17 trips).
- elasticnet: mean MAE 0.1673 kWh with a 95% bootstrap interval of [0.1126, 0.2319] kWh (n=17 trips).
- mlp_state: mean MAE 0.3722 kWh with a 95% bootstrap interval of [0.2652, 0.4857] kWh (n=17 trips).
- physics: mean MAE 0.7796 kWh with a 95% bootstrap interval of [0.6142, 0.9485] kWh (n=17 trips).
- pinn: mean MAE 0.2949 kWh with a 95% bootstrap interval of [0.2092, 0.3921] kWh (n=17 trips).
- pinn_fulltrip: mean MAE 0.2428 kWh with a 95% bootstrap interval of [0.1824, 0.3061] kWh (n=17 trips).
- pinn_no_dynamics: mean MAE 0.2777 kWh with a 95% bootstrap interval of [0.2149, 0.3415] kWh (n=17 trips).
- weak_mlp: mean MAE 0.4751 kWh with a 95% bootstrap interval of [0.3121, 0.6551] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.1802 kWh with a 95% bootstrap interval of [-0.3164, -0.0558] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus elasticnet: mean paired difference 0.1276 kWh with a 95% bootstrap interval of [0.0144, 0.2418] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -0.4846 kWh with a 95% bootstrap interval of [-0.6039, -0.3746] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=28, p=0.02016 (two-sided).
- pinn vs elasticnet: statistic=30, p=0.02667 (two-sided).
- pinn vs physics: statistic=0, p=1.526e-05 (two-sided).
