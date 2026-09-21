# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- constant: mean MAE 0.2488 kWh with a 95% bootstrap interval of [0.1682, 0.3369] kWh (n=17 trips).
- elasticnet: mean MAE 0.1673 kWh with a 95% bootstrap interval of [0.1126, 0.2319] kWh (n=17 trips).
- mlp_state: mean MAE 0.5053 kWh with a 95% bootstrap interval of [0.3118, 0.7269] kWh (n=17 trips).
- physics: mean MAE 0.7796 kWh with a 95% bootstrap interval of [0.6142, 0.9485] kWh (n=17 trips).
- pinn: mean MAE 0.3558 kWh with a 95% bootstrap interval of [0.2322, 0.5071] kWh (n=17 trips).
- pinn_fulltrip: mean MAE 0.2804 kWh with a 95% bootstrap interval of [0.2013, 0.3652] kWh (n=17 trips).
- pinn_no_dynamics: mean MAE 0.2583 kWh with a 95% bootstrap interval of [0.1754, 0.3441] kWh (n=17 trips).
- weak_mlp: mean MAE 0.4692 kWh with a 95% bootstrap interval of [0.2758, 0.7297] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.1134 kWh with a 95% bootstrap interval of [-0.3380, 0.0602] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus elasticnet: mean paired difference 0.1885 kWh with a 95% bootstrap interval of [0.0478, 0.3532] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -0.4238 kWh with a 95% bootstrap interval of [-0.5520, -0.3011] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=71, p=0.8176 (two-sided).
- pinn vs elasticnet: statistic=31, p=0.03052 (two-sided).
- pinn vs physics: statistic=1, p=3.052e-05 (two-sided).
