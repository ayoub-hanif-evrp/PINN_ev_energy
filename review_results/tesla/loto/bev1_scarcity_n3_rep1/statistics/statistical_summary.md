# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.5916 kWh with a 95% bootstrap interval of [0.2917, 0.8915] kWh (n=4 trips).
- ridge: mean MAE 3.3220 kWh with a 95% bootstrap interval of [0.1753, 8.1792] kWh (n=4 trips).
- weak_mlp: mean MAE 0.5226 kWh with a 95% bootstrap interval of [0.0779, 0.9674] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference 0.0690 kWh with a 95% bootstrap interval of [-0.0983, 0.2445] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -2.7304 kWh with a 95% bootstrap interval of [-7.3537, 0.1164] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -1.4910 kWh with a 95% bootstrap interval of [-2.7762, -0.5939] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=3, p=0.625 (two-sided).
- pinn vs ridge: statistic=3, p=0.625 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
