# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.1591 kWh with a 95% bootstrap interval of [0.0114, 0.4419] kWh (n=4 trips).
- ridge: mean MAE 0.8665 kWh with a 95% bootstrap interval of [0.2606, 1.4724] kWh (n=4 trips).
- weak_mlp: mean MAE 0.6446 kWh with a 95% bootstrap interval of [0.2122, 1.2735] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.4855 kWh with a 95% bootstrap interval of [-0.8316, -0.2008] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus ridge: mean paired difference -0.7074 kWh with a 95% bootstrap interval of [-1.4491, 0.0343] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -1.9235 kWh with a 95% bootstrap interval of [-3.2585, -1.1438] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=0, p=0.125 (two-sided).
- pinn vs ridge: statistic=2, p=0.375 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
