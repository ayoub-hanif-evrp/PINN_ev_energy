# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 1.0504 kWh with a 95% bootstrap interval of [0.6506, 1.4786] kWh (n=17 trips).
- pinn: mean MAE 0.7759 kWh with a 95% bootstrap interval of [0.4588, 1.1171] kWh (n=17 trips).
- ridge: mean MAE 1.2281 kWh with a 95% bootstrap interval of [0.7166, 1.8058] kWh (n=17 trips).
- weak_mlp: mean MAE 0.7701 kWh with a 95% bootstrap interval of [0.4579, 1.1280] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference 0.0058 kWh with a 95% bootstrap interval of [-0.4066, 0.4254] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -0.4522 kWh with a 95% bootstrap interval of [-1.0885, 0.1298] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -0.2745 kWh with a 95% bootstrap interval of [-0.7026, 0.1431] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=63, p=0.5477 (two-sided).
- pinn vs ridge: statistic=54, p=0.306 (two-sided).
- pinn vs physics: statistic=49, p=0.2069 (two-sided).
