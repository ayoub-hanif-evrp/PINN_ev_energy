# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 1.0515 kWh with a 95% bootstrap interval of [0.6537, 1.4796] kWh (n=17 trips).
- pinn: mean MAE 0.8275 kWh with a 95% bootstrap interval of [0.5355, 1.1536] kWh (n=17 trips).
- ridge: mean MAE 1.2270 kWh with a 95% bootstrap interval of [0.7177, 1.8023] kWh (n=17 trips).
- weak_mlp: mean MAE 0.9230 kWh with a 95% bootstrap interval of [0.5947, 1.2919] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.0955 kWh with a 95% bootstrap interval of [-0.3704, 0.1892] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -0.3995 kWh with a 95% bootstrap interval of [-1.0326, 0.1763] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -0.2241 kWh with a 95% bootstrap interval of [-0.5411, 0.0898] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=63, p=0.5477 (two-sided).
- pinn vs ridge: statistic=50, p=0.2247 (two-sided).
- pinn vs physics: statistic=45, p=0.1454 (two-sided).
