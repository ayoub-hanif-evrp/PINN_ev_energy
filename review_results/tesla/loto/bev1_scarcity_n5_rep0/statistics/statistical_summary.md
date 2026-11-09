# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.4833 kWh with a 95% bootstrap interval of [0.1337, 0.8328] kWh (n=4 trips).
- ridge: mean MAE 0.4086 kWh with a 95% bootstrap interval of [0.1794, 0.6597] kWh (n=4 trips).
- weak_mlp: mean MAE 0.6475 kWh with a 95% bootstrap interval of [0.0572, 1.3599] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.1642 kWh with a 95% bootstrap interval of [-0.5723, 0.0765] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference 0.0747 kWh with a 95% bootstrap interval of [-0.3832, 0.5325] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -1.5993 kWh with a 95% bootstrap interval of [-2.8625, -0.7665] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=5, p=1 (two-sided).
- pinn vs ridge: statistic=5, p=1 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
