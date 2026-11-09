# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.9843 kWh with a 95% bootstrap interval of [0.1882, 2.1750] kWh (n=4 trips).
- ridge: mean MAE 1.7054 kWh with a 95% bootstrap interval of [1.4354, 1.9594] kWh (n=4 trips).
- weak_mlp: mean MAE 0.6754 kWh with a 95% bootstrap interval of [0.3157, 1.0928] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference 0.3090 kWh with a 95% bootstrap interval of [-0.1772, 1.1251] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -0.7210 kWh with a 95% bootstrap interval of [-1.6852, 0.7408] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus physics: mean paired difference -1.0983 kWh with a 95% bootstrap interval of [-1.5022, -0.6943] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=4, p=0.875 (two-sided).
- pinn vs ridge: statistic=2, p=0.375 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
