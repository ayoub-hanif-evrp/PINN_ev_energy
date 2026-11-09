# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.9460 kWh with a 95% bootstrap interval of [0.1765, 2.1054] kWh (n=4 trips).
- ridge: mean MAE 3.1552 kWh with a 95% bootstrap interval of [0.5713, 7.9051] kWh (n=4 trips).
- weak_mlp: mean MAE 1.2142 kWh with a 95% bootstrap interval of [0.4134, 2.0315] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.2682 kWh with a 95% bootstrap interval of [-0.9581, 0.1572] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -2.2092 kWh with a 95% bootstrap interval of [-5.7997, -0.3841] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -1.1366 kWh with a 95% bootstrap interval of [-1.5745, -0.7386] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=4, p=0.875 (two-sided).
- pinn vs ridge: statistic=0, p=0.125 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
