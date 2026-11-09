# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.3030 kWh with a 95% bootstrap interval of [0.0871, 0.6278] kWh (n=4 trips).
- ridge: mean MAE 1.7718 kWh with a 95% bootstrap interval of [0.3644, 3.8472] kWh (n=4 trips).
- weak_mlp: mean MAE 0.9664 kWh with a 95% bootstrap interval of [0.1068, 2.6150] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.6634 kWh with a 95% bootstrap interval of [-1.9871, 0.0873] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -1.4687 kWh with a 95% bootstrap interval of [-3.2308, -0.2773] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -1.7796 kWh with a 95% bootstrap interval of [-3.0440, -0.9881] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=3, p=0.625 (two-sided).
- pinn vs ridge: statistic=0, p=0.125 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
