# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 1.2641 kWh with a 95% bootstrap interval of [0.6927, 1.9620] kWh (n=17 trips).
- pinn: mean MAE 0.6587 kWh with a 95% bootstrap interval of [0.4489, 0.8874] kWh (n=17 trips).
- ridge: mean MAE 1.7268 kWh with a 95% bootstrap interval of [0.8620, 3.0032] kWh (n=17 trips).
- weak_mlp: mean MAE 1.0059 kWh with a 95% bootstrap interval of [0.6714, 1.3595] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.3472 kWh with a 95% bootstrap interval of [-0.6126, -0.1153] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus ridge: mean paired difference -1.0681 kWh with a 95% bootstrap interval of [-2.3021, -0.3121] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -0.6054 kWh with a 95% bootstrap interval of [-1.3134, 0.0096] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=26, p=0.015 (two-sided).
- pinn vs ridge: statistic=7, p=0.0002899 (two-sided).
- pinn vs physics: statistic=43, p=0.1202 (two-sided).
