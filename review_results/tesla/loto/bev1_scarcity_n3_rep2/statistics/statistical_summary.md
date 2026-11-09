# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.5076 kWh with a 95% bootstrap interval of [0.0119, 1.1637] kWh (n=4 trips).
- ridge: mean MAE 2.6045 kWh with a 95% bootstrap interval of [0.7091, 5.6482] kWh (n=4 trips).
- weak_mlp: mean MAE 0.6717 kWh with a 95% bootstrap interval of [0.2045, 1.3533] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.1641 kWh with a 95% bootstrap interval of [-0.2156, -0.1167] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus ridge: mean paired difference -2.0969 kWh with a 95% bootstrap interval of [-4.4862, -0.6966] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -1.5750 kWh with a 95% bootstrap interval of [-2.4880, -0.9275] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=0, p=0.125 (two-sided).
- pinn vs ridge: statistic=0, p=0.125 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
