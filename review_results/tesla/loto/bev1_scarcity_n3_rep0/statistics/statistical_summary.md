# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 2.0826 kWh with a 95% bootstrap interval of [1.1618, 3.7037] kWh (n=4 trips).
- pinn: mean MAE 0.4350 kWh with a 95% bootstrap interval of [0.1681, 0.7019] kWh (n=4 trips).
- ridge: mean MAE 1.6283 kWh with a 95% bootstrap interval of [0.8919, 2.3648] kWh (n=4 trips).
- weak_mlp: mean MAE 0.6947 kWh with a 95% bootstrap interval of [0.3169, 1.1360] kWh (n=4 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.2597 kWh with a 95% bootstrap interval of [-0.5469, -0.0035] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus ridge: mean paired difference -1.1933 kWh with a 95% bootstrap interval of [-2.1967, -0.1900] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -1.6476 kWh with a 95% bootstrap interval of [-3.1228, -0.6848] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=1, p=0.25 (two-sided).
- pinn vs ridge: statistic=1, p=0.25 (two-sided).
- pinn vs physics: statistic=0, p=0.125 (two-sided).
