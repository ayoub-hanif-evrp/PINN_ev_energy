# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- physics: mean MAE 1.2636 kWh with a 95% bootstrap interval of [0.6922, 1.9615] kWh (n=17 trips).
- pinn: mean MAE 0.6753 kWh with a 95% bootstrap interval of [0.4265, 0.9600] kWh (n=17 trips).
- ridge: mean MAE 1.7286 kWh with a 95% bootstrap interval of [0.8662, 3.0041] kWh (n=17 trips).
- weak_mlp: mean MAE 0.5836 kWh with a 95% bootstrap interval of [0.3986, 0.7816] kWh (n=17 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference 0.0916 kWh with a 95% bootstrap interval of [-0.1405, 0.3438] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus ridge: mean paired difference -1.0534 kWh with a 95% bootstrap interval of [-2.3604, -0.1887] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -0.5884 kWh with a 95% bootstrap interval of [-1.2796, -0.0120] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=72, p=0.8536 (two-sided).
- pinn vs ridge: statistic=29, p=0.02322 (two-sided).
- pinn vs physics: statistic=46, p=0.1594 (two-sided).
