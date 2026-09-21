# Statistical summary

Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.
Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.

## MAE 95% trip-level bootstrap intervals

- constant: mean MAE 0.3594 kWh with a 95% bootstrap interval of [0.2096, 0.5365] kWh (n=3 trips).
- elasticnet: mean MAE 0.0660 kWh with a 95% bootstrap interval of [0.0084, 0.1474] kWh (n=3 trips).
- mlp_state: mean MAE 0.4808 kWh with a 95% bootstrap interval of [0.0823, 0.7208] kWh (n=3 trips).
- physics: mean MAE 0.8923 kWh with a 95% bootstrap interval of [0.5259, 1.1500] kWh (n=3 trips).
- pinn: mean MAE 0.1098 kWh with a 95% bootstrap interval of [0.0179, 0.1603] kWh (n=3 trips).
- pinn_fulltrip: mean MAE 0.2564 kWh with a 95% bootstrap interval of [0.2310, 0.2915] kWh (n=3 trips).
- pinn_no_dynamics: mean MAE 0.5297 kWh with a 95% bootstrap interval of [0.2716, 0.8077] kWh (n=3 trips).
- weak_mlp: mean MAE 0.8020 kWh with a 95% bootstrap interval of [0.0510, 1.6311] kWh (n=3 trips).

## Paired absolute-error differences (trip-level bootstrap)

- pinn minus weak_mlp: mean paired difference -0.6922 kWh with a 95% bootstrap interval of [-1.4799, 0.1093] kWh. Negative values mean the first method has lower absolute error.
  The interval contains zero.
- pinn minus elasticnet: mean paired difference 0.0438 kWh with a 95% bootstrap interval of [0.0096, 0.1089] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.
- pinn minus physics: mean paired difference -0.7826 kWh with a 95% bootstrap interval of [-1.1320, -0.3657] kWh. Negative values mean the first method has lower absolute error.
  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim.

## Wilcoxon signed-rank (secondary)

- pinn vs weak_mlp: statistic=1, p=0.5 (two-sided).
- pinn vs elasticnet: statistic=0, p=0.25 (two-sided).
- pinn vs physics: statistic=0, p=0.25 (two-sided).
