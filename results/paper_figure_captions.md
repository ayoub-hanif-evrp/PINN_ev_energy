# Figure captions

The 10-page manuscript uses the four captions below. The remaining captions are optional repository material.

## Main manuscript figures

### Paper Figure 1

`results/figures/fig01_method.png`

Fig. 1. How the estimator turns a trip into an energy prediction. The inputs are driving and environment variables. Vehicle physics proposes a power, a neural network adds a bounded correction, and the two are added. Integrating that power gives the trip energy. A second output, $\hat{D}(t)$, is the predicted SoC drop; it starts at zero and is built from the same network plus how far the trip has progressed. Training asks this drop to agree with the energy just used. The only supervision is the SoC change on short windows and on the full trip. SoC is not an input, and battery power is never measured.

### Paper Figure 2

`results/figures/fig02_observed_vs_predicted.png`

Fig. 2. Leave-one-trip-out observed versus predicted trip energy. Each point is one held-out trip. A point on the line is an exact prediction; a point below the line is an under-prediction. Neural predictions are averaged over three training seeds for display. All four panels use the same axes. Annotated MAE is the mean of per-trip absolute errors after averaging absolute error across seeds; annotated $R^2$ matches Table 3.

### Paper Figure 3

`results/figures/fig04_mae_ci.png`

Fig. 3. Leave-one-trip-out trip-energy MAE. Each marker is the mean absolute error across the 17 held-out trips, and the whisker is a 95% trip-level bootstrap interval. The number beside each marker is that MAE. Neural values average absolute error across seeds within each trip before the bootstrap. Methods are shown in a fixed order.

### Paper Figure 4

`results/figures/fig08_data_scarcity_learning_curve.png`

Fig. 4. Trip-energy error as the number of training trips grows. Panel (a) uses a logarithmic axis so that Ridge Regression, which reaches several kWh when few trips are available, can be shown together with the neural models; the printed numbers are the Ridge means above 1 kWh. Panel (b) uses a linear axis and shows only the Data-Driven MLP and the PINN. Markers are means across predefined training subsets, and error bars are the standard deviation of those subset means. Neural subsets also vary the initialization seed. Ridge is deterministic for a given subset, so the 16-trip point has no error bar. The dashed line is the Physics Model, which does not train on trips. A lower logarithmic error bar that would cross zero is omitted.

## Optional / repository figures

### `fig03_trip_errors.png`

Optional. Absolute trip-energy error for each of the 17 held-out trips. Points are individual trips. The horizontal mark is the median and the vertical bar is the interquartile range. Neural errors average absolute error across seeds within each trip.

### `fig05_residuals.png`

Optional. Trip-energy residual (predicted minus observed) against observed trip energy. The horizontal line is zero residual. Neural predictions are seed-averaged for display. Each panel uses the same axes.

### `fig06_error_ecdf.png`

Optional. Empirical distribution of absolute trip-energy error. The vertical axis is the fraction of held-out trips whose absolute error is at most the value on the horizontal axis. Neural errors average absolute error across seeds within each trip.

### `fig07_trip_error_heatmap.png`

Optional. Absolute trip-energy error for every held-out trip and method. Rows follow trajectory order T1, then T2, then T3. Neural cells average absolute error across seeds.

### `fig09_pinn_vs_mlp.png`

Optional. Paired absolute trip-energy error of the PINN and the Data-Driven MLP. Each marker is one held-out trip, with marker shape indicating trajectory. Absolute errors are averaged across seeds within the trip. Points below the diagonal have lower PINN absolute error.

### `fig10_soc_reconstruction.png`

Optional. Observed quantized SoC and SoC reconstructed by integrating predicted PINN power on one predetermined trip from each trajectory. When reconstructions for seeds 0, 1, and 2 are stored, the line is their mean and the band is ±1 standard deviation across seeds. Only the initial test SoC initializes the rollout. Test SoC is not a model input, and the curves are not a validation of instantaneous battery power.

### `fig11_window_consistency.png`

Optional. Neural window-energy consistency on held-out non-overlapping evaluation windows. Markers are the mean, across trips, of the seed-averaged window-energy MAE. Error bars are the standard deviation across held-out trips. Physics Model and Ridge Regression are absent because the stored window evaluation does not contain them.

### `fig12_feasibility.png`

Optional. Battery-feasibility decision errors at reserve margins of 5%, 10%, 15%, and 20% SoC. Panel (a) is the false-safe rate and panel (b) is the overly conservative rate, each as a percentage of held-out trips. This assessment is not an EV routing algorithm.

### `fig13_pinn_training_convergence.png`

Optional. PINN training record for the predetermined selection run on held-out trip T1_03_29_2021, seed 0. Panel (a) is the inner-validation multi-window energy MAE used for early stopping; the vertical line marks the epoch with the lowest value in this history. Panel (a) is training-only inner validation, not outer-test performance. Panel (b) shows the training objective and the weighted contributions that enter it, on a logarithmic axis.

### `fig14_validation_window_scales.png`

Optional. Inner-validation window-energy MAE by time scale for the same predetermined PINN selection run. This is training-only inner validation, not outer-test performance.

### `fig15_trip_energy_comparison.png`

Optional. Observed and predicted trip energy for every held-out trip, separated by trajectory. Neural predictions are averaged over three training seeds for display. The black marker is the observed energy.

### `fig16_trajectory_mae.png`

Optional. Absolute trip-energy error by trajectory. Small markers are individual held-out trips and large markers are the trajectory mean. Sample sizes are 4 (T1), 3 (T2), and 10 (T3). Neural errors average absolute error across seeds within each trip.
