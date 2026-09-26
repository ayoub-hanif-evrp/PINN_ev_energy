# Figure captions

Generated for the manuscript. Numbers described in the captions follow the result tables.

### Figure 1

`results/figures/fig01_method.png`

Fig. 1. Discrete-time physics-informed estimator. Driving and environment variables feed a longitudinal physics branch and a bounded neural correction. Predicted battery power is the sum of the two branches, integrated with the trapezoidal rule, and supervised by multi-window energy derived from SoC. SoC is training and evaluation supervision only; it is not a power-branch input. There are no instantaneous battery-power labels.

### Figure 2

`results/figures/fig02_observed_vs_predicted.png`

Fig. 2. Leave-one-trip-out observed versus predicted trip energy. Each marker is one held-out trip. Neural predictions are averaged over three training seeds for display. The diagonal line is perfect agreement. Annotated MAE is the mean of per-trip absolute errors after averaging absolute error across seeds; annotated R² matches the pooled-seed value in Table 3.

### Figure 3

`results/figures/fig03_trip_errors.png`

Fig. 3. Absolute trip-energy error for each of the 17 held-out trips. Points are individual trips. The horizontal mark is the median and the vertical bar is the interquartile range. Neural errors average absolute error across seeds within each trip.

### Figure 4

`results/figures/fig04_mae_ci.png`

Fig. 4. Leave-one-trip-out trip-energy MAE. The marker is the mean absolute error across held-out trips and the whisker is a 95% trip-level bootstrap interval. Neural values average absolute error across seeds within each trip before the bootstrap. Methods are shown in a fixed order, not sorted by error.

### Figure 5

`results/figures/fig05_residuals.png`

Fig. 5. Trip-energy residual (predicted minus observed) against observed trip energy. The horizontal line is zero residual. Neural predictions are seed-averaged for display. Each panel uses the same axes.

### Figure 6

`results/figures/fig06_error_ecdf.png`

Fig. 6. Empirical distribution of absolute trip-energy error. The vertical axis is the fraction of held-out trips whose absolute error is at most the value on the horizontal axis. The curve is a step function. Neural errors average absolute error across seeds within each trip.

### Figure 7

`results/figures/fig07_trip_error_heatmap.png`

Fig. 7. Absolute trip-energy error for every held-out trip and method. Rows follow trajectory order T1, then T2, then T3, and file order within each trajectory. Neural cells average absolute error across seeds. Darker cells are larger errors.

### Figure 8

`results/figures/fig08_data_scarcity_learning_curve.png`

Fig. 8. Held-out trip-energy MAE as a function of the number of training trips. Panel (a) shows all trainable methods on a logarithmic error scale because Ridge exhibits large errors in the severe low-data regime. Panel (b) is a linear-scale view of the neural comparison. Small markers are replicate-level mean MAEs; large markers are the mean of those replicates and error bars are their standard deviation. Neural replicates combine initialization seeds and predefined subset repeats. Ridge is deterministic given its training subset, so the full 16-trip fit is a single point. The dashed line is the Physics Model, which does not use training trips. A lower logarithmic error bar that would cross zero is omitted; the replicate markers still show that spread.

### Figure 9

`results/figures/fig09_pinn_vs_mlp.png`

Fig. 9. Paired absolute trip-energy error of the PINN and the Data-Driven MLP. Each marker is one held-out trip, with marker shape indicating trajectory. Absolute errors are averaged across seeds within the trip. Points below the diagonal have lower PINN absolute error.

### Figure 10

`results/figures/fig10_soc_reconstruction.png`

Fig. 10. Observed quantized SoC and SoC reconstructed by integrating predicted PINN power on one predetermined trip from each trajectory. When reconstructions for seeds 0, 1, and 2 are stored, the line is their mean and the band is ±1 standard deviation across seeds. Only the initial test SoC initializes the rollout. Test SoC is not a model input, and the curves are not a validation of instantaneous battery power.

### Figure 11

`results/figures/fig11_window_consistency.png`

Fig. 11. Neural window-energy consistency on held-out non-overlapping evaluation windows. Markers are the mean, across trips, of the seed-averaged window-energy MAE. Error bars are the standard deviation across held-out trips. Physics Model and Ridge Regression are absent because the stored window evaluation does not contain them. This is not a four-method benchmark.

### Figure 12

`results/figures/fig12_feasibility.png`

Fig. 12. Battery-feasibility decision errors at reserve margins of 5%, 10%, 15%, and 20% SoC. Panel (a) is the false-safe rate and panel (b) is the overly conservative rate, each as a percentage of held-out trips. This assessment is not an EV routing algorithm.

### Figure 13

`results/figures/fig13_pinn_training_convergence.png`

Fig. 13. PINN training record for the predetermined selection run on held-out trip T1_03_29_2021, seed 0. Panel (a) is the inner-validation multi-window energy MAE used for early stopping; the vertical line marks the epoch with the lowest value in this history. Panel (a) is training-only inner validation, not outer-test performance. Panel (b) shows the training objective and the weighted contributions that enter it, on a logarithmic axis.

### Figure 14

`results/figures/fig14_validation_window_scales.png`

Fig. 14. Inner-validation window-energy MAE by time scale for the same predetermined PINN selection run as Fig. 13. This is training-only inner validation, not outer-test performance.

### Figure 15

`results/figures/fig15_trip_energy_comparison.png`

Fig. 15. Observed and predicted trip energy for every held-out trip, separated by trajectory. Neural predictions are averaged over three training seeds for display. The black marker is the observed energy.

### Figure 16

`results/figures/fig16_trajectory_mae.png`

Fig. 16. Absolute trip-energy error by trajectory. Small markers are individual held-out trips and large markers are the trajectory mean. Sample sizes are 4 (T1), 3 (T2), and 10 (T3). Neural errors average absolute error across seeds within each trip.
