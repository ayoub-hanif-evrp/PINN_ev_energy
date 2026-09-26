# Results notes

Writing aid for a 10-page paper. The main text follows four figures: the method diagram, the observed-versus-predicted panels, the MAE summary, and the data-scarcity curve. Sentences below are generated from the canonical tables.

### Method

The estimator has no instantaneous battery-power labels. Driving and environment variables enter an analytical physics branch and a bounded neural residual; predicted power is their sum and is integrated to trip energy. A depletion head forms $\hat{D}(t)$ from the shared neural state and trip progress, with $\hat{D}(0)=0$. The conservation residual links that depletion to the integrated energy. Multi-window SoC energy supervises the integral and the depletion head. SoC is not a power-branch input.

### Main leave-one-trip-out comparison

On 17-fold leave-one-trip-out, the point-estimate trip-energy MAEs were 0.780 kWh (Physics Model), 0.368 kWh (Ridge Regression), 0.475 kWh (Data-Driven MLP), and 0.295 kWh (PINN).
PINN has the lowest point-estimate MAE. The observed-versus-predicted panels use one marker per held-out trip, with neural predictions averaged across seeds for display.

### Paired neural comparison

The paired mean difference, PINN absolute error minus Data-Driven MLP absolute error, was -0.180 kWh (95% paired bootstrap [-0.316, -0.056]; n = 17; Wilcoxon p = 0.020, a secondary paired test).
A negative difference means lower PINN absolute error. Seeds were averaged within each trip and were not treated as independent trips. Separate MAE intervals are not used as a test of the paired difference.

### Data-scarcity behaviour

With 3 training trips, held-out MAE was 2.247 ± 1.986 kWh for Ridge Regression, 0.710 ± 0.150 kWh for the Data-Driven MLP, and 0.439 ± 0.115 kWh for the PINN.
With 5 training trips, held-out MAE was 5.520 ± 3.031 kWh for Ridge Regression, 0.697 ± 0.174 kWh for the Data-Driven MLP, and 0.344 ± 0.048 kWh for the PINN.
With 8 training trips, held-out MAE was 2.031 ± 2.521 kWh for Ridge Regression, 0.434 ± 0.080 kWh for the Data-Driven MLP, and 0.297 ± 0.065 kWh for the PINN.
With 12 training trips, held-out MAE was 0.400 ± 0.061 kWh for Ridge Regression, 0.459 ± 0.122 kWh for the Data-Driven MLP, and 0.305 ± 0.055 kWh for the PINN.
With 16 training trips, held-out MAE was 0.368 ± 0.000 kWh for Ridge Regression, 0.475 ± 0.026 kWh for the Data-Driven MLP, and 0.295 ± 0.048 kWh for the PINN.
Ridge exhibited large variability in the subsets with 3, 5, and 8 training trips. The PINN point-estimate MAE stayed below the Data-Driven MLP at each reported training size. Performance of the neural models remained comparatively stable as the training set grew from 8 to 16 trips.

### Secondary supporting material

The paired mean difference, PINN minus Physics Model, was -0.485 kWh (95% paired bootstrap [-0.604, -0.375]; Wilcoxon p = 1.53e-05).
The paired mean difference, PINN minus Ridge Regression, was -0.073 kWh (95% paired bootstrap [-0.264, 0.110]; Wilcoxon p = 0.579). That interval includes zero, so the full-data PINN–Ridge difference is not resolved by this paired test.

On stored held-out evaluation windows, mean window-energy MAE after averaging seeds within each trip was 60 s (Data-Driven MLP 0.023 kWh; PINN 0.017 kWh); 120 s (Data-Driven MLP 0.040 kWh; PINN 0.030 kWh); 300 s (Data-Driven MLP 0.089 kWh; PINN 0.065 kWh); 600 s (Data-Driven MLP 0.112 kWh; PINN 0.071 kWh); full trip (Data-Driven MLP 0.475 kWh; PINN 0.295 kWh). These windows were not used to retune the models.

Battery-feasibility sensitivity is optional for the main text. At a 20% reserve, the false-safe rate was 17.65% for the Physics Model and 5.88% for the PINN. The corresponding overly conservative rates were 0.00% and 0.00%.
The Physics Model under-predicts trip energy on this split, which increases false-safe decisions when the reserve is large. This comparison is a feasibility sensitivity, not an EV routing solver.
