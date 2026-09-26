# Results notes

Draft sentences generated from the canonical tables. They are a writing aid, not a claim beyond those tables.

### Main trip-energy estimation

On 17-fold leave-one-trip-out, the point-estimate trip-energy MAEs were 0.780 kWh (Physics Model), 0.368 kWh (Ridge Regression), 0.475 kWh (Data-Driven MLP), and 0.295 kWh (PINN).
The paired mean difference in absolute error, PINN minus Physics Model, was -0.485 kWh (95% paired bootstrap [-0.604, -0.375]; Wilcoxon p = 1.53e-05).
The paired mean difference, PINN minus Ridge Regression, was -0.073 kWh (95% paired bootstrap [-0.264, 0.110]; Wilcoxon p = 0.579).
Separate MAE intervals are not used as a test of these paired differences.

### Data-scarcity behaviour

With 3 training trips, held-out MAE was 2.247 ± 1.986 kWh for Ridge Regression, 0.710 ± 0.150 kWh for the Data-Driven MLP, and 0.439 ± 0.115 kWh for the PINN.
With 5 training trips, held-out MAE was 5.520 ± 3.031 kWh for Ridge Regression, 0.697 ± 0.174 kWh for the Data-Driven MLP, and 0.344 ± 0.048 kWh for the PINN.
With 8 training trips, held-out MAE was 2.031 ± 2.521 kWh for Ridge Regression, 0.434 ± 0.080 kWh for the Data-Driven MLP, and 0.297 ± 0.065 kWh for the PINN.
With 12 training trips, held-out MAE was 0.400 ± 0.061 kWh for Ridge Regression, 0.459 ± 0.122 kWh for the Data-Driven MLP, and 0.305 ± 0.055 kWh for the PINN.
With 16 training trips, held-out MAE was 0.368 ± 0.000 kWh for Ridge Regression, 0.475 ± 0.026 kWh for the Data-Driven MLP, and 0.295 ± 0.048 kWh for the PINN.
Ridge exhibited large variability in the subsets with 3, 5, and 8 training trips. The PINN point-estimate MAE stayed below the Data-Driven MLP at each reported training size.

### PINN vs Data-Driven MLP

The paired mean difference, PINN absolute error minus Data-Driven MLP absolute error, was -0.180 kWh (95% paired bootstrap [-0.316, -0.056]; n = 17; Wilcoxon p = 0.020).
A negative difference means lower PINN absolute error. Seeds were averaged within each trip and were not treated as independent trips.

### Multi-window consistency

On stored held-out evaluation windows, mean window-energy MAE after averaging seeds within each trip was 60 s (Data-Driven MLP 0.023 kWh; PINN 0.017 kWh); 120 s (Data-Driven MLP 0.040 kWh; PINN 0.030 kWh); 300 s (Data-Driven MLP 0.089 kWh; PINN 0.065 kWh); 600 s (Data-Driven MLP 0.112 kWh; PINN 0.071 kWh); full trip (Data-Driven MLP 0.475 kWh; PINN 0.295 kWh). These windows were not used to retune the models.

### Battery-feasibility sensitivity

At a 20% reserve, the false-safe rate was 17.65% for the Physics Model and 5.88% for the PINN. The corresponding overly conservative rates were 0.00% and 0.00%.
The Physics Model under-predicts trip energy on this split, which increases false-safe decisions when the reserve is large. This comparison is a feasibility sensitivity, not an EV routing solver.
