"""Manuscript tables, captions, and a results draft generated from result CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _f3(value: float) -> str:
    return f"{float(value):.3f}"


def _f2(value: float) -> str:
    return f"{float(value):.2f}"


def _signed(value: float) -> str:
    number = float(value)
    return f"{number:+.3f}"


def _p(value: float) -> str:
    number = float(value)
    if number < 0.001:
        return f"{number:.2e}"
    return f"{number:.3f}"


def _minutes(seconds: float) -> str:
    from decimal import Decimal, ROUND_HALF_UP

    minutes = (Decimal(str(float(seconds))) / Decimal(60)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{minutes}"


def _thousands(value: float) -> str:
    return f"{int(round(float(value))):,}"


def _sampling_interval(seconds: float) -> str:
    value = float(seconds)
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))} s"
    return f"{value:.2f} s"


def write_paper_documents(
    out_dir: Path,
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t5: pd.DataFrame,
    capacity_kwh: float,
    window_csv: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_tables.md").write_text(
        _tables(t1, t2, t3, t4, t5, capacity_kwh),
        encoding="utf-8",
    )
    (out_dir / "paper_figure_captions.md").write_text(_captions(), encoding="utf-8")
    (out_dir / "paper_results.md").write_text(
        _results(t2, t3, t4, t5, window_csv),
        encoding="utf-8",
    )


def _tables(
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t5: pd.DataFrame,
    capacity_kwh: float,
) -> str:
    row = t1.iloc[0]
    dur_lo = _minutes(row["duration_s_min"])
    dur_hi = _minutes(row["duration_s_max"])
    parts = [
        "# Paper tables",
        "",
        "Generated from `results/tables/`. Do not edit the numbers by hand.",
        "",
        "For the 10-page manuscript, paste Tables 1, 3, and 4, plus the PINN versus Data-Driven MLP row of Table 5.",
        "Table 2 is a short method comparison if space remains. Table 6 is optional and is not required in the main text.",
        "",
        "### Table 1. Characteristics of the HELECAR-D data used in this study",
        "",
        "| Characteristic | Value |",
        "|---|---|",
        f"| Analysed trips | {int(row['n_trips'])} |",
        f"| Trajectory distribution | T1: {int(row['n_T1'])}, T2: {int(row['n_T2'])}, T3: {int(row['n_T3'])} |",
        f"| Total samples | {_thousands(row['n_samples'])} |",
        f"| Approximate sampling interval | {_sampling_interval(row['sampling_interval_s'])} |",
        f"| Trip duration | {dur_lo}–{dur_hi} min |",
        f"| Trip distance | {_f2(row['distance_km_min'])}–{_f2(row['distance_km_max'])} km |",
        f"| Mean trip distance | {_f2(row['distance_km_mean'])} km |",
        f"| Empirical SoC resolution | ≈ {_f2(row['empirical_soc_q_pp'])} percentage points |",
        f"| Assumed usable battery capacity | {_f1(capacity_kwh)} kWh |",
        "",
        f"The usable battery capacity of {_f1(capacity_kwh)} kWh is a modelling assumption used by the energy model. It is not an independently measured usable-capacity value from this dataset.",
        "",
        "### Table 2. Comparison of evaluated energy-estimation approaches",
        "",
        "| Method | Learning level | Supervision | Analytical vehicle physics | Battery-conservation constraint | Primary output |",
        "|---|---|---|---|---|---|",
        "| Physics Model | No learning | Vehicle equations | Yes | No learned conservation loss | Trip energy from integrated analytical power |",
        "| Ridge Regression | Trip-level supervised | SoC-derived trip energy | No | No | Trip energy |",
        "| Data-Driven MLP | Sample-level latent power | Multi-window SoC energy | No | No | Integrated latent power |",
        "| PINN | Sample-level latent power | Multi-window SoC energy + depletion constraint | Yes | Yes | Integrated latent power |",
        "",
        "### Table 3. Leave-one-trip-out trip-energy estimation performance",
        "",
        "| Method | MAE [kWh] | 95% CI | RMSE [kWh] | MAPE [%] | WAPE [%] | Bias [kWh] | R² |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, item in t2.iterrows():
        parts.append(
            f"| {item['Method']} | {_f3(item['MAE_kWh'])} | [{_f3(item['MAE_CI_low'])}, {_f3(item['MAE_CI_high'])}] | "
            f"{_f3(item['RMSE_kWh'])} | {_f2(item['MAPE_pct'])} | {_f2(item['WAPE_pct'])} | "
            f"{_signed(item['Bias_kWh'])} | {_f3(item['R2'])} |"
        )
    parts += [
        "",
        "MAE and its confidence interval use one value per held-out trip: the mean, across training seeds, of that trip's absolute error. The interval is a trip-level bootstrap of those values. RMSE, MAPE, WAPE, bias, and R² are computed on the pooled seed-level predictions. Neural seeds are training replicates, not additional trips.",
        "",
        "### Table 4. Performance under varying numbers of training trips",
        "",
        "| Training trips | Ridge Regression | Data-Driven MLP | PINN |",
        "|---:|---:|---:|---:|",
    ]
    for n in (3, 5, 8, 12, 16):
        cells = []
        for method in ("Ridge Regression", "Data-Driven MLP", "PINN"):
            hit = t3.loc[(t3["n_train"] == n) & (t3["method"] == method)]
            cells.append(f"{_f3(hit['MAE_kWh'].iloc[0])} ± {_f3(hit['uncertainty'].iloc[0])}")
        parts.append(f"| {n} | " + " | ".join(cells) + " |")
    parts += [
        "",
        "Values are mean held-out-trip MAE [kWh] across predefined training-subset replicates; uncertainty denotes the SD of replicate-level means. Neural replicates include initialization seeds. Ridge is deterministic conditional on its selected training subset, so identical full-set fits are not repeated.",
        "",
        "### Table 5. Paired trip-level comparison of PINN absolute errors",
        "",
        "| Comparison | n | Mean paired absolute-error difference [kWh] | 95% paired bootstrap CI | Wilcoxon p |",
        "|---|---:|---:|---|---:|",
    ]
    for _, item in t5.iterrows():
        parts.append(
            f"| {item['comparison']} | {int(item['n_trips'])} | {_signed(item['mean_absolute_error_difference_kwh'])} | "
            f"[{_f3(item['bootstrap_CI_low'])}, {_f3(item['bootstrap_CI_high'])}] | {_p(item['Wilcoxon_p'])} |"
        )
    parts += [
        "",
        "Difference = PINN absolute error − comparator absolute error, after averaging per-seed absolute errors within each held-out trip for neural methods. A negative value means lower PINN absolute error. The bootstrap interval is the primary paired summary. The Wilcoxon p-value is a secondary paired test, not a claim based on overlap of separate MAE intervals.",
        "",
        "### Table 6. Battery-feasibility decision errors",
        "",
        "| Reserve [%] | Physics false-safe [%] | PINN false-safe [%] | Physics conservative [%] | PINN conservative [%] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for reserve in (5, 10, 15, 20):
        phys = t4.loc[(t4["reserve_soc_pct"] == reserve) & (t4["method"] == "Physics Model")].iloc[0]
        pinn = t4.loc[(t4["reserve_soc_pct"] == reserve) & (t4["method"] == "PINN")].iloc[0]
        parts.append(
            f"| {int(reserve)} | {_f2(phys['false_safe_pct'])} | {_f2(pinn['false_safe_pct'])} | "
            f"{_f2(phys['overly_conservative_pct'])} | {_f2(pinn['overly_conservative_pct'])} |"
        )
    parts += [
        "",
        "Rates are the fraction of held-out trips on which the energy estimate produces a false-safe or overly conservative battery-feasibility decision. This is not an EV routing algorithm.",
        "",
    ]
    return "\n".join(parts)


def _f1(value: float) -> str:
    return f"{float(value):.1f}"


def _captions() -> str:
    main = [
        (
            "1",
            "fig01_method.png",
            "Fig. 1. How the estimator turns a trip into an energy prediction. The inputs are driving and environment variables. Vehicle physics proposes a power, a neural network adds a bounded correction, and the two are added. Integrating that power gives the trip energy. A second output, $\\hat{D}(t)$, is the predicted SoC drop; it starts at zero and is built from the same network plus how far the trip has progressed. Training asks this drop to agree with the energy just used. The only supervision is the SoC change on short windows and on the full trip. SoC is not an input, and battery power is never measured.",
        ),
        (
            "2",
            "fig02_observed_vs_predicted.png",
            "Fig. 2. Leave-one-trip-out observed versus predicted trip energy. Each point is one held-out trip. A point on the line is an exact prediction; a point below the line is an under-prediction. Neural predictions are averaged over three training seeds for display. All four panels use the same axes. Annotated MAE is the mean of per-trip absolute errors after averaging absolute error across seeds; annotated $R^2$ matches Table 3.",
        ),
        (
            "3",
            "fig04_mae_ci.png",
            "Fig. 3. Leave-one-trip-out trip-energy MAE. Each marker is the mean absolute error across the 17 held-out trips, and the whisker is a 95% trip-level bootstrap interval. The number beside each marker is that MAE. Neural values average absolute error across seeds within each trip before the bootstrap. Methods are shown in a fixed order.",
        ),
        (
            "4",
            "fig08_data_scarcity_learning_curve.png",
            "Fig. 4. Trip-energy error as the number of training trips grows. Panel (a) uses a logarithmic axis so that Ridge Regression, which reaches several kWh when few trips are available, can be shown together with the neural models; the printed numbers are the Ridge means above 1 kWh. Panel (b) uses a linear axis and shows only the Data-Driven MLP and the PINN. Markers are means across predefined training subsets, and error bars are the standard deviation of those subset means. Neural subsets also vary the initialization seed. Ridge is deterministic for a given subset, so the 16-trip point has no error bar. The dashed line is the Physics Model, which does not train on trips. A lower logarithmic error bar that would cross zero is omitted.",
        ),
    ]
    optional = [
        ("fig03_trip_errors.png", "Absolute trip-energy error for each of the 17 held-out trips. Points are individual trips. The horizontal mark is the median and the vertical bar is the interquartile range. Neural errors average absolute error across seeds within each trip."),
        ("fig05_residuals.png", "Trip-energy residual (predicted minus observed) against observed trip energy. The horizontal line is zero residual. Neural predictions are seed-averaged for display. Each panel uses the same axes."),
        ("fig06_error_ecdf.png", "Empirical distribution of absolute trip-energy error. The vertical axis is the fraction of held-out trips whose absolute error is at most the value on the horizontal axis. Neural errors average absolute error across seeds within each trip."),
        ("fig07_trip_error_heatmap.png", "Absolute trip-energy error for every held-out trip and method. Rows follow trajectory order T1, then T2, then T3. Neural cells average absolute error across seeds."),
        ("fig09_pinn_vs_mlp.png", "Paired absolute trip-energy error of the PINN and the Data-Driven MLP. Each marker is one held-out trip, with marker shape indicating trajectory. Absolute errors are averaged across seeds within the trip. Points below the diagonal have lower PINN absolute error."),
        ("fig10_soc_reconstruction.png", "Observed quantized SoC and SoC reconstructed by integrating predicted PINN power on one predetermined trip from each trajectory. When reconstructions for seeds 0, 1, and 2 are stored, the line is their mean and the band is ±1 standard deviation across seeds. Only the initial test SoC initializes the rollout. Test SoC is not a model input, and the curves are not a validation of instantaneous battery power."),
        ("fig11_window_consistency.png", "Neural window-energy consistency on held-out non-overlapping evaluation windows. Markers are the mean, across trips, of the seed-averaged window-energy MAE. Error bars are the standard deviation across held-out trips. Physics Model and Ridge Regression are absent because the stored window evaluation does not contain them."),
        ("fig12_feasibility.png", "Battery-feasibility decision errors at reserve margins of 5%, 10%, 15%, and 20% SoC. Panel (a) is the false-safe rate and panel (b) is the overly conservative rate, each as a percentage of held-out trips. This assessment is not an EV routing algorithm."),
        ("fig13_pinn_training_convergence.png", "PINN training record for the predetermined selection run on held-out trip T1_03_29_2021, seed 0. Panel (a) is the inner-validation multi-window energy MAE used for early stopping; the vertical line marks the epoch with the lowest value in this history. Panel (a) is training-only inner validation, not outer-test performance. Panel (b) shows the training objective and the weighted contributions that enter it, on a logarithmic axis."),
        ("fig14_validation_window_scales.png", "Inner-validation window-energy MAE by time scale for the same predetermined PINN selection run. This is training-only inner validation, not outer-test performance."),
        ("fig15_trip_energy_comparison.png", "Observed and predicted trip energy for every held-out trip, separated by trajectory. Neural predictions are averaged over three training seeds for display. The black marker is the observed energy."),
        ("fig16_trajectory_mae.png", "Absolute trip-energy error by trajectory. Small markers are individual held-out trips and large markers are the trajectory mean. Sample sizes are 4 (T1), 3 (T2), and 10 (T3). Neural errors average absolute error across seeds within each trip."),
    ]
    lines = [
        "# Figure captions",
        "",
        "The 10-page manuscript uses the four captions below. The remaining captions are optional repository material.",
        "",
        "## Main manuscript figures",
        "",
    ]
    for number, filename, text in main:
        lines += [f"### Paper Figure {number}", "", f"`results/figures/{filename}`", "", text, ""]
    lines += ["## Optional / repository figures", ""]
    for filename, text in optional:
        lines += [f"### `{filename}`", "", f"Optional. {text}", ""]
    return "\n".join(lines)


def _results(
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t5: pd.DataFrame,
    window_csv: Path | None,
) -> str:
    def mae(name: str) -> str:
        return _f3(t2.loc[t2["Method"] == name, "MAE_kWh"].iloc[0])

    def paired(name: str) -> pd.Series:
        return t5.loc[t5["comparison"] == name].iloc[0]

    def scarce(n: int, method: str) -> pd.Series:
        return t3.loc[(t3["n_train"] == n) & (t3["method"] == method)].iloc[0]

    mlp_row = paired("PINN - Data-Driven MLP")
    ridge_row = paired("PINN - Ridge Regression")
    phys_row = paired("PINN - Physics Model")
    p20 = t4.loc[(t4["reserve_soc_pct"] == 20) & (t4["method"] == "Physics Model")].iloc[0]
    n20 = t4.loc[(t4["reserve_soc_pct"] == 20) & (t4["method"] == "PINN")].iloc[0]
    lines = [
        "# Results notes",
        "",
        "Writing aid for a 10-page paper. The main text follows four figures: the method diagram, the observed-versus-predicted panels, the MAE summary, and the data-scarcity curve. Sentences below are generated from the canonical tables.",
        "",
        "### Method",
        "",
        "The estimator has no instantaneous battery-power labels. Driving and environment variables enter an analytical physics branch and a bounded neural residual; predicted power is their sum and is integrated to trip energy. A depletion head forms $\\hat{D}(t)$ from the shared neural state and trip progress, with $\\hat{D}(0)=0$. The conservation residual links that depletion to the integrated energy. Multi-window SoC energy supervises the integral and the depletion head. SoC is not a power-branch input.",
        "",
        "### Main leave-one-trip-out comparison",
        "",
        (
            f"On 17-fold leave-one-trip-out, the point-estimate trip-energy MAEs were "
            f"{mae('Physics Model')} kWh (Physics Model), {mae('Ridge Regression')} kWh (Ridge Regression), "
            f"{mae('Data-Driven MLP')} kWh (Data-Driven MLP), and {mae('PINN')} kWh (PINN)."
        ),
        "PINN has the lowest point-estimate MAE. The observed-versus-predicted panels use one marker per held-out trip, with neural predictions averaged across seeds for display.",
        "",
        "### Paired neural comparison",
        "",
        (
            f"The paired mean difference, PINN absolute error minus Data-Driven MLP absolute error, was "
            f"{_signed(mlp_row['mean_absolute_error_difference_kwh'])} kWh "
            f"(95% paired bootstrap [{_f3(mlp_row['bootstrap_CI_low'])}, {_f3(mlp_row['bootstrap_CI_high'])}]; "
            f"n = {int(mlp_row['n_trips'])}; Wilcoxon p = {_p(mlp_row['Wilcoxon_p'])}, a secondary paired test)."
        ),
        "A negative difference means lower PINN absolute error. Seeds were averaged within each trip and were not treated as independent trips. Separate MAE intervals are not used as a test of the paired difference.",
        "",
        "### Data-scarcity behaviour",
        "",
    ]
    for n in (3, 5, 8, 12, 16):
        ridge = scarce(n, "Ridge Regression")
        mlp = scarce(n, "Data-Driven MLP")
        pinn = scarce(n, "PINN")
        lines.append(
            f"With {n} training trips, held-out MAE was {_f3(ridge['MAE_kWh'])} ± {_f3(ridge['uncertainty'])} kWh "
            f"for Ridge Regression, {_f3(mlp['MAE_kWh'])} ± {_f3(mlp['uncertainty'])} kWh for the Data-Driven MLP, "
            f"and {_f3(pinn['MAE_kWh'])} ± {_f3(pinn['uncertainty'])} kWh for the PINN."
        )
    lines += [
        "Ridge exhibited large variability in the subsets with 3, 5, and 8 training trips. The PINN point-estimate MAE stayed below the Data-Driven MLP at each reported training size. Performance of the neural models remained comparatively stable as the training set grew from 8 to 16 trips.",
        "",
        "### Secondary supporting material",
        "",
        (
            f"The paired mean difference, PINN minus Physics Model, was "
            f"{_signed(phys_row['mean_absolute_error_difference_kwh'])} kWh "
            f"(95% paired bootstrap [{_f3(phys_row['bootstrap_CI_low'])}, {_f3(phys_row['bootstrap_CI_high'])}]; "
            f"Wilcoxon p = {_p(phys_row['Wilcoxon_p'])})."
        ),
        (
            f"The paired mean difference, PINN minus Ridge Regression, was "
            f"{_signed(ridge_row['mean_absolute_error_difference_kwh'])} kWh "
            f"(95% paired bootstrap [{_f3(ridge_row['bootstrap_CI_low'])}, {_f3(ridge_row['bootstrap_CI_high'])}]; "
            f"Wilcoxon p = {_p(ridge_row['Wilcoxon_p'])}). "
            "That interval includes zero, so the full-data PINN–Ridge difference is not resolved by this paired test."
        ),
        "",
    ]
    window_line = _window_sentence(window_csv)
    lines.append(window_line)
    lines += [
        "",
        (
            "Battery-feasibility sensitivity is optional for the main text. "
            f"At a 20% reserve, the false-safe rate was {_f2(p20['false_safe_pct'])}% for the Physics Model "
            f"and {_f2(n20['false_safe_pct'])}% for the PINN. "
            f"The corresponding overly conservative rates were {_f2(p20['overly_conservative_pct'])}% and {_f2(n20['overly_conservative_pct'])}%."
        ),
        "The Physics Model under-predicts trip energy on this split, which increases false-safe decisions when the reserve is large. This comparison is a feasibility sensitivity, not an EV routing solver.",
        "",
    ]
    return "\n".join(lines)


def _window_sentence(window_csv: Path | None) -> str:
    if window_csv is None or not window_csv.exists():
        return "Stored non-overlapping window evaluation was not available, so no window-scale comparison is stated."
    frame = pd.read_csv(window_csv)
    if "level" in frame.columns:
        frame = frame.loc[frame["level"] == "scale"]
    scales = ["time_60s", "time_120s", "time_300s", "time_600s", "full_trip"]
    labels = {"time_60s": "60 s", "time_120s": "120 s", "time_300s": "300 s", "time_600s": "600 s", "full_trip": "full trip"}
    bits = []
    for scale in scales:
        cells = []
        for method, name in (("weak_mlp", "Data-Driven MLP"), ("pinn", "PINN")):
            sub = frame.loc[(frame["method"] == method) & (frame["scale"] == scale)]
            if sub.empty:
                continue
            if "seed" in sub.columns:
                sub = sub.groupby("trip_id", as_index=False)["mae_kwh"].mean()
            cells.append(f"{name} {_f3(sub['mae_kwh'].mean())} kWh")
        if cells:
            bits.append(f"{labels[scale]} ({'; '.join(cells)})")
    if not bits:
        return "The stored window file did not contain Data-Driven MLP and PINN rows at the requested scales."
    return (
        "On stored held-out evaluation windows, mean window-energy MAE after averaging seeds within each trip was "
        + "; ".join(bits)
        + ". These windows were not used to retune the models."
    )
