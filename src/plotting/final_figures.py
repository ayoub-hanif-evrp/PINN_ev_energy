"""Conference-paper figures: PNG only, 400 dpi, white background."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data.preprocessing import ProcessedTrip, elapsed_seconds
from evaluation.bootstrap import bootstrap_mean_ci, trip_absolute_errors
from evaluation.metrics import summarize_energy_table
from evaluation.names import (
    FEASIBILITY_METHODS,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    PARITY_METHODS,
    SCARCITY_METHODS,
    label,
)
from evaluation.trip_agg import trip_level_predictions
from plotting.style import apply_style, save_png

TRAJ_MARKERS = {"T1": "o", "T2": "s", "T3": "^"}
TRAJ_COLORS = {"T1": "#000000", "T2": "#7570b3", "T3": "#1b9e77"}
LETTERS = "abcdefghij"


def _c(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def _tag(ax: plt.Axes, letter: str) -> None:
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")


def fig01_method(path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    def box(x, y, w, h, text, fc):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.25,rounding_size=0.7",
                linewidth=1.05,
                edgecolor="#222222",
                facecolor=fc,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.25,
                color="#222222",
                shrinkA=1.5,
                shrinkB=1.5,
                clip_on=False,
            )
        )

    box(
        1.5,
        32,
        26,
        36,
        "What the model sees\nspeed, acceleration, grade\nweather, traffic\n\nnot SoC\nnot battery power",
        "#eef3f8",
    )
    box(33, 52, 24, 16, "Vehicle physics\npower from the\nvehicle equations", "#f7f1e8")
    box(33, 32, 24, 16, "Learned correction\na bounded change\nto that power", "#e7f2ea")
    box(33, 14, 24, 14, "Predicted SoC drop\n$\\hat{D}(t)$, starts at 0", "#e7eef8")
    box(63, 52, 35, 16, "Predicted power\nphysics + correction\n$\\hat{P}=P_{\\mathrm{physics}}+\\delta P$", "#e8eef7")
    box(63, 32, 35, 16, "Trip energy\nintegrate that power\nover the trip", "#f4f4f4")
    box(63, 14, 35, 14, "Consistency check\nSoC drop must match\nthe energy used", "#f8ecec")
    box(
        1.5,
        1.2,
        96.5,
        10.2,
        "Training signal: how much SoC changed, on short windows and the full trip\n"
        "SoC supervises the model. It is never a power input. Battery power is never measured.",
        "#f7f7f7",
    )
    arrow(27.5, 60, 33, 60)
    arrow(27.5, 40, 33, 40)
    arrow(57, 60, 63, 60)
    arrow(57, 40, 63, 56)
    arrow(80.5, 52, 80.5, 48)
    arrow(80.5, 32, 80.5, 28)
    arrow(45, 32, 45, 28)
    arrow(57, 21, 63, 21)
    ax.text(46.4, 30.0, "same network + trip progress", fontsize=9, ha="left", va="center", color="#222222")
    save_png(fig, path)


def fig02_observed_vs_predicted(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    trip = trip_level_predictions(pred)
    methods = [m for m in PARITY_METHODS if m in set(trip["method"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2), sharex=True, sharey=True)
    lo = min(trip["observed_energy_kwh"].min(), trip["predicted_energy_kwh"].min())
    hi = max(trip["observed_energy_kwh"].max(), trip["predicted_energy_kwh"].max())
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    for i, (ax, method) in enumerate(zip(axes.ravel(), methods)):
        g = trip.loc[trip["method"] == method]
        ax.plot(lim, lim, color="0.45", lw=1.35, zorder=0)
        ax.scatter(
            g["observed_energy_kwh"],
            g["predicted_energy_kwh"],
            s=64,
            c=_c(method),
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolors="0.15",
            linewidths=0.45,
            zorder=2,
        )
        mae = float(trip_absolute_errors(pred, method).mean())
        r2 = float(summarize_energy_table(pred.loc[pred["method"] == method]).get(method, {}).get("r2", float("nan")))
        ax.text(
            0.04,
            0.96,
            f"({LETTERS[i]}) {label(method)}\nMAE = {mae:.3f} kWh\n$R^2$ = {r2:.3f}"
            + ("\nline = exact match" if i == 0 else ""),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("Observed trip energy (kWh)")
    axes[1, 1].set_xlabel("Observed trip energy (kWh)")
    axes[0, 0].set_ylabel("Predicted trip energy (kWh)")
    axes[1, 0].set_ylabel("Predicted trip energy (kWh)")
    fig.tight_layout()
    save_png(fig, path)


def fig03_trip_errors(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        err = trip_absolute_errors(pred, method).sort_index()
        x = np.full(len(err), i, dtype=float) + rng.uniform(-0.12, 0.12, len(err))
        ax.scatter(x, err.to_numpy(), s=36, c=_c(method), marker=METHOD_MARKERS.get(method, "o"), zorder=3, edgecolors="0.15", linewidths=0.4)
        q = err.quantile([0.25, 0.5, 0.75])
        ax.hlines(q[0.5], i - 0.22, i + 0.22, color="0.1", lw=1.8, zorder=4)
        ax.vlines(i, q[0.25], q[0.75], color="0.1", lw=1.4, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label(m) for m in methods])
    ax.set_ylabel("Absolute trip-energy error (kWh)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_png(fig, path)


def fig_mae_ci(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    means, lo, hi = [], [], []
    for method in methods:
        stats = bootstrap_mean_ci(trip_absolute_errors(pred, method).to_numpy(dtype=float))
        means.append(stats["mean"])
        lo.append(stats["ci_low"])
        hi.append(stats["ci_high"])
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xerr = np.vstack([np.array(means) - np.array(lo), np.array(hi) - np.array(means)])
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#222222", elinewidth=1.6, capsize=4.5, zorder=2)
    for i, method in enumerate(methods):
        ax.plot(means[i], y[i], marker=METHOD_MARKERS.get(method, "o"), color=_c(method), ms=11, zorder=3)
        ax.annotate(
            f"{means[i]:.3f}",
            (means[i], y[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            va="bottom",
            fontsize=11,
            annotation_clip=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label(m) for m in methods])
    ax.set_xlabel("Trip-energy MAE (kWh)")
    ax.text(0.99, 0.03, "whiskers: 95% CI", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.25")
    ax.set_xlim(left=0)
    ax.set_ylim(len(methods) - 0.55, -0.95)
    fig.tight_layout()
    save_png(fig, path)


def fig_residuals(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    trip = trip_level_predictions(pred)
    methods = [m for m in PARITY_METHODS if m in set(trip["method"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharex=True, sharey=True)
    resid = trip["predicted_energy_kwh"] - trip["observed_energy_kwh"]
    pad_y = 0.1 * (float(resid.max()) - float(resid.min()) + 1e-6)
    ylim = (float(resid.min()) - pad_y, float(resid.max()) + pad_y)
    xlim = (float(trip["observed_energy_kwh"].min()) - 0.2, float(trip["observed_energy_kwh"].max()) + 0.2)
    for i, (ax, method) in enumerate(zip(axes.ravel(), methods)):
        g = trip.loc[trip["method"] == method]
        residual = g["predicted_energy_kwh"] - g["observed_energy_kwh"]
        ax.axhline(0.0, color="0.25", lw=1.3)
        ax.scatter(g["observed_energy_kwh"], residual, s=42, c=_c(method), marker=METHOD_MARKERS.get(method, "o"), edgecolors="0.15", linewidths=0.4)
        ax.text(
            0.04,
            0.96,
            f"({LETTERS[i]}) {label(method)}\nmean bias {float(residual.mean()):+.2f} kWh",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[1, 0].set_xlabel("Observed trip energy (kWh)")
    axes[1, 1].set_xlabel("Observed trip energy (kWh)")
    axes[0, 0].set_ylabel("Residual (kWh)")
    axes[1, 0].set_ylabel("Residual (kWh)")
    fig.tight_layout()
    save_png(fig, path)


def fig_error_ecdf(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for method in methods:
        err = np.sort(trip_absolute_errors(pred, method).to_numpy(dtype=float))
        y = np.arange(1, len(err) + 1) / len(err)
        ax.step(err, y, where="post", color=_c(method), ls=METHOD_LINESTYLES.get(method, "-"), lw=2.0, label=label(method))
    ax.set_xlabel("Absolute trip-energy error (kWh)")
    ax.set_ylabel("Fraction of held-out trips with |error| ≤ x")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_png(fig, path)


def fig_trip_error_heatmap(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    err = pred.loc[pred["method"].isin(methods), ["trip_id", "trajectory", "method", "absolute_error_kwh"]]
    trip_err = err.groupby(["trajectory", "trip_id", "method"], as_index=False)["absolute_error_kwh"].mean()
    order = trip_err.sort_values(["trajectory", "trip_id"])["trip_id"].drop_duplicates().tolist()
    mat = trip_err.pivot(index="trip_id", columns="method", values="absolute_error_kwh").reindex(index=order, columns=methods)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    image = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="cividis", vmin=0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label(m) for m in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(list(mat.index), fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute error (kWh)")
    traj = trip_err.drop_duplicates("trip_id").set_index("trip_id").loc[order, "trajectory"]
    boundaries = [i for i in range(1, len(order)) if traj.iloc[i] != traj.iloc[i - 1]]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", lw=2.0)
    fig.tight_layout()
    save_png(fig, path)


def _replicate_values(frame: pd.DataFrame, method: str) -> np.ndarray:
    if frame.empty:
        return np.array([])
    if method == "ridge" and "subset_ids" in frame.columns:
        unique: dict[str, float] = {}
        for _, sub in frame.groupby(["seed", "repeat"], dropna=False):
            signature = str(tuple(sub["subset_ids"].astype(str).tolist()))
            unique[signature] = float(sub["absolute_error_kwh"].mean())
        return np.asarray(list(unique.values()), dtype=float)
    return frame.groupby(["seed", "repeat"], dropna=False)["absolute_error_kwh"].mean().to_numpy(dtype=float)


def fig_data_scarcity_learning_curve(raw: pd.DataFrame, path: Path) -> None:
    from evaluation.final_tables import complete_scarcity_frame

    raw = complete_scarcity_frame(raw)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.35))
    sizes = (3, 5, 8, 12, 16)
    phys = None
    if "physics" in set(raw["method"]):
        phys = float(raw.loc[raw["method"] == "physics", "absolute_error_kwh"].mean())
        for ax in axes:
            ax.axhline(phys, color=_c("physics"), ls="--", lw=1.6, label="Physics Model", zorder=1)

    def _summary(method: str, n: int) -> tuple[float, float, np.ndarray] | None:
        values = _replicate_values(raw.loc[(raw["method"] == method) & (raw["n_train"] == n)], method)
        if values.size == 0:
            return None
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        return mean, std, values

    def _draw(ax, methods, log_scale: bool) -> None:
        plotted_high = []
        for method in methods:
            xs, means, lowers, uppers = [], [], [], []
            for n in sizes:
                summary = _summary(method, n)
                if summary is None:
                    continue
                mean, std, _values = summary
                xs.append(n)
                means.append(mean)
                lowers.append(std if mean > std else 0.0)
                uppers.append(std)
                plotted_high.append(mean + std)
            ax.errorbar(
                xs,
                means,
                yerr=np.vstack([lowers, uppers]),
                color=_c(method),
                marker=METHOD_MARKERS.get(method, "o"),
                ms=9.5,
                ls=METHOD_LINESTYLES.get(method, "-"),
                lw=2.15,
                capsize=3.5,
                elinewidth=1.35,
                label=label(method),
                zorder=3,
            )
            if log_scale and method == "ridge":
                offsets = {3: (-18, 8), 5: (0, 7), 8: (18, 6)}
                align = {3: "right", 5: "center", 8: "left"}
                for n, mean, std in zip(xs, means, uppers):
                    if mean < 1.0 or n not in offsets:
                        continue
                    ax.annotate(
                        f"{mean:.2f}",
                        (n, mean + std),
                        textcoords="offset points",
                        xytext=offsets[n],
                        ha=align[n],
                        va="bottom",
                        fontsize=9.5,
                        color="#222222",
                        annotation_clip=False,
                    )
        ax.set_xlabel("Number of training trips")
        ax.set_xticks(list(sizes))
        if log_scale:
            from matplotlib.ticker import FixedLocator, FuncFormatter

            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
            ax.set_ylabel("Trip-energy MAE (kWh)")
            if plotted_high:
                ax.set_ylim(0.2, max(max(plotted_high) * 1.45, 10))
            ax.text(0.98, 0.97, "log scale", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        else:
            ax.set_ylabel("Trip-energy MAE (kWh)")
            highs = list(plotted_high)
            if phys is not None:
                highs.append(phys)
            top = max(max(highs) * 1.18, 0.85) if highs else 0.85
            ax.set_ylim(0, top)
            if phys is not None:
                ax.annotate(
                    "no training",
                    (sizes[-1], phys),
                    textcoords="offset points",
                    xytext=(-4, 4),
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    color=_c("physics"),
                )

    _draw(axes[0], [m for m in SCARCITY_METHODS if m in set(raw["method"])], True)
    _draw(axes[1], [m for m in ("weak_mlp", "pinn") if m in set(raw["method"])], False)
    axes[0].set_title("(a) All methods", loc="left", fontsize=12, fontweight="bold", pad=2)
    axes[1].set_title("(b) Neural models", loc="left", fontsize=12, fontweight="bold", pad=2)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_png(fig, path)


def fig_pinn_vs_mlp(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    a = trip_absolute_errors(pred, "weak_mlp")
    b = trip_absolute_errors(pred, "pinn")
    aligned = pd.concat([a.rename("mlp"), b.rename("pinn")], axis=1, join="inner").dropna()
    if aligned.empty:
        return
    traj = pred.drop_duplicates("trip_id").set_index("trip_id")["trajectory"]
    aligned["trajectory"] = aligned.index.map(traj)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    hi = float(max(aligned["mlp"].max(), aligned["pinn"].max())) * 1.08
    ax.plot([0, hi], [0, hi], color="0.45", lw=1.3)
    for name, g in aligned.groupby("trajectory"):
        ax.scatter(
            g["mlp"],
            g["pinn"],
            s=52,
            c=TRAJ_COLORS.get(str(name), "#333333"),
            marker=TRAJ_MARKERS.get(str(name), "o"),
            edgecolors="0.15",
            linewidths=0.4,
            label=str(name),
            zorder=3,
        )
    gaps = (aligned["pinn"] - aligned["mlp"]).abs().sort_values(ascending=False)
    for trip_id in gaps.index[:2]:
        if float(gaps.loc[trip_id]) < 0.35:
            continue
        ax.annotate(
            str(trip_id),
            (aligned.loc[trip_id, "mlp"], aligned.loc[trip_id, "pinn"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    ax.text(0.98, 0.55, "Below diagonal:\nlower PINN error", transform=ax.transAxes, ha="right", va="center", fontsize=10)
    ax.set_xlabel("Data-Driven MLP absolute error (kWh)")
    ax.set_ylabel("PINN absolute error (kWh)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(title="Trajectory", loc="upper left")
    fig.tight_layout()
    save_png(fig, path)


def fig06_soc_reconstruction(trips: list[ProcessedTrip], rec_dir: Path, path: Path) -> None:
    apply_style()
    from experiment.data import representative_test_ids

    ids = representative_test_ids(trips)
    by_id = {t.trip_id: t for t in trips}
    chosen = [i for i in ids if i in by_id and (rec_dir / f"{i}__pinn__seed0.npz").exists()]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.4, 2.45 * len(chosen)), sharex=False)
    if len(chosen) == 1:
        axes = [axes]
    for i, (ax, trip_id) in enumerate(zip(axes, chosen)):
        trip = by_id[trip_id]
        tmin = elapsed_seconds(trip.frame["dt_s"].to_numpy(dtype=float)) / 60.0
        curves = []
        for seed in (0, 1, 2):
            file = rec_dir / f"{trip_id}__pinn__seed{seed}.npz"
            if file.exists():
                curves.append(np.load(file)["soc_hat"])
        ax.plot(tmin, trip.frame["soc"], color="0.1", lw=1.8, label="Observed quantized SoC")
        if len(curves) >= 2:
            stacked = np.vstack([c[: min(len(c) for c in curves)] for c in curves])
            t_use = tmin[: stacked.shape[1]]
            mean = stacked.mean(axis=0)
            sd = stacked.std(axis=0, ddof=1)
            ax.fill_between(t_use, mean - sd, mean + sd, color=_c("pinn"), alpha=0.22, linewidth=0)
            ax.plot(t_use, mean, color=_c("pinn"), lw=1.8, label="PINN mean ± 1 SD")
        else:
            ax.plot(tmin[: len(curves[0])], curves[0], color=_c("pinn"), lw=1.8, ls="--", label="PINN seed 0")
        ax.set_ylabel("SoC (%)")
        _tag(ax, LETTERS[i])
        ax.text(0.12, 0.08, f"{trip_id} ({trip.trajectory})", transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("Time (min)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_png(fig, path)


def fig_window_consistency(window_csv: Path, path: Path) -> None:
    if not window_csv.exists():
        return
    frame = pd.read_csv(window_csv)
    if "level" in frame.columns:
        frame = frame.loc[frame["level"] == "scale"]
    if frame.empty or "scale" not in frame.columns:
        return
    apply_style()
    methods = [m for m in ("weak_mlp", "pinn") if m in set(frame["method"])]
    preferred = ["time_60s", "time_120s", "time_300s", "time_600s", "full_trip"]
    labels = {"time_60s": "60 s", "time_120s": "120 s", "time_300s": "300 s", "time_600s": "600 s", "full_trip": "full trip"}
    scales = [s for s in preferred if s in set(frame["scale"])]
    if not scales or not methods:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(scales))
    for method in methods:
        sub = frame.loc[frame["method"] == method]
        if "seed" in sub.columns:
            sub = sub.groupby(["trip_id", "scale"], as_index=False)["mae_kwh"].mean()
        means, stds = [], []
        for scale in scales:
            values = sub.loc[sub["scale"] == scale, "mae_kwh"].to_numpy(dtype=float)
            means.append(float(np.mean(values)) if values.size else np.nan)
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker=METHOD_MARKERS.get(method, "o"),
            ms=8,
            color=_c(method),
            ls=METHOD_LINESTYLES.get(method, "-"),
            lw=1.8,
            capsize=3,
            label=label(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in scales])
    ax.set_ylabel("Window-energy MAE (kWh)")
    ax.set_xlabel("Evaluation window scale")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig05_feasibility(table: pd.DataFrame, path: Path) -> None:
    apply_style()
    if table.empty:
        return
    method_col = "method" if "method" in table.columns else "Method"
    reserve_col = "reserve_soc_pct" if "reserve_soc_pct" in table.columns else "reserve_SoC_pct"
    false_col = "false_safe" if "false_safe" in table.columns else "false_safe_fraction"
    cons_col = "overly_conservative" if "overly_conservative" in table.columns else "overly_conservative_fraction"
    table = table.copy()
    table["_m"] = table[method_col].astype(str).str.lower().map(lambda val: {"physics": "physics", "pinn": "pinn"}.get(val, val))
    methods = [m for m in FEASIBILITY_METHODS if m in set(table["_m"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), sharex=True)
    titles = ["(a) False-safe decisions", "(b) Overly conservative decisions"]
    columns = [false_col, cons_col]
    ylabels = ["False-safe rate (%)", "Overly conservative rate (%)"]
    for ax, title, column, ylabel in zip(axes, titles, columns, ylabels):
        for method in methods:
            g = table.loc[table["_m"] == method].groupby(reserve_col, as_index=False)[column].mean()
            ax.plot(
                g[reserve_col],
                100.0 * g[column],
                marker=METHOD_MARKERS.get(method, "o"),
                ms=8,
                color=_c(method),
                ls=METHOD_LINESTYLES.get(method, "-"),
                lw=1.8,
                label=label(method),
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Reserve SoC (%)")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(0, None)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig_training_convergence(history_csv: Path, path: Path) -> None:
    if history_csv is None or not Path(history_csv).exists():
        return
    hist = pd.read_csv(history_csv)
    if "val_window_mae_kwh" not in hist.columns:
        return
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
    axes[0].set_title("(a)", loc="left", fontsize=12)
    axes[0].plot(hist["epoch"], hist["val_window_mae_kwh"], color=_c("pinn"), lw=2.0)
    best = int(hist["val_window_mae_kwh"].idxmin())
    best_epoch = float(hist.loc[best, "epoch"])
    best_value = float(hist.loc[best, "val_window_mae_kwh"])
    axes[0].axvline(best_epoch, color="0.25", ls="--", lw=1.3)
    axes[0].scatter([best_epoch], [best_value], s=46, color="0.1", zorder=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation multi-window MAE (kWh)")
    series = [
        ("train_loss", "training objective", "#222222"),
        ("w_window", "window", "#0072b2"),
        ("w_dynamics", "dynamics", "#d55e00"),
        ("w_state", "state", "#009e73"),
        ("w_prior", "prior", "#7a7a7a"),
    ]
    for col, name, color in series:
        if col not in hist.columns:
            continue
        y = hist[col].to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)
        if np.isfinite(y).any():
            axes[1].plot(hist["epoch"], y, label=name, color=color, lw=1.6)
    axes[1].set_yscale("log")
    axes[1].set_title("(b)", loc="left", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted training contribution")
    axes[1].text(0.98, 0.04, "Log scale", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    fig.tight_layout()
    save_png(fig, path)


def fig_validation_window_scales(history_csv: Path, path: Path) -> bool:
    if history_csv is None or not Path(history_csv).exists():
        return False
    hist = pd.read_csv(history_csv)
    columns = [
        ("val_mae_time_60s", "60 s"),
        ("val_mae_time_120s", "120 s"),
        ("val_mae_time_300s", "300 s"),
        ("val_mae_time_600s", "600 s"),
        ("val_mae_full_trip", "full trip"),
    ]
    present = [(col, name) for col, name in columns if col in hist.columns and hist[col].notna().any()]
    if not present:
        return False
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#000000", "#0072b2", "#d55e00", "#009e73", "#7a7a7a"]
    for (col, name), color in zip(present, colors):
        ax.plot(hist["epoch"], hist[col], label=name, color=color, lw=1.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation window-energy MAE (kWh)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_png(fig, path)
    return True


def fig_trip_energy_comparison(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    trip = trip_level_predictions(pred)
    methods = [m for m in PARITY_METHODS if m in set(trip["method"])]
    trajectories = [t for t in ("T1", "T2", "T3") if t in set(trip["trajectory"])]
    fig, axes = plt.subplots(1, len(trajectories), figsize=(7.5, 4.3), sharey=True, gridspec_kw={"width_ratios": [max(trip.loc[trip["trajectory"] == t, "trip_id"].nunique(), 1) for t in trajectories]})
    if len(trajectories) == 1:
        axes = [axes]
    offsets = np.linspace(-0.22, 0.22, len(methods) + 1)
    for i, (ax, trajectory) in enumerate(zip(axes, trajectories)):
        sub = trip.loc[trip["trajectory"] == trajectory]
        ids = sorted(sub["trip_id"].unique())
        xpos = {trip_id: k for k, trip_id in enumerate(ids)}
        observed = sub.drop_duplicates("trip_id").set_index("trip_id")
        ax.scatter([xpos[t] + offsets[0] for t in ids], [observed.loc[t, "observed_energy_kwh"] for t in ids], s=42, c="black", marker="o", label="Observed", zorder=4)
        for j, method in enumerate(methods):
            g = sub.loc[sub["method"] == method].set_index("trip_id")
            face = "white" if method == "ridge" else _c(method)
            edge = "black" if method == "ridge" else "0.15"
            ax.scatter(
                [xpos[t] + offsets[j + 1] for t in ids],
                [g.loc[t, "predicted_energy_kwh"] for t in ids],
                s=36,
                c=face,
                marker=METHOD_MARKERS.get(method, "o"),
                edgecolors=edge,
                linewidths=1.1 if method == "ridge" else 0.4,
                label=label(method),
                zorder=3,
            )
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=75, ha="right", fontsize=9)
        ax.set_title(f"({LETTERS[i]}) {trajectory}", loc="left", fontsize=12)
        ax.set_xlabel("Held-out trip")
    axes[0].set_ylabel("Trip energy (kWh)")
    handles, labels_ = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png(fig, path)


def fig_trajectory_mae(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    meta = pred.drop_duplicates("trip_id").set_index("trip_id")["trajectory"]
    trajectories = [t for t in ("T1", "T2", "T3") if t in set(meta)]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    rng = np.random.default_rng(2)
    counts = meta.value_counts()
    for i, trajectory in enumerate(trajectories):
        for method in methods:
            err = trip_absolute_errors(pred, method)
            err = err.loc[[idx for idx in err.index if meta.get(idx) == trajectory]]
            if err.empty:
                continue
            x = np.full(len(err), i, dtype=float) + rng.uniform(-0.16, 0.16, len(err))
            ax.scatter(x, err.to_numpy(), s=28, c=_c(method), marker=METHOD_MARKERS.get(method, "o"), alpha=0.85, zorder=3)
            ax.scatter([i], [float(err.mean())], s=80, c=_c(method), marker=METHOD_MARKERS.get(method, "o"), edgecolors="white", linewidths=0.8, zorder=4, label=label(method) if i == 0 else None)
    ax.set_xticks(range(len(trajectories)))
    ax.set_xticklabels([f"{t}\n(n = {int(counts[t])})" for t in trajectories])
    ax.set_ylabel("Absolute trip-energy error (kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)
