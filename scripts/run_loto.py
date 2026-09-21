#!/usr/bin/env python
"""Leave-one-trip-out energy estimation (Phases 5–8).

Quick debug runs:  python scripts/run_loto.py --config configs/quick.yaml
Do not treat quick-profile numbers as paper results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.discovery import discover_dataset  # noqa: E402
from data.features import trip_level_feature_table  # noqa: E402
from data.loader import load_selected_trips  # noqa: E402
from data.preprocessing import preprocess_trips  # noqa: E402
from data.quantization import combine_quantization, estimate_soc_quantization  # noqa: E402
from data.scaling import TripStandardScaler  # noqa: E402
from data.windows import generate_windows  # noqa: E402
from evaluation.metrics import reconstruct_soc, soc_reconstruction_metrics, summarize_energy_table, trip_energy_row  # noqa: E402
from manifest import write_manifest  # noqa: E402
from models.baselines import constant_consumption_predict, constant_wh_per_km, physics_only_predict  # noqa: E402
from models.elasticnet import TripElasticNet  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from physics.vehicle_model import observed_energy_kwh  # noqa: E402
from plotting.loto import plot_method_comparison, plot_trip_diagnostics  # noqa: E402
from training.cross_validation import leave_one_trip_out, trips_by_id  # noqa: E402
from training.trainer import make_trip_batch, predict_power, train_neural_model  # noqa: E402
from units import energy_kwh_from_power  # noqa: E402


def _print(msg: str) -> None:
    print(msg, flush=True)


def _load_processed(config: dict[str, Any]):
    discovery = discover_dataset(config)
    if discovery.n_selected == 0:
        raise SystemExit(
            "No HELECAR-D analysed trips found. Download the dataset locally "
            "(do not commit it) and re-run."
        )
    raw = load_selected_trips(discovery.selected)
    return preprocess_trips(raw, config)


def _trip_obs(trip, battery_capacity_kwh: float) -> dict[str, float]:
    df = trip.frame
    soc = df["soc"].to_numpy(dtype=float)
    distance_km = float(df["s_can_m"].iloc[-1] / 1000.0) if "s_can_m" in df.columns else 0.0
    duration_s = float(np.nansum(df["dt_s"].to_numpy(dtype=float)))
    e_obs = observed_energy_kwh(float(soc[0]), float(soc[-1]), battery_capacity_kwh)
    return {
        "distance_km": distance_km,
        "duration_s": duration_s,
        "e_obs_kwh": e_obs,
        "soc_start": float(soc[0]),
        "soc_end": float(soc[-1]),
    }


def _window_scale_filter(method: str) -> set[str] | None:
    if method == "pinn_fulltrip":
        return {"full_trip"}
    return None


def _neural_kind(method: str) -> str:
    return "mlp" if method == "weak_mlp" else "pinn"


def run_loto(config: dict[str, Any]) -> pd.DataFrame:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "base"))
    out_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    for sub in ("training_histories", "fold_manifests", "checkpoints", "soc_reconstruction", "diagnostic_figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    seed = int(config.get("experiment", {}).get("seed", 0))
    methods = list(config.get("experiment", {}).get("methods", ["constant", "physics", "elasticnet", "weak_mlp", "pinn"]))
    device = torch.device(str(config.get("experiment", {}).get("device", "cpu")))

    _print(f"=== LOTO profile={profile} seed={seed} methods={methods} ===")
    trips = _load_processed(config)
    params = load_vehicle_parameters(config=config)
    by_id = trips_by_id(trips)
    per_q = [estimate_soc_quantization(t.frame["soc"]) for t in trips]
    q = float(combine_quantization(per_q)["q"])
    _print(f"Trips={len(trips)}  empirical q={q}")

    folds = leave_one_trip_out(trips, seed=seed)
    rows: list[dict[str, Any]] = []
    soc_rows: list[dict[str, Any]] = []
    overlays: dict[str, dict[str, np.ndarray]] = {t.trip_id: {} for t in trips}

    for fold in folds:
        test = by_id[fold.test_id]
        outer_train = [by_id[i] for i in fold.train_ids]
        inner_train = [by_id[i] for i in fold.inner_train_ids]
        val = by_id[fold.val_id]
        obs = _trip_obs(test, params.battery_capacity_kwh)
        fold_info = {
            "test_id": fold.test_id,
            "train_ids": list(fold.train_ids),
            "inner_train_ids": list(fold.inner_train_ids),
            "val_id": fold.val_id,
            "scaler_trip_ids": [],
            "methods": {},
        }
        _print(f"\nFold test={fold.test_id}  val={fold.val_id}  n_train={len(outer_train)}")

        if "constant" in methods:
            mean_whkm = constant_wh_per_km(outer_train, params.battery_capacity_kwh)
            pred = constant_consumption_predict(mean_whkm, obs["distance_km"])
            rows.append(trip_energy_row(test.trip_id, test.trajectory, "constant", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred))
            fold_info["methods"]["constant"] = {"mean_wh_per_km": mean_whkm, "predicted_energy_kwh": pred}

        if "physics" in methods:
            phys = physics_only_predict(test, params, apply_bounds=bool(config.get("physics", {}).get("apply_power_bounds", False)))
            rows.append(trip_energy_row(test.trip_id, test.trajectory, "physics", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], phys["energy_kwh"]))
            overlays[test.trip_id]["p_physics"] = phys["p_battery_kw"]
            rec = reconstruct_soc(obs["soc_start"], phys["p_battery_kw"], test.frame["dt_s"].to_numpy(dtype=float), params.battery_capacity_kwh)
            sm = soc_reconstruction_metrics(test.frame["soc"].to_numpy(dtype=float), rec["soc_hat"], q, rec["soc_end_hat"])
            soc_rows.append({"trip_id": test.trip_id, "method": "physics", **sm})
            fold_info["methods"]["physics"] = {"predicted_energy_kwh": phys["energy_kwh"]}

        if "elasticnet" in methods:
            train_table = trip_level_feature_table(outer_train, battery_capacity_kwh=params.battery_capacity_kwh)
            test_table = trip_level_feature_table([test], battery_capacity_kwh=params.battery_capacity_kwh)
            model = TripElasticNet(random_state=seed).fit(train_table)
            model.assert_not_fitted_on(test.trip_id)
            pred = float(model.predict(test_table)[0])
            rows.append(trip_energy_row(test.trip_id, test.trajectory, "elasticnet", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred))
            fold_info["methods"]["elasticnet"] = {"best_params": model.best_params_, "predicted_energy_kwh": pred}

        neural_methods = [m for m in methods if m in {"weak_mlp", "pinn", "pinn_fulltrip"}]
        if neural_methods:
            scaler = TripStandardScaler().fit(inner_train)
            scaler.assert_not_fitted_on(test.trip_id)
            scaler.assert_not_fitted_on(val.trip_id)
            fold_info["scaler_trip_ids"] = list(scaler.fitted_trip_ids)
            train_windows = generate_windows(inner_train, params.battery_capacity_kwh, q, config=config, overlapping=True)
            val_windows = generate_windows([val], params.battery_capacity_kwh, q, config=config, overlapping=False)
            if any(w.trip_id == test.trip_id for w in train_windows + val_windows):
                raise AssertionError("Test-trip windows leaked into training/validation.")

            for method in neural_methods:
                scale_filter = _window_scale_filter(method)
                kind = _neural_kind(method)
                tr_batches = [
                    make_trip_batch(
                        t,
                        scaler.transform_trip(t),
                        params,
                        config,
                        train_windows,
                        device,
                        include_state=(kind == "pinn"),
                        window_scales=scale_filter,
                    )
                    for t in inner_train
                ]
                va_batches = [
                    make_trip_batch(
                        val,
                        scaler.transform_trip(val),
                        params,
                        config,
                        val_windows,
                        device,
                        include_state=(kind == "pinn"),
                        window_scales=scale_filter,
                    )
                ]
                for b in tr_batches:
                    b.assert_no_soc_input()
                trained = train_neural_model(
                    kind,
                    tr_batches,
                    va_batches,
                    config,
                    params.battery_capacity_kwh,
                    q,
                    seed,
                    lambda_prior=float(config.get("training", {}).get("lambda_prior", 0.1)),
                )
                hist = pd.DataFrame(trained["history"])
                hist.to_csv(out_dir / "training_histories" / f"{fold.test_id}__{method}.csv", index=False)
                torch.save(trained["model"].state_dict(), out_dir / "checkpoints" / f"{fold.test_id}__{method}.pt")

                test_batch = make_trip_batch(
                    test,
                    scaler.transform_trip(test),
                    params,
                    config,
                    [],
                    device,
                    include_state=False,
                    window_scales=scale_filter,
                )
                test_batch.assert_no_soc_input()
                pred_arr = predict_power(trained["model"], test_batch, kind)
                e_hat = energy_kwh_from_power(pred_arr["p_hat"], test.frame["dt_s"].to_numpy(dtype=float))
                rows.append(
                    trip_energy_row(
                        test.trip_id,
                        test.trajectory,
                        method,
                        seed,
                        obs["distance_km"],
                        obs["duration_s"],
                        obs["e_obs_kwh"],
                        e_hat,
                    )
                )
                rec = reconstruct_soc(
                    obs["soc_start"],
                    pred_arr["p_hat"],
                    test.frame["dt_s"].to_numpy(dtype=float),
                    params.battery_capacity_kwh,
                )
                sm = soc_reconstruction_metrics(test.frame["soc"].to_numpy(dtype=float), rec["soc_hat"], q, rec["soc_end_hat"])
                soc_rows.append(
                    {
                        "trip_id": test.trip_id,
                        "method": method,
                        "saturation_frac": pred_arr["saturation_frac"],
                        "p_hat_min": float(np.min(pred_arr["p_hat"])),
                        "p_hat_max": float(np.max(pred_arr["p_hat"])),
                        **sm,
                    }
                )
                np.savez(
                    out_dir / "soc_reconstruction" / f"{fold.test_id}__{method}.npz",
                    soc_hat=rec["soc_hat"],
                    p_hat=pred_arr["p_hat"],
                    delta_p=pred_arr["delta_p"],
                    d_hat=pred_arr["d_hat"],
                    p_phy=pred_arr["p_phy"],
                )
                if method == "weak_mlp":
                    overlays[test.trip_id]["p_mlp"] = pred_arr["p_hat"]
                if method == "pinn":
                    overlays[test.trip_id]["p_pinn"] = pred_arr["p_hat"]
                    overlays[test.trip_id]["delta_pinn"] = pred_arr["delta_p"]
                    overlays[test.trip_id]["d_hat"] = pred_arr["d_hat"]
                    overlays[test.trip_id]["soc_hat_pinn"] = rec["soc_hat"]
                fold_info["methods"][method] = {
                    "best_epoch": trained["best_epoch"],
                    "best_val_mae": trained["best_val_mae"],
                    "predicted_energy_kwh": e_hat,
                    "saturation_frac": pred_arr["saturation_frac"],
                    "lambdas": trained["lambdas"],
                }
                _print(
                    f"  {method}: E_hat={e_hat:.4f} kWh  E_obs={obs['e_obs_kwh']:.4f}  "
                    f"sat={pred_arr['saturation_frac']:.3f}  best_epoch={trained['best_epoch']}"
                )

        (out_dir / "fold_manifests" / f"{fold.test_id}.json").write_text(json.dumps(fold_info, indent=2, default=str), encoding="utf-8")

    pred_table = pd.DataFrame(rows)
    pred_table.to_csv(out_dir / "per_trip_predictions.csv", index=False)
    summary_map = summarize_energy_table(pred_table)
    summary = pd.DataFrame(
        [{"method": m, **vals} for m, vals in summary_map.items()]
    )
    if not summary.empty:
        order = {name: i for i, name in enumerate(methods)}
        summary["_ord"] = summary["method"].map(lambda m: order.get(m, 99))
        summary = summary.sort_values("_ord").drop(columns="_ord")
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    if soc_rows:
        pd.DataFrame(soc_rows).to_csv(out_dir / "soc_reconstruction_metrics.csv", index=False)

    _print("\n=== Quick LOTO summary (debug, not paper numbers) ===")
    if not summary.empty:
        show = summary.copy()
        _print(f"{'Method':<16} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Bias':>8}")
        _print("-" * 52)
        for _, r in show.iterrows():
            _print(f"{r['method']:<16} {r['mae_kwh']:8.4f} {r['rmse_kwh']:8.4f} {r['mape_pct']:8.2f} {r['bias_kwh']:8.4f}")
        plot_method_comparison(summary, out_dir / "diagnostic_figures" / "mae_by_method.png")

    _print("\n=== Per-trip absolute errors (kWh) ===")
    if not pred_table.empty:
        piv = pred_table.pivot(index="trip_id", columns="method", values="absolute_error_kwh")
        _print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
        piv.to_csv(out_dir / "per_trip_abs_error_kwh.csv")

    for trip in trips:
        arr = overlays.get(trip.trip_id, {})
        soc_hat = arr.get("soc_hat_pinn", np.full(trip.n_rows, np.nan))
        plot_trip_diagnostics(trip, arr, soc_hat, out_dir / "diagnostic_figures" / f"overlay_{trip.trip_id}.png")

    write_manifest(
        out_dir / "manifest.json",
        {
            "stage": "loto",
            "profile": profile,
            "seed": seed,
            "methods": methods,
            "n_trips": len(trips),
            "trip_ids": [t.trip_id for t in trips],
            "q": q,
            "note": "Quick-profile metrics are pipeline diagnostics, not paper results.",
        },
    )
    _print(f"\nWrote LOTO outputs to {out_dir}")
    return pred_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-trip-out EV energy experiment.")
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_loto(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
