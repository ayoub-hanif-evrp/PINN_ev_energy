"""Leave-one-trip-out energy estimation runner.

Per-fold q is estimated on training trips only. Neural models are selected on
inner_train + grouped validation, discarded, then retrained on all outer-training
trips for the selected number of epochs. Test-trip SoC is never a network input.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.feature_audit import assert_feature_units_and_coverage, feature_distribution_table
from data.features import trip_level_feature_table
from data.preprocessing import ProcessedTrip
from data.quantization import q_from_trips
from data.scaling import ProgressScaler, TripStandardScaler
from data.schema import MAIN_MODEL_FEATURES, PROGRESS_FEATURES
from data.windows import generate_windows
from evaluation.metrics import reconstruct_soc, soc_reconstruction_metrics, summarize_energy_table, trip_energy_row
from evaluation.window_eval import nonoverlapping_window_metrics
from experiment.cache import cache_fingerprint, default_cache_payload, load_cache, save_cache
from experiment.data import dataset_file_hashes, load_processed, representative_test_ids
from experiment.device import resolve_device, set_seeds
from manifest import git_commit, package_versions, write_manifest
from models.baselines import constant_consumption_predict, constant_wh_per_km, physics_only_predict
from models.elasticnet import TripElasticNet
from paths import project_root, resolve_under_root
from physics.parameters import VehicleParameters, load_vehicle_parameters
from physics.vehicle_model import observed_energy_kwh
from plotting.loto import plot_method_comparison, plot_trip_diagnostics
from training.cross_validation import leave_one_trip_out, trips_by_id
from training.trainer import (
    make_inference_batch,
    make_trip_batch,
    method_kind,
    predict_power,
    train_neural_model,
    uses_state,
    window_scale_filter,
)
from units import energy_kwh_from_power

NEURAL_METHODS = ("weak_mlp", "mlp_state", "pinn_no_dynamics", "pinn_fulltrip", "pinn")


def _print(msg: str) -> None:
    print(msg, flush=True)


def trip_obs(trip: ProcessedTrip, battery_capacity_kwh: float) -> dict[str, float]:
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


def _assert_no_overlap(test_id: str, *id_lists: list[str] | tuple[str, ...]) -> None:
    for ids in id_lists:
        if test_id in set(ids):
            raise AssertionError(f"Leakage: {test_id} found in {list(ids)}")


def _assert_train_windows(windows, forbidden: set[str]) -> None:
    leaked = {w.trip_id for w in windows} & forbidden
    if leaked:
        raise AssertionError(f"Window leakage from trips {leaked}")


def _pinn_diagnostics(pred: dict[str, np.ndarray], battery_capacity_kwh: float, soc_obs: np.ndarray) -> dict[str, float]:
    p_hat = np.asarray(pred["p_hat"], dtype=float)
    delta = np.asarray(pred["delta_p"], dtype=float)
    d_hat = np.asarray(pred.get("d_hat"), dtype=float)
    out = {
        "power_min": float(np.min(p_hat)) if p_hat.size else float("nan"),
        "power_max": float(np.max(p_hat)) if p_hat.size else float("nan"),
        "saturation_fraction": float(pred.get("saturation_frac", float("nan"))),
        "mean_abs_deltaP": float(np.mean(np.abs(delta))) if delta.size else float("nan"),
        "max_abs_deltaP": float(np.max(np.abs(delta))) if delta.size else float("nan"),
        "state_mae_pp": float("nan"),
        "dynamics_residual_mean": float("nan"),
        "dynamics_residual_rmse": float("nan"),
    }
    if d_hat.size and np.isfinite(d_hat).any() and soc_obs.size:
        d_obs = soc_obs[0] - soc_obs[: d_hat.size]
        finite = np.isfinite(d_hat) & np.isfinite(d_obs)
        if finite.any():
            out["state_mae_pp"] = float(np.mean(np.abs(d_hat[finite] - d_obs[finite])))
    return out


def _dynamics_from_arrays(p_hat: np.ndarray, d_hat: np.ndarray, dt: np.ndarray, e_batt: float) -> tuple[float, float]:
    if p_hat.size < 2 or d_hat.size < 2 or not np.isfinite(d_hat).any():
        return float("nan"), float("nan")
    de = np.zeros_like(p_hat)
    de[:-1] = 0.5 * (p_hat[:-1] + p_hat[1:]) * dt[:-1] / 3600.0
    r = d_hat[1:] - d_hat[:-1] - (100.0 / e_batt) * de[:-1]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(r)), float(np.sqrt(np.mean(r**2)))


def _neural_row(
    test: ProcessedTrip,
    method: str,
    seed: int,
    obs: dict[str, float],
    e_hat: float,
    pred: dict[str, np.ndarray],
    selected_epoch: int,
    retrain_epochs: int,
    val_score: float,
    params: VehicleParameters,
) -> dict[str, Any]:
    soc = test.frame["soc"].to_numpy(dtype=float)
    diag = _pinn_diagnostics(pred, params.battery_capacity_kwh, soc)
    mean_r, rmse_r = _dynamics_from_arrays(
        pred["p_hat"], pred["d_hat"], test.frame["dt_s"].to_numpy(dtype=float), params.battery_capacity_kwh
    )
    row = trip_energy_row(
        test.trip_id,
        test.trajectory,
        method,
        seed,
        obs["distance_km"],
        obs["duration_s"],
        obs["e_obs_kwh"],
        e_hat,
    )
    row.update(
        {
            "selected_epoch": int(selected_epoch),
            "final_retrain_epochs": int(retrain_epochs),
            "validation_score": float(val_score),
            **diag,
            "dynamics_residual_mean": mean_r,
            "dynamics_residual_rmse": rmse_r,
        }
    )
    return row


def run_loto(
    config: dict[str, Any],
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    test_ids: list[str] | None = None,
    output_subdir: str | None = None,
) -> pd.DataFrame:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "base"))
    out_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root)
    out_dir = out_dir / (output_subdir or profile)
    for sub in (
        "training_histories",
        "fold_manifests",
        "checkpoints",
        "soc_reconstruction",
        "diagnostic_figures",
        "loss_diagnostics",
        "window_eval",
        "failures",
        "cache",
    ):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    seeds = list(seeds if seeds is not None else config.get("experiment", {}).get("seeds", [config.get("experiment", {}).get("seed", 0)]))
    methods = list(methods if methods is not None else config.get("experiment", {}).get("methods", ["constant", "physics", "elasticnet", "weak_mlp", "pinn"]))
    device = resolve_device(config)
    config.setdefault("experiment", {})["resolved_device"] = str(device)
    retrain_outer = bool(config.get("experiment", {}).get("retrain_outer", True))
    use_cache = bool(config.get("experiment", {}).get("cache", True))

    _print(f"=== LOTO profile={profile} seeds={seeds} methods={methods} device={device} ===")
    trips = load_processed(config)
    feat_table = feature_distribution_table(trips)
    feat_table.to_csv(out_dir / "feature_distributions.csv", index=False)
    assert_feature_units_and_coverage(feat_table)
    params = load_vehicle_parameters(config=config)
    by_id = trips_by_id(trips)
    hashes = dataset_file_hashes(trips)
    git_sha = git_commit()

    requested = config.get("experiment", {}).get("test_trip_ids")
    if test_ids is None and requested:
        if requested == "smoke_representative":
            test_ids = representative_test_ids(trips)
        else:
            test_ids = list(requested)
    if test_ids:
        missing = [i for i in test_ids if i not in by_id]
        if missing:
            raise SystemExit(f"Requested test trips not found: {missing}")

    rows: list[dict[str, Any]] = []
    soc_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    overlays: dict[str, dict[str, np.ndarray]] = {t.trip_id: {} for t in trips}
    failures: list[dict[str, Any]] = []

    for seed in seeds:
        set_seeds(int(seed))
        folds = leave_one_trip_out(trips, seed=int(seed))
        if test_ids:
            folds = [f for f in folds if f.test_id in set(test_ids)]
        for fold in folds:
            test = by_id[fold.test_id]
            outer_train = [by_id[i] for i in fold.train_ids]
            inner_train = [by_id[i] for i in fold.inner_train_ids]
            val_trips = [by_id[i] for i in fold.val_ids]
            _assert_no_overlap(fold.test_id, fold.train_ids, fold.inner_train_ids, fold.val_ids)
            if set(fold.val_ids) & set(fold.inner_train_ids):
                raise AssertionError("validation IDs overlap inner training IDs")

            q_inner = q_from_trips(inner_train) if inner_train else q_from_trips(outer_train)
            q_outer = q_from_trips(outer_train)
            q_test_posthoc = q_from_trips([test])
            obs = trip_obs(test, params.battery_capacity_kwh)
            fold_info: dict[str, Any] = {
                "git_sha": git_sha,
                "profile": profile,
                "seed": int(seed),
                "test_id": fold.test_id,
                "train_ids": list(fold.train_ids),
                "inner_train_ids": list(fold.inner_train_ids),
                "val_ids": list(fold.val_ids),
                "q_inner_train": q_inner,
                "q_outer_train": q_outer,
                "q_test_posthoc": q_test_posthoc,
                "q_test_posthoc_note": "Evaluation-only. Never used to control training, windows, or Smooth-L1 scale.",
                "feature_list": list(MAIN_MODEL_FEATURES),
                "progress_features": list(PROGRESS_FEATURES),
                "vehicle_parameters": params.as_dict(),
                "window_settings": config.get("windows", {}),
                "device": str(device),
                "python_version": package_versions()["python"],
                "torch_version": package_versions().get("torch", "not-installed"),
                "dataset_hashes": {k: hashes.get(k) for k in [fold.test_id, *fold.train_ids]},
                "methods": {},
            }
            _print(
                f"\nFold test={fold.test_id} seed={seed} val={list(fold.val_ids)} "
                f"n_outer={len(outer_train)} n_inner={len(inner_train)} "
                f"q_inner={q_inner:.4g} q_outer={q_outer:.4g} q_test_posthoc={q_test_posthoc:.4g}"
            )

            if "constant" in methods:
                mean_whkm = constant_wh_per_km(outer_train, params.battery_capacity_kwh)
                pred_e = constant_consumption_predict(mean_whkm, obs["distance_km"])
                rows.append(trip_energy_row(test.trip_id, test.trajectory, "constant", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred_e))
                fold_info["methods"]["constant"] = {"mean_wh_per_km": mean_whkm, "predicted_energy_kwh": pred_e}

            if "physics" in methods:
                phys = physics_only_predict(test, params, apply_bounds=bool(config.get("physics", {}).get("apply_power_bounds", False)))
                rows.append(trip_energy_row(test.trip_id, test.trajectory, "physics", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], phys["energy_kwh"]))
                overlays[test.trip_id]["p_physics"] = phys["p_battery_kw"]
                rec = reconstruct_soc(obs["soc_start"], phys["p_battery_kw"], test.frame["dt_s"].to_numpy(dtype=float), params.battery_capacity_kwh)
                sm = soc_reconstruction_metrics(test.frame["soc"].to_numpy(dtype=float), rec["soc_hat"], q_test_posthoc, rec["soc_end_hat"])
                soc_rows.append({"trip_id": test.trip_id, "method": "physics", "seed": seed, **sm})
                fold_info["methods"]["physics"] = {"predicted_energy_kwh": phys["energy_kwh"]}

            if "elasticnet" in methods:
                train_table = trip_level_feature_table(outer_train, battery_capacity_kwh=params.battery_capacity_kwh)
                test_table = trip_level_feature_table([test], battery_capacity_kwh=params.battery_capacity_kwh)
                enet = TripElasticNet(random_state=seed).fit(train_table)
                enet.assert_not_fitted_on(test.trip_id)
                pred_e = float(enet.predict(test_table)[0])
                rows.append(trip_energy_row(test.trip_id, test.trajectory, "elasticnet", seed, obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred_e))
                fold_info["methods"]["elasticnet"] = {
                    "best_params": enet.best_params_,
                    "predicted_energy_kwh": pred_e,
                    "fitted_trip_ids": list(enet.fitted_trip_ids),
                }

            neural_methods = [m for m in methods if m in NEURAL_METHODS]
            for method in neural_methods:
                kind = method_kind(method)
                scale_filter = window_scale_filter(method)
                try:
                    result = _run_neural_fold(
                        method=method,
                        kind=kind,
                        scale_filter=scale_filter,
                        test=test,
                        outer_train=outer_train,
                        inner_train=inner_train,
                        val_trips=val_trips,
                        fold=fold,
                        q_inner=q_inner,
                        q_outer=q_outer,
                        params=params,
                        config=config,
                        device=device,
                        seed=int(seed),
                        retrain_outer=retrain_outer,
                        use_cache=use_cache,
                        out_dir=out_dir,
                        git_sha=git_sha,
                        profile=profile,
                    )
                except Exception as exc:  # noqa: BLE001 — record and continue so the paper fold is accounted for
                    tb = traceback.format_exc()
                    failures.append(
                        {
                            "trip_id": fold.test_id,
                            "seed": int(seed),
                            "model": method,
                            "exception": f"{type(exc).__name__}: {exc}",
                            "traceback": tb,
                        }
                    )
                    (out_dir / "failures" / f"{fold.test_id}__{method}__seed{seed}.txt").write_text(tb, encoding="utf-8")
                    _print(f"  FAIL {method} {fold.test_id} seed={seed}: {exc}")
                    continue

                pred = result["pred"]
                e_hat = result["e_hat"]
                row = _neural_row(
                    test,
                    method,
                    seed,
                    obs,
                    e_hat,
                    pred,
                    result["selected_epoch"],
                    result["retrain_epochs"],
                    result["validation_score"],
                    params,
                )
                rows.append(row)
                rec = reconstruct_soc(
                    obs["soc_start"],
                    pred["p_hat"],
                    test.frame["dt_s"].to_numpy(dtype=float),
                    params.battery_capacity_kwh,
                )
                sm = soc_reconstruction_metrics(
                    test.frame["soc"].to_numpy(dtype=float), rec["soc_hat"], q_test_posthoc, rec["soc_end_hat"]
                )
                soc_rows.append({"trip_id": test.trip_id, "method": method, "seed": seed, "saturation_frac": pred["saturation_frac"], **sm})
                np.savez(
                    out_dir / "soc_reconstruction" / f"{fold.test_id}__{method}__seed{seed}.npz",
                    soc_hat=rec["soc_hat"],
                    p_hat=pred["p_hat"],
                    delta_p=pred["delta_p"],
                    d_hat=pred["d_hat"],
                    p_phy=pred["p_phy"],
                )
                win_m = nonoverlapping_window_metrics(
                    test, pred["p_hat"], params.battery_capacity_kwh, q_outer, config
                )
                for item in win_m:
                    window_rows.append({"trip_id": test.trip_id, "method": method, "seed": seed, **item})
                if method == "weak_mlp":
                    overlays[test.trip_id]["p_mlp"] = pred["p_hat"]
                if method == "pinn":
                    overlays[test.trip_id]["p_pinn"] = pred["p_hat"]
                    overlays[test.trip_id]["delta_pinn"] = pred["delta_p"]
                    overlays[test.trip_id]["d_hat"] = pred["d_hat"]
                    overlays[test.trip_id]["soc_hat_pinn"] = rec["soc_hat"]
                fold_info["methods"][method] = {
                    "selected_epoch": result["selected_epoch"],
                    "final_retrain_epochs": result["retrain_epochs"],
                    "validation_score": result["validation_score"],
                    "validation_mae_by_scale": result["val_mae_by_scale"],
                    "predicted_energy_kwh": e_hat,
                    "saturation_frac": pred["saturation_frac"],
                    "lambdas": result["lambdas"],
                    "scaler_trip_ids_inner": result["scaler_ids_inner"],
                    "scaler_trip_ids_outer": result["scaler_ids_outer"],
                    "cached": result["cached"],
                }
                _print(
                    f"  {method}: E_hat={e_hat:.4f} kWh  E_obs={obs['e_obs_kwh']:.4f}  "
                    f"sat={pred['saturation_frac']:.3f}  sel_epoch={result['selected_epoch']}  "
                    f"retrain={result['retrain_epochs']}"
                )

            (out_dir / "fold_manifests" / f"{fold.test_id}__seed{seed}.json").write_text(
                json.dumps(fold_info, indent=2, default=str), encoding="utf-8"
            )

    pred_table = pd.DataFrame(rows)
    pred_table.to_csv(out_dir / "per_trip_predictions.csv", index=False)
    from evaluation.bootstrap import per_seed_metrics, seed_mean_std
    from evaluation.statistics import write_statistical_outputs

    seed_table = per_seed_metrics(pred_table) if not pred_table.empty else pd.DataFrame()
    if not seed_table.empty and int(seed_table["seed"].nunique()) > 1:
        summary = seed_mean_std(seed_table).rename(columns={"mae_kwh_mean": "mae_kwh", "rmse_kwh_mean": "rmse_kwh", "mape_pct_mean": "mape_pct", "wape_pct_mean": "wape_pct", "bias_kwh_mean": "bias_kwh"})
    else:
        summary_map = summarize_energy_table(pred_table)
        summary = pd.DataFrame([{"method": m, **vals} for m, vals in summary_map.items()])
    if not summary.empty and "method" in summary.columns:
        order = {name: i for i, name in enumerate(methods)}
        summary["_ord"] = summary["method"].map(lambda m: order.get(m, 99))
        summary = summary.sort_values("_ord").drop(columns="_ord")
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    if not pred_table.empty:
        write_statistical_outputs(pred_table, out_dir / "statistics")
    if soc_rows:
        pd.DataFrame(soc_rows).to_csv(out_dir / "soc_reconstruction_metrics.csv", index=False)
    if window_rows:
        pd.DataFrame(window_rows).to_csv(out_dir / "window_eval" / "nonoverlapping_test_windows.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures" / "failures.csv", index=False)

    _print(f"\n=== LOTO summary profile={profile} (not paper numbers unless profile=paper) ===")
    if not summary.empty:
        _print(f"{'Method':<18} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'WAPE':>8} {'Bias':>8}")
        _print("-" * 66)
        for _, r in summary.iterrows():
            _print(
                f"{r['method']:<18} {r['mae_kwh']:8.4f} {r['rmse_kwh']:8.4f} "
                f"{r.get('mape_pct', float('nan')):8.2f} {r.get('wape_pct', float('nan')):8.2f} {r['bias_kwh']:8.4f}"
            )
        plot_method_comparison(summary, out_dir / "diagnostic_figures" / "mae_by_method.png")

    if not pred_table.empty and "absolute_error_kwh" in pred_table.columns:
        try:
            piv = pred_table.pivot_table(index="trip_id", columns="method", values="absolute_error_kwh", aggfunc="mean")
            _print("\n=== Per-trip absolute errors (kWh, mean over seeds) ===")
            _print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
            piv.to_csv(out_dir / "per_trip_abs_error_kwh.csv")
        except ValueError:
            pred_table.to_csv(out_dir / "per_trip_abs_error_kwh.csv", index=False)

    for trip in trips:
        if test_ids and trip.trip_id not in set(test_ids):
            continue
        arr = overlays.get(trip.trip_id, {})
        soc_hat = arr.get("soc_hat_pinn", np.full(trip.n_rows, np.nan))
        plot_trip_diagnostics(trip, arr, soc_hat, out_dir / "diagnostic_figures" / f"overlay_{trip.trip_id}.png")

    write_manifest(
        out_dir / "manifest.json",
        {
            "stage": "loto",
            "profile": profile,
            "seeds": seeds,
            "methods": methods,
            "n_trips": len(trips),
            "trip_ids": [t.trip_id for t in trips],
            "test_ids_filter": test_ids,
            "device": str(device),
            "retrain_outer": retrain_outer,
            "n_failures": len(failures),
            "note": "Smoke/quick metrics are pipeline diagnostics, not paper results."
            if profile != "paper"
            else "Frozen paper LOTO.",
        },
    )
    _print(f"\nWrote LOTO outputs to {out_dir}")
    if failures:
        _print(f"WARNING: {len(failures)} neural fold(s) failed. See {out_dir / 'failures'}")
    return pred_table


def _run_neural_fold(
    *,
    method: str,
    kind: str,
    scale_filter: set[str] | None,
    test: ProcessedTrip,
    outer_train: list[ProcessedTrip],
    inner_train: list[ProcessedTrip],
    val_trips: list[ProcessedTrip],
    fold: Any,
    q_inner: float,
    q_outer: float,
    params: VehicleParameters,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    retrain_outer: bool,
    use_cache: bool,
    out_dir: Path,
    git_sha: str | None,
    profile: str,
) -> dict[str, Any]:
    payload = default_cache_payload(
        profile=profile,
        method=method,
        seed=seed,
        test_id=fold.test_id,
        train_ids=list(fold.train_ids),
        val_ids=list(fold.val_ids),
        q_inner=q_inner,
        q_outer=q_outer,
        feature_list=list(MAIN_MODEL_FEATURES),
        progress_features=list(PROGRESS_FEATURES),
        physics=config.get("physics", {}),
        windows=config.get("windows", {}),
        training=config.get("training", {}),
        git_sha=git_sha,
        extra={"retrain_outer": retrain_outer, "kind": kind},
    )
    fp = cache_fingerprint(payload)
    cached = load_cache(out_dir / "cache" / f"{fp}.json") if use_cache else None
    if cached and "p_hat" in cached:
        pred = {
            "p_hat": np.asarray(cached["p_hat"], dtype=float),
            "p_phy": np.asarray(cached["p_phy"], dtype=float),
            "delta_p": np.asarray(cached["delta_p"], dtype=float),
            "d_hat": np.asarray(cached["d_hat"], dtype=float),
            "saturation_frac": float(cached.get("saturation_frac", float("nan"))),
        }
        return {
            "pred": pred,
            "e_hat": float(cached["e_hat"]),
            "selected_epoch": int(cached["selected_epoch"]),
            "retrain_epochs": int(cached["retrain_epochs"]),
            "validation_score": float(cached.get("validation_score", float("nan"))),
            "val_mae_by_scale": cached.get("val_mae_by_scale", {}),
            "lambdas": cached.get("lambdas", {}),
            "scaler_ids_inner": cached.get("scaler_ids_inner", []),
            "scaler_ids_outer": cached.get("scaler_ids_outer", []),
            "cached": True,
        }

    include_state = uses_state(kind)
    sel_trips = inner_train if inner_train else outer_train
    q_sel = q_inner if inner_train else q_outer
    x_scaler_inner = TripStandardScaler().fit(sel_trips)
    p_scaler_inner = ProgressScaler().fit(sel_trips)
    x_scaler_inner.assert_not_fitted_on(test.trip_id)
    p_scaler_inner.assert_not_fitted_on(test.trip_id)
    for v in val_trips:
        if v.trip_id not in {t.trip_id for t in sel_trips}:
            x_scaler_inner.assert_not_fitted_on(v.trip_id)
            p_scaler_inner.assert_not_fitted_on(v.trip_id)

    train_windows = generate_windows(sel_trips, params.battery_capacity_kwh, q_sel, config=config, overlapping=True)
    val_windows = generate_windows(val_trips, params.battery_capacity_kwh, q_sel, config=config, overlapping=False)
    _assert_train_windows(train_windows + val_windows, {test.trip_id})
    if not train_windows:
        raise RuntimeError(f"No training windows for {method} on fold {test.trip_id}")

    tr_batches = [
        make_trip_batch(
            t,
            x_scaler_inner.transform_trip(t),
            params,
            config,
            train_windows,
            device,
            include_state=include_state,
            window_scales=scale_filter,
            progress_scaled=p_scaler_inner.transform_trip(t),
        )
        for t in sel_trips
    ]
    va_batches = [
        make_trip_batch(
            v,
            x_scaler_inner.transform_trip(v),
            params,
            config,
            val_windows,
            device,
            include_state=include_state,
            window_scales=scale_filter,
            progress_scaled=p_scaler_inner.transform_trip(v),
        )
        for v in val_trips
    ]
    selected = train_neural_model(
        kind,
        tr_batches,
        va_batches,
        config,
        params.battery_capacity_kwh,
        q_sel,
        seed,
        early_stopping=True,
    )
    hist_sel = pd.DataFrame(selected["history"])
    (out_dir / "training_histories").mkdir(parents=True, exist_ok=True)
    (out_dir / "loss_diagnostics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    hist_sel.to_csv(out_dir / "training_histories" / f"{test.trip_id}__{method}__seed{seed}__select.csv", index=False)
    selected_epoch = int(selected["best_epoch"])
    val_score = float(selected["best_val_mae"])
    val_mae_by_scale = {
        k.replace("val_mae_", ""): v
        for k, v in (selected["history"][selected_epoch] if selected["history"] else {}).items()
        if str(k).startswith("val_mae_")
    }

    if retrain_outer:
        x_scaler = TripStandardScaler().fit(outer_train)
        p_scaler = ProgressScaler().fit(outer_train)
        x_scaler.assert_not_fitted_on(test.trip_id)
        p_scaler.assert_not_fitted_on(test.trip_id)
        outer_windows = generate_windows(outer_train, params.battery_capacity_kwh, q_outer, config=config, overlapping=True)
        _assert_train_windows(outer_windows, {test.trip_id})
        retrain_epochs = max(int(selected_epoch) + 1, 1)
        outer_batches = [
            make_trip_batch(
                t,
                x_scaler.transform_trip(t),
                params,
                config,
                outer_windows,
                device,
                include_state=include_state,
                window_scales=scale_filter,
                progress_scaled=p_scaler.transform_trip(t),
            )
            for t in outer_train
        ]
        retrained = train_neural_model(
            kind,
            outer_batches,
            [],
            config,
            params.battery_capacity_kwh,
            q_outer,
            seed,
            max_epochs=retrain_epochs,
            early_stopping=False,
        )
        model = retrained["model"]
        lambdas = retrained["lambdas"]
        hist_re = pd.DataFrame(retrained["history"])
        hist_re.to_csv(out_dir / "training_histories" / f"{test.trip_id}__{method}__seed{seed}__retrain.csv", index=False)
        hist_re.to_csv(out_dir / "loss_diagnostics" / f"{test.trip_id}__{method}__seed{seed}.csv", index=False)
        scaler_outer_ids = list(x_scaler.fitted_trip_ids)
    else:
        model = selected["model"]
        lambdas = selected["lambdas"]
        retrain_epochs = int(selected["n_epochs_run"])
        x_scaler = x_scaler_inner
        p_scaler = p_scaler_inner
        scaler_outer_ids = list(x_scaler.fitted_trip_ids)
        hist_sel.to_csv(out_dir / "loss_diagnostics" / f"{test.trip_id}__{method}__seed{seed}.csv", index=False)

    torch.save(model.state_dict(), out_dir / "checkpoints" / f"{test.trip_id}__{method}__seed{seed}.pt")
    test_batch = make_inference_batch(
        test,
        x_scaler.transform_trip(test),
        params,
        config,
        device,
        progress_scaled=p_scaler.transform_trip(test),
    )
    test_batch.assert_no_soc_input()
    pred = predict_power(model, test_batch, kind)
    e_hat = energy_kwh_from_power(pred["p_hat"], test.frame["dt_s"].to_numpy(dtype=float))
    record = {
        "p_hat": pred["p_hat"].tolist(),
        "p_phy": pred["p_phy"].tolist(),
        "delta_p": pred["delta_p"].tolist(),
        "d_hat": np.asarray(pred["d_hat"], dtype=float).tolist(),
        "saturation_frac": pred["saturation_frac"],
        "e_hat": e_hat,
        "selected_epoch": selected_epoch,
        "retrain_epochs": retrain_epochs,
        "validation_score": val_score,
        "val_mae_by_scale": val_mae_by_scale,
        "lambdas": lambdas,
        "scaler_ids_inner": list(x_scaler_inner.fitted_trip_ids),
        "scaler_ids_outer": scaler_outer_ids,
    }
    save_cache(out_dir / "cache" / f"{fp}.json", record)
    return {
        "pred": pred,
        "e_hat": e_hat,
        "selected_epoch": selected_epoch,
        "retrain_epochs": retrain_epochs,
        "validation_score": val_score,
        "val_mae_by_scale": val_mae_by_scale,
        "lambdas": lambdas,
        "scaler_ids_inner": list(x_scaler_inner.fitted_trip_ids),
        "scaler_ids_outer": scaler_outer_ids,
        "cached": False,
    }
