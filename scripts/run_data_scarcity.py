#!/usr/bin/env python
"""Data-scarcity LOTO: train on n outer-training trips, evaluate the held-out trip.

Physics-only does not depend on n and is recorded once per test trip as a reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.features import trip_level_feature_table  # noqa: E402
from data.quantization import q_from_trips  # noqa: E402
from evaluation.metrics import trip_energy_row  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from experiment.device import resolve_device, set_seeds  # noqa: E402
from experiment.loto import _run_neural_fold, trip_obs  # noqa: E402
from experiment.subsets import stratified_subset  # noqa: E402
from manifest import git_commit, write_manifest  # noqa: E402
from models.baselines import physics_only_predict  # noqa: E402
from models.elasticnet import TripElasticNet  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from training.cross_validation import select_validation_ids, trips_by_id  # noqa: E402
from training.trainer import method_kind, window_scale_filter  # noqa: E402


def _print(msg: str) -> None:
    print(msg, flush=True)


def run_scarcity(config: dict[str, Any]) -> pd.DataFrame:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    out_dir = resolve_under_root(config.get("paths", {}).get("scarcity_dir", "outputs/scarcity"), root) / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    (out_dir / "training_histories").mkdir(exist_ok=True)
    (out_dir / "loss_diagnostics").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    seeds = list(config.get("scarcity", {}).get("seeds") or [config.get("experiment", {}).get("seeds", [0])[0]])
    sizes = list(config.get("scarcity", {}).get("n_train_trips", [3, 5, 8, 12, 16]))
    n_rep = int(config.get("scarcity", {}).get("n_subset_repeats", 3))
    methods = list(config.get("scarcity", {}).get("methods", ["elasticnet", "weak_mlp", "pinn"]))
    device = resolve_device(config)
    trips = load_processed(config)
    params = load_vehicle_parameters(config=config)
    by_id = trips_by_id(trips)
    rows: list[dict[str, Any]] = []
    git_sha = git_commit()

    loto_pred_path = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile / "per_trip_predictions.csv"
    loto_pred = pd.read_csv(loto_pred_path) if loto_pred_path.exists() else pd.DataFrame()

    for seed in seeds:
        set_seeds(int(seed))
        for test in trips:
            outer = [t for t in trips if t.trip_id != test.trip_id]
            obs = trip_obs(test, params.battery_capacity_kwh)
            if "physics" not in {r.get("method") for r in rows if r.get("trip_id") == test.trip_id}:
                phys = physics_only_predict(test, params)
                row = trip_energy_row(
                    test.trip_id, test.trajectory, "physics", seed,
                    obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], phys["energy_kwh"],
                )
                row["n_train"] = 0
                row["repeat"] = 0
                row["subset_ids"] = ""
                rows.append(row)
            for n in sizes:
                repeats = 1 if int(n) >= len(outer) else n_rep
                for rep in range(repeats):
                    subset_ids = stratified_subset(outer, int(n), test_id=test.trip_id, repeat=rep, seed=int(seed))
                    subset = [by_id[i] for i in subset_ids]
                    val_ids = select_validation_ids(subset, subset_ids, test.trip_id, seed=int(seed) + 1000 * rep)
                    inner_ids = tuple(i for i in subset_ids if i not in set(val_ids))
                    fold = SimpleNamespace(test_id=test.trip_id, train_ids=tuple(subset_ids), inner_train_ids=inner_ids, val_ids=val_ids)
                    q_inner = q_from_trips([by_id[i] for i in inner_ids] or subset)
                    q_outer = q_from_trips(subset)
                    manifest = {
                        "git_sha": git_sha,
                        "test_id": test.trip_id,
                        "n_train": int(n),
                        "repeat": rep,
                        "seed": int(seed),
                        "subset_ids": subset_ids,
                        "val_ids": list(val_ids),
                        "inner_train_ids": list(inner_ids),
                        "q_inner_train": q_inner,
                        "q_outer_train": q_outer,
                    }
                    if "elasticnet" in methods:
                        table = trip_level_feature_table(subset, battery_capacity_kwh=params.battery_capacity_kwh)
                        test_table = trip_level_feature_table([test], battery_capacity_kwh=params.battery_capacity_kwh)
                        model = TripElasticNet(random_state=int(seed)).fit(table)
                        model.assert_not_fitted_on(test.trip_id)
                        pred = float(model.predict(test_table)[0])
                        row = trip_energy_row(
                            test.trip_id, test.trajectory, "elasticnet", seed,
                            obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred,
                        )
                        row.update({"n_train": int(n), "repeat": rep, "subset_ids": ",".join(subset_ids)})
                        rows.append(row)
                    for method in methods:
                        if method == "elasticnet":
                            continue
                        reused = None
                        if int(n) >= len(outer) and not loto_pred.empty:
                            hit = loto_pred.loc[
                                (loto_pred["trip_id"] == test.trip_id)
                                & (loto_pred["method"] == method)
                                & (loto_pred["seed"] == int(seed))
                            ]
                            if not hit.empty:
                                reused = float(hit["predicted_energy_kwh"].iloc[0])
                        if reused is not None:
                            row = trip_energy_row(
                                test.trip_id, test.trajectory, method, seed,
                                obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], reused,
                            )
                            row.update({"n_train": int(n), "repeat": rep, "subset_ids": ",".join(subset_ids), "reused_loto": True})
                            rows.append(row)
                            continue
                        kind = method_kind(method)
                        result = _run_neural_fold(
                            method=method,
                            kind=kind,
                            scale_filter=window_scale_filter(method),
                            test=test,
                            outer_train=subset,
                            inner_train=[by_id[i] for i in inner_ids] or subset,
                            val_trips=[by_id[i] for i in val_ids],
                            fold=fold,
                            q_inner=q_inner,
                            q_outer=q_outer,
                            params=params,
                            config=config,
                            device=device,
                            seed=int(seed),
                            retrain_outer=True,
                            use_cache=True,
                            out_dir=out_dir,
                            git_sha=git_sha,
                            profile=f"{profile}_scarcity_n{n}_r{rep}",
                        )
                        row = trip_energy_row(
                            test.trip_id, test.trajectory, method, seed,
                            obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], result["e_hat"],
                        )
                        row.update({"n_train": int(n), "repeat": rep, "subset_ids": ",".join(subset_ids)})
                        rows.append(row)
                    (out_dir / "manifests" / f"{test.trip_id}__n{n}__r{rep}__seed{seed}.json").write_text(
                        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
                    )
                    _print(f"scarcity test={test.trip_id} n={n} rep={rep} seed={seed} subset={subset_ids}")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "scarcity_per_trip.csv", index=False)
    write_manifest(out_dir / "manifest.json", {"stage": "data_scarcity", "profile": profile, "sizes": sizes, "n_repeats": n_rep})
    _print(f"Wrote scarcity outputs to {out_dir}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Data-scarcity energy experiment.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    run_scarcity(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
