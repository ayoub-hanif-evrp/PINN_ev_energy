#!/usr/bin/env python
"""Cross-trajectory generalization: train on two trajectories, test the third.

These results are NOT mixed into the primary LOTO table.
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
from manifest import git_commit, write_manifest  # noqa: E402
from models.baselines import physics_only_predict  # noqa: E402
from models.elasticnet import TripElasticNet  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from training.cross_validation import select_validation_ids, trips_by_id  # noqa: E402
from training.trainer import method_kind, window_scale_filter  # noqa: E402

HOLDOUTS = (("T1", ("T2", "T3")), ("T2", ("T1", "T3")), ("T3", ("T1", "T2")))


def run_cross_trajectory(config: dict[str, Any]) -> pd.DataFrame:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    out_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root).parent / "cross_trajectory" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    (out_dir / "training_histories").mkdir(exist_ok=True)
    (out_dir / "loss_diagnostics").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    seeds = list(config.get("experiment", {}).get("cross_traj_seeds") or [config.get("experiment", {}).get("seeds", [0])[0]])
    methods = list(config.get("experiment", {}).get("cross_traj_methods", ["physics", "elasticnet", "weak_mlp", "pinn"]))
    device = resolve_device(config)
    trips = load_processed(config)
    params = load_vehicle_parameters(config=config)
    by_id = trips_by_id(trips)
    rows: list[dict[str, Any]] = []
    git_sha = git_commit()

    for seed in seeds:
        set_seeds(int(seed))
        for held, train_trajs in HOLDOUTS:
            train_trips = [t for t in trips if t.trajectory in train_trajs]
            test_trips = [t for t in trips if t.trajectory == held]
            train_ids = [t.trip_id for t in train_trips]
            val_ids = select_validation_ids(train_trips, train_ids, f"holdout_{held}", seed)
            inner_ids = tuple(i for i in train_ids if i not in set(val_ids))
            q_inner = q_from_trips([by_id[i] for i in inner_ids])
            q_outer = q_from_trips(train_trips)
            for test in test_trips:
                obs = trip_obs(test, params.battery_capacity_kwh)
                fold = SimpleNamespace(
                    test_id=test.trip_id,
                    train_ids=tuple(train_ids),
                    inner_train_ids=inner_ids,
                    val_ids=val_ids,
                )
                if "physics" in methods:
                    phys = physics_only_predict(test, params)
                    row = trip_energy_row(
                        test.trip_id, test.trajectory, "physics", seed,
                        obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], phys["energy_kwh"],
                    )
                    row["heldout_trajectory"] = held
                    rows.append(row)
                if "elasticnet" in methods:
                    table = trip_level_feature_table(train_trips, battery_capacity_kwh=params.battery_capacity_kwh)
                    test_table = trip_level_feature_table([test], battery_capacity_kwh=params.battery_capacity_kwh)
                    model = TripElasticNet(random_state=int(seed)).fit(table)
                    model.assert_not_fitted_on(test.trip_id)
                    pred = float(model.predict(test_table)[0])
                    row = trip_energy_row(
                        test.trip_id, test.trajectory, "elasticnet", seed,
                        obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred,
                    )
                    row["heldout_trajectory"] = held
                    rows.append(row)
                for method in methods:
                    if method in {"physics", "elasticnet"}:
                        continue
                    result = _run_neural_fold(
                        method=method,
                        kind=method_kind(method),
                        scale_filter=window_scale_filter(method),
                        test=test,
                        outer_train=train_trips,
                        inner_train=[by_id[i] for i in inner_ids],
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
                        profile=f"{profile}_xtraj_{held}",
                    )
                    row = trip_energy_row(
                        test.trip_id, test.trajectory, method, seed,
                        obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], result["e_hat"],
                    )
                    row["heldout_trajectory"] = held
                    rows.append(row)
                (out_dir / "manifests" / f"{held}__{test.trip_id}__seed{seed}.json").write_text(
                    json.dumps(
                        {
                            "heldout_trajectory": held,
                            "train_trajectories": list(train_trajs),
                            "train_ids": train_ids,
                            "val_ids": list(val_ids),
                            "test_id": test.trip_id,
                            "q_inner_train": q_inner,
                            "q_outer_train": q_outer,
                            "git_sha": git_sha,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                print(f"cross-traj holdout={held} test={test.trip_id} seed={seed}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "cross_trajectory_per_trip.csv", index=False)
    write_manifest(out_dir / "manifest.json", {"stage": "cross_trajectory", "profile": profile})
    print(f"Wrote cross-trajectory outputs to {out_dir}", flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-trajectory generalization.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    run_cross_trajectory(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
