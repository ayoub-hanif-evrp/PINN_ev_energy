#!/usr/bin/env python
"""Limited data-scarcity validation on the predetermined Tesla pool (review-only).

Uses the same held-out test trips across methods. Training subsets are sampled
from the remaining pool with a fixed seed before inspecting neural metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

from config import load_config  # noqa: E402
from experiment.loto import run_loto  # noqa: E402
from tesla_data import load_tesla_pool  # noqa: E402


def _apply_L(config: dict) -> None:
    vehicle = str(config["tesla"]["vehicle"])
    path = ROOT / "review_results" / "tesla" / f"residual_calibration_{vehicle.lower()}.json"
    if path.exists():
        config.setdefault("training", {})["residual_limit_kw"] = float(
            json.loads(path.read_text(encoding="utf-8"))["chosen_residual_limit_kw"]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=5, help="Fixed held-out test trips from pool end")
    args = parser.parse_args()

    config = load_config(args.config)
    _apply_L(config)
    # Avoid colliding with primary LOTO output profile.
    config.setdefault("experiment", {})["profile"] = str(config["experiment"]["profile"]) + "_scarcity"
    config["experiment"]["cache"] = True

    trips, pool_meta = load_tesla_pool(config, ROOT)
    by_id = {t.trip_id: t for t in trips}
    all_ids = list(pool_meta["trip_ids"])
    n_test = min(args.n_test, max(1, len(all_ids) // 3))
    test_ids = all_ids[-n_test:]
    train_universe = [i for i in all_ids if i not in set(test_ids)]

    n_list = list(config.get("scarcity", {}).get("n_train_trips", [3, 5, 8, 12]))
    methods = list(config.get("scarcity", {}).get("methods", ["physics", "ridge", "weak_mlp", "pinn"]))
    if "physics" not in methods:
        methods = ["physics", *methods]
    n_repeats = int(config.get("scarcity", {}).get("n_subset_repeats", 3))
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    out_root = ROOT / "review_results" / "tesla" / "scarcity" / str(config["tesla"]["vehicle"]).lower()
    out_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for n_train in n_list:
        if n_train > len(train_universe):
            continue
        for rep in range(n_repeats):
            chosen = [str(x) for x in rng.choice(train_universe, size=n_train, replace=False)]
            subset = [by_id[i] for i in chosen + test_ids]
            subdir = f"n{n_train}_rep{rep}"
            config["experiment"]["profile"] = f"{config['tesla']['vehicle'].lower()}_scarcity_{subdir}"
            table = run_loto(
                config,
                methods=methods,
                seeds=[seed],
                test_ids=test_ids,
                trips=subset,
                output_subdir=str(out_root.relative_to(ROOT) / subdir) if False else None,
            )
            # run_loto writes under paths.loto_dir / profile; keep a combined table too.
            table = table.copy()
            table["n_train"] = n_train
            table["repeat"] = rep
            # Attach measured energy.
            e_meas = {t.trip_id: float(t.stats.get("e_meas_kwh", np.nan)) for t in subset}
            table["e_meas_kwh"] = table["trip_id"].map(e_meas)
            frames.append(table)

    if not frames:
        raise SystemExit("No scarcity runs produced")
    all_tab = pd.concat(frames, ignore_index=True)
    all_tab.to_csv(out_root / "scarcity_trip_energy.csv", index=False)

    for ref in ("observed_energy_kwh", "e_meas_kwh"):
        rows = []
        for (n_train, method), g in all_tab.groupby(["n_train", "method"]):
            y = g[ref].to_numpy(dtype=float)
            yhat = g["predicted_energy_kwh"].to_numpy(dtype=float)
            rows.append(
                {
                    "n_train": int(n_train),
                    "method": method,
                    "ref": ref,
                    "mae_kwh": float(np.nanmean(np.abs(yhat - y))),
                    "n": int(len(g)),
                }
            )
        summary = pd.DataFrame(rows)
        tag = "soc" if ref.startswith("observed") else "meas"
        summary.to_csv(out_root / f"scarcity_summary_{tag}.csv", index=False)
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
