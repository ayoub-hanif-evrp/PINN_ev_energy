"""Machine-readable sanity checks for LOTO outputs. Never retunes to make PINN win."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ABSURD_POWER_KW = 200.0
SATURATION_FLAG = 0.95


def inspect_loto_dir(out_dir: Path, table: pd.DataFrame | None = None, profile: str = "") -> dict[str, Any]:
    out_dir = Path(out_dir)
    if table is None:
        pred_path = out_dir / "per_trip_predictions.csv"
        table = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    flags: list[str] = []
    report: dict[str, Any] = {"profile": profile, "out_dir": str(out_dir), "flags": flags}

    if table is None or table.empty:
        flags.append("empty_predictions")
        return report

    report["n_rows"] = int(len(table))
    report["methods"] = sorted(table["method"].unique().tolist())
    report["n_trips"] = int(table["trip_id"].nunique())
    report["n_seeds"] = int(table["seed"].nunique()) if "seed" in table.columns else 1

    if table[["predicted_energy_kwh", "observed_energy_kwh"]].isna().any().any():
        flags.append("nan_energy")
    if not np.isfinite(table["predicted_energy_kwh"].to_numpy(dtype=float)).all():
        flags.append("nonfinite_predicted_energy")

    neg = table.loc[table["predicted_energy_kwh"] < 0]
    report["n_negative_predicted_energy"] = int(len(neg))
    if len(neg):
        flags.append("negative_predicted_energy")
        report["negative_energy_rows"] = neg[["trip_id", "method", "predicted_energy_kwh"]].to_dict("records")

    if "power_max" in table.columns:
        absurd = table.loc[table["power_max"] > ABSURD_POWER_KW]
        report["n_absurd_power"] = int(len(absurd))
        if len(absurd):
            flags.append("absurd_power_hundreds_of_kw")
        if "saturation_fraction" in table.columns:
            sat = table.loc[table["saturation_fraction"] > SATURATION_FLAG]
            report["n_saturated"] = int(len(sat))
            if len(sat):
                flags.append("residual_head_saturated_gt_95pct")
                report["saturated_rows"] = sat[["trip_id", "method", "seed", "saturation_fraction"]].to_dict("records")

    manifests = list((out_dir / "fold_manifests").glob("*.json"))
    report["n_manifests"] = len(manifests)
    overlap = []
    for path in manifests:
        body = json.loads(path.read_text(encoding="utf-8"))
        test_id = body.get("test_id")
        for key in ("train_ids", "inner_train_ids", "val_ids"):
            if test_id in set(body.get(key) or []):
                overlap.append({"manifest": path.name, "leak_in": key, "test_id": test_id})
        for method, meta in (body.get("methods") or {}).items():
            for scaler_key in ("scaler_trip_ids_inner", "scaler_trip_ids_outer", "fitted_trip_ids"):
                ids = meta.get(scaler_key) or []
                if test_id in set(ids):
                    overlap.append({"manifest": path.name, "leak_in": f"{method}.{scaler_key}", "test_id": test_id})
    if overlap:
        flags.append("train_test_overlap")
        report["overlap"] = overlap

    failures = out_dir / "failures" / "failures.csv"
    report["n_recorded_failures"] = int(len(pd.read_csv(failures))) if failures.exists() else 0
    if report["n_recorded_failures"]:
        flags.append("recorded_fold_failures")

    report["ok"] = len(flags) == 0
    return report


def write_sanity_report(out_dir: Path, table: pd.DataFrame | None = None, profile: str = "") -> dict[str, Any]:
    report = inspect_loto_dir(out_dir, table=table, profile=profile)
    path = Path(out_dir) / "sanity_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Sanity report: ok={report.get('ok')} flags={report.get('flags')} -> {path}", flush=True)
    return report
