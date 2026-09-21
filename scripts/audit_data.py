"""Phase 1–4 dataset audit for HELECAR-D.

Never modifies original CSVs. Does not fabricate paper metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.discovery import discover_dataset, directory_tree_lines  # noqa: E402
from data.features import trip_level_feature_table  # noqa: E402
from data.loader import load_selected_trips  # noqa: E402
from data.preprocessing import cache_processed_trip, epoch_seconds, parse_trip_timestamps, preprocess_trips  # noqa: E402
from data.quantization import combine_quantization, estimate_soc_quantization  # noqa: E402
from data.windows import generate_windows, huber_delta_kwh, windows_to_frame  # noqa: E402
from data.paper_audit import paper_dataset_summary, trip_characteristics_table  # noqa: E402
from manifest import file_sha256, write_manifest  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from physics.vehicle_model import observed_energy_kwh, predict_trip_physics  # noqa: E402
from plotting.diagnostics import (  # noqa: E402
    plot_grade,
    plot_method_diagram,
    plot_physics_vs_soc_energy,
    plot_soc_transition_histogram,
    plot_speed_acceleration,
    plot_trip_overview,
)


DATA_PLACEMENT = """\
No HELECAR-D candidate CSV files were found.

Download the official HELECAR-D release locally (do not commit it), then place
it anywhere inside the project directory, for example:

  data/HELECAR-D/data/Analysed/T1/*.csv
  HELECAR-D/data/Analysed/T1/*.csv

Analysed files are preferred over raw CAN dumps.
Original files are never modified.
Unit tests can still run:  python -m pytest -q
This script will not fabricate paper results.
"""


def _print(msg: str) -> None:
    print(msg, flush=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sampling_intervals_from_trip(trip) -> np.ndarray:
    timestamps, _ = parse_trip_timestamps(trip.frame)
    if len(timestamps) < 2:
        return np.array([])
    diffs = np.diff(epoch_seconds(timestamps))
    return diffs[np.isfinite(diffs)]


def audit_loaded_trips(trips) -> dict:
    rows = []
    sampling = []
    missing_frames = []
    soc_increases = []
    quant_rows = []
    per_trip_q = []
    speed_agree = []
    alt_rows = []
    ts_rows = []
    modes: Counter[str] = Counter()

    for trip in trips:
        df = trip.frame
        n = len(df)
        soc = pd.to_numeric(df["soc"], errors="coerce") if "soc" in df.columns else pd.Series(dtype=float)
        qstats = estimate_soc_quantization(soc) if len(soc) else estimate_soc_quantization([])
        per_trip_q.append(qstats)
        quant_rows.append({"trip_id": trip.trip_id, "trajectory": trip.trajectory, **{k: v for k, v in qstats.items() if k not in {"unique_increments", "transition_values", "transition_counts"}}})

        if "mode" in df.columns:
            modes.update(str(v) for v in df["mode"].dropna().astype(str))

        miss = df.isna().sum()
        missing_frames.append({"trip_id": trip.trip_id, **{str(c): int(miss[c]) for c in miss.index}})

        diffs = _sampling_intervals_from_trip(trip)
        sampling.extend(float(x) for x in diffs.tolist())
        ts_rows.append(
            {
                "trip_id": trip.trip_id,
                "n_duplicate_timestamps": int(np.sum(np.round(diffs, 6) == 0)) if diffs.size else 0,
                "n_gaps_gt_2s": int(np.sum(diffs > 2.0)) if diffs.size else 0,
                "dt_median_s": float(np.median(diffs)) if diffs.size else np.nan,
                "dt_min_s": float(np.min(diffs)) if diffs.size else np.nan,
                "dt_max_s": float(np.max(diffs)) if diffs.size else np.nan,
            }
        )

        if "speed_kmh" in df.columns and "gps_speed_kmh" in df.columns:
            a = pd.to_numeric(df["speed_kmh"], errors="coerce")
            b = pd.to_numeric(df["gps_speed_kmh"], errors="coerce")
            mask = a.notna() & b.notna()
            err = (a - b)[mask]
            speed_agree.append(
                {
                    "trip_id": trip.trip_id,
                    "n": int(mask.sum()),
                    "mae_kmh": float(err.abs().mean()) if mask.any() else np.nan,
                    "rmse_kmh": float(np.sqrt((err**2).mean())) if mask.any() else np.nan,
                    "corr": float(a[mask].corr(b[mask])) if mask.sum() > 2 else np.nan,
                }
            )

        if "alt_m" in df.columns:
            alt = pd.to_numeric(df["alt_m"], errors="coerce")
            dalt = alt.diff()
            alt_rows.append(
                {
                    "trip_id": trip.trip_id,
                    "alt_min_m": float(alt.min()) if alt.notna().any() else np.nan,
                    "alt_max_m": float(alt.max()) if alt.notna().any() else np.nan,
                    "n_unique_alt": int(alt.nunique(dropna=True)),
                    "max_abs_step_m": float(dalt.abs().max()) if dalt.notna().any() else np.nan,
                    "n_abs_step_gt_5m": int((dalt.abs() > 5).sum()),
                }
            )

        if len(soc) >= 2:
            d = soc.diff()
            inc = d[d > 1e-6]
            soc_increases.append(
                {
                    "trip_id": trip.trip_id,
                    "n_increases": int(inc.shape[0]),
                    "max_increase_pp": float(inc.max()) if len(inc) else 0.0,
                    "sum_increases_pp": float(inc.sum()) if len(inc) else 0.0,
                }
            )

        rows.append(
            {
                "trip_id": trip.trip_id,
                "trajectory": trip.trajectory,
                "kind": trip.kind,
                "path": str(trip.source_path),
                "n_rows": n,
                "n_columns": int(df.shape[1]),
                "columns": ",".join(map(str, df.columns)),
                "soc_min": qstats["min"],
                "soc_max": qstats["max"],
                "soc_start": qstats["start"],
                "soc_end": qstats["end"],
                "n_soc_increases": qstats["n_increases"],
                "n_soc_decreases": qstats["n_decreases"],
            }
        )

    combined_q = combine_quantization(per_trip_q)
    traj_counts = Counter(t.trajectory for t in trips)
    dt_arr = np.asarray(sampling, dtype=float)
    return {
        "trip_summary": pd.DataFrame(rows),
        "quant_rows": pd.DataFrame(quant_rows),
        "combined_q": combined_q,
        "traj_counts": dict(traj_counts),
        "modes": dict(modes),
        "missing": pd.DataFrame(missing_frames),
        "speed_agree": pd.DataFrame(speed_agree),
        "altitude": pd.DataFrame(alt_rows),
        "timestamps": pd.DataFrame(ts_rows),
        "soc_increases": pd.DataFrame(soc_increases),
        "dt_values": dt_arr,
        "per_trip_q": per_trip_q,
    }


def run_audit(config: dict, do_preprocess: bool = True) -> int:
    root = project_root()
    out_dir = resolve_under_root(config.get("paths", {}).get("audit_dir", "outputs/data_audit"), root)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_dataset(config)
    tree = directory_tree_lines([c.path for c in discovery.candidates], root)
    _write_text(out_dir / "directory_structure.txt", "\n".join(tree) + ("\n" if tree else ""))

    _print("=== HELECAR-D discovery ===")
    _print(f"Project root: {root}")
    _print(f"Candidate CSVs: {len(discovery.candidates)}")
    _print(f"Selected trips (prefer analysed={config.get('discovery', {}).get('prefer_analysed', True)}): {discovery.n_selected}")
    if tree:
        _print("Detected files:")
        for line in tree:
            _print(f"  {line}")

    if discovery.n_selected == 0:
        _print(DATA_PLACEMENT)
        _write_text(out_dir / "audit_report.txt", DATA_PLACEMENT)
        write_manifest(
            out_dir / "manifest.json",
            {
                "stage": "audit",
                "n_selected_trips": 0,
                "config_path": config.get("_config_path"),
                "dataset_files": [],
            },
        )
        return 0

    trips = load_selected_trips(discovery.selected)
    stats = audit_loaded_trips(trips)

    stats["trip_summary"].to_csv(out_dir / "trip_summary.csv", index=False)
    stats["quant_rows"].to_csv(out_dir / "soc_quantization.csv", index=False)
    stats["missing"].to_csv(out_dir / "missing_values.csv", index=False)
    stats["speed_agree"].to_csv(out_dir / "speed_gps_agreement.csv", index=False)
    stats["altitude"].to_csv(out_dir / "altitude_quality.csv", index=False)
    stats["timestamps"].to_csv(out_dir / "timestamp_issues.csv", index=False)
    stats["soc_increases"].to_csv(out_dir / "soc_increases.csv", index=False)
    pd.Series(stats["dt_values"], name="dt_s").to_csv(out_dir / "sampling_intervals.csv", index=False)
    plot_soc_transition_histogram(stats["combined_q"], out_dir / "soc_transition_histogram.png")

    _print("\n=== Trip counts ===")
    _print(f"Total selected trips: {len(trips)}")
    for k, v in sorted(stats["traj_counts"].items()):
        _print(f"  {k}: {v}")
    _print("\n=== Rows per trip ===")
    for _, row in stats["trip_summary"].iterrows():
        _print(f"  {row['trip_id']}: {row['n_rows']} rows, {row['n_columns']} columns, SoC {row['soc_start']:.3f} -> {row['soc_end']:.3f}")
    _print("\n=== Sampling intervals (s) ===")
    dt = stats["dt_values"]
    if dt.size:
        _print(f"  n={dt.size}  min={dt.min():.3f}  median={np.median(dt):.3f}  max={dt.max():.3f}")
    _print("\n=== Unique Mode values ===")
    _print(f"  {stats['modes']}")
    _print("\n=== SoC quantization (empirical, not a BMS model) ===")
    cq = stats["combined_q"]
    _print(f"  estimated q = {cq['q']}")
    _print(f"  pooled mode = {cq['q_pooled_mode']}  median trip q = {cq['q_median_trip']}  median GCD = {cq['q_gcd_median']}")
    _print("\n=== SoC increases (possible regen or BMS noise) ===")
    if not stats["soc_increases"].empty:
        _print(stats["soc_increases"].to_string(index=False))
    _print("\n=== Speed vs GpsSpeed ===")
    if not stats["speed_agree"].empty:
        _print(stats["speed_agree"].to_string(index=False))
    _print("\n=== Altitude quality ===")
    if not stats["altitude"].empty:
        _print(stats["altitude"].to_string(index=False))
    _print("\n=== Timestamp duplicates / gaps ===")
    if not stats["timestamps"].empty:
        _print(stats["timestamps"].to_string(index=False))

    report_lines = [
        "HELECAR-D audit report",
        f"selected_trips={len(trips)}",
        f"trajectory_counts={stats['traj_counts']}",
        f"estimated_soc_q={cq['q']}",
        f"modes={stats['modes']}",
        "original_files_not_modified=true",
    ]
    _write_text(out_dir / "audit_report.txt", "\n".join(report_lines) + "\n")
    (out_dir / "soc_quantization_global.json").write_text(json.dumps(cq, indent=2, default=str), encoding="utf-8")

    physics_table = pd.DataFrame()
    window_frame = pd.DataFrame()
    processed = []
    if do_preprocess:
        _print("\n=== Preprocessing (SI units, acceleration, grade) ===")
        processed = preprocess_trips(trips, config)
        for p in processed:
            cache_processed_trip(p, config)
            _print(
                f"  {p.trip_id}: CAN={p.stats['distance_can_m']/1000:.2f} km  "
                f"GPSspeed={p.stats['distance_gps_speed_m']/1000:.2f} km  "
                f"Haversine={p.stats['distance_haversine_m']/1000:.2f} km  "
                f"duration={p.stats['duration_s']/60:.1f} min  "
                f"grade_clip_frac={p.stats['grade']['clip_fraction']:.4f}  "
                f"notes={p.notes}"
            )
        feat = trip_level_feature_table(processed)
        feat.to_csv(out_dir / "trip_characteristics_preprocessed.csv", index=False)

        params = load_vehicle_parameters(config=config)
        phys_rows = []
        apply_bounds = bool(config.get("physics", {}).get("apply_power_bounds", False))
        sharpness = float(config.get("physics", {}).get("bound_sharpness", 8.0))
        for p in processed:
            df = p.frame
            result = predict_trip_physics(
                df["speed_mps"].to_numpy(dtype=float),
                df["acc_mps2"].to_numpy(dtype=float),
                df["theta_rad"].to_numpy(dtype=float),
                df["dt_s"].to_numpy(dtype=float),
                params,
                apply_bounds=apply_bounds,
                bound_sharpness=sharpness,
            )
            soc = pd.to_numeric(df["soc"], errors="coerce")
            e_obs = observed_energy_kwh(float(soc.iloc[0]), float(soc.iloc[-1]), params.battery_capacity_kwh)
            phys_rows.append(
                {
                    "trip_id": p.trip_id,
                    "trajectory": p.trajectory,
                    "e_physics_kwh": result.energy_kwh,
                    "e_obs_kwh": e_obs,
                    "e_residual_kwh": result.energy_kwh - e_obs,
                    "mean_p_batt_kw": float(np.mean(result.p_battery_kw)),
                    "min_p_batt_kw": float(np.min(result.p_battery_kw)),
                    "max_p_batt_kw": float(np.max(result.p_battery_kw)),
                    "regen_fraction": float(np.mean(result.p_battery_kw < params.auxiliary_power_kw)),
                    "n_power_bounds": result.n_bound_applied,
                }
            )
        physics_table = pd.DataFrame(phys_rows)
        physics_table.to_csv(out_dir / "physics_only_vs_soc_energy_diagnostic.csv", index=False)
        _print("\n=== Physics-only vs SoC energy (diagnostic, not a paper table) ===")
        _print(physics_table.to_string(index=False))
        plot_physics_vs_soc_energy(physics_table, out_dir / "physics_vs_soc_energy_diagnostic.png")

        q = float(cq["q"]) if np.isfinite(cq["q"]) else 0.02
        windows = generate_windows(processed, params.battery_capacity_kwh, q, config=config, overlapping=True)
        window_frame = windows_to_frame(windows)
        window_frame.to_csv(out_dir / "training_windows.csv", index=False)
        _print("\n=== Weak energy windows ===")
        if window_frame.empty:
            _print("  none")
        else:
            counts = window_frame.groupby("scale").size().reindex(
                [c for c in ["soc_event_0.1", "soc_event_0.2", "soc_event_0.5", "time_60s", "time_120s", "time_300s", "time_600s", "full_trip"] if c in set(window_frame["scale"])]
                + [c for c in sorted(window_frame["scale"].unique()) if c not in {"soc_event_0.1", "soc_event_0.2", "soc_event_0.5", "time_60s", "time_120s", "time_300s", "time_600s", "full_trip"}]
            )
            _print(f"{'scale':<22} windows")
            _print("-" * 32)
            for scale, nwin in counts.items():
                _print(f"{scale:<22} {int(nwin)}")
            _print("\nWindows per trip:")
            per_trip = window_frame.groupby(["trip_id", "scale"]).size().unstack(fill_value=0)
            _print(per_trip.to_string())
            _print(f"  Huber delta hint (kWh) = {huber_delta_kwh(params.battery_capacity_kwh, q)}")
            per_trip.to_csv(out_dir / "windows_per_trip.csv")
            counts.to_csv(out_dir / "windows_by_scale.csv", header=["n_windows"])
            paper = paper_dataset_summary(processed, params.battery_capacity_kwh, q, config)
            paper.to_csv(out_dir / "paper_dataset_summary.csv", index=False)
            trip_characteristics_table(processed, params.battery_capacity_kwh).to_csv(
                out_dir / "trip_characteristics.csv", index=False
            )

        plot_trip_overview(processed, out_dir / "dataset_overview.png")
        plot_method_diagram(out_dir / "method_diagram.png")
        representative = processed[:3]
        for p in representative:
            plot_speed_acceleration(
                p,
                out_dir / f"speed_acc_{p.trip_id}.png",
                sensitivity_windows=config.get("preprocessing", {}).get("savgol_sensitivity_windows", [7, 11, 15]),
            )
            plot_grade(p, out_dir / f"grade_{p.trip_id}.png")

    hashes = {str(t.source_path): file_sha256(t.source_path) for t in trips}
    write_manifest(
        out_dir / "manifest.json",
        {
            "stage": "audit",
            "config_path": config.get("_config_path"),
            "n_selected_trips": len(trips),
            "trip_ids": [t.trip_id for t in trips],
            "trajectories": [t.trajectory for t in trips],
            "dataset_files": [str(t.source_path) for t in trips],
            "dataset_sha256": hashes,
            "quantization_estimate": cq,
            "n_windows": int(len(window_frame)),
            "vehicle_config": config.get("vehicle", {}),
            "wind_used_as_headwind": False,
            "preprocess_ran": do_preprocess,
        },
    )
    _print(f"\nWrote audit outputs to {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HELECAR-D CSVs found in the project.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    do_pre = False if args.no_preprocess else True
    return run_audit(config, do_preprocess=do_pre)


if __name__ == "__main__":
    raise SystemExit(main())
