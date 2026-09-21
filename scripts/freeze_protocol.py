#!/usr/bin/env python
"""Freeze the paper experimental protocol after quick debugging. Do not retune from paper test scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.schema import ELASTICNET_FEATURES, MAIN_MODEL_FEATURES, PROGRESS_FEATURES  # noqa: E402
from manifest import git_commit, package_versions  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402


def freeze(config_path: str) -> Path:
    config = load_config(config_path)
    sha = git_commit() or "unknown"
    params = load_vehicle_parameters(config=config)
    tr = config.get("training", {})
    exp = config.get("experiment", {})
    win = config.get("windows", {})
    lines = [
        "# Frozen paper protocol",
        "",
        "This file records the experimental protocol **before** the paper LOTO is used for conclusions.",
        "Do not tune these choices using outer-test paper metrics.",
        "",
        f"- commit SHA: `{sha}`",
        f"- config: `{config_path}`",
        f"- Python: {package_versions()['python']}",
        f"- PyTorch: {package_versions().get('torch', 'not-installed')}",
        "",
        "## Features",
        "",
        f"- Power-branch telemetry: {list(MAIN_MODEL_FEATURES)}",
        f"- Causal progress features (state head only): {list(PROGRESS_FEATURES)}",
        f"- ElasticNet whitelist: {list(ELASTICNET_FEATURES)}",
        "- Excluded from the power branch: SoC, LAT, LON, Date, absolute Time",
        "- WindSpeedAv is m/s and is **not** used as aerodynamic headwind.",
        "",
        "## Physics parameters (assumptions, not HELECAR measurements)",
        "",
    ]
    for k, v in params.as_dict().items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## SoC windows",
        "",
        f"- event ΔSoC thresholds (pp): {win.get('soc_event_dsoc_pct')}",
        f"- event min duration (s): {win.get('soc_event_min_duration_s')}",
        f"- fixed times (s): {win.get('fixed_time_s')}",
        f"- full trip: {win.get('include_full_trip')}",
        "- Observed window energy: `E_obs(a,b) = E_batt (SOC[a]-SOC[b])/100`",
        "- Predicted window energy: trapezoidal `E(a,b) = prefix[b]-prefix[a]`",
        "",
        "## Discrete-time PINN",
        "",
        "- `P_hat = P_physics + residual_limit * tanh(head_delta(h))`",
        "- `D_hat(t) = D_raw(t) - D_raw(0)` so `D_hat(0)=0`",
        "- Interval residual: `r_i = D_hat[i+1]-D_hat[i] - 100 ΔE_i / E_batt`",
        "- `ΔE_i = 0.5 (P_hat[i]+P_hat[i+1]) dt[i] / 3600`",
        "- Loss: `L = λ_window L_window + λ_dynamics L_dynamics + λ_state L_state + λ_prior L_prior`",
        f"- λ_window={tr.get('lambda_window')} λ_dynamics={tr.get('lambda_dynamics')} λ_state={tr.get('lambda_state')} λ_prior={tr.get('lambda_prior')} λ_boundary={tr.get('lambda_boundary')} (diagnostic only)",
        f"- hidden={tr.get('hidden_layers')} activation={tr.get('activation')} residual_limit_kw={tr.get('residual_limit_kw')}",
        f"- optimizer={tr.get('optimizer')} lr={tr.get('learning_rate')} weight_decay={tr.get('weight_decay')}",
        f"- max_epochs={tr.get('max_epochs')} patience={tr.get('early_stopping_patience')}",
        "",
        "## Validation / retraining",
        "",
        "- Grouped inner validation: one trip per available T1/T2/T3, deterministic in fold ID + seed",
        "- Validation score: mean over scales of window MAE (scale-balanced)",
        "- After selection, discard the model; refit scalers and q on all outer-training trips; retrain selected epochs",
        "- q_inner_train / q_outer_train from training trips only; q_test_posthoc is evaluation-only",
        "",
        f"- seeds: {exp.get('seeds')}",
        "- Primary metric: trip energy MAE (kWh). Also RMSE, bias, MAPE (exclude |E_obs|<=1e-12), WAPE, R²",
        "- Statistics: trip-level paired bootstrap (10,000), then optional Wilcoxon as secondary",
        "- Exclusions: none planned; failed folds are recorded, not dropped",
        "",
        "This is a discrete-time physics-informed neural network, not an automatic-differentiation PDE PINN.",
        "",
    ]
    out = ROOT / "reports" / "protocol_frozen.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    paper_copy = ROOT / "reports" / "paper" / "protocol_frozen.md"
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    paper_copy.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    freeze(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
