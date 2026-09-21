"""CSV + LaTeX tables. No automatic bolding of 'best' results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.metrics import summarize_energy_table


def _tex_escape(value) -> str:
    text = str(value)
    return text.replace("_", r"\_").replace("%", r"\%")


def dataframe_to_latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    if frame.empty:
        return f"% empty table {label}\n"
    cols = list(frame.columns)
    header = " & ".join(_tex_escape(c) for c in cols) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in frame.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.4g}")
            else:
                cells.append(_tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_table(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".tex").write_text(dataframe_to_latex(frame, caption, label), encoding="utf-8")


def main_results_table(pred: pd.DataFrame, bootstrap: pd.DataFrame | None = None) -> pd.DataFrame:
    summary = summarize_energy_table(pred)
    rows = []
    for method, vals in summary.items():
        row = {
            "Method": method,
            "MAE_kWh": vals["mae_kwh"],
            "RMSE_kWh": vals["rmse_kwh"],
            "MAPE_pct": vals["mape_pct"],
            "WAPE_pct": vals["wape_pct"],
            "Bias_kWh": vals["bias_kwh"],
            "R2": vals["r2"],
            "MAPE_n_undefined": vals.get("mape_n_undefined", 0.0),
        }
        if bootstrap is not None and not bootstrap.empty:
            sub = bootstrap.loc[bootstrap["method"] == method]
            if not sub.empty:
                row["MAE_CI_low"] = float(sub["ci_low"].iloc[0])
                row["MAE_CI_high"] = float(sub["ci_high"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)
