#!/usr/bin/env python3
"""Export mean±std benchmark tables from fold-level results CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("rmse", "mae", "r2", "runtime_s")
DEFAULT_FEATURIZER_ORDER = ("identity", "mordred", "morgan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True, help="Fold-level results CSV path.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Output summary CSV with mean/std columns.")
    parser.add_argument("--latex-out", type=Path, required=True, help="Output LaTeX table .tex path.")
    parser.add_argument("--caption", type=str, default="Benchmark results aggregated across folds (mean $\\pm$ std).")
    parser.add_argument("--label", type=str, default="tab:benchmark_mean_std")
    return parser.parse_args()


def clean_dict(row: dict[str, str]) -> dict[str, str]:
    return {str(k).strip(): (str(v).strip() if v is not None else "") for k, v in row.items()}


def _safe_float(value: str) -> float:
    return float(value.strip())


def _metric_text(mu: float, sigma: float, decimals: int) -> str:
    return f"{mu:.{decimals}f} $\\pm$ {sigma:.{decimals}f}"


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _sort_key(row: dict[str, object]) -> tuple:
    # Sort per dataset by r2_mean descending (higher is better), then by identifiers.
    feat = str(row["featurizer"])
    if feat in DEFAULT_FEATURIZER_ORDER:
        feat_order = DEFAULT_FEATURIZER_ORDER.index(feat)
    else:
        feat_order = len(DEFAULT_FEATURIZER_ORDER)
    return (str(row["dataset"]), -float(row["r2_mean"]), feat_order, feat, str(row["model"]))


def load_grouped_rows(results_csv: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}

    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw_row in reader:
            row = clean_dict(raw_row)
            key = (row["dataset"], row["featurizer"], row["model"])
            if key not in grouped:
                grouped[key] = defaultdict(list)
            for metric in METRICS:
                grouped[key][metric].append(_safe_float(row[metric]))

    rows: list[dict[str, object]] = []
    for (dataset, featurizer, model), metric_map in grouped.items():
        out_row: dict[str, object] = {
            "dataset": dataset,
            "featurizer": featurizer,
            "model": model,
        }
        for metric in METRICS:
            values = metric_map[metric]
            mu = mean(values)
            sigma = stdev(values) if len(values) > 1 else 0.0
            out_row[f"{metric}_mean"] = mu
            out_row[f"{metric}_std"] = sigma
        rows.append(out_row)

    rows.sort(key=_sort_key)
    return rows


def write_summary_csv(rows: list[dict[str, object]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "featurizer",
        "model",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "runtime_s_mean",
        "runtime_s_std",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "featurizer": row["featurizer"],
                    "model": row["model"],
                    "rmse_mean": f"{float(row['rmse_mean']):.10f}",
                    "rmse_std": f"{float(row['rmse_std']):.10f}",
                    "mae_mean": f"{float(row['mae_mean']):.10f}",
                    "mae_std": f"{float(row['mae_std']):.10f}",
                    "r2_mean": f"{float(row['r2_mean']):.10f}",
                    "r2_std": f"{float(row['r2_std']):.10f}",
                    "runtime_s_mean": f"{float(row['runtime_s_mean']):.10f}",
                    "runtime_s_std": f"{float(row['runtime_s_std']):.10f}",
                }
            )


def write_latex_table(rows: list[dict[str, object]], latex_out: Path, caption: str, label: str) -> None:
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Dataset & Featurizer & Model & RMSE $\\downarrow$ & MAE $\\downarrow$ & $R^2$ $\\uparrow$ & Runtime (s) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        line = " & ".join(
            [
                _escape_latex(str(row["dataset"])),
                _escape_latex(str(row["featurizer"])),
                _escape_latex(str(row["model"])),
                _metric_text(float(row["rmse_mean"]), float(row["rmse_std"]), 3),
                _metric_text(float(row["mae_mean"]), float(row["mae_std"]), 3),
                _metric_text(float(row["r2_mean"]), float(row["r2_std"]), 3),
                _metric_text(float(row["runtime_s_mean"]), float(row["runtime_s_std"]), 2),
            ]
        )
        lines.append(f"{line} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    latex_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_grouped_rows(args.results_csv)
    write_summary_csv(rows, args.summary_csv)
    write_latex_table(rows, args.latex_out, caption=args.caption, label=args.label)
    print(f"Saved: {args.summary_csv}")
    print(f"Saved: {args.latex_out}")


if __name__ == "__main__":
    main()
