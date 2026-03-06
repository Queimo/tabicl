#!/usr/bin/env python3
"""Create mean±std LaTeX tables and per-dataset plots for all benchmarks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from export_benchmark_latex_table import METRICS, load_grouped_rows, write_latex_table, write_summary_csv
from plot_polymer_benchmark_summary import load_rows, render_plot, write_dataset_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
        help="Root directory containing benchmark result folders.",
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="*_benchmark_results.csv",
        help="Pattern (recursive) for canonical benchmark result CSV files.",
    )
    parser.add_argument(
        "--plots-dirname",
        type=str,
        default="plots",
        help="Per-benchmark directory name where plot artifacts are written.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke-test result directories.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only export tables; skip PNG plot generation.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in str(name))
    safe = safe.strip("_")
    return safe or "item"


def benchmark_title_from_prefix(benchmark_prefix: str) -> str:
    stem = benchmark_prefix.removesuffix("_benchmark")
    return f"{stem.replace('_', ' ').title()} benchmark"


def normalize_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]

    if "dataset" not in normalized.columns:
        if {"source", "benchmark", "task"}.issubset(normalized.columns):
            normalized["dataset"] = (
                normalized["source"].astype(str)
                + "/"
                + normalized["benchmark"].astype(str)
                + "/"
                + normalized["task"].astype(str)
            )
        elif {"benchmark", "task"}.issubset(normalized.columns):
            normalized["dataset"] = normalized["benchmark"].astype(str) + "/" + normalized["task"].astype(str)
        elif "benchmark" in normalized.columns:
            normalized["dataset"] = normalized["benchmark"].astype(str)
        elif "source" in normalized.columns:
            normalized["dataset"] = normalized["source"].astype(str)
        else:
            normalized["dataset"] = "dataset"

    for col in ("featurizer", "model"):
        if col not in normalized.columns:
            raise ValueError(f"Missing required column '{col}' in results frame.")

    for metric in METRICS:
        if metric not in normalized.columns:
            normalized[metric] = pd.NA
        normalized[metric] = pd.to_numeric(normalized[metric], errors="coerce")

    keep = ["dataset", "featurizer", "model"] + list(METRICS)
    normalized = normalized[keep]
    normalized = normalized.dropna(subset=["dataset", "featurizer", "model"])
    normalized = normalized[~normalized[list(METRICS)].isna().all(axis=1)]
    return normalized.reset_index(drop=True)


def export_tables_for_benchmark(
    normalized_df: pd.DataFrame,
    benchmark_prefix: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    tmp_results = out_dir / f".{benchmark_prefix}_normalized_results.tmp.csv"
    tmp_results.write_text(normalized_df.to_csv(index=False, na_rep="nan"), encoding="utf-8")
    try:
        rows = load_grouped_rows(tmp_results)
    finally:
        tmp_results.unlink(missing_ok=True)

    summary_mean_std = out_dir / f"{benchmark_prefix}_summary_mean_std.csv"
    latex_out = out_dir / f"{benchmark_prefix}_final_table.tex"
    write_summary_csv(rows, summary_mean_std)
    write_latex_table(
        rows,
        latex_out,
        caption=f"{benchmark_title_from_prefix(benchmark_prefix)} results aggregated across folds (mean $\\pm$ std).",
        label=f"tab:{sanitize_name(benchmark_prefix)}_mean_std",
    )
    return summary_mean_std, latex_out


def make_plots_for_benchmark(
    normalized_df: pd.DataFrame,
    benchmark_prefix: str,
    out_dir: Path,
    plots_dirname: str,
) -> int:
    summary_for_plot = (
        normalized_df.groupby(["dataset", "featurizer", "model"], as_index=False)[list(METRICS)]
        .mean(numeric_only=True)
        .sort_values(["dataset", "featurizer", "rmse"])
    )
    if summary_for_plot.empty:
        return 0

    tmp_summary = out_dir / f".{benchmark_prefix}_plot_summary.tmp.csv"
    tmp_summary.write_text(summary_for_plot.to_csv(index=False, na_rep="nan"), encoding="utf-8")
    try:
        grouped = load_rows(tmp_summary)
    finally:
        tmp_summary.unlink(missing_ok=True)

    plots_dir = out_dir / plots_dirname
    plots_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for dataset, rows in sorted(grouped.items()):
        for metric in ("rmse", "mae", "r2", "runtime_s"):
            vals = []
            for row in rows:
                try:
                    vals.append(float(str(row.get(metric, "nan")).strip()))
                except Exception:
                    vals.append(float("nan"))
            if vals and all(not math.isfinite(v) for v in vals):
                print(
                    f"WARNING: {benchmark_prefix}/{dataset} has all-NaN {metric}; "
                    "using 0.0 for plotting fallback."
                )
                for row in rows:
                    row[metric] = "0.0"

        safe_dataset = sanitize_name(dataset)
        table_path = plots_dir / f"{safe_dataset}_benchmark_table.tsv"
        plot_path = plots_dir / f"{safe_dataset}_benchmark_bars.png"
        write_dataset_table(rows, table_path)
        try:
            render_plot(
                dataset=dataset,
                table_path=table_path,
                plot_path=plot_path,
                benchmark_title=benchmark_title_from_prefix(benchmark_prefix),
            )
            print(f"Saved {plot_path}")
            created += 1
        except Exception as exc:
            print(f"WARNING: failed plot for {benchmark_prefix}/{dataset}: {exc}")
    return created


def should_skip_path(path: Path, include_smoke: bool) -> bool:
    if include_smoke:
        return False
    lower_parts = [p.lower() for p in path.parts]
    return any("smoke" in p for p in lower_parts)


def main() -> None:
    args = parse_args()
    results_files = sorted(args.results_root.rglob(args.results_glob))
    if not results_files:
        print(f"WARNING: no files matched '{args.results_glob}' under {args.results_root}")
        return

    n_tables = 0
    n_plots = 0
    for results_csv in results_files:
        if should_skip_path(results_csv, include_smoke=args.include_smoke):
            continue
        benchmark_prefix = results_csv.stem.removesuffix("_results")
        out_dir = results_csv.parent
        print(f"\nProcessing {results_csv}")

        try:
            raw_df = pd.read_csv(results_csv)
            norm_df = normalize_results_frame(raw_df)
            if norm_df.empty:
                print(f"WARNING: no usable rows in {results_csv}; skipping.")
                continue

            summary_mean_std, latex_out = export_tables_for_benchmark(norm_df, benchmark_prefix, out_dir)
            print(f"Saved: {summary_mean_std}")
            print(f"Saved: {latex_out}")
            n_tables += 1

            if not args.skip_plots:
                created = make_plots_for_benchmark(norm_df, benchmark_prefix, out_dir, args.plots_dirname)
                n_plots += created
        except FileNotFoundError:
            print(f"WARNING: file disappeared during processing: {results_csv}")
        except Exception as exc:
            print(f"WARNING: failed processing {results_csv}: {exc}")
            continue

    print(f"\nCompleted. Benchmarks exported: {n_tables}, plots generated: {n_plots}")


if __name__ == "__main__":
    main()
