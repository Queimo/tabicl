#!/usr/bin/env python3
"""Create per-dataset bar plots from polymer_property_benchmark_summary.csv."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


def _clean_row(raw: dict[str, str]) -> dict[str, str]:
    return {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}


def load_rows(summary_csv: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw in reader:
            row = _clean_row(raw)
            dataset = row["dataset"]
            grouped[dataset].append(row)
    return grouped


def write_dataset_table(rows: list[dict[str, str]], out_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r["rmse"]))
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "rmse", "mae", "r2", "runtime_s"])
        for row in sorted_rows:
            label = f"{row['featurizer']}|{row['model']}"
            writer.writerow(
                [
                    label,
                    f"{float(row['rmse']):.8f}",
                    f"{float(row['mae']):.8f}",
                    f"{float(row['r2']):.8f}",
                    f"{float(row['runtime_s']):.8f}",
                ]
            )


def render_plot(dataset: str, table_path: Path, plot_path: Path, benchmark_title: str = "Polymer benchmark") -> None:
    gnuplot_script = f"""
set terminal pngcairo size 2200,1300
set output '{plot_path.as_posix()}'
set datafile separator '\\t'
set key off
set grid ytics lc rgb '#D9D9D9'
set style fill solid 0.85 border -1
set boxwidth 0.7
set multiplot layout 2,2 title '{benchmark_title}: {dataset}'

set xtics rotate by -30
set title 'RMSE (lower is better)'
plot '{table_path.as_posix()}' using 2:xticlabels(1) every ::1 with boxes lc rgb '#4C78A8'

set title 'MAE (lower is better)'
plot '{table_path.as_posix()}' using 3:xticlabels(1) every ::1 with boxes lc rgb '#F58518'

set title 'R2 (higher is better)'
set yrange [0:*]
plot '{table_path.as_posix()}' using 4:xticlabels(1) every ::1 with boxes lc rgb '#54A24B'
unset yrange

set title 'Runtime (s, log scale)'
set logscale y
plot '{table_path.as_posix()}' using 5:xticlabels(1) every ::1 with boxes lc rgb '#B279A2'
unset logscale y

unset multiplot
"""
    subprocess.run(["gnuplot"], input=gnuplot_script, text=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("benchmarks/results/polymers/polymer_property_benchmark_summary.csv"),
        help="Path to polymer benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/polymers/plots"),
        help="Output directory for per-dataset PNG bar plots.",
    )
    args = parser.parse_args()

    grouped = load_rows(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, rows in sorted(grouped.items()):
        table_path = args.output_dir / f"{dataset}_benchmark_table.tsv"
        plot_path = args.output_dir / f"{dataset}_benchmark_bars.png"
        write_dataset_table(rows, table_path)
        render_plot(dataset=dataset, table_path=table_path, plot_path=plot_path, benchmark_title="Polymer benchmark")
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
