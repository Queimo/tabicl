#!/usr/bin/env python3
"""Aggregate all benchmark fold-partial outputs under benchmarks/results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("benchmarks/results"))
    p.add_argument(
        "--partial-glob",
        type=str,
        default="*_benchmark_results_*.csv",
        help="Recursive glob used to locate fold-partial result CSVs.",
    )
    p.add_argument("--n-folds", type=int, default=None, help="Global expected fold count override")
    p.add_argument("--max-warning-lines", type=int, default=50)
    return p.parse_args()


def parse_partial_name(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if "_benchmark_results_" not in stem:
        return None
    head, _tag = stem.split("_benchmark_results_", 1)
    benchmark_prefix = f"{head}_benchmark"
    return benchmark_prefix, path.parent.as_posix()


def discover_partial_groups(results_root: Path, partial_glob: str) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(results_root.rglob(partial_glob)):
        parsed = parse_partial_name(path)
        if parsed is None:
            continue
        benchmark_prefix, parent = parsed
        groups[(benchmark_prefix, parent)].append(path)
    return groups


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return None


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_keys = [c for c in ("source", "benchmark", "task", "dataset", "featurizer", "fold", "model") if c in df.columns]
    if not dedupe_keys:
        return df
    return df.sort_values("__source_mtime").drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)


def infer_expected_folds(df: pd.DataFrame, config_files: list[Path], n_folds_override: int | None) -> list[int]:
    if n_folds_override is not None and n_folds_override > 0:
        return list(range(1, n_folds_override + 1))

    cfg_n_folds: int | None = None
    cfg_max_folds: int | None = None
    for path in config_files:
        try:
            cfg = json.loads(path.read_text())
        except Exception as exc:
            print(f"WARNING: failed to read config {path}: {exc}")
            continue
        n_folds = cfg.get("n_folds")
        max_folds = cfg.get("max_folds")
        if isinstance(n_folds, int) and n_folds > 0:
            cfg_n_folds = n_folds if cfg_n_folds is None else min(cfg_n_folds, n_folds)
        if isinstance(max_folds, int) and max_folds > 0:
            cfg_max_folds = max_folds if cfg_max_folds is None else min(cfg_max_folds, max_folds)

    if cfg_n_folds is not None:
        cap = cfg_n_folds
        if cfg_max_folds is not None:
            cap = min(cap, cfg_max_folds)
        return list(range(1, cap + 1))

    if "fold" in df.columns and not df.empty:
        folds = sorted(pd.to_numeric(df["fold"], errors="coerce").dropna().astype(int).unique().tolist())
        return folds
    return []


def warn_missing_folds(df: pd.DataFrame, expected_folds: list[int], max_warning_lines: int) -> None:
    if df.empty or "fold" not in df.columns:
        return
    if not expected_folds:
        print("WARNING: unable to infer expected folds; skipping missing-fold check.")
        return

    metric_candidates = {"rmse", "mae", "r2", "runtime_s", "accuracy", "roc_auc", "prc", "f1", "mse"}
    skip = {"fold", "n_train", "n_test", "n_features", "__source_file", "__source_mtime"} | metric_candidates
    combo_cols = [c for c in df.columns if c not in skip]
    if not combo_cols:
        combo_cols = [c for c in ("dataset", "model") if c in df.columns]
    if not combo_cols:
        return

    fold_series = pd.to_numeric(df["fold"], errors="coerce").dropna().astype(int)
    if fold_series.empty:
        return
    df = df.loc[fold_series.index].copy()
    df["__fold_int"] = fold_series.values

    missing_rows: list[tuple[dict, list[int]]] = []
    expected_set = set(expected_folds)
    grouped = df.groupby(combo_cols, dropna=False)["__fold_int"].apply(lambda s: set(s.tolist()))
    for keys, observed in grouped.items():
        missing = sorted(expected_set - observed)
        if missing:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = {combo_cols[i]: keys[i] for i in range(len(combo_cols))}
            missing_rows.append((key_map, missing))

    if not missing_rows:
        print("No missing fold runs detected.")
        return

    print(f"WARNING: detected {len(missing_rows)} combinations with missing folds.")
    for key_map, missing in missing_rows[:max_warning_lines]:
        combo_str = ", ".join(f"{k}={v}" for k, v in key_map.items())
        print(f"WARNING: missing folds [{', '.join(map(str, missing))}] for {combo_str}")
    if len(missing_rows) > max_warning_lines:
        print(f"WARNING: ... and {len(missing_rows) - max_warning_lines} more combinations.")


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    metric_candidates = ["rmse", "mae", "r2", "runtime_s", "accuracy", "roc_auc", "prc", "f1", "mse"]
    metric_cols = [c for c in metric_candidates if c in df.columns]
    if not metric_cols:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        metric_cols = [c for c in numeric_cols if c not in {"fold", "n_train", "n_test", "n_features"}]
    if not metric_cols:
        return pd.DataFrame()

    skip_cols = set(metric_cols) | {"fold", "n_train", "n_test", "n_features", "__source_file", "__source_mtime"}
    group_cols = [c for c in df.columns if c not in skip_cols]
    if not group_cols:
        summary = pd.DataFrame([df[metric_cols].mean(numeric_only=True)])
    else:
        summary = df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)

    if "rmse" in summary.columns:
        sort_cols = [c for c in group_cols if c in summary.columns] + ["rmse"]
        summary = summary.sort_values(sort_cols)
    elif group_cols:
        summary = summary.sort_values(group_cols)
    return summary


def aggregate_one_group(
    benchmark_prefix: str,
    parent_dir: Path,
    partial_files: list[Path],
    n_folds_override: int | None,
    max_warning_lines: int,
) -> None:
    frames = []
    for path in partial_files:
        df = read_csv_safe(path)
        if df is None:
            continue
        df["__source_file"] = path.name
        df["__source_mtime"] = path.stat().st_mtime
        frames.append(df)

    if not frames:
        print(f"WARNING: no readable result files for {benchmark_prefix} in {parent_dir}")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = dedupe_rows(combined)

    config_files = sorted(parent_dir.glob(f"{benchmark_prefix}_config*.json"))
    expected_folds = infer_expected_folds(combined, config_files, n_folds_override)
    warn_missing_folds(combined, expected_folds, max_warning_lines=max_warning_lines)

    detail_path = parent_dir / f"{benchmark_prefix}_results.csv"
    summary_path = parent_dir / f"{benchmark_prefix}_summary.csv"
    clean_df = combined.drop(columns=[c for c in ("__source_file", "__source_mtime") if c in combined.columns])
    summary = compute_summary(clean_df)

    clean_df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    groups = discover_partial_groups(args.results_root, args.partial_glob)
    if not groups:
        print(f"WARNING: no partial benchmark result files found under {args.results_root}")
        return

    print(f"Discovered {sum(len(v) for v in groups.values())} partial files across {len(groups)} benchmark groups.")
    for (benchmark_prefix, parent), files in sorted(groups.items()):
        parent_dir = Path(parent)
        print(f"\nAggregating {benchmark_prefix} ({len(files)} files) in {parent_dir}")
        aggregate_one_group(
            benchmark_prefix=benchmark_prefix,
            parent_dir=parent_dir,
            partial_files=files,
            n_folds_override=args.n_folds,
            max_warning_lines=args.max_warning_lines,
        )


if __name__ == "__main__":
    main()
