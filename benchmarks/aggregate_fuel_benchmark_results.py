#!/usr/bin/env python3
"""Aggregate partial fuel benchmark fold outputs into canonical result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_MODELS = ("TabICL", "XGBoost", "CatBoost", "RF")
EXPECTED_FEATURIZERS = ("morgan", "mordred")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("benchmarks/results/fuels"))
    p.add_argument(
        "--input-glob",
        type=str,
        default="fuel_property_benchmark_results_*.csv",
        help="Glob for partial fold result files. Falls back to canonical file if no matches.",
    )
    p.add_argument("--n-folds", type=int, default=None, help="Expected total CV folds")
    p.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help="Expected datasets/targets (default: infer from configs/results)",
    )
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return None


def load_partial_results(results_dir: Path, input_glob: str) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(results_dir.glob(input_glob))
    if not files:
        canonical = results_dir / "fuel_property_benchmark_results.csv"
        if canonical.exists():
            files = [canonical]
            print(f"WARNING: no partial files matched '{input_glob}', falling back to {canonical.name}")
        else:
            return pd.DataFrame(), []

    frames: list[pd.DataFrame] = []
    for path in files:
        df = safe_read_csv(path)
        if df is None:
            continue
        df["__source_file"] = path.name
        df["__source_mtime"] = path.stat().st_mtime
        frames.append(df)

    if not frames:
        return pd.DataFrame(), files

    combined = pd.concat(frames, ignore_index=True)
    required = {"dataset", "featurizer", "fold", "model"}
    if required.issubset(combined.columns):
        combined = (
            combined.sort_values("__source_mtime")
            .drop_duplicates(subset=["dataset", "featurizer", "fold", "model"], keep="last")
            .reset_index(drop=True)
        )
    return combined, files


def load_config_expectations(results_dir: Path) -> dict:
    config_paths = sorted(results_dir.glob("fuel_property_benchmark_config_*.json"))
    if not config_paths:
        fallback = results_dir / "fuel_property_benchmark_config.json"
        if fallback.exists():
            config_paths = [fallback]

    configs = []
    for path in config_paths:
        try:
            configs.append(json.loads(path.read_text()))
        except Exception as exc:
            print(f"WARNING: failed to read config {path}: {exc}")

    if not configs:
        return {}

    cfg = configs[0]
    n_folds = cfg.get("n_folds")
    max_folds = cfg.get("max_folds")
    targets = cfg.get("targets") or []
    max_datasets = cfg.get("max_datasets")
    if isinstance(max_datasets, int) and max_datasets >= 0:
        targets = targets[:max_datasets]

    expect_chemprop = not bool(cfg.get("disable_chemprop", False))
    expect_chemeleon = not bool(cfg.get("disable_chemprop_chemeleon", False))

    return {
        "n_folds": n_folds,
        "max_folds": max_folds,
        "targets": targets,
        "expect_chemprop": expect_chemprop,
        "expect_chemeleon": expect_chemeleon,
    }


def infer_expected_folds(n_folds_arg: int | None, cfg: dict, df: pd.DataFrame) -> list[int]:
    if n_folds_arg is not None and n_folds_arg > 0:
        return list(range(1, n_folds_arg + 1))

    cfg_n_folds = cfg.get("n_folds")
    cfg_max_folds = cfg.get("max_folds")
    if isinstance(cfg_n_folds, int) and cfg_n_folds > 0:
        cap = cfg_n_folds
        if isinstance(cfg_max_folds, int) and cfg_max_folds > 0:
            cap = min(cap, cfg_max_folds)
        return list(range(1, cap + 1))

    if "fold" in df.columns and not df.empty:
        folds = sorted({int(f) for f in pd.to_numeric(df["fold"], errors="coerce").dropna().astype(int).tolist()})
        return folds
    return []


def infer_expected_targets(targets_arg: list[str] | None, cfg: dict, df: pd.DataFrame) -> list[str]:
    if targets_arg:
        return list(targets_arg)
    if cfg.get("targets"):
        return list(cfg["targets"])
    if "dataset" in df.columns and not df.empty:
        return sorted(df["dataset"].dropna().astype(str).unique().tolist())
    return []


def warn_missing_runs(df: pd.DataFrame, expected_folds: list[int], expected_targets: list[str], cfg: dict) -> None:
    if df.empty:
        print("WARNING: no rows available to validate run completeness.")
        return
    if not expected_folds:
        print("WARNING: unable to infer expected folds; skipping missing-run check.")
        return
    if not expected_targets:
        print("WARNING: unable to infer expected targets; skipping missing-run check.")
        return

    expected_pairs = [(feat, model) for feat in EXPECTED_FEATURIZERS for model in BASE_MODELS]
    if cfg.get("expect_chemprop", False):
        expected_pairs.append(("identity", "ChempropGNN"))
        if cfg.get("expect_chemeleon", False):
            expected_pairs.append(("identity", "ChempropCheMeleon"))

    observed = set(
        zip(
            df["dataset"].astype(str),
            df["featurizer"].astype(str),
            pd.to_numeric(df["fold"], errors="coerce").fillna(-1).astype(int),
            df["model"].astype(str),
        )
    )

    missing = []
    for ds in expected_targets:
        for feat, model in expected_pairs:
            for fold in expected_folds:
                key = (ds, feat, fold, model)
                if key not in observed:
                    missing.append(key)

    if not missing:
        print("No missing runs detected.")
        return

    print(f"WARNING: detected {len(missing)} missing dataset/featurizer/fold/model runs.")
    for ds, feat, fold, model in missing[:50]:
        print(f"WARNING: missing -> dataset={ds}, featurizer={feat}, fold={fold}, model={model}")
    if len(missing) > 50:
        print(f"WARNING: ... and {len(missing) - 50} more missing combinations.")


def write_outputs(results_dir: Path, df: pd.DataFrame) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    detail_path = results_dir / "fuel_property_benchmark_results.csv"
    summary_path = results_dir / "fuel_property_benchmark_summary.csv"

    if "__source_file" in df.columns:
        df = df.drop(columns=[c for c in ("__source_file", "__source_mtime") if c in df.columns])

    if df.empty:
        df = pd.DataFrame(
            columns=["dataset", "featurizer", "fold", "model", "n_train", "n_test", "n_features", "rmse", "mae", "r2", "runtime_s"]
        )
        summary = pd.DataFrame(columns=["dataset", "featurizer", "model", "rmse", "mae", "r2", "runtime_s"])
    else:
        summary = (
            df.groupby(["dataset", "featurizer", "model"], as_index=False)[["rmse", "mae", "r2", "runtime_s"]]
            .mean(numeric_only=True)
            .sort_values(["dataset", "featurizer", "rmse"])
        )

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    df, files = load_partial_results(args.results_dir, args.input_glob)

    if files:
        print(f"Loaded {len(files)} result file(s).")
    else:
        print("WARNING: no result files found to aggregate.")

    cfg_expectations = load_config_expectations(args.results_dir)
    expected_folds = infer_expected_folds(args.n_folds, cfg_expectations, df)
    expected_targets = infer_expected_targets(args.targets, cfg_expectations, df)
    warn_missing_runs(df, expected_folds, expected_targets, cfg_expectations)
    write_outputs(args.results_dir, df)


if __name__ == "__main__":
    main()
