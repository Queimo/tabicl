"""Polymer property prediction benchmark.

Features:
- Two featurizers: RDKit Morgan + Mordred
- 20-fold CV (default)
- PolyCL dataset ingestion (all CSV datasets in the repo, except pretraining corpus)
- Optional T_g dataset ingestion

Typical usage:
1) Prepare PolyCL-derived polymetrics datasets:
   python -u benchmarks/polymer_property_prediction_benchmark.py --prepare-only

2) Smoke test pipeline:
   python -u benchmarks/polymer_property_prediction_benchmark.py --smoke-test --max-folds 1

3) Full run (all prepared datasets + T_g if provided):
   python -u benchmarks/polymer_property_prediction_benchmark.py --tg-path path/to/Tg.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

POLYCL_REPO = "https://github.com/JiajunZhou96/PolyCL"
POLYCL_LOCAL = Path("benchmarks/.cache/PolyCL")
POLYMETRICS_DIR = Path("benchmarks/datasets/polymetrics")
DEFAULT_OUTPUT_DIR = Path("benchmarks/results/polymers")
TABICL_OFFLOAD_DIR = Path("benchmarks/.cache/tabicl_offload")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from tabicl import TabICLRegressor

try:
    import ctypes

    LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    LIBC = None


@dataclass
class DatasetPack:
    name: str
    smiles: pd.Series
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=20)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--n-morgan-bits", type=int, default=1024)
    p.add_argument("--max-datasets", type=int, default=None)
    p.add_argument("--tg-path", type=Path, default=None, help="Path to T_g CSV (columns: smiles + target/value/tg)")
    p.add_argument("--tabicl-batch-size", type=int, default=4, help="TabICL inference batch size (lower = less RAM)")
    p.add_argument(
        "--tabicl-offload-mode",
        type=str,
        choices=["auto", "cpu", "disk"],
        default="cpu",
        help="TabICL offload mode",
    )
    p.add_argument("--tabicl-n-jobs", type=int, default=1, help="CPU threads for TabICL")
    p.add_argument(
        "--tabicl-max-train-rows",
        type=int,
        default=2000,
        help="Cap TabICL training rows per fold to reduce memory (set <=0 to disable)",
    )
    p.add_argument("--xgb-n-jobs", type=int, default=1, help="CPU threads for XGBoost")
    p.add_argument("--rf-n-jobs", type=int, default=1, help="CPU threads for RandomForest")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def trim_memory() -> None:
    gc.collect()
    if LIBC is not None and hasattr(LIBC, "malloc_trim"):
        try:
            LIBC.malloc_trim(0)
        except Exception:
            pass


def ensure_polycl_repo() -> None:
    if POLYCL_LOCAL.exists():
        try:
            subprocess.run(["git", "-C", str(POLYCL_LOCAL), "pull", "--ff-only"], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: unable to update cached PolyCL repo ({exc}). Using local cache at {POLYCL_LOCAL}.")
    else:
        POLYCL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "--depth", "1", POLYCL_REPO, str(POLYCL_LOCAL)], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Unable to clone PolyCL repository from {POLYCL_REPO}. "
                "Please provide internet access or pre-populate benchmarks/.cache/PolyCL."
            ) from exc


def normalize_polycl_datasets() -> list[str]:
    """Create normalized polymetrics datasets for all PolyCL benchmark CSVs."""
    src_dir = POLYCL_LOCAL / "datasets"
    POLYMETRICS_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    for csv_path in sorted(src_dir.glob("*.csv")):
        name = csv_path.stem
        if name == "pretrain_1m":
            continue

        df = pd.read_csv(csv_path)
        if "smiles" in df.columns:
            smiles_col = "smiles"
        elif "enumeration" in df.columns:
            smiles_col = "enumeration"
        else:
            continue

        target_col = "value" if "value" in df.columns else None
        if target_col is None:
            continue

        out = pd.DataFrame(
            {
                "dataset": name,
                "smiles": df[smiles_col].astype(str),
                "target": pd.to_numeric(df[target_col], errors="coerce"),
                "source": "PolyCL",
            }
        ).dropna(subset=["smiles", "target"])

        out_path = POLYMETRICS_DIR / f"{name}.csv"
        out.to_csv(out_path, index=False)
        created.append(name)

    manifest = {
        "source_repo": POLYCL_REPO,
        "datasets": created,
    }
    (POLYMETRICS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return created


def infer_tg_columns(df: pd.DataFrame) -> tuple[str, str]:
    smiles_candidates = ["smiles", "SMILES", "polymer_smiles", "Polymer_SMILES", "enumeration"]
    target_candidates = ["Tg", "tg", "T_g", "target", "value", "y"]

    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if smiles_col is None or target_col is None:
        raise ValueError(
            f"Unable to infer columns for Tg dataset. Columns found: {df.columns.tolist()}"
        )
    return smiles_col, target_col


def load_all_datasets(tg_path: Path | None, smoke_test: bool) -> list[DatasetPack]:
    if smoke_test:
        s = pd.Series(["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCO", "CCN", "CCC", "CCCl", "CCBr"])  # noqa: E501
        y = np.array([0.10, 0.20, 0.15, 0.35, 0.50, 0.12, 0.18, 0.17, 0.33, 0.55], dtype=np.float32)
        return [DatasetPack("smoke_polymer", s, y)]

    packs: list[DatasetPack] = []
    for csv_path in sorted(POLYMETRICS_DIR.glob("*.csv")):
        stem = csv_path.stem.lower()
        if "template" in stem or stem.endswith("_enum"):
            continue
        df = pd.read_csv(csv_path)
        try:
            smiles_col, target_col = infer_tg_columns(df)
        except ValueError:
            print(f"Skipping {csv_path.name}: unable to infer smiles/target columns")
            continue

        clean = pd.DataFrame(
            {
                "smiles": df[smiles_col].astype(str),
                "target": pd.to_numeric(df[target_col], errors="coerce"),
            }
        ).dropna(subset=["smiles", "target"])
        if clean.empty:
            print(f"Skipping {csv_path.name}: no valid rows after cleaning")
            continue

        packs.append(
            DatasetPack(
                csv_path.stem,
                clean["smiles"],
                clean["target"].to_numpy(dtype=np.float32),
            )
        )

    auto_tg = POLYMETRICS_DIR / "Tg.csv"
    if tg_path is None and auto_tg.exists():
        tg_path = auto_tg

    if tg_path is not None and tg_path.exists():
        tg_df = pd.read_csv(tg_path)
        smiles_col, target_col = infer_tg_columns(tg_df)
        clean_tg = pd.DataFrame(
            {
                "smiles": tg_df[smiles_col].astype(str),
                "target": pd.to_numeric(tg_df[target_col], errors="coerce"),
            }
        ).dropna(subset=["smiles", "target"])
        if clean_tg.empty:
            print(f"Skipping Tg dataset from {tg_path}: no valid rows after cleaning")
            return packs
        packs.append(
            DatasetPack(
                "Tg",
                clean_tg["smiles"],
                clean_tg["target"].to_numpy(dtype=np.float32),
            )
        )

    return packs


def smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def featurize_morgan(smiles: pd.Series, n_bits: int) -> tuple[np.ndarray, list[int]]:
    rows, keep = [], []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep.append(idx)
    if not rows:
        return np.empty((0, n_bits), dtype=np.float32), keep
    return np.vstack(rows), keep


def featurize_mordred(smiles: pd.Series) -> tuple[np.ndarray, list[int]]:
    calc = Calculator(descriptors, ignore_3D=True)
    mols, keep = [], []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        mols.append(mol)
        keep.append(idx)
    if not mols:
        return np.empty((0, 0), dtype=np.float32), keep

    desc = calc.pandas(mols, nproc=1)
    desc = desc.apply(pd.to_numeric, errors="coerce")
    desc = desc.replace([np.inf, -np.inf], np.nan)
    desc = desc.loc[:, desc.isna().mean(axis=0) <= 0.4]
    X = SimpleImputer(strategy="median").fit_transform(desc)
    return X.astype(np.float32), keep


def models(args: argparse.Namespace):
    return {
        "TabICL": lambda: TabICLRegressor(
            n_estimators=1,
            random_state=args.random_state,
            device="cpu",
            n_jobs=args.tabicl_n_jobs,
            batch_size=args.tabicl_batch_size,
            offload_mode=args.tabicl_offload_mode,
            disk_offload_dir=str(TABICL_OFFLOAD_DIR),
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state,
            n_jobs=args.xgb_n_jobs,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=args.random_state,
            thread_count=1,
            allow_writing_files=False,
            verbose=0,
        ),
        "RF": lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=args.rf_n_jobs,
        ),
    }


def run_cv(
    args: argparse.Namespace,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    featurizer: str,
    n_folds: int,
    max_folds: int | None,
    random_state: int,
) -> list[dict]:
    y = np.asarray(y)
    valid = np.isfinite(y)
    X, y = X[valid], y[valid]
    if len(y) < 2:
        print(f"Skipping {dataset_name} | {featurizer}: not enough valid samples ({len(y)})")
        return []

    effective_folds = max(2, min(n_folds, len(y)))
    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    recs = []
    for fold, (tr, te) in enumerate(kf.split(X, y), start=1):
        if max_folds is not None and fold > max_folds:
            break
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        for model_name, ctor in models(args).items():
            print(f"Running {dataset_name} | {featurizer} | fold {fold} | {model_name}", flush=True)
            model = None
            pred = None
            try:
                model = ctor()
                t0 = time.time()
                X_tr_fit, y_tr_fit = X_tr, y_tr
                if model_name == "TabICL" and args.tabicl_max_train_rows > 0 and len(y_tr) > args.tabicl_max_train_rows:
                    rng = np.random.default_rng(random_state + fold)
                    sel = rng.choice(len(y_tr), size=args.tabicl_max_train_rows, replace=False)
                    X_tr_fit = X_tr[sel]
                    y_tr_fit = y_tr[sel]
                    print(
                        f"TabICL train rows capped: {len(y_tr)} -> {len(y_tr_fit)} "
                        f"for {dataset_name} fold {fold}",
                        flush=True,
                    )

                model.fit(X_tr_fit, y_tr_fit)
                pred = model.predict(X_te)
                dt = time.time() - t0
                rec = {
                    "dataset": dataset_name,
                    "featurizer": featurizer,
                    "fold": fold,
                    "model": model_name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_features": int(X.shape[1]),
                    "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "mae": float(mean_absolute_error(y_te, pred)),
                    "r2": float(r2_score(y_te, pred)) if len(y_te) > 1 else np.nan,
                    "runtime_s": float(dt),
                }
                recs.append(rec)
                print(json.dumps(rec))
            finally:
                # Egc can be memory-heavy; aggressively release model and prediction buffers.
                del model
                del pred
                trim_memory()
    return recs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    TABICL_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    ensure_polycl_repo()
    created = normalize_polycl_datasets()

    if args.prepare_only:
        print(f"Prepared {len(created)} normalized PolyCL datasets in {POLYMETRICS_DIR}")
        return

    packs = load_all_datasets(tg_path=args.tg_path, smoke_test=args.smoke_test)
    if args.max_datasets is not None:
        packs = packs[: args.max_datasets]

    records = []
    for pack in packs:
        X_morgan, idx_m = featurize_morgan(pack.smiles, args.n_morgan_bits)
        y_m = pack.y[np.array(idx_m)]
        records.extend(
            run_cv(args, pack.name, X_morgan, y_m, "morgan", args.n_folds, args.max_folds, args.random_state)
        )
        del X_morgan, idx_m, y_m
        trim_memory()

        X_mordred, idx_d = featurize_mordred(pack.smiles)
        y_d = pack.y[np.array(idx_d)]
        records.extend(
            run_cv(args, pack.name, X_mordred, y_d, "mordred", args.n_folds, args.max_folds, args.random_state)
        )
        del X_mordred, idx_d, y_d
        trim_memory()

    df = pd.DataFrame(records)
    detail = args.output_dir / "polymer_property_benchmark_results.csv"
    df.to_csv(detail, index=False)

    summary = (
        df.groupby(["dataset", "featurizer", "model"], as_index=False)[["rmse", "mae", "r2", "runtime_s"]]
        .mean(numeric_only=True)
        .sort_values(["dataset", "featurizer", "rmse"])
    )
    summary_path = args.output_dir / "polymer_property_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "n_folds": args.n_folds,
        "random_state": args.random_state,
        "n_morgan_bits": args.n_morgan_bits,
        "smoke_test": args.smoke_test,
        "max_datasets": args.max_datasets,
        "max_folds": args.max_folds,
        "tabicl_batch_size": args.tabicl_batch_size,
        "tabicl_offload_mode": args.tabicl_offload_mode,
        "tabicl_n_jobs": args.tabicl_n_jobs,
        "tabicl_max_train_rows": args.tabicl_max_train_rows,
        "xgb_n_jobs": args.xgb_n_jobs,
        "rf_n_jobs": args.rf_n_jobs,
        "tg_path": str(args.tg_path) if args.tg_path else None,
        "polycl_repo": POLYCL_REPO,
        "prepared_polycl_datasets": created,
    }
    config_path = args.output_dir / "polymer_property_benchmark_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved: {detail}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
