"""Interaction benchmark for polymer-solvent property prediction.

Target:
- average_IP

Inputs:
- polymer_smiles
- solvent_smiles
- T_K
- volume_fraction

Featurizers:
- morgan_pair: Morgan(polymer) + Morgan(solvent) + [T_K, volume_fraction]
- mordred_pair: Mordred(polymer) + Mordred(solvent) + [T_K, volume_fraction]

Models:
- TabICL, XGBoost, CatBoost, RandomForest
- Optional ChempropGNN + ChempropCheMeleon
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
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

DEFAULT_DATASET_PATH = Path("benchmarks/datasets/polymetrics/chi_clean.csv")
DEFAULT_OUTPUT_DIR = Path("benchmarks/results/interactions")
TABICL_OFFLOAD_DIR = Path("benchmarks/.cache/tabicl_offload")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--fold-index", type=int, default=None, help="Run only this 1-based CV fold")
    p.add_argument(
        "--results-tag",
        type=str,
        default=None,
        help="Optional tag appended to output filenames (e.g., fold_1)",
    )
    p.add_argument("--n-morgan-bits", type=int, default=1024)
    p.add_argument("--featurizers", choices=["morgan", "mordred", "both"], default="both")
    p.add_argument("--tabicl-batch-size", type=int, default=4)
    p.add_argument("--tabicl-n-jobs", type=int, default=1)
    p.add_argument("--tabicl-max-train-rows", type=int, default=4000)
    p.add_argument("--xgb-n-jobs", type=int, default=1)
    p.add_argument("--rf-n-jobs", type=int, default=1)
    p.add_argument("--disable-chemprop", action="store_true", help="Disable Chemprop GNN model")
    p.add_argument(
        "--disable-chemprop-chemeleon",
        action="store_true",
        help="Disable Chemprop CheMeleon foundation model",
    )
    p.add_argument("--chemprop-bin", type=str, default="chemprop", help="Chemprop CLI executable")
    p.add_argument("--chemprop-featurizers", choices=["morgan", "mordred", "both"], default="morgan")
    p.add_argument(
        "--chemprop-chemeleon-name",
        type=str,
        default="CheMeleon",
        help="Value passed to Chemprop --from-foundation for CheMeleon model",
    )
    p.add_argument("--chemprop-epochs", type=int, default=50)
    p.add_argument("--chemprop-batch-size", type=int, default=64)
    p.add_argument("--chemprop-num-workers", type=int, default=0)
    p.add_argument("--chemprop-accelerator", type=str, default="cpu")
    p.add_argument("--chemprop-devices", type=str, default="1")
    p.add_argument(
        "--chemprop-val-fraction",
        type=float,
        default=0.1,
        help="Fraction of fold-train rows duplicated as validation rows for Chemprop",
    )
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def trim_memory() -> None:
    gc.collect()
    if LIBC is not None and hasattr(LIBC, "malloc_trim"):
        try:
            LIBC.malloc_trim(0)
        except Exception:
            pass


def parse_decimal_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def load_dataset(path: Path, smoke_test: bool) -> pd.DataFrame:
    if smoke_test:
        return pd.DataFrame(
            {
                "polymer_smiles": ["*CC*", "*CC*", "*C=C*", "*C=C*", "*CCO*", "*CCO*"],
                "solvent_smiles": ["CCO", "CC(C)=O", "O", "CCN", "CCO", "O"],
                "T_K": [298.15, 308.15, 298.15, 313.15, 303.15, 323.15],
                "volume_fraction": [0.1, 0.3, 0.2, 0.4, 0.15, 0.5],
                "average_IP": [0.25, 0.15, 0.30, 0.05, 0.28, 0.10],
            }
        )

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed:")]]
    required = ["polymer_smiles", "solvent_smiles", "T_K", "volume_fraction", "average_IP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {df.columns.tolist()}")

    clean = pd.DataFrame(
        {
            "polymer_smiles": df["polymer_smiles"].astype(str).str.strip(),
            "solvent_smiles": df["solvent_smiles"].astype(str).str.strip(),
            "T_K": parse_decimal_series(df["T_K"]),
            "volume_fraction": parse_decimal_series(df["volume_fraction"]),
            "average_IP": parse_decimal_series(df["average_IP"]),
        }
    )
    clean = clean.replace({"polymer_smiles": {"": np.nan}, "solvent_smiles": {"": np.nan}})
    clean = clean.dropna(subset=["polymer_smiles", "solvent_smiles", "T_K", "volume_fraction", "average_IP"])
    clean = clean.reset_index(drop=True)
    return clean


def smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def featurize_morgan_pair(df: pd.DataFrame, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    keep_idx = []
    for idx, row in df.iterrows():
        polymer = smiles_to_mol(row["polymer_smiles"])
        solvent = smiles_to_mol(row["solvent_smiles"])
        if polymer is None or solvent is None:
            continue

        fp_poly = AllChem.GetMorganFingerprintAsBitVect(polymer, radius=2, nBits=n_bits)
        fp_solvent = AllChem.GetMorganFingerprintAsBitVect(solvent, radius=2, nBits=n_bits)

        arr_poly = np.zeros((n_bits,), dtype=np.float32)
        arr_solvent = np.zeros((n_bits,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp_poly, arr_poly)
        Chem.DataStructs.ConvertToNumpyArray(fp_solvent, arr_solvent)

        numeric = np.array([row["T_K"], row["volume_fraction"]], dtype=np.float32)
        feat = np.concatenate([arr_poly, arr_solvent, numeric], axis=0)
        rows.append(feat)
        keep_idx.append(idx)

    if not rows:
        return np.empty((0, 2 * n_bits + 2), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(rows), np.array(keep_idx, dtype=int)


def build_mordred_lookup(smiles_values: pd.Series) -> dict[str, np.ndarray]:
    calc = Calculator(descriptors, ignore_3D=True)
    unique_smiles = pd.Series(smiles_values.astype(str).unique())
    mols = []
    valid_smiles = []
    for s in unique_smiles:
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        valid_smiles.append(s)
        mols.append(mol)

    if not mols:
        return {}

    desc = calc.pandas(mols, nproc=1)
    desc = desc.apply(pd.to_numeric, errors="coerce")
    desc = desc.replace([np.inf, -np.inf], np.nan)
    desc = desc.loc[:, desc.isna().mean(axis=0) <= 0.4]
    X = SimpleImputer(strategy="median").fit_transform(desc).astype(np.float32)
    return {s: X[i] for i, s in enumerate(valid_smiles)}


def featurize_mordred_pair(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    poly_lookup = build_mordred_lookup(df["polymer_smiles"])
    solvent_lookup = build_mordred_lookup(df["solvent_smiles"])
    if not poly_lookup or not solvent_lookup:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int)

    rows = []
    keep_idx = []
    for idx, row in df.iterrows():
        p = poly_lookup.get(row["polymer_smiles"])
        s = solvent_lookup.get(row["solvent_smiles"])
        if p is None or s is None:
            continue
        numeric = np.array([row["T_K"], row["volume_fraction"]], dtype=np.float32)
        rows.append(np.concatenate([p, s, numeric], axis=0))
        keep_idx.append(idx)

    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(rows).astype(np.float32), np.array(keep_idx, dtype=int)


def model_builders(args: argparse.Namespace) -> dict:
    return {
        "TabICL": lambda: TabICLRegressor(
            n_estimators=1,
            random_state=args.random_state,
            device="cpu",
            n_jobs=args.tabicl_n_jobs,
            batch_size=args.tabicl_batch_size,
            offload_mode="cpu",
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


def sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in str(name))
    safe = safe.strip("_")
    return safe or "item"


def should_run_chemprop(args: argparse.Namespace, featurizer: str) -> bool:
    if args.disable_chemprop:
        return False
    if args.chemprop_featurizers == "both":
        return True
    if args.chemprop_featurizers == "morgan":
        return featurizer == "morgan_pair"
    return featurizer == "mordred_pair"


def chemprop_variants(args: argparse.Namespace) -> list[tuple[str, str, str | None]]:
    variants: list[tuple[str, str, str | None]] = [("ChempropGNN", "chemprop_default", None)]
    if not args.disable_chemprop_chemeleon:
        variants.append(("ChempropCheMeleon", "chemprop_chemeleon", args.chemprop_chemeleon_name))
    return variants


def run_chemprop_fold(
    args: argparse.Namespace,
    dataset_name: str,
    featurizer: str,
    fold: int,
    train_rows: pd.DataFrame,
    train_y: np.ndarray,
    test_rows: pd.DataFrame,
    random_state: int,
    run_name: str,
    from_foundation: str | None,
) -> np.ndarray:
    chemprop_home = args.output_dir / "chemprop_home"
    chemprop_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(chemprop_home.resolve())
    env.setdefault("XDG_CACHE_HOME", str((chemprop_home / ".cache").resolve()))

    fold_dir = (
        args.output_dir
        / "chemprop_runs"
        / sanitize_name(run_name)
        / sanitize_name(dataset_name)
        / sanitize_name(featurizer)
        / f"fold_{fold}"
    )
    shutil.rmtree(fold_dir, ignore_errors=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    val_fraction = float(np.clip(args.chemprop_val_fraction, 0.0, 1.0))
    n_val = max(1, int(round(len(train_rows) * val_fraction)))
    n_val = min(n_val, len(train_rows))
    rng = np.random.default_rng(random_state + fold)
    val_idx = rng.choice(len(train_rows), size=n_val, replace=False)

    train_df = train_rows.copy().reset_index(drop=True)
    train_df["target"] = train_y
    train_df["split"] = "train"
    if len(train_df) < 2:
        extra = train_df.iloc[rng.choice(len(train_df), size=2 - len(train_df), replace=True)].copy()
        extra["split"] = "train"
        train_df = pd.concat([train_df, extra], ignore_index=True)

    val_df = train_df.iloc[val_idx].copy()
    if len(val_df) < 2:
        extra = train_df.iloc[rng.choice(len(train_df), size=2 - len(val_df), replace=True)].copy()
        val_df = pd.concat([val_df, extra], ignore_index=True)
    val_df["split"] = "val"
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    fit_csv = fold_dir / "fit.csv"
    fit_df.to_csv(fit_csv, index=False)

    test_df = test_rows.copy().reset_index(drop=True)
    test_df["_row_id"] = np.arange(len(test_df), dtype=np.int32)
    test_csv = fold_dir / "test.csv"
    test_df.to_csv(test_csv, index=False)

    model_dir = fold_dir / "model"
    pred_csv = fold_dir / "predictions.csv"
    warmup_epochs = min(2, max(0, int(args.chemprop_epochs) - 1))
    batch_size = max(2, min(int(args.chemprop_batch_size), len(train_df)))

    train_cmd = [
        args.chemprop_bin,
        "train",
        "--data-path",
        str(fit_csv),
        "--output-dir",
        str(model_dir),
        "--smiles-columns",
        "polymer_smiles",
        "solvent_smiles",
        "--descriptors-columns",
        "T_K",
        "volume_fraction",
        "--target-columns",
        "target",
        "--splits-column",
        "split",
        "--task-type",
        "regression",
        "--epochs",
        str(args.chemprop_epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.chemprop_num_workers),
        "--accelerator",
        str(args.chemprop_accelerator),
        "--devices",
        str(args.chemprop_devices),
    ]
    if from_foundation:
        train_cmd.extend(["--from-foundation", from_foundation])

    predict_cmd = [
        args.chemprop_bin,
        "predict",
        "--test-path",
        str(test_csv),
        "--output",
        str(pred_csv),
        "--model-paths",
        str(model_dir),
        "--smiles-columns",
        "polymer_smiles",
        "solvent_smiles",
        "--descriptors-columns",
        "T_K",
        "volume_fraction",
        "--batch-size",
        str(max(1, min(batch_size, len(test_rows)))),
        "--num-workers",
        str(args.chemprop_num_workers),
        "--accelerator",
        str(args.chemprop_accelerator),
        "--devices",
        str(args.chemprop_devices),
    ]

    train_log = fold_dir / "chemprop_train.log"
    predict_log = fold_dir / "chemprop_predict.log"
    try:
        with train_log.open("w") as fh:
            subprocess.run(train_cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Chemprop train failed for fold {fold}. See {train_log}") from exc

    try:
        with predict_log.open("w") as fh:
            subprocess.run(predict_cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Chemprop predict failed for fold {fold}. See {predict_log}") from exc

    pred_df = pd.read_csv(pred_csv)
    if "_row_id" in pred_df.columns:
        pred_df = pred_df.sort_values("_row_id")

    pred_cols = [c for c in pred_df.columns if c not in {"polymer_smiles", "solvent_smiles", "T_K", "volume_fraction", "_row_id"}]
    non_unc_cols = [c for c in pred_cols if not c.endswith("_unc")]
    if "target" in non_unc_cols:
        pred_col = "target"
    elif len(non_unc_cols) == 1:
        pred_col = non_unc_cols[0]
    else:
        raise RuntimeError(f"Unable to infer Chemprop prediction column from {pred_cols} in {pred_csv}")

    pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float32)
    if len(pred) != len(test_rows) or np.isnan(pred).any():
        raise RuntimeError(f"Invalid Chemprop predictions in {pred_csv}: shape={pred.shape}, n_test={len(test_rows)}")
    return pred


def run_cv(
    args: argparse.Namespace,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    featurizer: str,
    rows_for_chemprop: pd.DataFrame | None,
    n_folds: int,
    max_folds: int | None,
    random_state: int,
    selected_folds: set[int] | None = None,
) -> list[dict]:
    if len(y) < 2:
        print(f"Skipping {dataset_name} | {featurizer}: not enough rows ({len(y)})")
        return []

    y = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]
    if rows_for_chemprop is not None:
        rows_for_chemprop = rows_for_chemprop.iloc[np.where(valid)[0]].reset_index(drop=True)
    if len(y) < 2:
        print(f"Skipping {dataset_name} | {featurizer}: no valid target values after filtering")
        return []

    effective_folds = max(2, min(n_folds, len(y)))
    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    records: list[dict] = []
    executed_folds: set[int] = set()
    for fold, (tr, te) in enumerate(kf.split(X, y), start=1):
        if max_folds is not None and fold > max_folds:
            break
        if selected_folds is not None and fold not in selected_folds:
            continue
        executed_folds.add(fold)

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        for model_name, builder in model_builders(args).items():
            print(f"Running {dataset_name} | {featurizer} | fold {fold} | {model_name}", flush=True)
            model = None
            pred = None
            try:
                model = builder()
                X_fit, y_fit = X_tr, y_tr
                if model_name == "TabICL" and args.tabicl_max_train_rows > 0 and len(y_tr) > args.tabicl_max_train_rows:
                    rng = np.random.default_rng(args.random_state + fold)
                    sel = rng.choice(len(y_tr), size=args.tabicl_max_train_rows, replace=False)
                    X_fit = X_tr[sel]
                    y_fit = y_tr[sel]
                t0 = time.time()
                model.fit(X_fit, y_fit)
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
                records.append(rec)
                print(json.dumps(rec))
            finally:
                del model
                del pred
                trim_memory()

        if should_run_chemprop(args, featurizer):
            if rows_for_chemprop is None:
                print(f"Skipping Chemprop for {dataset_name} | {featurizer}: missing row metadata")
            else:
                for model_name, run_name, from_foundation in chemprop_variants(args):
                    print(f"Running {dataset_name} | {featurizer} | fold {fold} | {model_name}", flush=True)
                    t0 = time.time()
                    pred = run_chemprop_fold(
                        args=args,
                        dataset_name=dataset_name,
                        featurizer=featurizer,
                        fold=fold,
                        train_rows=rows_for_chemprop.iloc[tr].reset_index(drop=True),
                        train_y=y[tr],
                        test_rows=rows_for_chemprop.iloc[te].reset_index(drop=True),
                        random_state=random_state,
                        run_name=run_name,
                        from_foundation=from_foundation,
                    )
                    dt = time.time() - t0
                    rec = {
                        "dataset": dataset_name,
                        "featurizer": "identity",
                        "fold": fold,
                        "model": model_name,
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "n_features": int(X.shape[1]),
                        "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                        "mae": float(mean_absolute_error(y[te], pred)),
                        "r2": float(r2_score(y[te], pred)) if len(y[te]) > 1 else np.nan,
                        "runtime_s": float(dt),
                    }
                    records.append(rec)
                    print(json.dumps(rec))
                    del pred
                    trim_memory()

    if selected_folds is not None and not executed_folds:
        print(
            f"Skipping {dataset_name} | {featurizer}: requested folds {sorted(selected_folds)} "
            f"outside available range 1..{effective_folds}"
        )
    return records


def output_file_with_tag(output_dir: Path, stem: str, suffix: str, tag: str | None) -> Path:
    tagged = f"{stem}_{sanitize_name(tag)}" if tag else stem
    return output_dir / f"{tagged}.{suffix}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    TABICL_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if args.fold_index is not None and args.fold_index < 1:
        raise ValueError("--fold-index must be >= 1")

    if not args.disable_chemprop:
        resolved_chemprop = shutil.which(args.chemprop_bin)
        if resolved_chemprop is None and not Path(args.chemprop_bin).exists():
            raise RuntimeError(
                f"Chemprop executable not found: '{args.chemprop_bin}'. "
                "Activate the environment that has Chemprop or pass --chemprop-bin."
            )
        if resolved_chemprop is not None:
            args.chemprop_bin = resolved_chemprop

    df = load_dataset(args.dataset_path, smoke_test=args.smoke_test)
    print(f"Loaded {len(df)} rows from {args.dataset_path}")

    selected_folds = {args.fold_index} if args.fold_index is not None else None
    effective_tag = args.results_tag or (f"fold_{args.fold_index}" if args.fold_index is not None else None)
    dataset_name = Path(args.dataset_path).stem
    all_records = []

    if args.featurizers in {"morgan", "both"}:
        X_m, keep_m = featurize_morgan_pair(df, args.n_morgan_bits)
        y_m = df.loc[keep_m, "average_IP"].to_numpy(dtype=np.float32)
        rows_m = df.loc[keep_m, ["polymer_smiles", "solvent_smiles", "T_K", "volume_fraction"]].reset_index(drop=True)
        all_records.extend(
            run_cv(
                args,
                dataset_name=dataset_name,
                X=X_m,
                y=y_m,
                featurizer="morgan_pair",
                rows_for_chemprop=rows_m,
                n_folds=args.n_folds,
                max_folds=args.max_folds,
                random_state=args.random_state,
                selected_folds=selected_folds,
            )
        )
        del X_m, keep_m, y_m, rows_m
        trim_memory()

    if args.featurizers in {"mordred", "both"}:
        X_d, keep_d = featurize_mordred_pair(df)
        y_d = df.loc[keep_d, "average_IP"].to_numpy(dtype=np.float32)
        rows_d = df.loc[keep_d, ["polymer_smiles", "solvent_smiles", "T_K", "volume_fraction"]].reset_index(drop=True)
        all_records.extend(
            run_cv(
                args,
                dataset_name=dataset_name,
                X=X_d,
                y=y_d,
                featurizer="mordred_pair",
                rows_for_chemprop=rows_d,
                n_folds=args.n_folds,
                max_folds=args.max_folds,
                random_state=args.random_state,
                selected_folds=selected_folds,
            )
        )
        del X_d, keep_d, y_d, rows_d
        trim_memory()

    out_df = pd.DataFrame(all_records)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=["dataset", "featurizer", "fold", "model", "n_train", "n_test", "n_features", "rmse", "mae", "r2", "runtime_s"]
        )
        print("Warning: no benchmark records were generated.")

    detail_path = output_file_with_tag(args.output_dir, "interaction_benchmark_results", "csv", effective_tag)
    out_df.to_csv(detail_path, index=False)

    if out_df.empty:
        summary = pd.DataFrame(columns=["dataset", "featurizer", "model", "rmse", "mae", "r2", "runtime_s"])
    else:
        summary = (
            out_df.groupby(["dataset", "featurizer", "model"], as_index=False)[["rmse", "mae", "r2", "runtime_s"]]
            .mean(numeric_only=True)
            .sort_values(["dataset", "featurizer", "rmse"])
        )
    summary_path = output_file_with_tag(args.output_dir, "interaction_benchmark_summary", "csv", effective_tag)
    summary.to_csv(summary_path, index=False)

    config = {
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "n_rows_loaded": int(len(df)),
        "target": "average_IP",
        "feature_columns": ["polymer_smiles", "solvent_smiles", "T_K", "volume_fraction"],
        "n_folds": args.n_folds,
        "max_folds": args.max_folds,
        "fold_index": args.fold_index,
        "results_tag": effective_tag,
        "random_state": args.random_state,
        "n_morgan_bits": args.n_morgan_bits,
        "featurizers": args.featurizers,
        "smoke_test": args.smoke_test,
        "tabicl_batch_size": args.tabicl_batch_size,
        "tabicl_n_jobs": args.tabicl_n_jobs,
        "tabicl_max_train_rows": args.tabicl_max_train_rows,
        "xgb_n_jobs": args.xgb_n_jobs,
        "rf_n_jobs": args.rf_n_jobs,
        "disable_chemprop": args.disable_chemprop,
        "disable_chemprop_chemeleon": args.disable_chemprop_chemeleon,
        "chemprop_bin": args.chemprop_bin,
        "chemprop_featurizers": args.chemprop_featurizers,
        "chemprop_chemeleon_name": args.chemprop_chemeleon_name,
        "chemprop_epochs": args.chemprop_epochs,
        "chemprop_batch_size": args.chemprop_batch_size,
        "chemprop_num_workers": args.chemprop_num_workers,
        "chemprop_accelerator": args.chemprop_accelerator,
        "chemprop_devices": args.chemprop_devices,
        "chemprop_val_fraction": args.chemprop_val_fraction,
    }
    config_path = output_file_with_tag(args.output_dir, "interaction_benchmark_config", "json", effective_tag)
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
