"""Mega-benchmark scaffold for Polaris + MoleculeACE with 20-fold CV.

This script is intended to be *experiment ready* rather than quick to run.
Use `--smoke-test` to validate wiring in seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from tabicl import TabICLClassifier, TabICLRegressor
from xgboost import XGBClassifier, XGBRegressor

# Optional imports; only needed for full benchmark runs.
try:
    import polaris as po
except Exception:
    po = None


POLARIS_BENCHMARKS = [
    "polaris/pkis2-ret-wt-cls-v2",
    "polaris/pkis2-ret-wt-reg-v2",
    "polaris/pkis2-kit-wt-cls-v2",
    "polaris/pkis2-kit-wt-reg-v2",
    "polaris/pkis2-egfr-wt-reg-v2",
    "polaris/adme-fang-solu-1",
    "polaris/adme-fang-rppb-1",
    "polaris/adme-fang-hppb-1",
    "polaris/adme-fang-perm-1",
    "polaris/adme-fang-rclint-1",
    "polaris/adme-fang-hclint-1",
    "tdcommons/lipophilicity-astrazeneca",
    "tdcommons/ppbr-az",
    "tdcommons/clearance-hepatocyte-az",
    "tdcommons/cyp2d6-substrate-carbonmangels",
    "tdcommons/half-life-obach",
    "tdcommons/cyp2c9-substrate-carbonmangels",
    "tdcommons/clearance-microsome-az",
    "tdcommons/dili",
    "tdcommons/bioavailability-ma",
    "tdcommons/vdss-lombardo",
    "tdcommons/cyp3a4-substrate-carbonmangels",
    "tdcommons/pgp-broccatelli",
    "tdcommons/caco2-wang",
    "tdcommons/herg",
    "tdcommons/bbb-martins",
    "tdcommons/ames",
    "tdcommons/ld50-zhu",
]

MOLECULEACE_BENCHMARKS = [
    "CHEMBL1862_Ki",
    "CHEMBL1871_Ki",
    "CHEMBL2034_Ki",
    "CHEMBL2047_EC50",
    "CHEMBL204_Ki",
    "CHEMBL2147_Ki",
    "CHEMBL214_Ki",
    "CHEMBL218_EC50",
    "CHEMBL219_Ki",
    "CHEMBL228_Ki",
    "CHEMBL231_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL235_EC50",
    "CHEMBL236_Ki",
    "CHEMBL237_EC50",
    "CHEMBL237_Ki",
    "CHEMBL238_Ki",
    "CHEMBL239_EC50",
    "CHEMBL244_Ki",
    "CHEMBL262_Ki",
    "CHEMBL264_Ki",
    "CHEMBL2835_Ki",
    "CHEMBL287_Ki",
    "CHEMBL2971_Ki",
    "CHEMBL3979_EC50",
    "CHEMBL4005_Ki",
    "CHEMBL4203_Ki",
    "CHEMBL4616_EC50",
    "CHEMBL4792_Ki",
]


@dataclass
class DatasetPack:
    source: str
    benchmark_name: str
    smiles: pd.Series
    y: np.ndarray
    task: str  # regression | classification


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/mega"))
    p.add_argument("--n-folds", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-morgan-bits", type=int, default=1024)
    p.add_argument("--max-benchmarks", type=int, default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--models", type=str, default="TabICL,XGBoost,CatBoost,RF", help="Comma-separated subset of TabICL,XGBoost,CatBoost,RF")
    return p.parse_args()


def smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def featurize_morgan(smiles: pd.Series, n_bits: int) -> tuple[np.ndarray, list[int]]:
    rows, keep_idx = [], []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep_idx.append(idx)
    return np.vstack(rows), keep_idx


def featurize_mordred(smiles: pd.Series) -> tuple[np.ndarray, list[int]]:
    calc = Calculator(descriptors, ignore_3D=True)
    mols, keep_idx = [], []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        mols.append(mol)
        keep_idx.append(idx)
    desc_df = calc.pandas(mols)
    desc_df = desc_df.apply(pd.to_numeric, errors="coerce")
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.loc[:, desc_df.isna().mean(axis=0) <= 0.4]
    X = SimpleImputer(strategy="median").fit_transform(desc_df)
    return X.astype(np.float32), keep_idx

def model_factory(task: str, random_state: int, selected: set[str]) -> dict[str, Any]:
    if task == "classification":
        all_models = {
            "TabICL": lambda: TabICLClassifier(n_estimators=1, random_state=random_state),
            "XGBoost": lambda: XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "CatBoost": lambda: CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_seed=random_state,
                verbose=0,
            ),
            "RF": lambda: RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        return {k: v for k, v in all_models.items() if k in selected}

    all_models = {
        "TabICL": lambda: TabICLRegressor(n_estimators=1, random_state=random_state),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=0,
        ),
        "RF": lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return {k: v for k, v in all_models.items() if k in selected}
def polaris_task_type(benchmark: Any, target_col: str) -> str:
    t = str(benchmark.target_types[target_col]).lower()
    return "classification" if "class" in t else "regression"


def load_polaris_dataset(name: str) -> DatasetPack:
    if po is None:
        raise ImportError("polaris is required for Polaris benchmark loading")
    benchmark = po.load_benchmark(name)
    train, test = benchmark.get_train_test_split()
    df = pd.concat([train.as_dataframe(), test.as_dataframe()], ignore_index=True)
    smiles_col = list(benchmark.input_cols)[0]
    target_col = list(benchmark.target_cols)[0]
    task = polaris_task_type(benchmark, target_col)
    y = df[target_col].to_numpy()
    return DatasetPack("polaris", name, df[smiles_col], y, task)


def load_moleculeace_dataset(name: str) -> DatasetPack:
    url = (
        "https://raw.githubusercontent.com/molML/MoleculeACE/"
        "7e6de0bd2968c56589c580f2a397f01c531ede26/"
        f"MoleculeACE/Data/benchmark_data/{name}.csv"
    )
    df = pd.read_csv(url)
    return DatasetPack("moleculeace", name, df["smiles"], df["y"].to_numpy(), "regression")


def metric_row(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    if task == "classification":
        out = {"accuracy": float(accuracy_score(y_true, y_pred))}
        if y_proba is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                out["roc_auc"] = np.nan
        else:
            out["roc_auc"] = np.nan
        out["rmse"] = np.nan
        out["mae"] = np.nan
        out["r2"] = np.nan
        return out

    return {
        "accuracy": np.nan,
        "roc_auc": np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_dataset_cv(
    pack: DatasetPack,
    featurizer_name: str,
    X: np.ndarray,
    n_folds: int,
    max_folds: int | None,
    random_state: int,
    selected_models: set[str],
) -> list[dict[str, Any]]:
    y = pack.y
    n_samples = len(y)
    effective_folds = min(n_folds, n_samples)
    if pack.task == "classification":
        class_counts = pd.Series(y).value_counts()
        effective_folds = min(effective_folds, int(class_counts.min()))
        effective_folds = max(effective_folds, 2)
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    else:
        effective_folds = max(effective_folds, 2)
        splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    models = model_factory(pack.task, random_state, selected_models)
    records: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        if max_folds is not None and fold_id > max_folds:
            break

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, ctor in models.items():
            print(
                f"Running {pack.source}/{pack.benchmark_name} | {featurizer_name} | fold {fold_id} | {model_name}",
                flush=True,
            )
            model = ctor()
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            runtime_s = time.time() - t0

            y_proba = None
            if pack.task == "classification" and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_proba = proba[:, 1]
                except Exception:
                    y_proba = None

            rec = {
                "source": pack.source,
                "benchmark": pack.benchmark_name,
                "task": pack.task,
                "featurizer": featurizer_name,
                "fold": fold_id,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(X.shape[1]),
                "runtime_s": float(runtime_s),
            }
            rec.update(metric_row(pack.task, y_test, y_pred, y_proba))
            records.append(rec)
            print(json.dumps(rec))

    return records


def smoke_pack() -> DatasetPack:
    smiles = pd.Series(["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCO", "CCN", "CCC", "CCCl", "CCBr"])
    y = np.array([0.1, 0.2, 0.15, 0.35, 0.5, 0.12, 0.18, 0.17, 0.33, 0.55], dtype=np.float32)
    return DatasetPack("smoke", "toy_regression", smiles, y, "regression")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        packs = [smoke_pack()]
    else:
        polaris_names = POLARIS_BENCHMARKS
        moleculeace_names = MOLECULEACE_BENCHMARKS
        if args.max_benchmarks is not None:
            polaris_names = polaris_names[: args.max_benchmarks]
            moleculeace_names = moleculeace_names[: args.max_benchmarks]
        packs = []
        for x in polaris_names:
            try:
                packs.append(load_polaris_dataset(x))
            except Exception as e:
                print(f"Skipping Polaris benchmark {x}: {e}", flush=True)
        for x in moleculeace_names:
            try:
                packs.append(load_moleculeace_dataset(x))
            except Exception as e:
                print(f"Skipping MoleculeACE benchmark {x}: {e}", flush=True)

    selected_models = {x.strip() for x in args.models.split(",") if x.strip()}
    all_records: list[dict[str, Any]] = []
    for pack in packs:
        X_morgan, keep_morgan = featurize_morgan(pack.smiles, args.n_morgan_bits)
        pack_morgan = DatasetPack(pack.source, pack.benchmark_name, pack.smiles.iloc[keep_morgan], pack.y[keep_morgan], pack.task)
        all_records.extend(
            run_dataset_cv(
                pack_morgan,
                featurizer_name="morgan",
                X=X_morgan,
                n_folds=args.n_folds,
                max_folds=args.max_folds,
                random_state=args.random_state,
                selected_models=selected_models,
            )
        )

        X_mordred, keep_mordred = featurize_mordred(pack.smiles)
        pack_mordred = DatasetPack(pack.source, pack.benchmark_name, pack.smiles.iloc[keep_mordred], pack.y[keep_mordred], pack.task)
        all_records.extend(
            run_dataset_cv(
                pack_mordred,
                featurizer_name="mordred",
                X=X_mordred,
                n_folds=args.n_folds,
                max_folds=args.max_folds,
                random_state=args.random_state,
                selected_models=selected_models,
            )
        )

    out_df = pd.DataFrame(all_records)
    detail_path = args.output_dir / "mega_benchmark_results.csv"
    out_df.to_csv(detail_path, index=False)

    summary = (
        out_df.groupby(["source", "task", "featurizer", "model"], as_index=False)[
            ["rmse", "mae", "r2", "accuracy", "roc_auc", "runtime_s"]
        ]
        .mean(numeric_only=True)
        .sort_values(["source", "task", "featurizer", "model"])
    )
    summary_path = args.output_dir / "mega_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "n_folds": args.n_folds,
        "random_state": args.random_state,
        "n_morgan_bits": args.n_morgan_bits,
        "smoke_test": args.smoke_test,
        "n_polaris_benchmarks": len(POLARIS_BENCHMARKS),
        "n_moleculeace_benchmarks": len(MOLECULEACE_BENCHMARKS),
        "polaris_benchmarks": POLARIS_BENCHMARKS,
        "moleculeace_benchmarks": MOLECULEACE_BENCHMARKS,
        "models": sorted(selected_models),
    }
    config_path = args.output_dir / "mega_benchmark_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
