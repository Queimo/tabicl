import json
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
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor
from tdc.single_pred import ADME
from xgboost import XGBRegressor

RANDOM_STATE = 42
MAX_SAMPLES = 200
N_MORGAN_BITS = 1024
TEST_SIZE = 0.2
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetSpec:
    name: str


DATASETS = [
    DatasetSpec("solubility_aqsoldb"),  # ESOL-like
    DatasetSpec("lipophilicity_astrazeneca"),
    DatasetSpec("hydrationfreeenergy_freesolv"),
    DatasetSpec("caco2_wang"),
    DatasetSpec("ppbr_az"),
]


MODELS = {
    "TabICL": lambda: TabICLRegressor(n_estimators=1, random_state=RANDOM_STATE),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=200,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
    ),
    "RF": lambda: RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    return ADME(name=spec.name).get_data()


def smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def featurize_morgan(smiles: pd.Series) -> tuple[np.ndarray, list[int]]:
    rows = []
    keep_idx = []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_MORGAN_BITS)
        arr = np.zeros((N_MORGAN_BITS,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
        keep_idx.append(idx)
    return np.vstack(rows), keep_idx


def featurize_mordred(smiles: pd.Series) -> tuple[np.ndarray, list[int]]:
    calc = Calculator(descriptors, ignore_3D=True)
    mols = []
    keep_idx = []
    for idx, s in smiles.items():
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        mols.append(mol)
        keep_idx.append(idx)

    desc_df = calc.pandas(mols)
    desc_df = desc_df.apply(pd.to_numeric, errors="coerce")
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    missing_ratio = desc_df.isna().mean(axis=0)
    desc_df = desc_df.loc[:, missing_ratio <= 0.4]
    X = SimpleImputer(strategy="median").fit_transform(desc_df)
    return X.astype(np.float32), keep_idx


def run_one(X: np.ndarray, y: np.ndarray, model_name: str) -> tuple[float, float, float, float, int, int]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = MODELS[model_name]()
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return rmse, mae, r2, elapsed, len(y_train), len(y_test)


def main() -> None:
    records = []
    for spec in DATASETS:
        df = load_dataset(spec)
        if len(df) > MAX_SAMPLES:
            df = df.sample(MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

        for feat_name, feat_func in [("morgan", featurize_morgan), ("mordred", featurize_mordred)]:
            X, keep_idx = feat_func(df["Drug"])
            y = df.loc[keep_idx, "Y"].to_numpy(dtype=np.float32)

            for model_name in MODELS:
                print(f"Running {spec.name} | {feat_name} | {model_name}", flush=True)
                rmse, mae, r2, elapsed, n_train, n_test = run_one(X, y, model_name)
                rec = {
                    "dataset": spec.name,
                    "featurizer": feat_name,
                    "model": model_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "runtime_s": elapsed,
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "n_features": int(X.shape[1]),
                }
                records.append(rec)
                print(json.dumps(rec))

    out_df = pd.DataFrame(records).sort_values(["dataset", "featurizer", "rmse"])
    detail_path = RESULTS_DIR / "molecular_regression_benchmark.csv"
    out_df.to_csv(detail_path, index=False)

    summary = (
        out_df.groupby(["featurizer", "model"], as_index=False)[["rmse", "mae", "r2", "runtime_s"]]
        .mean()
        .sort_values(["featurizer", "rmse"])
    )
    summary_path = RESULTS_DIR / "molecular_regression_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
