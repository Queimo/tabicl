from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

DATA_DIR = Path('benchmarks/datasets/polymetrics')
OUT_DIR = Path('benchmarks/results/polycl_reproduction')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['Eea', 'Egb', 'Egc', 'Ei', 'EPS', 'Nc', 'Xc']
N_BITS = 1024
RADIUS = 2
N_SPLITS = 5
RANDOM_STATE = 42


def morgan_features(smiles: pd.Series):
    X=[]; keep=[]
    for i,s in smiles.items():
        mol=Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        fp=AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
        arr=np.zeros((N_BITS,),dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp,arr)
        X.append(arr); keep.append(i)
    return np.vstack(X), keep

rows=[]
for ds in DATASETS:
    df=pd.read_csv(DATA_DIR/f'{ds}.csv')
    X, keep = morgan_features(df['smiles'])
    y = pd.to_numeric(df.loc[keep,'target'], errors='coerce').to_numpy(dtype=np.float32)
    finite=np.isfinite(y)
    X,y = X[finite], y[finite]

    kf=KFold(n_splits=min(N_SPLITS, len(y)), shuffle=True, random_state=RANDOM_STATE)
    fold_scores=[]
    for fold,(tr,te) in enumerate(kf.split(X,y), start=1):
        model=RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X[tr], y[tr])
        pred=model.predict(X[te])
        r2=float(r2_score(y[te], pred))
        fold_scores.append(r2)
        rows.append({'dataset':ds,'fold':fold,'r2':r2,'n_train':int(len(tr)),'n_test':int(len(te))})

    print(ds, np.mean(fold_scores))

detail=pd.DataFrame(rows)
summary=detail.groupby('dataset', as_index=False)['r2'].mean()
summary=summary.set_index('dataset').loc[DATASETS].reset_index()
summary.loc[len(summary)] = {'dataset':'Avg. R2', 'r2': summary['r2'].mean()}

paper={'Eea':0.8401,'Egb':0.8643,'Egc':0.8704,'Ei':0.7421,'EPS':0.6840,'Nc':0.7540,'Xc':0.4345,'Avg. R2':0.7413}
summary['paper_rfecfp_r2']=summary['dataset'].map(paper)
summary['delta_ours_minus_paper']=summary['r2']-summary['paper_rfecfp_r2']

detail.to_csv(OUT_DIR/'rfecfp_5fold_detail.csv', index=False)
summary.to_csv(OUT_DIR/'rfecfp_5fold_summary.csv', index=False)
(OUT_DIR/'rfecfp_config.json').write_text(json.dumps({
    'datasets':DATASETS,'n_splits':N_SPLITS,'n_bits':N_BITS,'radius':RADIUS,
    'model':'RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)'
}, indent=2))

md=['# PolyCL Table 1 RFECFP reproduction (attempt)','',
    '| Dataset | Ours R2 (5-fold) | Paper RFECFP R2 | Delta |',
    '|---|---:|---:|---:|']
for _,r in summary.iterrows():
    md.append(f"| {r['dataset']} | {r['r2']:.4f} | {r['paper_rfecfp_r2']:.4f} | {r['delta_ours_minus_paper']:.4f} |")
(OUT_DIR/'rfecfp_reproduction_report.md').write_text('\n'.join(md))
print('saved', OUT_DIR)
