# Molecular regression benchmark

This benchmark compares **TabICL**, **XGBoost**, **CatBoost**, and **Random Forest** on 5 molecular regression datasets using:

- RDKit Morgan fingerprints (radius=2, 1024 bits)
- Mordred descriptors (`mordredcommunity` package)

Datasets (TDC ADME):

1. `solubility_aqsoldb` (ESOL-like solubility)
2. `lipophilicity_astrazeneca`
3. `hydrationfreeenergy_freesolv`
4. `caco2_wang`
5. `ppbr_az`

## Run configuration

- `MAX_SAMPLES=200` per dataset
- single random train/test split (`test_size=0.2`, `random_state=42`)
- Models evaluated: TabICL, XGBoost, CatBoost, RF

## Outputs

- `molecular_regression_benchmark.csv`: full per-dataset/per-model metrics
- `molecular_regression_summary.csv`: aggregate mean metrics by featurizer/model
- `../../analysis.md`: qualitative analysis and takeaways
