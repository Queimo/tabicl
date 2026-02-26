# Molecular regression benchmark analysis

## Setup
- 5 ADME molecular regression datasets from TDC: solubility_aqsoldb, lipophilicity_astrazeneca, hydrationfreeenergy_freesolv, caco2_wang, ppbr_az.
- Featurizers: RDKit Morgan (radius=2, 1024 bits) and Mordred descriptors (mordredcommunity).
- Models: TabICL, XGBoost, CatBoost, Random Forest.
- Train/test split: 80/20 with random_state=42.
- Maximum samples per dataset: 200 molecules.

## Aggregate results (mean across datasets)

| featurizer | model | rmse | mae | r2 | runtime_s |
|---|---|---|---|---|---|
| mordred | TabICL | 3.8309 | 2.3416 | 0.5915 | 65.2729 |
| mordred | RF | 3.9386 | 2.5025 | 0.5280 | 10.2096 |
| mordred | XGBoost | 4.0572 | 2.5115 | 0.5030 | 10.2778 |
| mordred | CatBoost | 4.1298 | 2.5046 | 0.4939 | 55.0076 |
| morgan | TabICL | 4.6494 | 2.7708 | 0.2615 | 42.3187 |
| morgan | CatBoost | 4.9424 | 3.1175 | 0.2033 | 1.2715 |
| morgan | RF | 5.1047 | 3.1984 | 0.1646 | 1.0711 |
| morgan | XGBoost | 5.4345 | 3.3456 | 0.1324 | 0.2252 |

## Best model per dataset/featurizer (by RMSE)

| dataset | featurizer | model | rmse | r2 | runtime_s |
|---|---|---|---|---|---|
| caco2_wang | mordred | TabICL | 0.4987 | 0.6967 | 68.5649 |
| caco2_wang | morgan | TabICL | 0.8163 | 0.1874 | 46.1773 |
| hydrationfreeenergy_freesolv | mordred | TabICL | 1.2028 | 0.9486 | 58.9253 |
| hydrationfreeenergy_freesolv | morgan | CatBoost | 3.8103 | 0.4842 | 0.9386 |
| lipophilicity_astrazeneca | mordred | TabICL | 0.9163 | 0.3224 | 65.8401 |
| lipophilicity_astrazeneca | morgan | TabICL | 1.0425 | 0.1229 | 47.2742 |
| ppbr_az | mordred | RF | 14.4445 | 0.2932 | 13.6926 |
| ppbr_az | morgan | TabICL | 15.1034 | 0.2272 | 48.2441 |
| solubility_aqsoldb | mordred | TabICL | 1.3503 | 0.7711 | 66.2830 |
| solubility_aqsoldb | morgan | XGBoost | 2.2261 | 0.3778 | 0.2550 |

## Key findings
- **Morgan features:** best average RMSE is **TabICL** (4.649).
- **Mordred features:** best average RMSE is **TabICL** (3.831).
- TabICL is competitive and often strong on descriptor-rich inputs, but it is substantially slower than tree ensembles in this CPU benchmark.
- Mordred descriptors improved average predictive performance over Morgan fingerprints for most model families in this run.

## Runtime/quality trade-off
- XGBoost and RF are consistently fastest.
- CatBoost usually improves over XGBoost/RF on some datasets but incurs a runtime increase.
- TabICL obtains strong scores on some tasks (e.g., hydrationfreeenergy_freesolv with Mordred) but has the highest runtime in all settings.

## Caveats
- This benchmark is single-split (no repeated CV), so small ranking differences should not be over-interpreted.
- Results are sensitive to descriptor filtering/imputation choices and default model hyperparameters.
- A follow-up benchmark should add repeated splits (or scaffold split), confidence intervals, and hyperparameter tuning budgets per method.