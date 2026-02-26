# All Benchmarks

## Molecular regression (ADME)

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

## Mega molecular benchmark

| source | task | featurizer | model | rmse | mae | r2 | accuracy | roc_auc | runtime_s |
|---|---|---|---|---|---|---|---|---|---|
| smoke | regression | mordred | RF | 0.0164 | 0.0164 | nan | nan | nan | 0.4129 |
| smoke | regression | morgan | RF | 0.0511 | 0.0511 | nan | nan | nan | 0.3995 |

## Polymer property benchmark

| dataset | featurizer | model | rmse | mae | r2 | runtime_s |
|---|---|---|---|---|---|---|
| smoke_polymer | mordred | RF | 0.0164 | 0.0164 | nan | 0.4565 |
| smoke_polymer | morgan | RF | 0.0511 | 0.0511 | nan | 0.4146 |

## PolyCL RFECFP reproduction

| dataset | r2 | paper_rfecfp_r2 | delta_ours_minus_paper |
|---|---|---|---|
| Eea | 0.8339 | 0.8401 | -0.0062 |
| Egb | 0.8425 | 0.8643 | -0.0218 |
| Egc | 0.8627 | 0.8704 | -0.0077 |
| Ei | 0.7549 | 0.7421 | 0.0128 |
| EPS | 0.7395 | 0.6840 | 0.0555 |
| Nc | 0.7960 | 0.7540 | 0.0420 |
| Xc | 0.3049 | 0.4345 | -0.1296 |
| Avg. R2 | 0.7335 | 0.7413 | -0.0078 |

## Execution notes
- Full Polaris-hosted benchmark execution was attempted, but Polaris dataset host access failed in this environment (`data.polarishub.io` unreachable).
- To avoid blocking, smoke validation was re-run and result files were generated for mega/polymer scripts.
- Run full commands in your target environment with Polaris network access:
  - `python -u benchmarks/mega_molecular_benchmark.py --n-folds 20 --models RF`
  - `python -u benchmarks/polymer_property_prediction_benchmark.py --n-folds 20 --models RF`