# Benchmark scripts

## Mega molecular benchmark (Polaris + MoleculeACE)

`mega_molecular_benchmark.py` implements a full experiment scaffold for:

- **Polaris benchmarks** (including `polaris/*` and `tdcommons/*` IDs used by CheMeleon)
- **MoleculeACE benchmarks** (30 tasks)

### What it runs

- Featurizers:
  - RDKit Morgan fingerprints (`radius=2`, configurable bit-size)
  - Mordred descriptors (`mordredcommunity`, missing-value filtering + median imputation)
- Models:
  - TabICL
  - XGBoost
  - CatBoost
  - Random Forest
- Protocol:
  - **20-fold CV** by default (`--n-folds 20`)
  - Stratified folds for classification, standard KFold for regression

### Benchmark IDs source

The benchmark IDs mirror the lists used in:
- `JacksonBurns/CheMeleon` (`models/*/evaluate.py`)

### Run examples

Smoke test (quick wiring check):

```bash
python -u benchmarks/mega_molecular_benchmark.py --smoke-test --n-folds 20 --max-folds 1
```

Full run (all Polaris + MoleculeACE, 20-fold CV):

```bash
python -u benchmarks/mega_molecular_benchmark.py --n-folds 20
# optionally restrict model set for faster runs
python -u benchmarks/mega_molecular_benchmark.py --n-folds 20 --models RF
```

### Outputs

Saved under `benchmarks/results/mega/`:

- `mega_benchmark_results.csv`: fold-level results
- `mega_benchmark_summary.csv`: aggregated metrics
- `mega_benchmark_config.json`: benchmark IDs and config snapshot

## Polymer property prediction benchmark

`polymer_property_prediction_benchmark.py` benchmarks polymer property prediction with:

- Featurizers: Morgan fingerprints + Mordred descriptors
- Models: TabICL, XGBoost, CatBoost, RF
- Protocol: 20-fold CV (default)

It automatically clones `JiajunZhou96/PolyCL`, normalizes all property datasets in
`PolyCL/datasets/*.csv` (except `pretrain_1m.csv`) into:

- `benchmarks/datasets/polymetrics/<dataset>.csv`
- `benchmarks/datasets/polymetrics/manifest.json`

Optional: include a `T_g` dataset via `--tg-path` (CSV with `smiles`/`SMILES` and one of `Tg`, `tg`, `T_g`, `target`, `value`, `y`).

If `--tg-path` is not provided, the script auto-loads `benchmarks/datasets/polymetrics/Tg.csv` if present.
A starter template is provided at `benchmarks/datasets/polymetrics/Tg_template.csv`.

Prepare datasets only:

```bash
python -u benchmarks/polymer_property_prediction_benchmark.py --prepare-only
```

Smoke test:

```bash
python -u benchmarks/polymer_property_prediction_benchmark.py --smoke-test --max-folds 1 --max-datasets 1
```

Full run (all PolyCL-derived polymetrics + optional Tg):

```bash
python -u benchmarks/polymer_property_prediction_benchmark.py --n-folds 20 --tg-path path/to/Tg.csv
# optionally restrict model set for faster runs
python -u benchmarks/polymer_property_prediction_benchmark.py --n-folds 20 --models RF
```


## PolyCL Table 1 RFECFP reproduction

`reproduce_polycl_table1_rfecfp.py` runs a dedicated RandomForest + ECFP baseline on
seven PolyCL datasets (`Eea`, `Egb`, `Egc`, `Ei`, `EPS`, `Nc`, `Xc`) with 5-fold CV,
and compares against the RFECFP line from the PolyCL paper table.

Run:

```bash
python -u benchmarks/reproduce_polycl_table1_rfecfp.py
```

Outputs under `benchmarks/results/polycl_reproduction/`:

- `rfecfp_5fold_detail.csv`
- `rfecfp_5fold_summary.csv`
- `rfecfp_reproduction_report.md`
- `rfecfp_config.json`


Both benchmark scripts accept `--models` as a comma-separated subset of `TabICL,XGBoost,CatBoost,RF`.
