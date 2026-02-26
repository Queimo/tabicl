# Full run status

Attempted command:

```bash
python -u benchmarks/mega_molecular_benchmark.py --n-folds 20 --models RF
```

Status: **failed in this environment** while loading Polaris datasets.

Observed error class:
- `FileNotFoundError` on Polaris dataset URLs hosted at `https://data.polarishub.io/...`
- underlying connector error indicated host/network unreachability from this container.

Smoke-test command succeeded:

```bash
python -u benchmarks/mega_molecular_benchmark.py --smoke-test --n-folds 20 --max-folds 1 --models RF
```
