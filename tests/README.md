# Regression tests

The original **38 test modules** are grouped by behavior in six importable subpackages.
No tests were removed during the October 2026 directory reorganization.

| Directory | Modules | Focus |
| --- | ---: | --- |
| `tests/pricing/` | 5 | Recursive/backward valuation, vanilla prices and options |
| `tests/lattices/` | 10 | Trees, node identity, probabilities, pruning and invariants |
| `tests/dividends/` | 5 | Calendars, event steps, dividends and exercise |
| `tests/risk/` | 3 | Greeks, risk kernel and hedging |
| `tests/frontend/` | 8 | Workflows, model comparison, visualizations and UI pricing |
| `tests/research/` | 7 | Experimental correctness, reproducibility, Pareto and research views |
| **Total** | **38** | |

## Running the suite

Always run from the repository root:

```bash
python -m compileall -q src frontend experiments tests
python -m pytest --collect-only -q
python -m pytest -q
```

The accepted pre-move baseline is **1,171 passing cases**. Preserve all existing
parametrizations and behavioral assertions. Because these are now importable
packages, Pytest collects modules as `tests.<subpackage>.test_*`.

The change is organizational only: market parameters, pricing engines, datasets,
export directories and archived evidence under `results/` remain unchanged.
