# Validation

Use the project's virtual environment and run these commands from the
repository root (on Windows Git Bash, activate with
`source .venv/Scripts/activate`):

```bash
python -m pytest -q
python -m compileall -q src frontend experiments tests
git diff --check
git status --short
python -m streamlit run frontend/app.py
```

There is **no `scripts/` release audit**. The maintained regression
suite checks current financial invariants and application contracts.
A passing test suite or high CheckStyle grade is not proof of exact
continuous-time pricing or market calibration.

## Package relocation smoke checks

From the repository root, verify the new fully qualified modules and test discovery
before running costly offline research. These checks do not overwrite historical `results/` files:

```bash
python -m compileall -q src frontend experiments tests
python -m pytest --collect-only -q
python -m pytest -q
python -m experiments.performance.pricing_accuracy_cost --help
python -m experiments.dividends.dividend_grid_robustness --help
```

## Manual desktop acceptance

1. Price European calls/puts and American puts in Pricing Terminal.
   Check price, model, algorithm, unit conventions, ticket and hedge sign.
2. Compare a European no-dividend contract with its eligible
   Black–Scholes benchmark. Expect finite-step lattice differences.
3. Exercise the dividend-compatible Hybrid and Trinomial models.
   Confirm unsupported binomial/cash-dividend configurations fail
   explicitly instead of silently omitting dividends.
4. Check calendar date mode and legacy time-in-years mode against
   **the same economic maturity**. Test recursive and backward
   execution paths wherever supported.
5. Open Model Diagnostics. Inspect a like-for-like model comparison,
   real allocated versus connected nodes, and lattice explorer edges.
6. Open all three Quant Research sections. No lengthy offline
   experiments should run merely because navigation changes.
   Confirm old datasets remain readable through the optional archive.
7. Inspect an American early-exercise comparison and verify that
   exercise-region probability and first-exercise distribution are
   clearly distinguished.

## Before merging

Review `git diff main...lilian --stat`; inspect numerical and
frontend changes; run all maintained tests; check the Streamlit
presentation. Ensure historical `results/` observations are unmodified.
Only merge `lilian` into `main` after those checks pass.

For architectural assumptions see [architecture.md](architecture.md);
for experiment-specific invariants see [experiments.md](experiments.md).
