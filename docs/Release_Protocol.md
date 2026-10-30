# LatticePricer272 — Release and numerical validation protocol (Phase 6D)

## Scope and trust boundary

This project is a **quantitative-finance teaching/research prototype**, not a
calibrated or production trading model. The source of truth for finance is the
existing tested `src/` engine; Streamlit and archived CSVs must not silently
invent prices, impute failed cases, or relabel finite-N discrepancies as true
errors. The Phase 6D patch **does not modify the pricing engines** or recreate
any historical results.

The final scientific dataset is
`results/data/dividend_grid_robustness.csv`, generated independently of
Streamlit. Its two panels represent different experimental contrasts:

- `grid`: actual cash ex-date stays fixed, the lattice depth varies;
- `date`: lattice depth stays at N=100 while the **true** contractual ex-date
  varies by calendar days. It is not an estimate of discretization error.

## Local installation and release checks

Use the project's Windows Git Bash virtual environment, from the repository
root. **Do not run full empirical benchmarks just to validate a code patch.**

```bash
source .venv/Scripts/activate
python -m pytest -q
python -m compileall -q src frontend experiments tests
git diff --check
git status --short
python -m streamlit run frontend/app.py
```

The `pytest` suite checks the maintained financial invariants and app
contracts; no separate historical release-audit runner is required. Archived
Phase 6C evidence stays versioned under `results/`, and the scientific
assumptions are documented in the corresponding experiment protocol.
Before delivery, run the application manually with calendar-date contracts.

## Streamlit acceptance checklist

1. Three workspaces: Pricing Terminal, Model Diagnostics and Quant Research.
2. Research has three main sections; **Numerical reliability** offers pricing
   convergence, Greek robustness and market scenarios. No expensive rerun on
   switching views; user submissions initiate live experiments.
3. **Computational efficiency** distinguishes allocations from reachable
   pricing nodes. Cross-model run times are recorded evidence, not universal
   performance or speed rankings.
4. **Exercise & dividends** offers live optimal stopping and recorded
   dividend-grid robustness. Verify the latter loads 26 successful triplets
   and both charts have readable, full-width bottom legends in contrasting
   colors; no labels clipped at the right edge.
5. Recorded charts show actual samples only. A reference against the largest
   N is **not** an exact solution; a moved contractual date is **not** a
   numerical-grid perturbation.
6. The optional historical evidence toggle shows six archived studies and
   preserves their CSV exports and schema. Switching sections never deletes
   prior session results.
7. Run one vanilla European Call without dividends and one American Put with
   a positive cash dividend; compare against previously validated prices.
8. Check pricing with both **calendar dates** and **legacy time-in-years**
   and both pricing traversals when supported.

## Scientific and performance claims

- Black-Scholes provides a price benchmark only under the relevant European
  no-cash-dividend assumptions. The normal engine implements known discrete
  dividends; it does not implement a continuous dividend yield.
- A probability of being *in* the exercise region is not the stopped-path
  probability of *first* exercise. Do not sum date-wise region mass to obtain
  an overall exercise probability.
- Price accuracy, convergence order, Pareto-front membership, node savings,
  runtime scaling and Greek stability are distinct experimental questions.
  An empirical error/runtime frontier uses **observed** configurations only.
- Historical run times depend on hardware, environment and load. Do not
  combine their sample distributions with single-run Streamlit timings.
- Financial invariants and a passing regression suite **do not prove** perfect
  discretization or market accuracy. Report model-conditional observations.

## Final Git checklist

Stage only relevant source, tests and protocol files; the versioned historical
CSV and PNG evidence should remain unmodified by this release. `.gitignore`
excludes checkstyle outputs and temporary quick-study data. Inspect staged
changes with `git diff --cached --stat` and `git diff --cached --check` before
commit. After visual review and green tests, push `lilian` and optionally tag
an agreed version (do not overwrite an existing tag).

Historical `docs/archive/Frontend_*` files document prior iterations. They are not
current source of truth for navigation. Use the README and this protocol.
