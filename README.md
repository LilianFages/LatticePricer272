# LatticePricer272

**Numerical option pricing, discrete cash dividends, lattice visualization, risk and quantitative experiments.**

A research/engineering project built around binomial, trinomial, allocation-pruned and hybrid trees. The Python pricing engines, Streamlit interface and offline experimental benchmarks are deliberately kept separate. This repository is an educational quantitative-finance project, **not** a market-calibrated trading system.

## Run locally

Use a Python version compatible with the pinned dependencies in `requirements.txt` (the validated development checkout uses its own `.venv`). In the repository root:

```bash
python -m venv .venv
# Activate .venv using your platform's shell command.
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run frontend/app.py
```

On Windows Git Bash, activate with `source .venv/Scripts/activate`. Run the full test suite before and after every pricing-engine or UI refactor. Streamlit may open a browser window automatically.

## Application

- **Pricing Terminal:** specify contracts, exercise, model family, optional discrete cash dividends and pricing algorithm. Review results and risk calculations.
- **Model Diagnostics:** cross-model comparisons and an interactive node-by-node lattice explorer.
- **Quant Research:** three complementary research questions, with experiments explicitly triggered by the user:
  - **Numerical reliability:** price convergence, stability of finite-difference Greeks and supplementary market repricing scenarios.
  - **Computational efficiency:** pruning outcomes, allocated nodes and observed runtime trade-offs.
  - **Exercise & dividends:** early exercise regions and stopping distributions with/without discrete dividends.

Saved experimental evidence is available through an **optional, read-only archive**. The six legacy archived views and their CSV files remain accessible without automatically rerunning expensive experiments. The research UI does not substitute failed experiments or missing CSV cells with zero.

## Repository organization

| Directory | Purpose |
| --- | --- |
| `src/` | Three lattice families, one recursive pricer, one backward pricer and shared risk primitives |
| `frontend/` | Streamlit views, validated request/result adapters, analytical services and presentation |
| `experiments/` | Offline research protocols and command-line experimental studies |
| `tests/` | Pricing, dividend-boundary, numerical, frontend and provenance regression tests |
| `results/data/` | Versioned CSV measurements and experiment outputs |
| `results/figures/` | Versioned visual evidence from offline studies |
| `docs/` | Current architecture, quantitative research, experiment and validation guides |

### Python subpackages

| Package | Responsibility |
| --- | --- |
| `src/contracts/` | Option payoffs, discrete cash dividends and exercise timing |
| `src/lattices/` | Binomial, trinomial, allocation-pruned and hybrid trees |
| `src/pricing/` | Recursive/backward pricing and Black–Scholes references |
| `src/analytics/` | Numerical accuracy and Pareto metrics |
| `src/risk/` | Greeks, risk sensitivities and hedging |
| `experiments/convergence/` | Finite-step prices, errors and accuracy thresholds |
| `experiments/performance/` | Timings, recursion limits and accuracy versus cost |
| `experiments/pruning/` | Branch/allocation pruning benchmarks |
| `experiments/dividends/` | Cash-dividend impact, timing and stopping profiles |
| `experiments/comparisons/` | Comparison of lattice families |
| `tests/{pricing,lattices,dividends,risk,frontend,research}/` | Domain-focused regression suites |

Run scripts as modules from the repository root, for example:

```bash
python -m experiments.performance.pricing_accuracy_cost --quick
python -m experiments.dividends.dividend_grid_robustness --quick
```

### Pricing Terminal lattice depth

The Pricing Terminal allows **N ≤ 700** with the recursive pricer and
**N ≤ 1,000** with iterative backward induction. Selecting an algorithm
updates the editor's allowed maximum without changing the financial model.
Other research and model-comparison workspaces retain their independent
limits (typically N ≤ 350, with smaller visualization limits).
The same algorithm is used for the premium, Greeks and optional spot
revaluations. Very deep lattices, especially unpruned trinomial trees,
can require considerable memory and computation time.

## Numerical integrity and limitations

- **Analytical references:** Black–Scholes comparisons are meaningful only when the vanilla European/no-cash-dividend assumptions of the relevant implementation apply. Otherwise, a high-resolution numerical value is a *reference gap*, not a known pricing error.
- **Cash dividends:** the engines support discrete cash payments, not a continuous dividend yield; ex-dates use a consistent right-closed lattice-step convention. Event-time and node-time exercise may differ.
- **Exercise probabilities:** the mass in an exercise region on a date can include paths that would have been stopped earlier. First-exercise mass is a separate stopping-time statistic.
- **Performance:** recorded timings depend on hardware, Python version, run order and system load. Single-run live timings are not statistically equivalent to repeated offline benchmarks.
- **Pruning:** allocation savings and price differences must be measured on comparable successful cases. Failed reference calculations are never interpreted as a zero gap.

## Documentation

- [Architecture](docs/architecture.md): code structure and numerical conventions.
- [Research](docs/research.md): current Streamlit workspaces and financial interpretation.
- [Experiments](docs/experiments.md): quantitative designs and recorded evidence.
- [Validation](docs/validation.md): maintained tests and manual acceptance checks.

## Final verification

```bash
python -m pytest -q
python -m compileall -q src frontend experiments tests
python -m streamlit run frontend/app.py
```

Research datasets and figures remain versioned under `results/`.
Do not rerun full historical benchmarks just to validate a code patch.
