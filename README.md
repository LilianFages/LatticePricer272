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
| `src/` | Pricing trees, transition probabilities, option objects, backward algorithms and numerical risk primitives |
| `frontend/` | Streamlit views, validated request/result adapters, analytical services and presentation |
| `experiments/` | Offline research protocols and command-line experimental studies |
| `tests/` | Pricing, dividend-boundary, numerical, frontend and provenance regression tests |
| `results/data/` | Versioned CSV measurements and experiment outputs |
| `results/figures/` | Versioned visual evidence from offline studies |
| `docs/` | Experiment protocols, model assumptions, historical release notes and research design |

## Numerical integrity and limitations

- **Analytical references:** Black–Scholes comparisons are meaningful only when the vanilla European/no-cash-dividend assumptions of the relevant implementation apply. Otherwise, a high-resolution numerical value is a *reference gap*, not a known pricing error.
- **Cash dividends:** the engines support discrete cash payments, not a continuous dividend yield; ex-dates use a consistent right-closed lattice-step convention. Event-time and node-time exercise may differ.
- **Exercise probabilities:** the mass in an exercise region on a date can include paths that would have been stopped earlier. First-exercise mass is a separate stopping-time statistic.
- **Performance:** recorded timings depend on hardware, Python version, run order and system load. Single-run live timings are not statistically equivalent to repeated offline benchmarks.
- **Pruning:** allocation savings and price differences must be measured on comparable successful cases. Failed reference calculations are never interpreted as a zero gap.

## Reproducible offline experiments

Each entry point has its own input grid and output protocol. Two examples:

```bash
python -m experiments.accuracy_vs_cost --quick
python -m experiments.option_value_exercise_by_dates --steps 100
```

These commands can generate files; review their corresponding protocols and output paths first. Do **not** indiscriminately rerun full historical benchmarks or overwrite versioned results when you only want to explore the dashboard.

For the current Quant Research organization, see [`docs/Quant_Research_Phase6.md`](docs/Quant_Research_Phase6.md). Existing detailed protocols include `docs/Accuracy_vs_Cost_protocol.md`, `docs/Accuracy_vs_Cost_Phase2_protocol.md` and `docs/Temporal_Experiments_Protocol.md`. Older `Frontend_*` notes are retained as historical release documentation; they do not necessarily describe the current navigation.

## Final release verification (Phase 6D)

After the normal test suite, run the **read-only** scientific and dependency
check against committed experiment evidence:

```bash
python -m pytest -q
python -m scripts.release_audit
```

The release audit checks the 26 recorded Phase 6C configurations (complete
experimental grid, European call-put parity, American early-exercise premium,
finite-N reference gaps and contractual ex-date price shifts), their exported
summary and PNGs, and static first-party imports. It never reprices options,
rewrites CSVs or updates figures. This complements rather than replaces full
`pytest` and desktop Streamlit acceptance tests.

For the authoritative current documentation, see
[`docs/README.md`](docs/README.md),
[`docs/Release_Protocol.md`](docs/Release_Protocol.md) and
[`docs/Architecture_Snapshot.md`](docs/Architecture_Snapshot.md).
The current **Exercise & dividends** section also offers the recorded
[dividend-grid robustness study](docs/Dividend_Grid_Robustness_Protocol.md).
