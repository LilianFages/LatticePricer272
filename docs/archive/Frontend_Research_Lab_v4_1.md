# LatticePricer272 — Research Lab v4.1

## Goal

Read, filter, compare and export **existing versioned experiment results** from `results/data/` in a desk-style Streamlit workspace. The Research Lab is an evidence explorer, **not** a separate pricing engine or a benchmark runner. No changes in `src/` and no automatic long calculations.

## Installation

1. Unzip at repository root, preserving folders.
2. Execute `python install_research_lab.py` once. The patch is idempotent, changes only the Streamlit navigation in `frontend/app.py`, and refuses an unknown layout.
3. Check `python -m pytest tests/test_research_lab_service.py -q` and `python -m pytest -q`.
4. Run `python -m streamlit run frontend/app.py` and open **05 / Research lab**.

## Tabs

- **Overview:** dataset counts and distinctions between tested cases and comparable cases.
- **Pruning robustness:** filters for dividend schedules, N and option kind, status breakdown, allocation savings only when both pricing engines succeed. Allocation-only success is NOT proof of price equivalence.
- **Accuracy vs cost:** observed multi-strike error vs worst median valuation time (log–log), observed non-dominated frontier *within the currently selected models*, fastest observed configurations at chosen thresholds, and CSV export. No extrapolation. Zero errors remain in calculations but cannot be drawn on the logarithmic error axis.
- **Complexity:** measured node counts, wall-clock, and memory for the pruning benchmark, plus independently measured recursive/backward runtime series. Those runs may come from different machines/sessions and are not interchangeable with controlled microbenchmarks.
- **Temporal exercise:** discounted expected values, exercise-region probability and stopped first-exercise mass. The former probability can include paths that would already have stopped: do **not** sum over dates. N-sensitivity is supplied by an independent CSV.
- **Data library:** whitelisted repository datasets, schema checks, previews and UTF-8 CSV downloads.

## Integrity and units

- Reuse `results/data/*.csv` from version control; do not hardcode research conclusions or manufacture missing measurements.
- Original time units: seconds in files, converted to milliseconds *only in labeled presentations*.
- Accuracy multi-strike metric: maximum absolute error across exactly `K=80,100,120` for European calls without cash dividends. Runtime metric: largest median per-strike valuation time at a given model and N; **not** total cost of pricing three strikes together.
- The Pareto frontier is based on the sampled grid, with strictly positive runtime and nonnegative error. Zero-error points are excluded only from log-axis rendering.
- Missing, null, or failed comparisons remain missing; never interpret a baseline failure as zero price gap.
- N=100 temporal profiles use a **common terminal forward**, achieved with different starting spots for dividend vs no-dividend scenarios.
- If files are absent or schemas change, the affected tab shows a descriptive notice; it does not reconstruct prices from defaults.

## Tests

Tests cover empirical status semantics, Pareto dominance and ties, thresholds, probability conservation, data-schema validation, CSV export, and safe idempotent app integration. They do not rerun full pricing experiments.

## Follow-up

After v4.1 validation, add curated scientific insights and optional explicit benchmark re-runs with progress/time bounds. Avoid automatically launching costly experiments or blocking navigation.
