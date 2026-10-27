# Phase 6B — Shared lattice and empirical primitives

## Scope

This refactor reduces duplication without changing contract validation,
pricing equations, transition moments, pruning thresholds or research data.

1. **Dividend lookup**: both hybrid and trinomial trees use
   `dividend_for_lattice_step`. The original right-closed mapping and duplicate
   event error are preserved. `TransitionMoments.at_lattice_step` only constructs
   the identical `(step - 1) * dt, step * dt` interval, then delegates to the
   existing `TransitionMoments.compute` formula.
2. **Identity-based graph traversal**: frontend pricing and two experiments
   delegate to `src.lattice_graph.count_reachable_nodes`, following up/mid/down
   references but **not** lower-neighbor allocation links. Probability-aware
   tracing in the visualizer is intentionally left untouched.
3. **Empirical Pareto**: four existing research consumers delegate dominance
   decisions to `src.observed_frontier.nondominated_indices`. Filtering, error
   meaning, latency definition, tie retention, sorting and exports remain
   specific to each original caller.

## Reproducibility

This patch does **not** rerun the historical experiments or alter any files
under `results/`. Previously recorded figures, CSVs and comparisons remain
unchanged. In addition to the complete suite, use:

```bash
python -m pytest tests/test_phase6b_lattice_primitives.py tests/test_phase6b_observed_frontier.py -q
python -m pytest -q
git diff --check
```

The patch deliberately does not refactor tree-construction layouts or mutate
Streamlit navigation (Phase 6A). Numeric equivalence is enforced through the
existing regression suite, notably date-boundary and multi-model repricing tests.
