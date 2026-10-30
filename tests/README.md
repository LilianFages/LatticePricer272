# Regression tests

The maintained suite is organized by **financial or application behavior**, not
by implementation phase or historical experiment number. The October 2026
consolidation reduced the suite from 64 to **38 Python test modules** while
retaining the existing test functions, parametrizations and regression scenarios.
Where several older modules were consolidated, section comments record their
original filenames for traceability.

## Test areas

| Area | Modules | Representative coverage |
| --- | ---: | --- |
| Pricing engines and lattices | 14 | Binomial/trinomial/hybrid trees, pruning, recursion/backward parity, transition moments and contract primitives |
| Dividends, calendars and exercise | 5 | Ex-date boundaries, event timing, cash dividends, American exercise and dated contracts |
| Frontend and workflows | 9 | Pricing and comparison adapters, saved references, lattice visualization, diagnostics and analytics |
| Risk and quantitative research | 7 | Greeks, hedging, risk isolation, observed accuracy, runtime/memory studies and research integrity |
| Offline experiments | 3 | Multi-strike accuracy/cost, Pareto fronts, shared experimental pricers and dividend-grid studies |
| **Total** | **38** | |

## Quick navigation

- Pricing trees: `test_binomial_tree.py`, `test_hybrid_tree.py`,
  `test_trinomial_tree.py`, `test_trinomial_pricing.py`,
  `test_trinomial_probability.py`, `test_trinomial_pruning.py`,
  `test_allocation_pruning.py`.
- Valuation correctness: `test_backward_pricing.py`,
  `test_recursive_pricer.py`, `test_black_scholes.py`,
  `test_transition_moments.py`, `test_lattice_invariants.py`,
  `test_node.py`, `test_option.py`.
- Dividends and exercise: `test_calendar_contracts.py`,
  `test_dividend_grid_validation.py`, `test_dividend_exercise.py`,
  `test_option_exercise_dates.py`, `test_early_exercise_boundary.py`.
- Frontend behavior: `test_frontend_pricing.py`,
  `test_contract_workflows.py`, `test_model_comparison.py`,
  `test_diagnostics_contract.py`, `test_quant_analytics.py`,
  `test_pricing_sensitivity.py`, `test_lattice_explorer.py`,
  `test_lattice_visualization.py`, `test_lattice_allocation_trace.py`.
- Risk and research: `test_risk_kernel.py`, `test_trading_risk.py`,
  `test_greek_numerics.py`, `test_research_accuracy.py`,
  `test_research_efficiency.py`, `test_research_lab.py`,
  `test_research_data_integrity.py`.
- Offline studies: `test_experiment_accuracy_cost.py`,
  `test_experimental_binomial_pricers.py`,
  `test_dividend_grid_research.py`.

## Validation

Run from the repository root in the active virtual environment:

```bash
python -m compileall -q src frontend experiments tests
python -m pytest --collect-only -q
python -m pytest -q
git diff --check
git status --short
```

The pre-consolidation reference was **1,171 passing pytest cases**. Confirm the
same collection and passing count after this structural change before treating
it as validated. Do not remove regressions to improve a style score; CheckStyle
is a separate follow-up pass after behavioral validation.
