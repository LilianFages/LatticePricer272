# Phase 6A — Quant Research: three scientific questions

## Scope / baseline

Based on branch `lilian` at commit `4dc5ffd144712f19d930803d5c01e23785974597`.
The previous Streamlit page contained four top-level tabs and a six-view optional archive. This phase **consolidates presentation only**: `src/`, pricing equations, risk bump schemes, Monte Carlo assumptions (none), reference rules, dividends, output data and old session-state keys remain unchanged. No empirical experiment is rerun by this patch.

## Scientific map

| Main section | Interactive investigations retained | Primary question |
| --- | --- | --- |
| Numerical reliability | Price convergence; finite-difference Delta/Gamma bump robustness; controlled market repricing scenarios | Where do model prices and numerical sensitivities become reliable? |
| Computational efficiency | Same-N pruning and allocation/time study | What resources do pruning variants actually save, and at what observed price discrepancy? |
| Exercise & dividends | Early-exercise analysis, with/without-dividend comparison, date probabilities and detailed exports | How do exercise decisions and ex-dividend events shape option value and stopping? |

The existing `frontend.analytics_views.render_convergence_tab` keeps its **measured accuracy-versus-cost** panels: do not move the cost chart into a second live experiment that would reprice the same contract. The `Market scenarios` investigation is supplementary, not a Greek estimator.

### Reference discipline

- Black–Scholes is a valid analytical benchmark only when assumptions apply (e.g. eligible European vanilla contracts without cash dividends).
- A finite-depth numerical baseline supplies only a *discrepancy to that baseline*. Do not call it the true pricing error.
- Live runtime measurements are not directly comparable to saved offline multi-repeat CPU benchmarks.
- An exercise-region probability at date `t` is **not** a stopped-path first-exercise probability. Their date-wise sums have different interpretations.
- Error, runtime and allocation comparisons exclude invalid/failed reference outcomes rather than filling missing values with zeros.

## Archive compatibility

The archive remains behind `quant_research_archive_visible` and uses the original selector key `quant_research_archive_choice`. The six archived selection **values** are unchanged; display labels are qualified with an explanatory scientific theme only:

- Numerical reliability: Accuracy vs cost
- Computational efficiency: Pruning robustness, Complexity
- Exercise & dividends: Temporal exercise
- Research library: Overview, Data library

`frontend.research_lab.RESEARCH_ARCHIVE_HANDLERS` and `frontend.research_lab_service.CATALOG` are **not edited**. All experiment CSV files and exports remain in place. Historical views are loaded only when the user opts in.

The new within-section selector is `quant_research_numerical_view`. Existing result keys such as `qa_conv_result`, `qa_efficiency_result`, `qa_exb_result` and `qa_stab_result` are left intact. No session-state migration or clearing is performed.

## Acceptance checklist

1. Run the patch's `--check` and then `--apply`; confirm no non-target files changed.
2. Run `python -m pytest -q tests/test_quant_research_navigation.py tests/test_research_lab_service.py tests/test_research_lab_v42.py`.
3. Run `python -m pytest -q` and `git diff --check`.
4. Launch `python -m streamlit run frontend/app.py`. Confirm three tabs, one visible numerical investigation at a time, valid forms and familiar results/exports.
5. Visit each of the three numerical choices, the pruning study and the stopping study; ensure calculations occur only upon submission, not tab changes.
6. Activate the optional archive and verify all six choices; exercise representative controls and exported CSVs. Switch archive off and verify no archived page is displayed.
7. Test two contracts: a European call without dividends and an American put with a valid cash dividend. Check that reference language and stopping conventions remain correct.
8. Stage only `frontend/quant_research.py`, `tests/test_quant_research_navigation.py`, `README.md` and `docs/Quant_Research_Phase6.md`; commit only after tests **and** Streamlit review.

## Next work

**6B:** dependency/duplication analysis (particularly common tree operations, counted node reachability and offline pricing/benchmark boilerplate), then controlled refactors with dedicated invariant tests. **6C:** evaluate one complementary quantitative experiment on exact-grid vs off-grid ex-dividend timing and its effect on option values, Greeks and convergence, without altering transition conventions. **6D:** documentation pruning, evidence provenance and final reproducibility review. Do not delete old experiments, CSVs or historic docs before proving they are no longer referenced.
