# LatticePricer272 — Research Lab v4.2 (targeted finish)

## Scope and installation

This release is a **presentation and data-provenance refinement** of the validated Research Lab v4.1. It changes neither pricing engines (`src/`) nor navigation (`frontend/app.py`), and it does not start any experiments. The ZIP includes only:

- `frontend/research_lab.py` — complete updated six-tab page
- `frontend/research_lab_service.py` — two read-only missingness diagnostics
- `tests/test_research_lab_v42.py` — new regression tests
- `docs/Frontend_Research_Lab_v4_2.md` — release protocol

Prerequisite: install and validate Research Lab v4.1 first. With your `.venv` activated, from the **repository root**:

```bash
python -m zipfile -e "$HOME/Downloads/LatticePricer272_Streamlit_Research_Lab_v4_2.zip" .
python -m pytest tests/test_research_lab_service.py tests/test_research_lab_v42.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

There is no installer for this release because `05 / Research lab` is already registered in v4.1. Do **not** run `install_research_lab.py` again for this patch (the original script is idempotent, but is not needed).

## Changes by tab

| Tab | Change | Integrity guarantee |
| --- | --- | --- |
| Overview | Three derived experimental highlights and two compact evidence charts now use unused screen space. | Derived from recorded comparable cases and empirical Pareto points only. |
| Pruning robustness | The outcome histogram is horizontal, with unrotated labels. | Comparison count and savings remain restricted to compatible branch/allocation successes. |
| Accuracy vs cost | Gold Pareto points are larger, with a stronger dashed frontier and an enlarged chart. | Selected-model frontier is still recomputed from selected observed data; no interpolation. |
| Complexity | Shorter model labels, descriptive units, horizontal legends and named strategies. | Raw model codes and resource values in tables/CSV are unchanged. |
| Temporal exercise | The discounted value chart (and premium-stability mini-chart) uses a non-zero vertical baseline; scenario labels are human-readable. | Axis rescaling affects presentation only. Exercise mass and first-exercise definitions unchanged. |
| Data library | Curated default column preview; an explicit explanation of missing CSV values; downloadable chosen columns **and** all source columns in a full re-export. | No missing values are filled; full original data remain available. |

## Interpreting the 384 empty cells of `allocation_pruning_robustness.csv`

The saved robustness dataset has **108 rows and 22 columns**. For the validated snapshot, the 384 blank fields are:

- **300 empty error/diagnostic text cells**: `branch_error`, `allocation_error`, or `error` is empty because that diagnostic was not populated for that run.
- **84 unavailable comparison fields** across 12 `allocation_only` cases at 200 steps with two dividends (7 comparison-dependent columns per row, because the branch baseline failed).
- **0 unclassified blanks** in that recorded snapshot.

Other datasets' empty fields are **not** assigned these explanations. In the UI they remain unclassified unless another dataset-specific interpretation is validated. All exported values preserve NaNs as blank fields, never as zero.

## Validation checklist

1. **Overview**: coverage 108 / 60 / 9; max comparable price gap 0 EUR, average savings ~29.4%, worst-strike minimum ~0.00094 EUR. Verify the two new charts do not crowd the dashboard.
2. **Pruning robustness**: statuses 96 compared / 12 allocation-only / 0 other; readable horizontal status categories.
3. **Accuracy vs cost**: 60 observations; 9 selected-model non-dominated points with all models selected; gold points and line visible; at target 0.005 EUR select binomial N=175 (~25.48 ms).
4. **Complexity**: switch resource between total time and allocated nodes; legend labels are readable and units appropriate. Recorded numbers remain unchanged.
5. **Temporal exercise**: original lines remain unchanged; the value chart now highlights variation in the 8.28–8.61 EUR region rather than compressing it against zero. Keep the equal-forward/adjusted-spot caveat.
6. **Data library**: select `Robustness`; 108 rows, 22 columns, 384 blank fields; expand column diagnostics and confirm 300+84+0 breakdown. Export selected columns and all source columns separately. Switch datasets to verify non-Robustness blanks are not explained as allocation-only.

Run both targeted and full pytest suites on the user's real checkout and check Streamlit renders all six tabs. The release was syntax-checked and its service tests passed in a standalone copy; the graphical Streamlit/Altair render must be confirmed in the user's environment.

## Non-goals / outstanding work

- No engine changes or changes to the experiments/CSV generation protocols.
- No claiming universal speed ranking or asymptotic proofs from these sampled charts.
- No re-running heavy experiments in a Streamlit rerun.
- No commit/push until the v4.2 six-tab manual review passes.
