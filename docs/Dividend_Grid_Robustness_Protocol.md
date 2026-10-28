# LatticePricer272 — Phase 6C: Dividend ex-date / grid robustness

## Research question

How stable are **European and American put valuations** across lattice depths
when a **real discrete cash ex-date remains fixed**? And how sensitive is the
same discrete lattice to a **genuine contractual ex-date change** near a grid
boundary? The two experiments must not be confused.

This is an original complementary study, not a rerun of `dividend_impact`,
`option_by_dates_sensitivity`, or the general convergence analysis. It isolates
the interaction of **cash ex-date timing, lattice discretization, and early
exercise** while checking one fundamental European financial invariant.

## Fixed market and reproducibility

- `S0=100`, `K=100`, `r=2%` continuously compounded, `sigma=20%`, `T=1` year.
- One known **cash dividend** `D=3 EUR/share`; reference actual ex-date `t=0.5` years.
- Compare existing `Hybrid` and `Trinomial · Allocation pruning` implementations.
- Each observation is three **fresh** valuations: European put, American put,
  and European call; **same dividend and market conditions** within a triplet.
- Pruning threshold `1e-18` for allocation-pruned trinomial.
- No continuous dividend yield, stochastic dividends, calibration, or market data.
- Ex-date placement follows the core right-closed intervals `(t[k-1],t[k]]`
  and the existing ULP boundary policy. The experiment does not implement any
  new dividend timing logic.

### Cohort A — change grid only

Fix the **actual** dividend payment time at `t=0.5`. Sweep:
`N = 25, 40, 50, 75, 100, 125, 150, 200`. All market and contractual
inputs are identical; only numerical resolution changes. Store the event's
assigned lattice step and its within-step relative position to interpret
odd/even or event-boundary oscillations.

`gap_to_finest_sample` = European put price at N minus the **largest successful
sampled N for that same model**, not absolute error against a true price. The
reference is finite, model-specific, and only an observational comparison.

### Cohort B — change contractual ex-date only

Fix `N=100`. Shift actual ex-date by `-2,-1,0,+1,+2` calendar days using
`shift_days/365`. Spot, dividend amount, strike and other parameters remain
constant; no artificial correction keeps forward prices equal.

`date_shift_put` = European put price with shifted dividend date minus the
same-model unshifted price at N=100. This is a **difference between contracts**,
not discretization error. The input date also changes the PV of the dividend.

## Financial diagnostics

For each successful triplet:

- **European put–call parity**, for deterministic cash dividend:
  `C_E - P_E = S0 - D exp(-r t_D) - K exp(-r T)`;
  record signed `parity_residual`, never overwrite it with zero.
- **American early exercise premium** `P_A-P_E` under the identical model,
  step count and cash dividend. Record negative values if observed, rather than
  forcing a theoretical sign numerically.
- Any failed leg marks the entire triplet `status=failed`. The input grid
  coordinate, model, and error message remain in the CSV. Unavailable prices
  remain missing, not zero; no finite extrapolated replacement is made.

The cash-dividend trinomial engine uses an established within-step variance
proxy. Small parity residuals or close inter-model prices are **diagnostics**,
not proof of accurate exercise boundaries or a theorem about convergence.

## Runs

From repo root with the project's virtual environment active:

```bash
python -m pytest tests/test_phase6c_dividend_grid_robustness.py tests/test_phase6c_grid_views.py -q
python -m pytest -q
python -m experiments.dividend_grid_robustness --quick
python -m experiments.dividend_grid_robustness
```

The quick run writes **only** under `results/phase6c_quick/`. Full outputs:

- `results/data/dividend_grid_robustness.csv`
- `results/dividend_grid_robustness_summary.txt`
- `results/figures/dividend_grid_prices.png`
- `results/figures/dividend_exdate_shifts.png`

Re-running a previously completed study requires explicit `--overwrite`.
For a chartless run, append `--no-plots`. Each successful row implies **three
separate option valuations**. The optional Streamlit viewer in Quant Research
only loads **the full CSV**; it never launches pricing on page navigation.

## Acceptance conditions and scientific interpretation

1. Verify paired contracts differ only in `N` (cohort A) or actual ex-date
   (cohort B), with all date changes explicitly documented.
2. Check `status`, failure messages, missing values, and parity residuals.
3. Check `P_A-P_E`; investigate negative early-exercise premiums beyond
   numerical tolerance rather than hiding them.
4. Inspect prices for grid/event alignment oscillations. No asymptotic
   convergence order may be inferred from a few sampled points.
5. Distinguish numerical-reference discrepancies from exact errors, and
   date-shift price changes from numerical discrepancies.
6. Confirm no existing `results/` data or `src/` pricing modules were edited.
7. After full run, inspect **Quant Research > Exercise & dividends >
   Dividend-grid stability (recorded)** and export the CSV.

All figures are visualizations of recorded points; connecting lines are
visual guides, not evidence of price availability between configurations.
