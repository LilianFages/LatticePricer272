# LatticePricer272 — Quant Analytics v3.3 / Exercise Boundary

## Scope

Adds the fifth **Exercise boundary** tab to the existing `04 / Quant analytics` workspace. All calculations call the established `src/` tree/pricer code; no engine equations, graph topology, source files, or other Streamlit pages are changed. The v3.2 **Greeks snapshot**, **Market sensitivities**, **Convergence**, and **Greeks stability** tabs remain present.

**New files**

- `frontend/exercise_boundary_service.py`: separate priced-lattice traversal for optimal stopping and probability accounting.
- `frontend/exercise_boundary_views.py`: desk-style Streamlit panel with plots, contract entry, date table and exports.
- `tests/test_exercise_boundary_service.py`: 12 synthetic unit tests and 2 real-engine integration tests.
- `docs/Frontend_Quant_Analytics_Exercise_Boundary_v3_3.md`.

**Updated file**: `frontend/quant_analytics.py` (adds the fifth tab; no change to existing Greeks logic).

## Mathematics and probability conventions

For each time step `i`, the engine has already priced the option at its connected nodes. The diagnostics propagate two **different** distributions:

1. **Unconditional reach** `q_i(n)`: the probability of the stock arriving at node `n` without stopping the path when an option would exercise.
2. **Alive reach** `a_i(n)`: the mass of paths not exercised at earlier dates and arriving at `n`.

An American option is exercised **strictly** when its intrinsic payoff is positive and exceeds the one-step discounted continuation value by a scale-dependent tolerance of `1e-10`. An exercise/continuation tie is classified as continuation. European options never exercise early. At maturity, exercise requires a strictly positive payoff for either style.

The three quantities charted by date are:

- **Exercise-region probability** `R_i = sum_{n in exercise_set_i} q_i(n)`. This can count the same underlying paths at several dates because the stock continues even if an actual option holder would have exercised. **Never sum R_i over time.**
- **First-exercise probability** `F_i = sum_{n in exercise_set_i} a_i(n)`. Here paths already exercised are removed before propagating to `i+1`; `sum_i F_i` is the probability of a positive-payoff exercise by expiry.
- **Survival probability** `A_i` after date `i`: probability still unexercised. `A_i + sum_{k<=i} F_k = 1`, including worthless expiry outcomes at maturity.

The discounted expected option value is `exp(-r*t_i) sum_n q_i(n) V_i(n)`. Its expectation is constant for European options (up to model numerics), and non-increasing for American options as a supermartingale.

**Boundary envelope:** for puts, highest spot at which early exercise is optimal among positive-probability connected nodes; for calls, lowest spot. Only steps before maturity appear in the boundary plot. This is a **discrete observed envelope**, not an analytic/free-boundary solution or an interpolated continuum. Non-contiguous sets are flagged; a single threshold should **not** be inferred from such dates. Gaps without exercise are not connected by line segments.

### Dividend handling

Cash ex-dividend dates and amounts are passed unchanged to the existing engine. A dividend belongs to the engine's step-convention interval `(t_{i-1}, t_i]`. A 3-euro cash payment does *not* equal a 3% continuous dividend yield. Numerical anomalies (including nonpositive expected prices) remain explicit engine failures and are **not** silently clamped. The Binomial engine does not support discrete cash dividends; select Hybrid or a Trinomial variant.

### Plot and performance controls

- Interactive upper bound: **N = 200** to avoid oversized graphical payloads. The underlying engines retain their original range.
- The node map can omit nodes below a chosen unconditional probability and subsample if necessary (approximately 14,000 visible points maximum). **The full nodes CSV is never sampled.**
- The figure displays *reachable* nodes and their priced states, not all constructed/allotted nodes. For allocation geometry use the **Lattice Explorer**.
- The probability plots use the exact per-date total before any visual sampling.
- Results persist only in Streamlit `session_state` and are recomputed only after **RUN EXERCISE ANALYTICS**.

## Installation in the repository

Stop the running Streamlit process with `Ctrl+C`. In Git Bash at repo root:

```bash
python -m zipfile -e "$HOME/Downloads/LatticePricer272_Streamlit_Exercise_Boundary_v3_3.zip" .
python -m pytest tests/test_exercise_boundary_service.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

The ZIP replaces **only** `frontend/quant_analytics.py` and adds four files, with no changes to `src/`, Git branches, or existing experiment outputs. No Git commit is made by the archive.

## Acceptance plan

1. **American Put, discrete cash dividend**: `S=K=100`, `r=2%`, `sigma=20%`, `T=1`, `D=3 at t=0.5`, `N=100`, Trinomial Allocation Pruning, threshold `1e-18`. Expect premium near **8.60668465 EUR**. First exercise probability spikes at `t=0.5` (~10.834%), and total **early** exercise probability near **52.975%** (from the project's earlier temporal study). The plot must not show a pre-dividend exercise boundary on this configuration.
2. **No-dividend American Put**: same inputs, no dividend. Early exercise is possible before maturity; premium differs from the cash-dividend option. Note: previous professor comparison at matched forward uses a **different spot** for the no-dividend contract, and must not be confused with a fixed-spot comparison in this tab.
3. **European Put without dividend**: no early boundary or early exercise. The discounted average value should be constant to numerical precision. At maturity, first exercise occurs only for in-the-money paths.
4. **American Call without dividends**: with positive interest rate, there should generally be no early exercise. Do not present the discrete result as an exact theorem for arbitrary negative-rate or dividend configurations.
5. **Exports and filters**: changing plot probability cutoff must not alter premium or date probabilities; both CSV exports must include their complete underlying records.
6. Check all existing `pytest` tests. Do **not** commit before verifying the real-engine tests and charts on the user's PC.
