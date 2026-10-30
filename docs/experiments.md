# Reproducible numerical experiments

Numerical experiments are executed explicitly through `experiments/`.
The versioned `results/` CSVs, summaries and figures must not be
overwritten simply to validate a code change or open the Streamlit UI.

## Accuracy versus computational cost

Scripts: `experiments.pricing_accuracy_cost`,
`experiments.accuracy_cost_multistrike`.

No-dividend European options use `S0=100`, `r=2%`, `sigma=20%`,
`T=1`, `K=80,100,120`. Compare binomial, full trinomial,
branch-pruned trinomial and allocation-pruned trinomial. Recursive
pricing is fixed across the four methods.

The exact analytical comparator is Black–Scholes, and the three-strike
aggregate error is:

```text
E_max(model, N) = max_{K in {80,100,120}} |V_lattice(model,N,K) - V_BS(K)|
```

The first study uses `N=20,40,60,100,150,200,300` (calls and puts).
The cross-strike extension adds `N=80,125,175,225,250,275,325,350`
(European calls), reusing existing first-study call observations.

Costs include fresh-tree construction and pricing. Repeated runtimes
are summarized by their median. `max_median_runtime_seconds` is the
largest per-strike median for **one** contract; summed medians are
estimates of sequential workload. Observed Pareto frontiers are
sample-dependent, not continuous theoretical optima. Exact zero
errors are retained as zero in CSVs but omitted from logarithmic plots.

```bash
python -m experiments.pricing_accuracy_cost --quick
python -m experiments.accuracy_cost_multistrike --baseline-only
python -m experiments.accuracy_cost_multistrike --quick
```

Main outputs: `results/data/accuracy_cost_benchmark.csv` and
`results/data/accuracy_cost_multistrike_aggregate.csv`, raw timing
CSVs, Pareto/target tables and associated figures.

## Option value and exercise by date

Scripts: `experiments.option_value_exercise_profiles` and
`experiments.exercise_profile_step_sensitivity`.

Compare an American put with a known cash dividend, an American put
without dividend and a European put without dividend. For the
no-dividend cases, adjust the spot by the dividend present value to
match the **initial terminal forward**; intermediate distributions
are not identical.

Baseline: `S0=100` for the dividend case, `K=100`, `r=2%`,
`sigma=20%`, `T=1`, `N=100`, cash `D=3` at `t=0.5`.
Use allocation-pruned trinomial and recursive pricing.

For date `t_i`, unconditional node probabilities `p_ij` and priced
option values `V_ij`, the discounted mean is:

```text
M_i = exp(-r*t_i) * sum_j p_ij * V_ij
P_region(i) = sum_{j in exercise region} p_ij
P_first(i) = P(first optimal exercise occurs at t_i)
```

European discounted mean values should be martingales; the
American option's discounted mean is a supermartingale under the
specified stopping rule. `P_region` is **not** the stopping-time
distribution: its date-wise sum is not a lifetime exercise probability.
First-exercise mass plus worthless expiry accounts for the initial
probability population.

The sensitivity protocol compares `N=50,100,200`, reading the
already recorded N=100 baseline. This is a numerical stability
comparison, **not** proof of convergence.

```bash
python -m experiments.option_value_exercise_profiles --steps 100
python -m experiments.exercise_profile_step_sensitivity
```

Main outputs: `results/data/option_by_dates_profiles.csv` and
`results/data/option_by_dates_sensitivity_summary.csv`, alongside
summary text and exercise-probability figures.

## Dividend ex-date versus numerical grid

Script: `experiments.dividend_grid_robustness`.

Common conditions: `S0=K=100`, `r=2%`, `sigma=20%`, `T=1`,
cash dividend `D=3`, baseline `t_D=0.5`. Compare Hybrid and
allocation-pruned trinomial (threshold `1e-18`). Each observation
prices a matched European put, American put and European call.

- **Grid study:** keep actual `t_D=0.5`; vary
  `N=25,40,50,75,100,125,150,200`.
- **Contract-date study:** keep `N=100`; change actual ex-date
  by `-2,-1,0,+1,+2` calendar days, divided by 365.

The grid gap is against the **finest successful sampled N for that
model**. The date-shift difference is against the unshifted
contract under the same model and N; it is not numerical error.

Matched prices must be checked against the cash-dividend parity and
the American early-exercise premium:

```text
C_E - P_E = S0 - D*exp(-r*t_D) - K*exp(-r*T)
American put premium = P_A - P_E
```

Failed triplets retain missing values and error status; no prices
are imputed. The full design covers **26 configurations** (16 grid
and 10 date-shift model cases).

```bash
python -m experiments.dividend_grid_robustness --quick
python -m experiments.dividend_grid_robustness
```

Full evidence:
`results/data/dividend_grid_robustness.csv`,
`results/dividend_grid_robustness_summary.txt`,
`results/figures/dividend_grid_prices.png` and
`results/figures/dividend_exdate_shifts.png`.

## Other self-contained studies

Each file defines one numerical question. The filenames below distinguish
the underlying lattice, the measured quantity and the experiment's purpose.

| Script (under `experiments/`) | Focus |
| --- | --- |
| `binomial_convergence.py` | Binomial call prices against Black–Scholes across N |
| `binomial_convergence_order.py` | Empirical order of binomial convergence |
| `binomial_error_gap.py` | Strike-dependent binomial approximation-error gaps |
| `binomial_precision_target.py` | Required N for prescribed binomial error tolerances |
| `binomial_runtime_scaling.py` | Binomial valuation runtime as N increases |
| `pricing_recursion_limits.py` | Recursive versus backward pricing at large depths |
| `trinomial_convergence.py` | Binomial/trinomial convergence comparison |
| `branch_pruning_sensitivity.py` | Branch-pruning threshold and valuation impact |
| `allocation_pruning_benchmark.py` | Allocation-pruning timing and memory benchmark |
| `allocation_pruning_robustness.py` | Robustness against the branch-pruned reference |
| `hybrid_vs_trinomial.py` | Hybrid versus trinomial with cash dividends |
| `dividend_impact.py` | Effect of discrete dividend amounts on vanilla prices |
| `american_vs_european.py` | Exercise-style premium across interest rates |

Study modules were renamed for clarity without renaming historical output
CSVs and figures. Paths under `results/` remain stable for the Research Lab.

## General safeguards

Use `python -m pytest -q` to validate code, not a full rerun of
historical research. Quick and full modes may write different paths;
check each script's arguments before invoking `--overwrite`.
The `experiments/` directory contains further standalone studies;
their parameters and outputs are defined by their own entry points.
Measured runtime, node count, reference gap, mathematical model error
and actual contract-value change are distinct quantities.
