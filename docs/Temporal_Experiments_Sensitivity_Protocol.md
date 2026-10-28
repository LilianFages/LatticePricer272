# LatticePricer272 — Stability of the professor's temporal experiments

## Purpose

The previous run (`N=100`) validated three put scenarios with matched **initial terminal forwards**, according to the professor's two experiments: (i) expected, discounted option value by date and (ii) exercise-region probability by date. A supplementary experiment computed the **distribution of first optimal exercise**, accounting for stopped paths.

Here we study **sensitivity to time discretization** using `N = 50, 100, 200`. This is a limited numerical stability check, not a convergence proof and not a test of all strikes, volatility regimes or dividend specifications.

## Reproducibility and file safety

- `N=100` is **read** from `results/data/option_by_dates_profiles.csv`. The script checks its existing `results/option_by_dates_summary.txt` for compatible parameters. Neither file is modified.
- `N=50` and `N=200` are freshly computed by calling the existing `run_experiment` in `experiments/option_value_exercise_by_dates.py`. No `src/` code is modified.
- All outputs carry an unambiguous `option_by_dates_sensitivity_` prefix.
- The root directory is the repository root; run commands from there.

## Metrics compared

For each of the three scenarios and time-step count:

1. **Initial put premium:** `V_0(N)`.
2. **Probability of first exercise before maturity:** `sum_{i=0}^{N-1} P(tau=t_i)` (as a proportion of the initial risk-neutral population).
3. **Probability of first exercise on the discrete cash-dividend date** `t_d = 0.5` (only for the dividend-paying American put).
4. **Total exercise probability:** early + at maturity; remaining paths expire without payoff.
5. **European discounted-mean martingale residual** and **American nonincreasing discounted-mean check**.
6. **Probability mass conservation** and matched initial terminal forward.

The plotted probabilities measure **different events**. Probability of being in the exercise region at a given date may count paths previously optimally exercised; probabilities of **first** exercise are calculated from surviving paths only. Exercise at maturity is not classified as **early** exercise.

## Input conventions

- `S0=100` for the cash-dividend option, `K=100`, `r=0.02`, `sigma=0.20`, `T=1`.
- A **discrete cash dividend** `D=3` at `t=0.5`, not a continuous dividend yield `q=3%`.
- No-dividend spot adjusted by the present value of the cash dividend so initial terminal forwards match.
- The same allocation-pruned trinomial pricing engine and exercise tolerance as the existing script.

## Run

```bash
python -m pytest tests/test_option_by_dates_sensitivity.py -q
python -m pytest -q
python -m experiments.option_by_dates_sensitivity
```

## New output files

- `results/data/option_by_dates_sensitivity_N50.csv`
- `results/data/option_by_dates_sensitivity_N200.csv`
- `results/data/option_by_dates_sensitivity_summary.csv` (nine model-depth summaries)
- `results/option_by_dates_sensitivity_summary.txt`
- `results/figures/option_by_dates_sensitivity_prices.png`
- `results/figures/option_by_dates_sensitivity_early_exercise.png`
- `results/figures/option_by_dates_sensitivity_dividend_spike.png`

The baseline `N=100` plots and profile CSV remain untouched. Review the tables and figures **before** committing. In the LaTeX report, describe sensitivity findings as conditional on three sampled values of `N`.
