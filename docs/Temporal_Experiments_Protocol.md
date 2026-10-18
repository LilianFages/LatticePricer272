# LatticePricer272 — Experiments proposed by the professor

## Source: three teaching slides

1. **Average option value by dates:** at each date, average option values on the nodes, express in the valuation-date numeraire. A European option's discounted average should be constant. The professor shows a put's premium during the backward move.
2. **Three schemes, same forward price (adjusted spot):** American put with dividends, American put without dividends, European put without dividends.
3. **Exercise probability by dates:** sum the probabilities of nodes at which immediate/early exercise is optimal, with the same three schemes.

## Exact quantitative definitions

Time grid: `t_i = i * T / N` with risk-neutral transition probabilities. `p_{i,j}` is the probability of reaching node `j` on date `i` in the underlying tree *without stopping when an American holder exercises*. `V_{i,j}` is the option value obtained by backward induction.

- **Mean value in pricing-date numeraire:** `M_i = exp(-r*t_i) * sum_j p_{i,j} V_{i,j}`.
- **Exercise region:** `E_i = {j: intrinsic(S_{i,j}) > hold(S_{i,j}) + tol}`, with `intrinsic>0`; at maturity `E_N = {j: intrinsic>0}`. A European option has empty early exercise regions by construction.
- **Professor-style exercise probability:** `P_region(i) = sum_{j in E_i} p_{i,j}`.
- **Additional stopping-time statistic (NOT the same metric):** `P_first(i) = P(tau = t_i, option exercised)` under the optimal early exercise rule; stop and remove paths from the live distribution on their first exercise. Remaining terminal out-of-money paths expire worthless. `sum_i P_first(i) + P(expiry worthless) = 1`.

`P_region(i)` can count paths that would already have stopped at earlier dates. Therefore **do not sum `P_region` across dates** and interpret it as a lifetime exercise probability.

## Comparable forward: discrete cash dividend, not 3% proportional yield

The original illustration says "Am 3% div." The current engine supports **discrete cash dividends** and does not implement a continuous dividend yield `q=3%`. To avoid changing `src/` in this experimental step, this experiment uses **D=3 EUR at t=0.5**, from `S0=100 EUR` for the dividend scheme.

Under the specified dividend convention, `F_0(T) = S0 * exp(r*T) - D * exp(r*(T-t_d))`. For no-dividend schemes, set `S0_adjusted = S0 - D*exp(-r*t_d)` to match **terminal forward at inception**. This does NOT imply equal starting spots, intermediate forwards or distributions.

Default parameters: `K=100`, `r=0.02`, `sigma=0.20`, `T=1`, `N=100`, dividend at `t=0.5`. All three schemes use `AllocationPrunedTrinomialTree` and `RecursivePricer` so price models and CPU costs do not differ by the implementation architecture. A common pruning threshold `1e-18` is applied.

## Financial and numerical validation

- At each date, check unconditional probability sum = 1.
- Check initial recorded mean value equals `V0`.
- Check the no-dividend European discounted-mean martingale within `1e-7 EUR` on this experiment.
- Check matched terminal forward within `1e-7 EUR` across three schemes.
- Check first-exercise masses plus terminal worthless expiry = 1.
- Treat ties as continuation before maturity; exercise at maturity iff payoff > 0.
- Distinguish raw node exercise region from first-exercise stopping distribution.
- Do not infer exact match to slide curves; slides are illustrative, engine conventions differ.

## Outputs

Run from repository root:

```bash
python -m pytest tests/test_option_value_exercise_by_dates.py -q
python -m pytest -q
python -m experiments.option_value_exercise_by_dates --steps 100
```

Output files:

- `results/data/option_by_dates_profiles.csv` (3 schemes × N+1 dates, with all key measures)
- `results/option_by_dates_summary.txt` (parameters, comparisons, interpretation caveats)
- `results/figures/option_by_dates_average_pv.png`
- `results/figures/option_by_dates_exercise_region.png`
- `results/figures/option_by_dates_first_exercise.png`

**Interpretation:** the European discounted expectation is flat (risk-neutral martingale), American expected discounted node value is nonincreasing as a supermartingale, and first-exercise probabilities are meaningful as a single stopping-time distribution. Its sum need not be 100% because paths ending out of the money expire worthless.

Do not commit before tests and review of empirical results. No file under `src/` is edited by this experiment.
