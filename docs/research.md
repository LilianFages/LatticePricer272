# Research and application

Start Streamlit from the repository root:

```bash
python -m streamlit run frontend/app.py
```

The current application has **three workspaces**.

## Pricing Terminal

Select the contract, exercise style, model, numerical depth and pricing
algorithm. Choose calendar dates or time in years and, for compatible
models, discrete cash dividends. Results include price, model metadata,
eligible comparisons, Greeks and Delta-hedging tools.

Respect the displayed units: Vega per 1 percentage point of volatility,
Rho per 1 basis point and Theta per calendar day. Hedging depends on
option position, number of contracts, multiplier and existing shares.

## Model Diagnostics

Compare models under an identical contract and inspect the actual lattice.
Physical allocation, priced/reachable nodes and model runtime are
separate dimensions. Visual sampling must not fabricate nodes or edges.

## Quant Research

| Section | Current investigations |
| --- | --- |
| Numerical reliability | Pricing convergence, finite-difference Greek stability, market scenarios |
| Computational efficiency | Same-N pruning, allocation counts, measured runtime and price deviations |
| Exercise & dividends | Optimal stopping, with/without-dividend comparisons, ex-date/grid sensitivity |

The optional historical **data archive** remains accessible in Streamlit.
It loads recorded research data and figures from `results/` without
launching historical experiments. This is not the removed collection of
obsolete frontend release notes.

## Interpreting outputs

- Use Black–Scholes as a price benchmark only when its assumptions apply.
- The difference to a finite-step numerical reference is a *reference gap*,
  not a certified pricing error.
- Exercise-region probability at a date can include paths that would have
  exercised before then. First-exercise probabilities track the actual
  stopping time and cannot be replaced by exercise-region mass.
- A change in the contractual dividend ex-date changes the economic
  contract, not just the numerical grid.
- Empirical Pareto frontiers use observed successful configurations only.
  Runtime rankings depend on the measurement environment.

For specific experiment definitions see [experiments.md](experiments.md).
