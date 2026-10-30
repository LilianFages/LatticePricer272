# Architecture

LatticePricer272 separates numerical pricing, financial risk, Streamlit and
offline numerical research. The UI calls the existing pricing engines; it does
not maintain a separate pricing implementation.

| Directory | Responsibility |
| --- | --- |
| `src/` | Option payoffs, binomial/trinomial/hybrid trees, probabilities, exercise, valuation |
| `src/risk/` | Finite-difference and eligible analytical Greeks, desk risk, share hedging |
| `frontend/pricing_service.py` | Contract validation, model and recursive/backward algorithm routing |
| `frontend/` | Streamlit pricing, diagnostics, research services, visualization and exports |
| `experiments/` | Independent, explicitly launched numerical research scripts |
| `results/` | Versioned experiment CSVs, summaries and figures |
| `tests/` | Current numerical invariants and application integration tests |

## Supported models

- **Binomial:** vanilla European/American calls and puts, without discrete
  cash dividends in this implementation.
- **Trinomial:** full, branch-pruned and allocation-pruned versions. Physical
  allocations and connected/reachable pricing nodes are different metrics.
- **Hybrid:** combines binomial and dividend-compatible trinomial transitions
  to handle known cash-dividend events.
- **Valuation:** one `RecursivePricer` and one `BackwardPricer` handle all
  three tree families. The backward algorithm traverses the built lattice
  from expiry to valuation, computing discounted conditional expectations
  and the American early-exercise maximum.

The unique backward implementation builds the complete graph, so its memory
use grows with lattice size. The older binomial-only column-rebuilding
implementation has been removed; historical timing measurements in `results/`
are retained but must not be attributed to the current unified engine.

The core reuses `src/dividend_timing.py` for dividend-to-step mapping,
`src/transition_moments.py` for local transition moments,
`src/lattice_graph.py` for reachable-node counting and
`src/observed_frontier.py` for empirical Pareto dominance.

## Financial conventions

- Dividends are known **cash amounts** per share, not a continuous yield;
  ex-dates follow a common right-closed lattice-step convention.
- European and American exercise rights are different. The immediate
  exercise decision compares intrinsic and continuation value.
- A numerical finite-N reference is not an exact price. Black–Scholes is
  a suitable reference only for eligible European vanilla contracts.
- Failed pricing or unavailable research observations stay explicit;
  no missing value may be silently filled with zero.
- Opening a Streamlit view never intentionally executes a full offline
  research benchmark. Previously recorded `results/` files are read-only.

This is an educational quantitative finance application, not a market
calibrated production trading system.

See [research.md](research.md), [experiments.md](experiments.md) and
[validation.md](validation.md).
