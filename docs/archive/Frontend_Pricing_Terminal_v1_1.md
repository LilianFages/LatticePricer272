# Lattice 272 — Pricing Terminal v1.1

This incremental package replaces four files under `frontend/` and adds one new test module.
It does **not** modify `src/` or replace `.streamlit/config.toml`.

## Updates

- Price and benchmarks: signed and relative gap from Black–Scholes where applicable.
- Numerical diagnostics: theoretical **full trinomial node count (N+1)^2**, exact for **no cash dividends, positive volatility**. Saving % is shown only for Allocation Pruning with this reference. For dividend cases, no misleading comparison is shown.
- Explicit separation of reachable nodes, allocated nodes and simplified transitions.
- Reset ticket button (sidebar): creates a new form identity with default values; pricing-history blotter remains.
- Optional spot revaluation, **nine actual independent calls to the same src/ engine** at 80%–120% of entered spot. All other parameters remain fixed. This is not a Greeks/delta measure and not the terminal payoff. Some dividend scenarios may fail independently and are reported rather than replaced with fictitious numbers.
- Terminal payoff moved to an optional expander to avoid unnecessary dashboard crowding.
- Larger labels and metrics; updated `width="stretch"` Streamlit API.

## Run

From the existing repository root in Git Bash:

```bash
python -m pytest tests/test_frontend_v11.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

**In-browser checks:**
1. European call, allocation pruning, S=K=100, r=2%, vol=20%, T=1, N=100: price near 8.925141 EUR. Check benchmark, relative deviation and allocation savings.
2. Click `Calculate spot value curve`; review the curve and export the CSV.
3. American put, cash dividend 3 EUR at 0.5 yr, N=100: price near 8.606685 EUR; no Black–Scholes benchmark or fabricated allocation savings.
4. Click `Reset ticket to defaults`: form resets; session history stays.
5. Set an invalid model/dividend combination and verify readable validation message.

Streamlit version per project requirements is 1.64; interface is optimized for desktop first. Runtime numbers are diagnostic single-run observations, not benchmark statistics. There is no real-time market-data feed or trading connectivity.

## Future Model Comparison: geometric lattice explorer (not implemented in v1.1)

Show the **actual** node graph constructed by `src/` for small N (suggest N<=10), with re-combination, up/mid/down links, reach probability and node option value on selection. For pruning, distinguish removed allocations from redirected monomial transitions; nodes with low reach probability must not be conflated with nodes never allocated. Color exercise-region nodes for American options. Build a separate small tree for visualization without changing the pricing N. This is deliberately postponed to Model Comparison.

Do not commit until real-engine tests and UI walkthrough pass.
