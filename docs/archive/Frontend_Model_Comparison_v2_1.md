# LatticePricer272 — Model Comparison (2.1)

## Goal
Add a desk-style multi-engine comparison **inside the existing Streamlit app**. No changes to `src/` and no independent application. The `Pricing Terminal` page (1.1) is retained.

## Files

- `frontend/app.py` — modified sidebar navigation, allows switching between modules 01/02.
- `frontend/comparison_service.py` — pure-Python orchestration and typed report/results, calls existing `frontend.pricing_service.price_contract()`.
- `frontend/model_comparison.py` — Streamlit page (model selection, common market data, results, plots, export).
- `tests/test_comparison_service.py` — real-engine tests, numerical comparisons, failure-isolation tests.

## Execution

```bash
python -m pytest tests/test_comparison_service.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

Open module `02 / Model comparison` in the sidebar. Compare a European ATM call with `S=K=100`, `r=2%`, `sigma=20%`, `T=1`, `N=100`; reference should be Black–Scholes. Try an American put with dividend `D=3` at `t=0.5` (Binomial is explicitly unavailable).

### Benchmark convention

Each selected model prices the same option and market parameters. The service performs 1/3/5 **fresh pricing runs**, rotating model order and displaying the median of timings measured by the existing `price_contract()` service (tree construction and recursive pricing). Node traversal and chart rendering are not inside the timer. These timings are indicative; they are not a statistically controlled microbenchmark. Price is checked for numerical consistency across repeats.

### Reference

- `Auto`: Black–Scholes for European, no-dividend contracts (when at least one engine prices successfully); otherwise the first **successful** selected model.
- Manual model baseline: reference is **unavailable** if that model fails; the application never silently substitutes another baseline.
- Black–Scholes is not used for American or discrete-cash-dividend contracts.
- Absolute/signed and percentage differences are not silently set to zero if the reference is absent or has zero value.

### Numerical limitations

- `BinomialTree` does **not** support discrete cash dividends. It is marked `unsupported`, not interpreted as a failed financial model.
- A full-trinomial construction may fail in extreme low-probability tails with multiple cash dividends. The other models still price; report records each status and reason.
- `Reachable nodes` are unique nodes connected through pricing branches. They are **not** necessarily the number of actually allocated nodes.
- Actual allocation counts and avoided-allocation percentages are shown only if the existing adapter exposes them for a defensible comparison (optimized trinomial in the no-dividend, positive-volatility regime).
- No built-in Greek, P&L or bid/ask claims. Input market data are supplied manually.

### Next step (2.2)

Build an optional **Interactive Lattice Explorer** with small `N <= 10`, visualizing **actual** node/transition graphs from `src/`: binomial, trinomial, hybrid, branching/pruning; hover inspection of underlying, option value, reach and transition probabilities. Larger N are priced normally but not rendered exhaustively.
