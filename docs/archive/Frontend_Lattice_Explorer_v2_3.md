# LatticePricer272 — Lattice Explorer v2.3

## Purpose

This is an incremental upgrade of the already working v2.2 explorer, not a replacement app. The engine remains unchanged in `src/`. It adds **actual allocated-column geometry** to the existing positive-probability connected graph, and it keeps the initial price/spot scale identical for both comparisons.

### New controls

- **Display nodes → All allocated / Connected only.** All allocated shows real nodes captured from each built column. A **hollow grey circle** is a node allocated in Python but lacking any positive-probability path from the root. These are *not hypothetical locations* constructed using a pricing formula. No edge to such a node is drawn unless it has positive transition probability (in which case it would be connected).
- **Workspace view → Side by side / Focus left / Focus right.** Focus mode enlarges a chart without changing its lattice resolution or price. Both models use the **same initial Y-axis limits**, computed over both snapshots and the selected display mode. Subsequent user zoom/pan is independent per chart.
- **Simplified outgoing transitions** are drawn as thicker dashed rose edges; simplified nodes retain rose markers. An unconnected node is not the same as a simplified transition.
- **Node inspector:** inspect normal priced nodes or inspect actually allocated nodes with no positive-probability path. The latter may or may not carry a cached option value (because a recursion can traverse a zero-probability link); they never contribute positive probability mass to the pricing expectation.
- **Pruning comparison KPIs:** when comparing a full/branch trinomial to an allocation-pruned trinomial, show the actual `Node` allocation counts and the observed savings. A price discrepancy is explicitly warned about, rather than hidden.

## Instrumentation: no alterations to the engines

During `RecursivePricer().price(...)` only, `capture_allocated_columns(tree)` installs a temporary wrapper **on that tree instance** around its existing column-build method (`_build_column`, `_build_next_column`, or the hybrid step builders). Each wrapper invokes the original method exactly once and records the newly built top-to-bottom `Node` chain. All instance overrides are restored in a `finally` block, including after exceptions. No probability, volatility, pricing, or early-exercise formula is recomputed by the tracer.

The visualizer checks every priced connected node exists among those allocations and checks the engine's own allocation count where one is available. This is a small-`N` *geometry inspection*, **not** a runtime or peak-RAM benchmark. Build instrumentation slightly changes wall-clock time and must **not** be used to estimate pruning speedups; use `Model Comparison` and the published experiments for that purpose.

## Expected N=10 illustration

For a put American, `S=K=100`, `r=2%`, `sigma=20%`, `T=1`, `N=10`, no cash dividends and threshold `1e-5`:

- Branch-pruned trinomial: full 121 actual allocated objects; about 113 connected positive-mass nodes.
- Allocation-pruned trinomial: about 113 allocated objects; about 113 connected positive-mass nodes.
- About 8 avoided objects or 6.61% of the branch allocation count. **These values are reference expectations from the earlier local run, not a fresh execution**.
- The two model prices should agree within floating-point tolerance for this example, but that is not guaranteed for arbitrary large pruning thresholds.

## Installation on the current `lilian` branch

Stop Streamlit first. Extract the ZIP **at the repository root**, preserving folders. It replaces only `frontend/lattice_explorer.py` and `frontend/lattice_explorer_service.py`; adds `frontend/lattice_allocation.py`, this document, and `tests/test_lattice_explorer_v23.py`. It does not replace the working `frontend/app.py`, `frontend/model_comparison.py`, or `frontend/pricing_service.py`, nor any `src/` file.

```bash
python -m pytest tests/test_lattice_explorer_v23.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

### Manual validation checklist

1. Binomial vs Full Trinomial with `N=5`; counts should be 21 and 36; positive-mass view and full-allocation view should coincide.
2. Branch vs Allocation with `N=10`, `1e-5`; prices should match and hollow circles should appear **only** on the Branch graph; actual allocations should differ.
3. Switch to Focus left/right; check initial Y-axis limits remain equal across the pair. Hover over one hollow point and open its Node inspector.
4. Optionally use a larger pruning threshold (`1e-3`) to see more pink dashed links. Their number and location are determined by the engine; this is a pedagogy setting that can perturb values.
5. Check `pytest` and all existing pages: Pricing Terminal, Model Comparison, Lattice Explorer.

The app is a research/demo tool with user-supplied market data, not executable trading quotes.
