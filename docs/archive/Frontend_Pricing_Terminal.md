# Lattice 272 — Pricing Terminal (Streamlit v1)

## Intent

A restrained **desk-style interface** integrated into the existing `LatticePricer272` Git repository. The Python pricing engines in `src/` are used **as-is**; all frontend orchestration is isolated in `frontend/`.

## Project additions

```text
frontend/
    __init__.py
    app.py                   # Streamlit presentation, market ticket, session blotter
    pricing_service.py       # Typed, validated adapter to existing src/ engines
    components.py            # Pricing result, payoff chart and CSV download
    styles.py                # Restricted CSS for a compact dark UI
.streamlit/
    config.toml              # Native Streamlit dark theme
tests/
    test_frontend_pricing.py # Tests against REAL src/ engines
docs/
    Frontend_Pricing_Terminal.md
```

No existing files need to be replaced. Streamlit, Altair and pandas are already listed in the repository's `requirements.txt`.

## Setup (Windows 11 / VS Code / Git Bash)

1. Open repository root on branch `lilian`.
2. Unzip this archive **into the repository root**, preserving the subdirectories.
3. Activate the existing `.venv` (if necessary, run `source .venv/Scripts/activate` from Git Bash).
4. Test the new adapter: `python -m pytest tests/test_frontend_pricing.py -q`.
5. Regression check: `python -m pytest -q`.
6. Start from the **repository root**:

```bash
python -m streamlit run frontend/app.py
```

A browser page should open at `http://localhost:8501`. Stop the server with `Ctrl+C`.

If dependencies are missing, with the venv active, run `python -m pip install -r requirements.txt`.

## Functional scope (v1)

- European and American calls/puts.
- Five choices / four model families: Binomial; Trinomial full; Trinomial branch pruning; Trinomial allocation pruning; Hybrid.
- Market data: spot, strike, risk-free rate, annualized volatility, years to maturity, lattice time steps.
- Cash dividend schedule edited in a table; only **positive** amounts count (a zero-amount row is ignored). The table supports adding and deleting rows.
- The BinomialTree in the project **does not** support dividends; the app refuses this combination rather than silently dropping cashflows.
- Positive forward assumption, duplicate dividends within a lattice step, invalid input and numerical engine errors are surfaced to users.
- Theoretical price, runtime (constructor + build + pricing), unique nodes reachable via pricing branches, allocation/pruning diagnostics where natively provided by engines.
- Black–Scholes reference and signed error **only for European options without dividends**.
- Terminal payoff chart (not an MTM curve, not delta), CSV export and last-12 session blotter.

## Technical choices / limitations

- **No pricing equations copied** to the UI. `price_contract` creates a **fresh tree per run**, passes it to the existing `RecursivePricer`, and only then reads diagnostics.
- The default pruning threshold is `1e-18`; Full Trinomial forces pruning threshold `0`.
- `reachable_nodes` means unique nodes reachable **from the root through transitions**. It is *not* the total number of nodes allocated in memory. `allocated_nodes` is shown only where the optimized tree exposes a true counter.
- Execution time excludes the extra node count and dashboard rendering, but includes tree construction and recursive pricing.
- Interactive `N` is limited to 350 for acceptable UI responsiveness; this is **not a theoretical limit**.
- Cash dividends may still trigger existing lattice limitations on low-price tails; errors are intentionally surfaced, not patched silently.
- This is a **local model valuation tool**, not a real-time quote service. There is no market-data feed, trading connection, implied-vol calibration, Greeks calculation or persistence beyond the active browser session.
- Current model reference/price agreement does not imply correctness for all inputs: the original research and unit tests remain authoritative.

## Next increments after validation

1. Model comparison panel: run matched models/inputs, compare prices, latency, node counts.
2. Quant analytics: convergence curves, sensitivity and Greeks where supported/implemented and tested.
3. Research Lab: reuse the existing CSV experiments (pruning, accuracy/runtime, temporal exercise).

## Git strategy

Only commit after the above tests and a manual browser walkthrough. For v1 the commit should contain the new `frontend/`, `.streamlit/config.toml`, `tests/test_frontend_pricing.py` and this document. Do **not** add local diagnostic artifacts.
