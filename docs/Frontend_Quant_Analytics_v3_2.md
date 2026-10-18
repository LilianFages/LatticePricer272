# LatticePricer272 — Quant Analytics v3.2

## Scope

Adds **three tabs** beside the existing v3.1.1 Greeks snapshot. Everything stays in `frontend/`; no changes to `src/`, model calibration, or pricing conventions.

- **Market sensitivities:** independently reprice a Call/Put (European/American) using up to five models and an odd-numbered grid (5, 9 or 13) of spot, volatility, interest rate or maturity levels.
- **Convergence:** select 2–12 increasing values of N (interactive maximum 350), compare prices, single-run end-to-end runtimes, and gaps to a reference.
- **Greeks stability:** study Delta and Gamma as a function of spot bump and lattice discretization N. The original v3.1.1 five-Greek snapshot remains unchanged.

### Numerical conventions

| Study | Convention |
| --- | --- |
| Sensitivity spot | Base ±20% relative, optionally ×0.5 or ×1.5 |
| Sensitivity volatility | Base ±5 **percentage points** absolute, optionally scaled |
| Sensitivity rate | Base ±2 **percentage points** absolute, optionally scaled |
| Sensitivity maturity | Base ±25% relative, optionally scaled; ex-dividend dates unchanged relative to today |
| Sensitivity other parameters | Unchanged, including N and pruning threshold |
| Convergence analytic reference | Black–Scholes **only** for European contracts without cash dividends, when returned by an existing valid pricing result |
| Convergence numerical reference | Selected model at the highest sampled N. A **price gap**, not an error against the true price |
| Convergence runtime | One cold/fresh construction + pricing for each configuration, **not** a statistically repeated benchmark |
| Greeks bump sweep | Same central spot bump used for both Delta and Gamma, retaining the finite-lattice behavior |
| Greeks N sweep | Delta uses ±1% spot; Gamma uses automatically selected grid-aware width at **each** N. The Gamma bump itself changes with N. |

No curves are computed automatically upon navigation. Each button runs a bounded batch and stores its result in Streamlit `session_state` until another calculation or server restart. Models, values, and failures are exported to CSV (no silent substitution of zero for a failed scenario). Black–Scholes must not be presented as an American/cash-dividend reference.

### Important caveats

- Finite lattices have oscillating convergence and discrete payoffs. Numerical Gamma may be sensitive to N, strike alignment, and bump size, particularly near early-exercise boundaries and cash dividends.
- With cash dividends, the existing binomial engine is unsupported. Other models still run and the invalid rows remain explicitly marked.
- Changing maturity holds cash ex-dividend dates fixed (years from today). Dates beyond the shortened maturity invalidate that scenario and are reported rather than moved or silently dropped.
- Some coarse lattices fail on cash dividends because the positive stock-price condition is violated. Those cases appear as errors, not fabricated values.
- Single-run runtimes fluctuate with CPU load; repeat measurements for a formal speed comparison in *Model Comparison* or the existing research benchmarks.
- Numerical reference at finite N does not guarantee convergence. On log-error charts, zero gaps cannot be shown and remain present in the CSV.
- Spot bumps use **fresh pricing engines** and do not represent infinitesimal derivatives when the lattice is nonsmooth.
- No live market data are loaded. Output is not an executable quote.

## Installation

From the root of the local checkout on branch `lilian`:

```bash
# Stop the running Streamlit server with Ctrl+C first.
python -m zipfile -e "$HOME/Downloads/LatticePricer272_Streamlit_Quant_Analytics_v3_2.zip" .
python -m pytest tests/test_analytics_service.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

Only `frontend/quant_analytics.py` is replaced. The ZIP adds `frontend/analytics_service.py`, `frontend/analytics_views.py`, `tests/test_analytics_service.py`, and this document. Original `frontend/greeks_service.py` and the financial engines in `src/` are unchanged.

## Initial acceptance tests

1. **European Call, ATM, no dividend:** `S=K=100, r=2%, σ=20%, T=1, N=100`, model = Allocation pruning. The existing Greeks snapshot still gives a price around €8.925141 and Gamma around 0.019532 with the grid-aware bump.
2. **Market sensitivities:** choose spot ±20%, 9 points and Binomial + Allocation pruning. Both price curves should be increasing for a Call. CSV should have 18 rows plus header.
3. **Convergence:** choose N=[20,40,80,100,150,200], and Binomial + Allocation pruning. Black–Scholes is the analytic reference. Expect oscillatory errors, not monotonic decrease.
4. **Greeks stability:** compare bumps [0.5%,1%,3.5%,5%] and N=[40,80,100,150]. For the fixed-N bump sweep, small Gamma bumps can yield near-zero values in a discrete tree; the grid-aware N sweep should be more informative.
5. **American Put, cash dividend €3 at 0.5 years:** convergence/reference should explicitly read *numerical reference*, not Black–Scholes, and an unsupported Binomial row should be reported while other models continue.

## Project development rules

Don't commit until the user confirms both the UI and the full test suite locally. On completion, stage only the five files above. `git push origin lilian` is a separate user action.
