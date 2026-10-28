# Quant Analytics — Greeks (v3.1)

**Scope.** Introduces the fourth Streamlit workspace (`04 / Quant analytics`) in the existing LatticePricer272 repository. It does not alter `src/`, Black–Scholes pricing, the existing frontend service, or the earlier tests. Numerical sensitivities always reprice a **fresh** lattice via `frontend.pricing_service.price_contract`.

## Usage

From the repo root and inside `.venv`:

```bash
python -m pytest tests/test_quant_greeks.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

Choose `04 / Quant analytics`, enter the option and market parameters, and click **CALCULATE GREEKS**. Defaults are `S=K=100`, `r=2%`, `sigma=20%`, `T=1y`, `N=100`, European call, trinomial with allocation pruning.

**The Greeks are bump-dependent numerical estimates**, not hedges inferred from observed market option quotes. Cases with dividends or early exercise have no Black–Scholes analytical benchmark in this interface.

## Units and stencils

Let `V(S, sigma, r, T; dividend schedule)` be the *freshly priced* option. The default differentiation steps are:

- Spot: `hS = 0.01*S` (1% of spot).
- Volatility: `hσ = 0.005` (0.5 **volatility percentage point**, not 0.5%).
- Continuously compounded rate: `hr = 0.001` (10 **basis points**, not 10%).
- Calendar roll for Theta: `hT = 1/365` years.

| Reported Greek | Stencil | Unit |
|---|---|---|
| Delta | `[V(S+hS) - V(S-hS)]/(2 hS)` | € premium / € spot |
| Gamma | `[V(S+hS)-2V(S)+V(S-hS)]/hS²` | € premium / € spot² |
| Vega / 1pp | `0.01*[V(σ+hσ)-V(σ-hσ)]/(2 hσ)` | € premium per +1 vol percentage point |
| Rho / 1bp | `0.0001*[V(r+hr)-V(r-hr)]/(2 hr)` | € premium per +1 rate basis point |
| Theta / 1d | `V(t+1 day; S held fixed) - V(t)` | € premium change over one calendar day |

The algorithm uses the same **number of lattice steps** `N` for every shocked run. This ensures an apples-to-apples pricing engine comparison but may cause discretization effects for very small maturity changes. Theta is a **one-day roll-down scenario**, not a central derivative. For the BS comparison, analytic Theta per year is divided by 365; small differences from the one-day discrete roll-down are expected. The spot and volatility ladder charts reuse the scenario prices already computed; no hidden extra pricing calls are introduced by chart rendering.

At `sigma < hσ`, negative-volatility shocks are invalid, so Vega uses the second-order one-sided stencil `0.01*[-3V(σ)+4V(σ+hσ)-V(σ+2hσ)]/(2hσ)`.

For Theta with cash dividends, the remaining time to maturity is reduced by one day **and** all future dividend times are shifted back one calendar day. If an ex-dividend date falls during that day, or maturity is one day or less, Theta is reported as unavailable rather than silently estimating an economically ambiguous jump at unchanged spot. The local model uses **cash dividends**; do not reinterpret them as continuous dividend yields.

## Validation and limitations

Analytic reference uses BS closed-form Greeks only if all are true: European exercise, no dividends, `sigma > 0`, `K > 0`. In the table the comparison is `lattice Greek - analytic Greek` with identical reporting units. Price itself is determined by the selected lattice.

If a *shock* is invalid or fails to build (e.g. cash-dividend tail problems), that Greek becomes **N/A** with an explicit warning. Other Greeks remain available if their shocks succeeded. Base-contract failures abort the run. CSV export includes inputs, differentiation bump sizes, model estimates, analytic references, repriced shocks and warnings.

Lattice Greeks, especially Gamma and Theta, can be noisy and nonmonotone in `N` due to grid alignment and exercise boundaries. A result should not be interpreted as a stable trading hedge without checking robustness to bumps, discretization, calibrated inputs and model choice. Use the advanced bump selector for sensitivity checks. The engine timing shown is the **sum of fresh pricing engine runtimes**, not a market execution latency.

## Next modules

After user validation and commit of this v3.1, expand Quant Analytics with spot/vol/rate/maturity sensitivity grids, price convergence by N, and exercise-boundary analysis. Research Lab remains the fifth workspace.
