# Quant Analytics — Gamma correction v3.1.1

## Why the initial Gamma was zero

At the ATM European call benchmark (S=K=100, sigma=20%, r=2%, T=1y, N=100),
the trinomial lattice with a spot bump of +/-1% gives Gamma ~= 0, whereas
Black--Scholes Gamma is approximately 0.0196 EUR/EUR^2. This is not proof of a
wrong pricing engine. At fixed N, the lattice is locally piecewise affine as a
function of spot: small perturbations do not move any terminal node across the
strike's payoff kink. The central second difference then vanishes.

## Fix

Delta continues to use its own small spot perturbation (+/-1% by default).
Gamma uses a separate spot stencil whose default width is tied to an estimate
of terminal lattice spacing. For trinomial models, the estimate uses:

`h_Gamma / S = exp(sigma * sqrt(3 * T/N)) - 1`.

For the binomial and hybrid models it uses the approximate ratio between
adjacent terminal binomial levels `exp(2*sigma*sqrt(T/N)) - 1`.

The width is **not fitted to Black--Scholes** and can be overridden in Advanced.
A cap at +/-20% avoids implausibly wide finite differences; a warning is raised
when the suggested width is above the cap. This is a numerical approximation,
not a guarantee of a converged market hedge ratio, especially for American
exercise regions, dividend steps, small N or high volatility.

This change adds two scenario pricings when the Gamma and Delta spot widths
differ. The UI shows both widths and the CSV exports them explicitly.

## Validation

For the ATM test, expected Gamma with the grid-aware bump is approximately
0.0195 EUR/EUR^2, close to Black--Scholes 0.0196. Confirm on the user's
real `src/` engines by running the tests and clicking **CALCULATE GREEKS** again.

No changes to `src/` or the original tree/pricer implementation.
