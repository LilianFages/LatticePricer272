# LatticePricer272 — Interactive Lattice Explorer (v2.2)

The new **03 / Lattice explorer** Streamlit workspace visualizes the **real priced
object graphs** created by the engines in `src/`, with a shared geometry scale
for two side-by-side models. No new lattice pricing model is introduced.

## Install

From the `lilian` branch at the repository root, extract the ZIP. It replaces
`frontend/app.py` from Model Comparison v2.1 and adds two frontend modules,
one test module, and this document. It does not change any `src/` file.

```bash
python -m pytest tests/test_lattice_explorer_service.py -q
python -m pytest -q
python -m streamlit run frontend/app.py
```

## Controls and interpretation

- Enter a put/call and European/American exercise style, market parameters,
  2–10 steps, and two architectures from the existing 5 model families.
- Press **Build & Compare Lattices**. A failure of one model does not suppress
  the successful result from the other model.
- Hover over a node to inspect spot, valued option, probability of reaching
  the node from the root, and outgoing probabilities. Zoom and pan directly
  in the charts. The optional inspector lets you pick a date and node to
  read the same values in a table.
- Blue nodes: continuation; amber: **strict** optimal early exercise;
  grey: maturity; pink: node with simplified/collapsed outgoing transition.
  Maturity and pruning visualization are independent concepts; the color
  priority is simplified first.
- Positive-probability transitions are shown; duplicate successor pointers
  to the same node are grouped, with their positive weights added.
- For Binomial and Hybrid, forward node probabilities are recalculated by
  propagating the engine's **existing** transition probabilities; the
  `Node.reach_probability` attribute is not assumed to be populated.
- The chart is the **positive-probability connected geometry**, **not** the
  total number of objects allocated during construction. This distinction
  is critical when interpreting branch vs allocation pruning. The allocation
  count is only displayed when the engine itself exposes it.
- Use `1e-3` selectively to *visualize* pruning in very small trees; a high
  threshold may change prices. The usual `1e-18` benchmark threshold often
  does **not** visibly prune a tree with just 5 steps.
- Optional one cash dividend: the model uses `(step_start, step_end]` dates.
  Binomial is incompatible and shows an isolated, explicit error.

## Validation and next scope

Test both a **binomial vs full trinomial** comparison and **branch pruning vs
allocation pruning**, with the same market parameters. Check that price and
model geometry match existing behavior, and verify conservation of each
layer's probability mass. The next extension can add highlighted dividend
columns, an explicit comparison of allocated (including unreachable) nodes,
color-by-moneyness and dedicated exercise-boundary overlays.

The UI is for course research and testing, not live financial quotes.
