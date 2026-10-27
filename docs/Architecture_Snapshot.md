# LatticePricer272 — Final architecture snapshot (Phase 6D)

## Layered responsibilities

| Layer | Main components | Ownership |
|---|---|---|
| Models and lattice primitives | `src/node.py`, `src/*_tree.py`, `src/transition_moments.py`, `src/dividend_timing.py`, `src/dividend_exercise.py`, `src/lattice_graph.py` | Time discretization, dividends, probabilities, recombining-node graph, price propagation and stopping |
| Financial risk | `src/risk/` | Typed risk interfaces and bumped/analytical sensitivities; no Streamlit UI logic |
| Pricing adapter | `frontend/pricing_service.py` | Validates UI requests and routes models/recursive/backward methods; not a second pricing engine |
| Research services | `frontend/*_service.py`, `frontend/research_accuracy_service.py` | Bounded interactive studies; observable costs, diagnostics and errors |
| Views and navigation | `frontend/app.py`, `frontend/quant_research.py`, `frontend/*_views.py` | Desktop-first Streamlit forms/charts, preservation of session state |
| Offline evidence | `experiments/`, `results/` | Independent protocols, versioned tables and visual evidence |
| Tests and release audit | `tests/`, `scripts/release_audit.py` | Numerical regressions and read-only release evidence verification |

## Consolidations already completed

- Common dividend-step placement and transition moments are shared by the
  trinomial and hybrid engines (6B), preserving the existing right-closed
  ULP event-time convention.
- Reachable-node counting is separate from physical node allocation and is
  centralized in `src/lattice_graph.py` (6B).
- Empirical Pareto dominance is centralized in `src/observed_frontier.py`
  (6B); data adapters retain their public APIs and reference conventions.
- Quant Research now has **three** themes instead of many competing main tabs
  (6A). The ex-date stability view is recorded evidence, not an extra pricing
  call on navigation (6C).
- Phase 6D adds strict CSV checks, legible categorical chart legends and an
  executable release audit. No pricing equations change in this phase.

## Dependency cleanup decision

**No source module or versioned experiment is deleted in 6D.** Static import
resolution can discover broken direct imports but cannot establish that a
module is dead: Streamlit view routing can be dynamic, historical scripts
may be run directly, serialized outputs may depend on schema stability and
external users can import existing APIs. Such deletions would have a larger
risk/reward ratio than a final deliverable needs.

Historical release notes are retained as archival evidence. The new
`docs/README.md` identifies authoritative current documents. Old results
and their underlying scripts remain paired. A future deletion should have
an explicit incoming-dependency inventory, a replacement/migration plan and
successful full tests, not just a low CheckStyle score.

## Boundaries and limitations

- The tree grid and its within-step dividend variance approximation remain
  model-dependent. Financial parity tests are a diagnostic, not proof of
  convergence to the true exercise boundary.
- `scripts/release_audit.py` checks *recorded* Phase 6C science and statically
  follows direct first-party imports. It does not execute a benchmark, inspect
  dynamic dispatch or replace `pytest`/Streamlit visual review.
- Development style scores should not drive algorithm changes; comment
  density and long test methods can distort such heuristics.
