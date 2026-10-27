# Documentation index

## Current reference material

- [Root README](../README.md): installation, workspaces and project scope.
- [Release protocol](Release_Protocol.md): authoritative acceptance checks,
  reproducibility and supported scientific statements.
- [Architecture snapshot](Architecture_Snapshot.md): module responsibilities,
  completed factorizations and deletion decisions.
- [Quant Research phase 6](Quant_Research_Phase6.md): rationale and navigation
  of the three-section research workspace (written at the start of phase 6).
- [Dividend grid robustness](Dividend_Grid_Robustness_Protocol.md): fixed
  ex-date versus truly shifted contractual ex-date and diagnostics.
- [Accuracy versus cost](Accuracy_vs_Cost_protocol.md) and
  [cross-strike phase 2](Accuracy_vs_Cost_Phase2_protocol.md): empirical
  precision, runtime and sampled Pareto methodology.
- [Temporal experiments](Temporal_Experiments_Protocol.md) and
  [sensitivity](Temporal_Experiments_Sensitivity_Protocol.md): stopping-time,
  exercise region and convergence semantics.

## Historical design and release records

Files beginning `Frontend_` describe successive implementations and should
be treated as historical release evidence, **not current navigation guides**.
They remain tracked to preserve the traceability of prior development.
Course/reference PDF documents are also preserved.

Each versioned CSV/figure in `results/` corresponds to a specific experiment
in `experiments/`. Do not overwrite these data simply to refresh Streamlit.
