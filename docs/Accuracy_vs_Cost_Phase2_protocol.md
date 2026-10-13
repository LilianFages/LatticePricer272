# Accuracy vs computational cost — Phase 2: robust multi-strike study

## Objective

Extend the existing European no-dividend accuracy/cost experiment **without rewriting any pricing engine** or rerunning all 168 configurations.

For each model \(m\) and number of steps \(N\), use the three European calls with strikes \(K\in\{80,100,120\}\) and Black–Scholes analytical prices. Define

\[
E_{\max}(m,N)=\max_{K\in\{80,100,120\}}|V_{m,N}(K)-V_{BS}(K)|.
\]

Also retain the mean and root-mean-square (RMS) error to provide context. *Worst-case* means worst **of these three strikes only**; it is not a guarantee for every strike or market regime.

## Cost metrics

Each option is priced separately, starting from a fresh tree. Each phase-2 configuration has one unmeasured warmup and five end-to-end time measurements (constructor + tree construction + recursive pricing), with the median retained.

- `max_median_runtime_seconds`: maximum of the three per-strike medians, used as a conservative latency measure for **one** of the sampled contracts.
- `sum_median_runtime_seconds`: sum of the three medians, an *estimate* of a sequential basket cost, **not** a directly timed vectorized/batched operation.
- `allocated_nodes_per_tree`: physical lattice nodes for one option valuation, not the cumulative three-option total.

Minimize \(E_{\max}\) and `max_median_runtime_seconds` to produce the observed multi-strike Pareto frontier.

Because phase-1 and phase-2 timings are collected during different experimental runs, the combined frontier is empirical and sensitive to machine load. Differences close to timing variability should not be interpreted causally; a single-session rerun of tied/close candidates would be appropriate.

## Existing market configuration (must be unchanged)

- Spot 100 €, rate 2%, volatility 20%, maturity 1 year.
- No cash dividends; **European calls only in this phase**.
- Strikes: 80, 100 and 120 €.
- Four models: binomial, full trinomial, branch-pruned trinomial, allocation-pruned trinomial.
- Pruning threshold: \(10^{-18}\).
- Phase-1 steps: 20, 40, 60, 100, 150, 200 and 300.
- Newly measured steps: **80, 125, 175, 225, 250, 275, 325 and 350**.

This adds **96 measured configurations** (8 new step counts × 3 strikes × 4 models), with **480 additional timed runs** at the default five repetitions. It reuses the 84 phase-1 European-call results and does not repeat them. The 84 previously measured puts are intentionally excluded from the new multi-strike timing comparison; the phase-1 put results remain available unchanged.

## Files

Add these two files under the existing repository root:

- `experiments/accuracy_cost_multistrike.py`
- `tests/test_accuracy_cost_multistrike.py`

This protocol belongs in `docs/Accuracy_vs_Cost_Phase2_protocol.md`.

The script requires the existing files:

- `experiments/accuracy_vs_cost.py`
- `results/data/accuracy_cost_benchmark.csv`
- `results/accuracy_cost_summary.txt`

It checks that phase-1 call combinations are complete, nonduplicated and consistent with Black–Scholes and the original experimental configuration.

## Git Bash commands (from repository root)

```bash
python -m pytest tests/test_accuracy_cost_multistrike.py -q
python -m pytest -q
python -m experiments.accuracy_cost_multistrike --quick
python -m experiments.accuracy_cost_multistrike
```

Optional preview (loads phase-1 data, **does not run new pricing**):

```bash
python -m experiments.accuracy_cost_multistrike --baseline-only
```

`--quick` and `--baseline-only` write to **separate** subfolders under `results/` and do not overwrite full results.

## Outputs (full run)

- `results/data/accuracy_cost_multistrike_new.csv`: 96 new individual configurations, including any failures.
- `results/data/accuracy_cost_multistrike_timings.csv`: 480 new raw measurements (if every configuration succeeds).
- `results/data/accuracy_cost_multistrike_aggregate.csv`: one row for each complete model × step-count group with errors and costs; combines phase-1 calls and phase-2 calls.
- `results/data/accuracy_cost_multistrike_targets.csv`: fastest configuration among the sampled points meeting each *three-strike maximum-error* target.
- `results/data/accuracy_cost_multistrike_pareto.csv`: sampled non-dominated points.
- `results/figures/accuracy_cost_multistrike_error_runtime.png`: worst error vs conservative runtime (log-log), connected in ascending \(N\).
- `results/figures/accuracy_cost_multistrike_error_nodes.png`: worst error vs allocated nodes (log-log).
- `results/figures/accuracy_cost_multistrike_pareto.png`: all configurations and observed Pareto frontier (log-log).
- `results/accuracy_cost_multistrike_summary.txt`: experimental metadata, tables of target winners and front, warnings and caveats.

**Zero errors:** never replace a zero error with an arbitrary positive floor. Any exact-zero error observations are excluded from logarithmic axes, while their original values remain in CSVs and Pareto/target calculations.

## How to interpret

1. Verify all 14 new tests, then run the full suite. The previous full suite had 221 passing tests before phase 2 was added.
2. Check that 96 newly measured configurations succeed and that the output shows no missing (model, N, strike) groups.
3. Use the *new* summary/target table to identify the fastest model for a **given maximum-error tolerance across all three sampled strikes**, not a single ATM strike.
4. Compare results at equal \(N\) (algorithmic efficiency) and at similar run time (practical efficiency); do not conflate the two.
5. Keep phase-1 graphs/CSVs and this phase-2 extension separate, and only then select the most informative graphs for the evolving LaTeX report.

**No direct Git write, commit or push is performed by this script.**
