"""Experiment accuracy/cost, aggregation and Pareto-front invariants.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import pytest
from experiments.pricing_accuracy_cost import MODELS, Result
from experiments.accuracy_cost_multistrike import (
    ADDITIONAL_STEPS,
    Aggregate,
    aggregate_cross_strike,
    load_baseline,
    measure_new_steps,
    pareto_multi,
    target_choices,
)
import math
from experiments.pricing_accuracy_cost import (
    MODELS,
    Result,
    _positive_points,
    allocated_nodes,
    bs_reference,
    contract_results,
    create_option,
    fastest_at_error_target,
    make_tree,
    pareto_front,
    parity_deviations,
)
from src.recursive_pricer import RecursivePricer
import pandas as pd
from experiments.accuracy_cost_multistrike import Aggregate, pareto_multi
from experiments.pricing_accuracy_cost import Result, pareto_front
from frontend.research_lab_service import observed_pareto
from src.accuracy_metrics import nondominated_indices


# test_accuracy_cost_multistrike.py (tests preserved)
# Construct controlled one-strike observations with independently specified errors.
def _sample_result(model: str, strike: float, error: float,
                  runtime: float, nodes: int = 100, n: int = 50) -> Result:
    return Result(
        nb_steps=n, strike=strike, option_kind="call", model=model,
        reference_price=10.0, price=10.0 + error,
        absolute_error=error, relative_error=error / 10.0,
        median_runtime_seconds=runtime,
        min_runtime_seconds=runtime, max_runtime_seconds=runtime,
        allocated_nodes=nodes, status="success", failure_reason="",
    )


def _sample_aggregate(model: str, n: int, error: float, runtime: float) -> Aggregate:
    return Aggregate(
        model=model, nb_steps=n, worst_absolute_error=error,
        mean_absolute_error=error/2, root_mean_square_error=error/1.5,
        worst_strike=100.0, max_median_runtime_seconds=runtime,
        sum_median_runtime_seconds=3 * runtime,
        allocated_nodes_per_tree=n*n, source="newly_measured",
    )


# Preserve the separation between original and incremental benchmark grids.
def test_new_steps_are_disjoint_from_old() -> None:
    from experiments.pricing_accuracy_cost import STEP_COUNTS
    assert not set(STEP_COUNTS) & set(ADDITIONAL_STEPS)
    assert len(ADDITIONAL_STEPS) == len(set(ADDITIONAL_STEPS))


# Worst-strike error and aggregate timing answer different research questions.
def test_aggregation_uses_worst_strike_and_separate_latency_metrics() -> None:
    records = [
        _sample_result("binomial", 80.0, 0.004, 0.01),
        _sample_result("binomial", 100.0, 0.009, 0.02),
        _sample_result("binomial", 120.0, 0.006, 0.03),
    ]
    values = aggregate_cross_strike(records, models=("binomial",))
    assert len(values) == 1
    result = values[0]
    assert result.worst_strike == 100.0
    assert result.worst_absolute_error == pytest.approx(0.009)
    assert result.mean_absolute_error == pytest.approx((0.004+0.009+0.006)/3)
    assert result.max_median_runtime_seconds == pytest.approx(0.03)
    assert result.sum_median_runtime_seconds == pytest.approx(0.06)


# Cross-strike comparisons require complete, unique strike coverage.
@pytest.mark.parametrize("records", [
    [_sample_result("binomial", 80.0, 0.01, 0.01)],
    [_sample_result("binomial", 80.0, 0.01, 0.01)]*3,
])
def test_incomplete_or_duplicate_strikes_rejected(records: list[Result]) -> None:
    with pytest.raises(ValueError, match="strike coverage"):
        aggregate_cross_strike(records, models=("binomial",))


# Allocated-node counts must remain coherent within one model/grid run.
def test_varying_node_allocations_rejected() -> None:
    records = [
        _sample_result("binomial", 80.0, 0.002, 0.01, nodes=100),
        _sample_result("binomial", 100.0, 0.002, 0.01, nodes=101),
        _sample_result("binomial", 120.0, 0.002, 0.01, nodes=100),
    ]
    with pytest.raises(ValueError, match="allocation changed"):
        aggregate_cross_strike(records, models=("binomial",))


# Failed pricing attempts cannot be treated as ordinary numeric observations.
def test_failed_record_rejected() -> None:
    rows = [_sample_result("binomial", k, .01, .02) for k in (80.0, 100.0, 120.0)]
    rows[1] = replace(rows[1], status="failed")
    with pytest.raises(ValueError, match="failed"):
        aggregate_cross_strike(rows, models=("binomial",))


# The frontier must exclude models worse on both measured dimensions.
def test_pareto_excludes_dominated_points() -> None:
    a = _sample_aggregate("binomial", 20, 0.03, 0.001)
    b = _sample_aggregate("binomial", 40, 0.01, 0.002)
    c = _sample_aggregate("trinomial_full", 60, 0.02, 0.003)
    d = _sample_aggregate("trinomial_allocation", 100, 0.005, 0.01)
    front = pareto_multi([d, c, b, a])
    assert [p.nb_steps for p in front] == [20, 40, 100]


# Choose the fastest actually measured feasible configuration.
def test_uniform_target_selects_fastest_observed() -> None:
    points = [
        _sample_aggregate("binomial", 60, 0.011, 0.001),
        _sample_aggregate("trinomial_full", 80, 0.006, 0.04),
        _sample_aggregate("trinomial_allocation", 100, 0.007, 0.025),
    ]
    options = target_choices(points, targets=(0.01, 0.001))
    assert options[0]["model"] == "trinomial_allocation"
    assert options[1]["status"] == "not_reached"


# Prevent double-counting of baseline and new experiment steps.
def test_new_measurements_cannot_repeat_phase1_steps() -> None:
    with pytest.raises(ValueError, match="overlap"):
        measure_new_steps((20,), repeats=1)


# Only positive repetition counts yield valid timing samples.
def test_negative_repetitions_rejected() -> None:
    with pytest.raises(ValueError, match="repetition"):
        measure_new_steps((80,), repeats=0)


# The benchmark must never invent a missing first-phase dataset.
def test_baseline_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="phase-1"):
        load_baseline(tmp_path / "absent.csv")


# Baseline contract metadata must identify unique comparable runs.
def test_baseline_incomplete_or_duplicate_rejected(tmp_path: Path) -> None:
    path = tmp_path / "phase1.csv"
    # Minimal CSV with duplicate entries. Baseline validation fails without
    # a costly engine execution; no phase-1 output is modified.
    fields = list(Result.__dataclass_fields__)
    item = _sample_result("binomial", 100.0, 0.01, 0.001, n=20)
    from dataclasses import asdict
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(asdict(item))
        writer.writerow(asdict(item))
    with pytest.raises(ValueError):
        load_baseline(path)


# A completed experiment must export all expected reproducibility artifacts.
def test_output_pipeline_writes_complete_tables_and_graphs(tmp_path: Path) -> None:
    from experiments.accuracy_cost_multistrike import save_phase2

    baseline = [
        _sample_result(model, strike, 0.01 + .001 * idx, .01 + .002 * idx,
                      nodes=150, n=40)
        for model in MODELS
        for idx, strike in enumerate((80.0, 100.0, 120.0))
    ]
    paths = save_phase2(baseline, [], [], tmp_path, repeats=0, added_steps=())
    assert all(path.is_file() for path in paths.values())
    with paths["aggregated"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(MODELS)
    assert all(float(row["worst_absolute_error"]) == pytest.approx(0.012)
               for row in rows)
    assert "not reached" in paths["summary"].read_text(encoding="utf-8")


# Exact-zero numerical errors must remain zero after aggregation.
def test_absence_of_zero_numerical_floor_in_aggregation() -> None:
    rows = [_sample_result("binomial", k, 0.0, .02) for k in (80., 100., 120.)]
    aggregated = aggregate_cross_strike(rows, models=("binomial",))
    assert aggregated[0].worst_absolute_error == 0.0


# test_accuracy_vs_cost.py (tests preserved)
# Build synthetic error/runtime points without invoking the numerical engines.
def _fake_result(model: str, steps: int, runtime: float, error: float) -> Result:
    return Result(
        nb_steps=steps,
        strike=100.0,
        option_kind="call",
        model=model,
        reference_price=10.0,
        price=10.0 + error,
        absolute_error=error,
        relative_error=error / 10.0,
        median_runtime_seconds=runtime,
        min_runtime_seconds=runtime,
        max_runtime_seconds=runtime,
        allocated_nodes=steps * steps,
        status="success",
        failure_reason="",
    )


# The protocol must include all four pricing implementations.
def test_all_four_models_available() -> None:
    assert len(MODELS) == 4
    assert len(set(MODELS)) == 4
    for model in MODELS:
        assert make_tree(model, 4).nb_steps == 4


@pytest.mark.parametrize("kind", ["call", "put"])
def test_black_scholes_contract_is_european(kind: str) -> None:
    option = create_option(kind, 100.0)
    assert not option.is_american
    assert bs_reference(kind, 100.0) > 0.0


def test_invalid_option_type_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported option kind"):
        create_option("digital", 100.0)


def test_pareto_front_keeps_only_nondominated_points() -> None:
    points = [
        _fake_result("binomial", 20, 0.01, 0.12),
        _fake_result("binomial", 40, 0.02, 0.09),
        _fake_result("trinomial_full", 40, 0.03, 0.10),  # dominated
        _fake_result("trinomial_allocation", 100, 0.05, 0.01),
    ]
    frontier = pareto_front(points)
    assert [p.nb_steps for p in frontier] == [20, 40, 100]
    assert all(p.model != "trinomial_full" for p in frontier)


def test_target_selects_fastest_observed_feasible_point() -> None:
    points = [
        _fake_result("binomial", 20, 0.02, 0.03),
        _fake_result("binomial", 60, 0.05, 0.001),
        _fake_result("trinomial_allocation", 40, 0.03, 0.005),
    ]
    selected = fastest_at_error_target(points, 0.01)
    assert selected is not None
    assert selected.model == "trinomial_allocation"
    assert selected.nb_steps == 40
    # The selected configuration must be the fastest actually observed feasible run.
    assert fastest_at_error_target(points, 1e-5) is None


def test_exact_zero_error_not_put_on_log_axis() -> None:
    points = [
        _fake_result("binomial", 20, 0.02, 0.0),
        _fake_result("binomial", 40, 0.04, 0.005),
    ]
    xs, ys = _positive_points(points, "median_runtime_seconds", "absolute_error")
    assert xs == [0.04]
    assert ys == [0.005]


def test_contract_selection_excludes_other_options() -> None:
    points = [_fake_result("binomial", 20, 0.01, 0.1)]
    assert len(contract_results(points, "call", 100.0)) == 1
    # A model comparison must not mix different option contracts.
    assert contract_results(points, "put", 100.0) == []


@pytest.mark.parametrize("model", MODELS)
def test_no_dividend_atm_prices_and_node_counts(model: str) -> None:
    steps = 40
    tree = make_tree(model, steps)
    price = RecursivePricer().price(tree, create_option("call", 100.0))
    assert math.isfinite(price)
    assert abs(price - bs_reference("call", 100.0)) < 0.25
    node_count = allocated_nodes(model, steps, tree)
    assert node_count > 0
    if model == "binomial":
        assert node_count == (steps + 1) * (steps + 2) // 2
    elif model in ("trinomial_full", "trinomial_branch"):
        assert node_count == (steps + 1) ** 2
    else:
        # Use the same at-the-money contract when comparing prices and lattice sizes.
        assert node_count <= (steps + 1) ** 2


def test_allocation_and_branch_agree_for_atm_put() -> None:
    steps = 100
    option = create_option("put", 100.0)
    branch = RecursivePricer().price(make_tree("trinomial_branch", steps), option)
    allocation = RecursivePricer().price(make_tree("trinomial_allocation", steps), option)
    assert allocation == pytest.approx(branch, abs=1e-9)


def test_put_call_parity_diagnostics() -> None:
    reference_call = bs_reference("call", 100.0)
    reference_put = bs_reference("put", 100.0)
    target = 100.0 - 100.0 * math.exp(-0.02)
    assert reference_call - reference_put == pytest.approx(target, abs=1e-10)
    assert parity_deviations([]) == []


# test_pareto_frontier_consistency.py (tests preserved)
def test_pure_frontier_retains_ties_zero_errors_and_tradeoffs() -> None:
    points = [(1., .2), (2., .1), (3., .15), (1., .2), (4., 0.)]
    assert nondominated_indices(points) == (0, 1, 3, 4)
    assert nondominated_indices([]) == ()
    assert nondominated_indices([(1, 0), (1, 0)]) == (0, 1)


def test_live_archive_accept_same_observations_and_duplicate_coordinates() -> None:
    frame = pd.DataFrame({
        'model': ['A', 'B', 'C', 'D', 'E'],
        'max_median_runtime_seconds': [1., 2., 3., 1., 4.],
        'worst_absolute_error': [.2, .1, .15, .2, 0.],
    })
    assert observed_pareto(frame)['model'].tolist() == ['A', 'D', 'B', 'E']


def test_phase_one_frontier_keeps_observation_order_on_equal_runtimes() -> None:
    points = []
    for steps, runtime, gap in [(20, 1., .2), (40, 2., .1),
                                (60, 3., .15), (80, 1., .2)]:
        points.append(Result(
            nb_steps=steps, strike=100., option_kind='call', model='binomial',
            reference_price=10., price=10. + gap, absolute_error=gap,
            relative_error=gap/10., median_runtime_seconds=runtime,
            min_runtime_seconds=runtime, max_runtime_seconds=runtime,
            allocated_nodes=100, status='success', failure_reason='',
        ))
    assert [r.nb_steps for r in pareto_front(points)] == [20, 80, 40]


def test_phase_two_frontier_retains_original_secondary_error_sort() -> None:
    rows = []
    for steps, runtime, gap in [(20, 1., .2), (40, 2., .1),
                                (60, 3., .15), (80, 1., .2)]:
        rows.append(Aggregate(
            model='binomial', nb_steps=steps, worst_absolute_error=gap,
            mean_absolute_error=gap, root_mean_square_error=gap,
            worst_strike=100., max_median_runtime_seconds=runtime,
            sum_median_runtime_seconds=runtime * 3,
            allocated_nodes_per_tree=100, source='phase1',
        ))
    assert [r.nb_steps for r in pareto_multi(rows)] == [20, 80, 40]
