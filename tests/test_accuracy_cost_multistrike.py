"""Regression tests for phase-2 multi-strike accuracy/cost aggregation."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.accuracy_vs_cost import MODELS, Result
from experiments.accuracy_cost_multistrike import (
    ADDITIONAL_STEPS,
    Aggregate,
    aggregate_cross_strike,
    load_baseline,
    measure_new_steps,
    pareto_multi,
    target_choices,
)


def sample_result(model: str, strike: float, error: float,
                  runtime: float, nodes: int = 100, n: int = 50) -> Result:
    return Result(
        nb_steps=n, strike=strike, option_kind="call", model=model,
        reference_price=10.0, price=10.0 + error,
        absolute_error=error, relative_error=error / 10.0,
        median_runtime_seconds=runtime,
        min_runtime_seconds=runtime, max_runtime_seconds=runtime,
        allocated_nodes=nodes, status="success", failure_reason="",
    )


def sample_aggregate(model: str, n: int, error: float, runtime: float) -> Aggregate:
    return Aggregate(
        model=model, nb_steps=n, worst_absolute_error=error,
        mean_absolute_error=error/2, root_mean_square_error=error/1.5,
        worst_strike=100.0, max_median_runtime_seconds=runtime,
        sum_median_runtime_seconds=3 * runtime,
        allocated_nodes_per_tree=n*n, source="newly_measured",
    )


def test_new_steps_are_disjoint_from_old() -> None:
    from experiments.accuracy_vs_cost import STEP_COUNTS
    assert not set(STEP_COUNTS) & set(ADDITIONAL_STEPS)
    assert len(ADDITIONAL_STEPS) == len(set(ADDITIONAL_STEPS))


def test_aggregation_uses_worst_strike_and_separate_latency_metrics() -> None:
    records = [
        sample_result("binomial", 80.0, 0.004, 0.01),
        sample_result("binomial", 100.0, 0.009, 0.02),
        sample_result("binomial", 120.0, 0.006, 0.03),
    ]
    values = aggregate_cross_strike(records, models=("binomial",))
    assert len(values) == 1
    result = values[0]
    assert result.worst_strike == 100.0
    assert result.worst_absolute_error == pytest.approx(0.009)
    assert result.mean_absolute_error == pytest.approx((0.004+0.009+0.006)/3)
    assert result.max_median_runtime_seconds == pytest.approx(0.03)
    assert result.sum_median_runtime_seconds == pytest.approx(0.06)


@pytest.mark.parametrize("records", [
    [sample_result("binomial", 80.0, 0.01, 0.01)],
    [sample_result("binomial", 80.0, 0.01, 0.01)]*3,
])
def test_incomplete_or_duplicate_strikes_rejected(records: list[Result]) -> None:
    with pytest.raises(ValueError, match="strike coverage"):
        aggregate_cross_strike(records, models=("binomial",))


def test_varying_node_allocations_rejected() -> None:
    records = [
        sample_result("binomial", 80.0, 0.002, 0.01, nodes=100),
        sample_result("binomial", 100.0, 0.002, 0.01, nodes=101),
        sample_result("binomial", 120.0, 0.002, 0.01, nodes=100),
    ]
    with pytest.raises(ValueError, match="allocation changed"):
        aggregate_cross_strike(records, models=("binomial",))


def test_failed_record_rejected() -> None:
    rows = [sample_result("binomial", k, .01, .02) for k in (80.0, 100.0, 120.0)]
    rows[1] = replace(rows[1], status="failed")
    with pytest.raises(ValueError, match="failed"):
        aggregate_cross_strike(rows, models=("binomial",))


def test_pareto_excludes_dominated_points() -> None:
    a = sample_aggregate("binomial", 20, 0.03, 0.001)
    b = sample_aggregate("binomial", 40, 0.01, 0.002)
    c = sample_aggregate("trinomial_full", 60, 0.02, 0.003)
    d = sample_aggregate("trinomial_allocation", 100, 0.005, 0.01)
    front = pareto_multi([d, c, b, a])
    assert [p.nb_steps for p in front] == [20, 40, 100]


def test_uniform_target_selects_fastest_observed() -> None:
    points = [
        sample_aggregate("binomial", 60, 0.011, 0.001),
        sample_aggregate("trinomial_full", 80, 0.006, 0.04),
        sample_aggregate("trinomial_allocation", 100, 0.007, 0.025),
    ]
    options = target_choices(points, targets=(0.01, 0.001))
    assert options[0]["model"] == "trinomial_allocation"
    assert options[1]["status"] == "not_reached"


def test_new_measurements_cannot_repeat_phase1_steps() -> None:
    with pytest.raises(ValueError, match="overlap"):
        measure_new_steps((20,), repeats=1)


def test_negative_repetitions_rejected() -> None:
    with pytest.raises(ValueError, match="repetition"):
        measure_new_steps((80,), repeats=0)


def test_baseline_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="phase-1"):
        load_baseline(tmp_path / "absent.csv")


def test_baseline_incomplete_or_duplicate_rejected(tmp_path: Path) -> None:
    path = tmp_path / "phase1.csv"
    # Minimal CSV with duplicate entries. Baseline validation fails without
    # a costly engine execution; no phase-1 output is modified.
    fields = list(Result.__dataclass_fields__)
    item = sample_result("binomial", 100.0, 0.01, 0.001, n=20)
    from dataclasses import asdict
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(asdict(item))
        writer.writerow(asdict(item))
    with pytest.raises(ValueError):
        load_baseline(path)


def test_output_pipeline_writes_complete_tables_and_graphs(tmp_path: Path) -> None:
    from experiments.accuracy_cost_multistrike import save_phase2

    baseline = [
        sample_result(model, strike, 0.01 + .001 * idx, .01 + .002 * idx,
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


def test_absence_of_zero_numerical_floor_in_aggregation() -> None:
    rows = [sample_result("binomial", k, 0.0, .02) for k in (80., 100., 120.)]
    aggregated = aggregate_cross_strike(rows, models=("binomial",))
    assert aggregated[0].worst_absolute_error == 0.0
