"""Focused regressions for the accuracy-versus-cost experiment."""

import math

import pytest

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
