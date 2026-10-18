"""Small analytical lattice checks plus an integration smoke test.

Run at the repository root:
    python -m pytest tests/test_option_value_exercise_by_dates.py -q
"""
from __future__ import annotations

import math

import pytest

from src.node import Node
from src.option import PutOption
from experiments.option_value_exercise_by_dates import (
    DIVIDEND_AMOUNT,
    DIVIDEND_TIME,
    RATE,
    SPOT,
    Scenario,
    exercise_decision,
    make_scenarios,
    profile_from_priced_tree,
    propagate,
    run_experiment,
    transitions,
)


def simple_two_step_tree(american: bool) -> tuple[Node, PutOption, Scenario]:
    """Two-step 1/2-1/2 tree, recombining at S=100, with r=0."""
    root = Node(100.0)
    up = Node(110.0)
    down = Node(80.0)
    upup = Node(120.0)
    middle = Node(100.0)
    downdown = Node(80.0)

    root.next_up, root.next_down = up, down
    up.next_up, up.next_down = upup, middle
    down.next_up, down.next_down = middle, downdown
    for node in (root, up, down):
        node.up_probability = 0.5
        node.down_probability = 0.5

    upup.option_value = 0.0
    middle.option_value = 0.0
    downdown.option_value = 20.0
    up.option_value = 0.0
    down.option_value = 20.0 if american else 10.0
    root.option_value = 10.0 if american else 5.0
    option = PutOption(100.0, is_american=american)
    scenario = Scenario("test", "test", 100.0, american)
    return root, option, scenario


def test_matched_initial_terminal_forward() -> None:
    s = make_scenarios()
    adjusted = SPOT - DIVIDEND_AMOUNT * math.exp(-RATE * DIVIDEND_TIME)
    assert s[0].spot == SPOT
    assert s[1].spot == pytest.approx(adjusted)
    assert s[2].spot == pytest.approx(adjusted)
    div_forward = SPOT * math.exp(RATE) - DIVIDEND_AMOUNT * math.exp(
        RATE * (1.0 - DIVIDEND_TIME)
    )
    assert adjusted * math.exp(RATE) == pytest.approx(div_forward)


def test_recombination_probabilities_merge() -> None:
    root, _, _ = simple_two_step_tree(american=False)
    first = propagate({root: 1.0})
    second = propagate(first)
    assert sorted(first.values()) == [0.5, 0.5]
    assert sorted(second.values()) == [0.25, 0.25, 0.5]


def test_monomial_pruning_conserves_mass() -> None:
    root = Node(100.0)
    sole = Node(101.0)
    root.next_up = root.next_mid = root.next_down = sole
    root.up_probability = 0.0
    root.mid_probability = 1.0
    root.down_probability = 0.0
    assert transitions(root) == [(sole, 1.0)]
    assert propagate({root: 1.0})[sole] == pytest.approx(1.0)


def test_duplicate_positive_successors_sum_weights() -> None:
    root = Node(100.0)
    child = Node(105.0)
    root.next_up = root.next_mid = root.next_down = child
    root.up_probability = 0.2
    root.mid_probability = 0.5
    root.down_probability = 0.3
    assert propagate({root: 1.0})[child] == pytest.approx(1.0)


def test_reject_invalid_probability_sum() -> None:
    root = Node(100.0)
    root.next_up = Node(105.0)
    root.next_down = Node(95.0)
    root.up_probability = 0.3
    root.down_probability = 0.3
    with pytest.raises(AssertionError, match="Outgoing probability mass"):
        transitions(root)


def test_european_discounted_average_is_martingale() -> None:
    root, option, scenario = simple_two_step_tree(american=False)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert [r["discounted_mean_option_value"] for r in result.rows] == pytest.approx(
        [5.0, 5.0, 5.0]
    )
    assert [r["exercise_region_probability"] for r in result.rows] == pytest.approx(
        [0.0, 0.0, 0.25]
    )
    assert result.total_first_exercise_probability == pytest.approx(0.25)
    assert result.expiry_without_exercise_probability == pytest.approx(0.75)


def test_american_discounted_mean_can_decrease() -> None:
    root, option, scenario = simple_two_step_tree(american=True)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert [r["discounted_mean_option_value"] for r in result.rows] == pytest.approx(
        [10.0, 10.0, 5.0]
    )


def test_exercise_region_not_stopping_probability() -> None:
    root, option, scenario = simple_two_step_tree(american=True)
    result = profile_from_priced_tree(root, option, 2, 1.0, 0.0, scenario)
    assert result.rows[1]["exercise_region_probability"] == pytest.approx(0.5)
    assert result.rows[2]["exercise_region_probability"] == pytest.approx(0.25)
    assert result.rows[1]["first_exercise_probability"] == pytest.approx(0.5)
    assert result.rows[2]["first_exercise_probability"] == pytest.approx(0.0)
    assert result.total_first_exercise_probability == pytest.approx(0.5)
    assert result.expiry_without_exercise_probability == pytest.approx(0.5)


def test_european_never_exercises_early() -> None:
    root, option, _ = simple_two_step_tree(american=False)
    low = root.next_down
    assert low is not None
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=False) is False
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=True) is True


def test_american_strict_exercise_over_continuation() -> None:
    root, option, _ = simple_two_step_tree(american=True)
    low = root.next_down
    assert low is not None
    assert exercise_decision(low, option, dt=0.5, rate=0, is_maturity=False) is True


def test_experiment_integration_small_tree() -> None:
    """Real src/ integration, deliberately light for the everyday test suite."""
    results = run_experiment(nb_steps=20)
    assert len(results) == 3
    for result in results:
        assert len(result.rows) == 21
        assert math.isclose(
            result.total_first_exercise_probability
            + result.expiry_without_exercise_probability,
            1.0, abs_tol=2e-10,
        )
    assert results[1].initial_price + 1e-9 >= results[2].initial_price
