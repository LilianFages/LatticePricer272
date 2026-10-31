"""Exercise Boundary Analytics: exact small-tree stopping-time invariants.

Tests exercise logic without relying on lattice pricing convergence or
Black–Scholes; can run against the actual repo in the user's .venv.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import replace

import pytest

from frontend.exercise_boundary_service import (
    _advance, _exercise, _successors, analyze_priced_tree, dates_csv, nodes_csv,
    run_exercise_study,
)
from frontend.pricing_service import PricingInputError, PricingRequest


# Preserve the event-date and exercise-policy conventions in this case.
class Node:
    def __init__(self: object, spot: float, value: object):
        self.price = float(spot)
        self.option_value = float(value)
        self.next_up = None
        self.next_mid = None
        self.next_down = None
        self.up_probability = None
        self.mid_probability = None
        self.down_probability = None


# Set both transition probabilities on the controlled binomial toy tree.
def _link_binomial(node: Node, up: Node, down: Node, p: float = 0.5) -> None:
    node.next_up = up
    node.next_down = down
    node.up_probability = p
    node.down_probability = 1.0 - p


# Use a two-step toy tree to isolate stopping decisions from pricing accuracy.
def _basic_american() -> tuple[Node, PricingRequest]:
    request = PricingRequest(
        spot=100., strike=100., rate=.02, maturity=1., steps=2,
        model="Trinomial · Allocation pruning", option_type="Put",
        exercise="American",
    )
    terminal = Node(110, 0)
    left = Node(80, 20)
    right = Node(120, 0)
    dt = request.maturity / request.steps
    root = Node(100, math.exp(-.02 * dt)*10)
    _link_binomial(root, right, left)
    _link_binomial(left, terminal, terminal)
    _link_binomial(right, terminal, terminal)
    return root, request


# First-exercise mass and continuation survival form a stopping partition.
def test_american_stopping_not_double_counted() -> None:
    root, request = _basic_american()
    dates, points, err = analyze_priced_tree(root, request)
    assert err < 1e-14
    assert len(dates) == 3
    assert dates[0].region_probability == 0
    assert dates[1].region_probability == pytest.approx(.5)
    assert dates[1].first_exercise_probability == pytest.approx(.5)
    assert dates[1].boundary_spot == pytest.approx(80.)
    assert dates[2].survival_probability == pytest.approx(.5)
    assert dates[2].cumulative_exercise_probability == pytest.approx(.5)
    assert dates[0].discounted_mean_value == pytest.approx(root.option_value)
    assert dates[2].discounted_mean_value == 0
    assert any(p.state == "Early exercise" for p in points)


# European exercise cannot occur before contractual maturity.
def test_european_only_itm_expiry_exercises() -> None:
    request = PricingRequest(spot=100, strike=100, rate=0., steps=1,
                             exercise="European", option_type="Put")
    left, right = Node(90, 10), Node(110, 0)
    root = Node(100, 5)
    _link_binomial(root, right, left)
    dates, points, _ = analyze_priced_tree(root, request)
    assert dates[0].first_exercise_probability == 0
    assert dates[0].boundary_spot is None
    assert dates[1].first_exercise_probability == pytest.approx(.5)
    assert dates[1].region_probability == pytest.approx(.5)
    assert dates[1].survival_probability == pytest.approx(.5)
    assert dates[1].boundary_spot == 90
    assert dates[0].discounted_mean_value == dates[1].discounted_mean_value
    assert sum(p.state == "Maturity · ITM" for p in points) == 1


# Recombining successor links must not manufacture probability mass.
def test_duplicate_branches_preserve_mass() -> None:
    request = PricingRequest(spot=100, strike=100, rate=0., steps=1,
                             exercise="European", option_type="Put")
    child = Node(90, 10)
    root = Node(100, 10)
    root.next_up = child
    root.next_mid = child
    root.next_down = child
    root.up_probability = .3
    root.mid_probability = .4
    root.down_probability = .3
    assert len(_successors(root)) == 3
    assert _advance({root: 1.0})[child] == pytest.approx(1.)
    dates, _, _ = analyze_priced_tree(root, request)
    assert dates[1].first_exercise_probability == pytest.approx(1.)


# Reject transitions violating probability axioms.
def test_invalid_transition_probability_rejected() -> None:
    root = Node(100, 0)
    child = Node(90, 0)
    root.next_up, root.next_down = child, child
    root.up_probability, root.down_probability = .4, .4
    with pytest.raises(RuntimeError, match="sum to 1"):
        _successors(root)


# Positive transition mass must have a reachable successor.
def test_nonzero_prob_without_successor_rejected() -> None:
    root = Node(100, 0)
    root.up_probability = 1.
    with pytest.raises(RuntimeError, match="without child"):
        _successors(root)


# Define a deterministic convention for exact exercise/hold ties.
def test_tie_means_continuation() -> None:
    request = PricingRequest(strike=100, rate=0., steps=1,
                             exercise="American", option_type="Put")
    root = Node(90, 10)
    terminal = Node(90, 10)
    _link_binomial(root, terminal, terminal)
    assert _exercise(root, request, 0, 1.) is False


# A zero-payoff maturity state does not represent positive exercise value.
def test_worthless_maturity_is_not_exercised() -> None:
    req = PricingRequest(strike=100, rate=0., steps=1,
                         exercise="American", option_type="Call")
    node = Node(99, 0)
    assert _exercise(node, req, 1, 1.) is False


# Boundary extrema must respect the chosen payoff direction.
def test_put_and_call_boundary_extrema() -> None:
    # Each case has a real early-exercise node at the chosen price.
    for kind, spot in (("Put", 70.), ("Call", 130.)):
        req = PricingRequest(strike=100., rate=.02, steps=1,
                             option_type=kind, exercise="American")
        node = Node(spot, 30)
        terminal = Node(100., 0)
        _link_binomial(node, terminal, terminal)
        assert _exercise(node, req, 0, 1.)


# CSV labels must distinguish region occupancy and first exercise.
def test_csv_provides_clearly_named_probability_conventions() -> None:
    root, request = _basic_american()
    dates, points, err = analyze_priced_tree(root, request)
    from frontend.exercise_boundary_service import ExerciseStudy
    study = ExerciseStudy(request, root.option_value, dates, points, .5, .5, .5,
                          err, 0)
    rows = list(csv.DictReader(io.StringIO(dates_csv(study))))
    assert len(rows) == 3
    assert rows[1]["first_exercise_probability"] == "0.5"
    assert rows[2]["survival_probability"] == "0.5"
    node_rows = list(csv.DictReader(io.StringIO(nodes_csv(study))))
    assert len(node_rows) == len(points)
    assert "risk_neutral_unconditional_reach_probability" in node_rows[0]


# Absence of early exercise must not create a fictitious boundary.
def test_not_misleading_when_no_early_exercise() -> None:
    req = PricingRequest(strike=100., rate=0., steps=1,
                         option_type="Call", exercise="European")
    itm = Node(110, 10)
    otm = Node(90, 0)
    root = Node(100, 5)
    _link_binomial(root, itm, otm)
    dates, _, _ = analyze_priced_tree(root, req)
    assert dates[0].boundary_spot is None
    assert dates[-1].first_exercise_probability == pytest.approx(.5)
    assert dates[-1].boundary_spot == 110


# Zero probability of exercise is not evidence of a boundary location.
def test_zero_early_mass_does_not_invent_boundary() -> None:
    root, request = _basic_american()
    request = replace(request, exercise="European")
    dates, _, _ = analyze_priced_tree(root, request)
    assert all(x.boundary_spot is None for x in dates[:-1])


# Guard costly lattice construction against out-of-range requests.
def test_build_service_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    request = PricingRequest(steps=201)
    import frontend.exercise_boundary_service as service
    monkeypatch.setattr(service, "validate_request", lambda _: None)
    with pytest.raises(PricingInputError, match="at most"):
        run_exercise_study(request)


# Cash-dividend exercise profiles are checked against real engine outputs.
def test_live_engine_cash_dividend_scenario_matches_course_results() -> None:
    """Actual engine integration; do not confuse with the synthetic checks above."""
    from frontend.pricing_service import DividendInput
    request = PricingRequest(
        spot=100., strike=100., rate=.02, volatility=.20, maturity=1.,
        steps=50, model="Trinomial · Allocation pruning", option_type="Put",
        exercise="American", dividends=(DividendInput(.5, 3.),),
        pruning_threshold=1e-18,
    )
    study = run_exercise_study(request)
    assert study.premium == pytest.approx(8.60049447, abs=3e-5)
    assert study.early_exercise_probability == pytest.approx(.51538, abs=3e-4)
    assert study.dates[25].first_exercise_probability == pytest.approx(.07450, abs=3e-4)
    assert study.worst_probability_mass_error < 2e-10
    assert study.total_exercise_probability + study.expired_without_exercise_probability == pytest.approx(1.)


# Discounted European option value obeys the martingale consistency check.
def test_live_engine_european_martingale_no_dividends() -> None:
    request = PricingRequest(spot=100., strike=100., rate=.02,
                             volatility=.2, maturity=1., steps=40,
                             model="Trinomial · Allocation pruning",
                             option_type="Put", exercise="European")
    study = run_exercise_study(request)
    assert study.early_exercise_probability == 0.0
    assert all(d.boundary_spot is None for d in study.dates[:-1])
    assert max(abs(d.discounted_mean_value - study.premium)
               for d in study.dates) < 1e-7
