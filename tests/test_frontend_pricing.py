"""Frontend pricing requests, algorithm routing and model selection.

Consolidated without dropping individual regression cases.
"""

from __future__ import annotations

import csv
import io
import math
import pytest
from frontend.pricing_service import (
    MODELS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    count_reachable_nodes,
    price_contract,
    ticket_csv,
    validate_request,
)
from src.node import Node
from dataclasses import replace
from functools import partial
from frontend.greeks_service import calculate_greeks
from frontend.pricing_service import (
    PRICING_ALGORITHMS,
    DividendInput,
    PricingInputError,
    PricingRequest,
    PricingResult,
    price_contract,
    ticket_csv,
)
from frontend.trading_risk import compute_desk_risk, compute_greek_curve
from frontend.model_selection import (
    FAMILIES, PRUNING_THRESHOLDS, resolve_model_selection,
)
from frontend.pricing_service import MODELS, PricingInputError, PricingRequest


# test_frontend_pricing.py (tests preserved)
# Check nonnegative prices, populated references and coherent lattice sizes.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kind", ("Call", "Put"))
def test_european_models_return_finite_price_and_valid_node_count(
    model: str, kind: str
) -> None:
    result = price_contract(PricingRequest(model=model, option_type=kind, steps=20))
    assert math.isfinite(result.price)
    assert result.price >= 0.0
    assert result.reachable_nodes >= result.request.steps + 1
    assert result.elapsed_ms >= 0.0
    assert result.black_scholes is not None
    if model == "Trinomial · Allocation pruning":
        assert result.allocated_nodes is not None
        assert result.allocated_nodes >= result.reachable_nodes
    else:
        assert result.allocated_nodes is None


# Immediate exercise gives a rigorous lower bound for an American put.
@pytest.mark.parametrize("model", MODELS)
def test_american_put_not_below_intrinsic_value(model: str) -> None:
    result = price_contract(PricingRequest(
        model=model, steps=25, exercise="American",
        option_type="Put", spot=90.0, strike=100.0,
    ))
    assert result.price >= 10.0 - 1e-10
    assert result.black_scholes is None


# Cash dividends invalidate the no-dividend analytic benchmark.
@pytest.mark.parametrize("model", (
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning", "Hybrid",
))
def test_supported_models_price_with_cash_dividend(model: str) -> None:
    result = price_contract(PricingRequest(
        model=model, steps=20,
        dividends=(DividendInput(time=0.5, amount=3.0),),
    ))
    assert math.isfinite(result.price)
    assert result.black_scholes is None


# At zero pruning, the allocation and full-grid prices must coincide.
def test_variants_match_at_zero_pruning_threshold() -> None:
    common = dict(steps=25, option_type="Put", pruning_threshold=0.0)
    full = price_contract(PricingRequest(**common, model="Trinomial · Full"))
    allocation = price_contract(PricingRequest(
        **common, model="Trinomial · Allocation pruning",
    ))
    assert allocation.price == pytest.approx(full.price, abs=1e-10)


# The allocation engine must handle the two-ex-date contract.
def test_two_dividends_can_be_priced_by_allocation_tree() -> None:
    result = price_contract(PricingRequest(
        model="Trinomial · Allocation pruning", steps=80,
        dividends=(DividendInput(0.25, 2.0), DividendInput(0.75, 3.0)),
    ))
    assert result.price > 0.0


# Disallow cash-dividend pricing when the engine does not support it.
def test_binomial_dividends_are_rejected_with_clear_message() -> None:
    with pytest.raises(PricingInputError, match="does not support cash dividends"):
        validate_request(PricingRequest(
            model="Binomial", dividends=(DividendInput(0.5, 3.0),),
        ))


# Enforce the engine convention of one cash dividend per time step.
def test_dividends_cannot_share_lattice_step() -> None:
    with pytest.raises(PricingInputError, match="same lattice time step"):
        validate_request(PricingRequest(
            model="Hybrid", steps=10,
            dividends=(DividendInput(0.51, 1.0), DividendInput(0.55, 1.0)),
        ))


# Inputs must be rejected before the numerical engine is invoked.
@pytest.mark.parametrize("override", [
    {"spot": 0.0}, {"volatility": -0.1}, {"maturity": 0.0},
    {"steps": 0}, {"steps": 351}, {"steps": 12.5},
    {"strike": -1.0}, {"rate": math.nan},
    {"pruning_threshold": 1.0}, {"exercise": "Bermudan"},
])
def test_invalid_contract_parameters_are_rejected(override: dict) -> None:
    with pytest.raises(PricingInputError):
        validate_request(PricingRequest(**override))


# A cash payment cannot destroy positivity of the adjusted stock process.
def test_dividend_greater_than_spot_is_rejected() -> None:
    with pytest.raises(PricingInputError, match="Discounted cash dividends"):
        validate_request(PricingRequest(
            model="Hybrid", dividends=(DividendInput(0.5, 200.0),),
        ))


# Reachable-node counts must not double-count shared successors.
def test_reachable_count_deduplicates_recombining_links() -> None:
    root = Node(100.0)
    a = Node(110.0)
    b = Node(90.0)
    shared = Node(100.0)
    root.next_up = a
    root.next_down = b
    a.next_down = shared
    b.next_up = shared
    assert count_reachable_nodes(root) == 4


# A downloadable ticket must retain the full contract specification.
def test_ticket_export_contains_market_parameters_and_price() -> None:
    result = price_contract(PricingRequest(steps=15))
    fields = dict(csv.reader(io.StringIO(ticket_csv(result))))
    assert fields["model"] == "Trinomial · Allocation pruning"
    assert fields["steps"] == "15"
    assert float(fields["price"]) == pytest.approx(result.price)
    assert float(fields["reachable_nodes"]) >= 16


# Displayed price gaps must preserve the intended benchmark sign.
def test_black_scholes_difference_sign_is_correct() -> None:
    result = price_contract(PricingRequest(steps=30, model="Binomial"))
    assert result.black_scholes_error == pytest.approx(
        result.price - result.black_scholes
    )


# test_pricing_algorithm_routing.py (tests preserved)
MODELS = (
    "Binomial",
    "Hybrid",
    "Trinomial · Full",
    "Trinomial · Branch pruning",
    "Trinomial · Allocation pruning",
)


def make_request(
    model: str,
    option_type: str,
    exercise: str,
    *,
    dividends: tuple[DividendInput, ...] = (),
    steps: int = 14,
) -> PricingRequest:
    return PricingRequest(
        model=model,
        option_type=option_type,
        exercise=exercise,
        dividends=dividends,
        steps=steps,
        spot=103.0,
        strike=100.0,
        rate=0.02,
        volatility=0.22,
        maturity=1.0,
    )


# Recursive and backward algorithms should agree on prices and node accounting.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("option_type", ("Call", "Put"))
@pytest.mark.parametrize("exercise", ("European", "American"))
def test_public_api_parity_no_dividends(
    model: str, option_type: str, exercise: str,
) -> None:
    request = make_request(model, option_type, exercise)
    recursive = price_contract(request, algorithm="Recursive")
    backward = price_contract(request, algorithm="Backward")
    assert math.isclose(recursive.price, backward.price, rel_tol=1e-12, abs_tol=1e-12)
    assert recursive.request == backward.request == request
    assert recursive.algorithm == "Recursive"
    assert backward.algorithm == "Backward"
    assert recursive.reachable_nodes == backward.reachable_nodes
    assert recursive.allocated_nodes == backward.allocated_nodes
    assert recursive.pruned_transitions == backward.pruned_transitions


# Cash-dividend parity also guards ex-date event ordering across engines.
@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize("option_type", ("Call", "Put"))
@pytest.mark.parametrize("exercise", ("European", "American"))
def test_public_api_parity_cash_dividends(
    model: str, option_type: str, exercise: str,
) -> None:
    request = make_request(
        model, option_type, exercise,
        dividends=(DividendInput(time=0.5, amount=3.0),),
        steps=20,
    )
    recursive = price_contract(request, algorithm="Recursive")
    backward = price_contract(request, algorithm="Backward")
    assert math.isclose(recursive.price, backward.price, rel_tol=1e-12, abs_tol=1e-12)
    assert recursive.reachable_nodes == backward.reachable_nodes
    assert recursive.allocated_nodes == backward.allocated_nodes
    assert recursive.pruned_transitions == backward.pruned_transitions


# Backward pricing is opt-in and must not change the public default.
def test_default_stays_recursive() -> None:
    request = PricingRequest(steps=8)
    default = price_contract(request)
    explicit = price_contract(request, algorithm="Recursive")
    assert default.algorithm == explicit.algorithm == "Recursive"
    assert default.price == explicit.price


@pytest.mark.parametrize("bad", ("Unknown", "forward", "", None))
def test_invalid_algorithm_is_rejected(bad: object) -> None:
    with pytest.raises(PricingInputError, match="pricing algorithm"):
        price_contract(PricingRequest(steps=6), algorithm=bad)


def test_csv_exports_executed_algorithm() -> None:
    result = price_contract(PricingRequest(steps=7), algorithm="Backward")
    fields = dict(csv.reader(io.StringIO(ticket_csv(result))))
    assert fields["pricing_algorithm"] == "Backward"
    assert fields["price"] == str(result.price)


# Algorithm selection is execution metadata, never mutable contract state.
def test_immutable_request_is_shared_between_methods() -> None:
    request = PricingRequest(
        model="Hybrid", steps=14, option_type="Put", exercise="American",
        dividends=(DividendInput(time=0.5, amount=2.0),),
    )
    baseline = replace(request)
    price_contract(request, algorithm="Backward")
    price_contract(request, algorithm="Recursive")
    assert request == baseline
    assert not hasattr(request, "algorithm")


# Every Greek shock must use the chosen pricing algorithm consistently.
@pytest.mark.parametrize("algorithm", PRICING_ALGORITHMS)
def test_seven_greeks_use_injected_pricing_algorithm(algorithm: str) -> None:
    request = PricingRequest(
        model="Trinomial · Allocation pruning", steps=14, volatility=0.22,
    )
    observed: list[str] = []

    def observed_pricer(shocked: PricingRequest) -> PricingResult:
        value = price_contract(shocked, algorithm=algorithm)
        observed.append(value.algorithm)
        return value

    snapshot = compute_desk_risk(request, pricer=observed_pricer)
    assert len(observed) == snapshot.base.evaluations + snapshot.extra_evaluations
    assert len(observed) > 8
    assert set(observed) == {algorithm}
    for name in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"):
        assert snapshot.value(name) is not None


# Verify that the pricing workflow keeps its input contract intact.
def test_greek_values_agree_across_algorithms() -> None:
    request = PricingRequest(
        model="Hybrid", option_type="Put", exercise="American",
        spot=104.0, strike=100.0, steps=20,
        dividends=(DividendInput(time=0.5, amount=2.0),),
    )
    recursive = compute_desk_risk(
        request, pricer=partial(price_contract, algorithm="Recursive"),
    )
    backward = compute_desk_risk(
        request, pricer=partial(price_contract, algorithm="Backward"),
    )
    for name in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"):
        r_value, b_value = recursive.value(name), backward.value(name)
        if r_value is None or b_value is None:
            assert r_value is b_value
        else:
            assert math.isclose(r_value, b_value, rel_tol=1e-7, abs_tol=1e-7)


# The callback must propagate through selected-curve repricing.
def test_selected_greek_curve_uses_backward_callback() -> None:
    request = PricingRequest(model="Binomial", steps=9)
    seen = []

    def backward(shocked: PricingRequest) -> PricingResult:
        result = price_contract(shocked, algorithm="Backward")
        seen.append(result.algorithm)
        return result

    curve = compute_greek_curve(request, "Delta", count=3, pricer=backward)
    assert len(curve) == 3
    assert seen and set(seen) == {"Backward"}
    assert all(point.value is not None for point in curve)


def test_binomial_cash_dividend_restriction_remains() -> None:
    request = PricingRequest(
        model="Binomial", steps=8,
        dividends=(DividendInput(time=0.5, amount=3.0),),
    )
    # Reject invalid cases explicitly rather than silently substituting values.
    with pytest.raises(PricingInputError, match="BinomialTree"):
        price_contract(request, algorithm="Backward")

# LOT_5C3_GUARDED_REFACTOR


# test_model_selection.py (tests preserved)
# Model routing: pricing families and non-pruned defaults must remain distinct.
def test_three_distinct_families() -> None:
    assert FAMILIES == ("Binomial", "Hybrid", "Trinomial")


def test_binomial_without_pruning_maps_exactly() -> None:
    model, threshold = resolve_model_selection("Binomial")
    assert model == "Binomial" and threshold == 1e-18


def test_hybrid_without_pruning_maps_exactly() -> None:
    model, threshold = resolve_model_selection("Hybrid")
    assert model == "Hybrid" and threshold == 1e-18


def test_trinomial_full_requires_no_pruning() -> None:
    assert resolve_model_selection("Trinomial", False) == (
        "Trinomial · Full", 1e-18,
    )


# Trinomial pruning selections must retain exact method and threshold.
def test_trinomial_on_defaults_to_allocation() -> None:
    assert resolve_model_selection("Trinomial", True) == (
        "Trinomial · Allocation pruning", 1e-18,
    )


def test_trinomial_branch_preserves_threshold() -> None:
    assert resolve_model_selection("Trinomial", True, "Branch", 1e-12) == (
        "Trinomial · Branch pruning", 1e-12,
    )


def test_trinomial_allocation_preserves_threshold() -> None:
    assert resolve_model_selection("Trinomial", True, "Allocation", 1e-15) == (
        "Trinomial · Allocation pruning", 1e-15,
    )


# Reject ambiguous, unsupported, or non-finite pruning configurations.
@pytest.mark.parametrize("threshold", PRUNING_THRESHOLDS)
def test_all_supported_thresholds_map_identically(threshold: float) -> None:
    _, selected = resolve_model_selection("Trinomial", True, "Branch", threshold)
    assert selected == threshold


@pytest.mark.parametrize("family", ("Binomial", "Hybrid"))
def test_non_trinomial_cannot_silently_enable_pruning(family: object) -> None:
    with pytest.raises(PricingInputError, match="only for Trinomial"):
        resolve_model_selection(family, True)


@pytest.mark.parametrize("family", ("Unknown", "", "trinomial"))
def test_unknown_family_rejected(family: object) -> None:
    with pytest.raises(PricingInputError, match="Unknown model"):
        resolve_model_selection(family)


def test_unknown_method_rejected() -> None:
    # Reject invalid cases explicitly rather than silently substituting values.
    with pytest.raises(PricingInputError, match="implementation"):
        resolve_model_selection("Trinomial", True, "Guess")


@pytest.mark.parametrize("threshold", (0.0, -1e-18, 0.5, float("nan"), float("inf")))
def test_unsupported_or_nonfinite_pruning_threshold_rejected(threshold: float) -> None:
    with pytest.raises(PricingInputError):
        resolve_model_selection("Trinomial", True, threshold=threshold)


@pytest.mark.parametrize("pruning", (0, 1, "On", None))
def test_pruning_must_be_a_boolean(pruning: object) -> None:
    with pytest.raises(PricingInputError, match="boolean"):
        resolve_model_selection("Trinomial", pruning)


# Backwards-compatible engine coverage after translating UI selections.
@pytest.mark.parametrize(
    ("family", "enabled", "method"),
    [
        ("Binomial", False, "Allocation"),
        ("Hybrid", False, "Allocation"),
        ("Trinomial", False, "Allocation"),
        ("Trinomial", True, "Branch"),
        ("Trinomial", True, "Allocation"),
    ],
)
def test_every_existing_engine_remains_available(family: object, enabled: object, method: object) -> None:
    model, threshold = resolve_model_selection(family, enabled, method)
    assert model in MODELS
    request = PricingRequest(model=model, pruning_threshold=threshold)
    assert request.model == model


def test_full_is_never_mislabeled_as_pruned_when_off() -> None:
    for method in ("Allocation", "Branch"):
        # Verify that the pricing workflow keeps its input contract intact.
        model, threshold = resolve_model_selection(
            "Trinomial", False, method, 1e-12,
        )
        assert model == "Trinomial · Full" and threshold == 1e-18

# LOT_5C4_GUARDED_REFACTOR
