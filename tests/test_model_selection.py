"""Phase 1B-D2.2: exact mappings, bounds and no engine side effects."""

import pytest

from frontend.model_selection import (
    FAMILIES, PRUNING_THRESHOLDS, resolve_model_selection,
)
from frontend.pricing_service import MODELS, PricingInputError, PricingRequest


# Model routing: pricing families and non-pruned defaults must remain distinct.
def test_three_distinct_families():
    assert FAMILIES == ("Binomial", "Hybrid", "Trinomial")


def test_binomial_without_pruning_maps_exactly():
    model, threshold = resolve_model_selection("Binomial")
    assert model == "Binomial" and threshold == 1e-18


def test_hybrid_without_pruning_maps_exactly():
    model, threshold = resolve_model_selection("Hybrid")
    assert model == "Hybrid" and threshold == 1e-18


def test_trinomial_full_requires_no_pruning():
    assert resolve_model_selection("Trinomial", False) == (
        "Trinomial · Full", 1e-18,
    )


# Trinomial pruning selections must retain exact method and threshold.
def test_trinomial_on_defaults_to_allocation():
    assert resolve_model_selection("Trinomial", True) == (
        "Trinomial · Allocation pruning", 1e-18,
    )


def test_trinomial_branch_preserves_threshold():
    assert resolve_model_selection("Trinomial", True, "Branch", 1e-12) == (
        "Trinomial · Branch pruning", 1e-12,
    )


def test_trinomial_allocation_preserves_threshold():
    assert resolve_model_selection("Trinomial", True, "Allocation", 1e-15) == (
        "Trinomial · Allocation pruning", 1e-15,
    )


# Reject ambiguous, unsupported, or non-finite pruning configurations.
@pytest.mark.parametrize("threshold", PRUNING_THRESHOLDS)
def test_all_supported_thresholds_map_identically(threshold):
    _, selected = resolve_model_selection("Trinomial", True, "Branch", threshold)
    assert selected == threshold


@pytest.mark.parametrize("family", ("Binomial", "Hybrid"))
def test_non_trinomial_cannot_silently_enable_pruning(family):
    with pytest.raises(PricingInputError, match="only for Trinomial"):
        resolve_model_selection(family, True)


@pytest.mark.parametrize("family", ("Unknown", "", "trinomial"))
def test_unknown_family_rejected(family):
    with pytest.raises(PricingInputError, match="Unknown model"):
        resolve_model_selection(family)


def test_unknown_method_rejected():
    with pytest.raises(PricingInputError, match="implementation"):
        resolve_model_selection("Trinomial", True, "Guess")


@pytest.mark.parametrize("threshold", (0.0, -1e-18, 0.5, float("nan"), float("inf")))
def test_unsupported_or_nonfinite_pruning_threshold_rejected(threshold):
    with pytest.raises(PricingInputError):
        resolve_model_selection("Trinomial", True, threshold=threshold)


@pytest.mark.parametrize("pruning", (0, 1, "On", None))
def test_pruning_must_be_a_boolean(pruning):
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
def test_every_existing_engine_remains_available(family, enabled, method):
    model, threshold = resolve_model_selection(family, enabled, method)
    assert model in MODELS
    request = PricingRequest(model=model, pruning_threshold=threshold)
    assert request.model == model


def test_full_is_never_mislabeled_as_pruned_when_off():
    for method in ("Allocation", "Branch"):
        model, threshold = resolve_model_selection(
            "Trinomial", False, method, 1e-12,
        )
        assert model == "Trinomial · Full" and threshold == 1e-18

# LOT_5C4_GUARDED_REFACTOR
