"""Module 2.1: cross-model comparison with the existing REAL src/ engines."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import replace

import pytest

import frontend.comparison_service as service
from frontend.comparison_service import (
    compare_models,
    comparison_csv,
    display_rows,
)
from frontend.pricing_service import (
    DividendInput,
    PricingInputError,
    PricingRequest,
    price_contract,
)


# Cross-check each dashboard quote against the independent single-model entry point.
@pytest.mark.parametrize("option_type", ["Call", "Put"])
def test_all_models_price_same_european_contract(option_type: str) -> None:
    request = PricingRequest(
        steps=20, option_type=option_type, exercise="European"
    )
    models = (
        "Binomial", "Trinomial · Full", "Trinomial · Branch pruning",
        "Trinomial · Allocation pruning", "Hybrid",
    )
    report = compare_models(request, models, repetitions=1)
    assert len(report.successful) == len(models)
    assert report.reference_label == "Black–Scholes"
    assert report.reference_price is not None
    assert all(line.median_ms >= 0.0 for line in report.successful)
    for line in report.successful:
        direct = price_contract(replace(request, model=line.model))
        assert line.result.price == pytest.approx(direct.price, abs=1e-10)


# Equivalent grids must agree when no dividend disrupts recombination.
def test_trinomial_variants_have_equal_prices_without_dividends() -> None:
    models = (
        "Trinomial · Full", "Trinomial · Branch pruning",
        "Trinomial · Allocation pruning",
    )
    report = compare_models(PricingRequest(steps=30), models, 1)
    prices = [line.result.price for line in report.successful]
    assert max(prices) - min(prices) < 1e-9


# Unsupported contracts must be reported without suppressing usable alternatives.
def test_binomial_dividend_marked_unsupported_others_price() -> None:
    request = PricingRequest(
        steps=20, option_type="Put", exercise="American",
        dividends=(DividendInput(0.5, 3.0),),
    )
    models = ("Binomial", "Trinomial · Allocation pruning", "Hybrid")
    report = compare_models(request, models, repetitions=1)
    assert report.lines[0].status == "unsupported"
    assert len(report.successful) == 2
    assert report.reference_label == "Trinomial · Allocation pruning"
    assert report.successful[0].result.black_scholes is None


# One engine failing cannot erase another engine’s valid computed price.
def test_full_trinomial_dividend_error_does_not_hide_allocation() -> None:
    request = PricingRequest(
        steps=200, option_type="Call", exercise="European",
        dividends=(DividendInput(0.25, 2.0), DividendInput(0.75, 3.0)),
    )
    report = compare_models(
        request, ("Trinomial · Full", "Trinomial · Allocation pruning"), 1
    )
    assert report.lines[0].status == "error"
    assert report.lines[1].status == "ok"
    assert report.reference_label == "Trinomial · Allocation pruning"
    assert report.reference_price == pytest.approx(6.51469873429, abs=1e-7)


# Relative price gaps must use the exact benchmark selected by the user.
def test_user_selected_reference_changes_gap_sign() -> None:
    request = PricingRequest(steps=20)
    report = compare_models(
        request, ("Binomial", "Trinomial · Allocation pruning"),
        1, reference="Binomial",
    )
    rows = display_rows(report)
    assert report.reference_label == "Binomial"
    assert rows[0]["Δ vs reference (€)"] == pytest.approx(0.0)
    assert rows[1]["Δ vs reference (€)"] == pytest.approx(
        rows[1]["Price (€)"] - rows[0]["Price (€)"]
    )


# Reported timing summary must remain bounded by measured extrema.
def test_median_of_three_runs_is_recorded() -> None:
    report = compare_models(
        PricingRequest(steps=15), ("Binomial",), repetitions=3
    )
    line = report.lines[0]
    assert line.status == "ok"
    assert line.min_ms <= line.median_ms <= line.max_ms


# Reject invalid model combinations before running any engine.
@pytest.mark.parametrize("models", [(), ("Binomial", "Binomial"), ("Unknown",)])
def test_invalid_model_selection(models: tuple[str, ...]) -> None:
    with pytest.raises(PricingInputError):
        compare_models(PricingRequest(steps=10), models, 1)


# Timing repetitions require a strictly valid positive count.
@pytest.mark.parametrize("repetitions", [0, 2, 6, 1.5])
def test_invalid_repeat_count(repetitions: int) -> None:
    with pytest.raises(PricingInputError):
        compare_models(PricingRequest(steps=10), ("Binomial",), repetitions)


# Black–Scholes is not an admissible exercise-style benchmark here.
def test_black_scholes_not_allowed_for_american_options() -> None:
    with pytest.raises(PricingInputError, match="Black–Scholes"):
        compare_models(
            PricingRequest(steps=10, exercise="American"),
            ("Binomial",), reference="Black–Scholes",
        )


# Reference prices must come from a model actually selected.
def test_nonselected_reference_rejected() -> None:
    with pytest.raises(PricingInputError, match="Reference"):
        compare_models(
            PricingRequest(steps=10), ("Binomial",),
            reference="Trinomial · Full",
        )


# A zero price denominator makes the relative deviation undefined, not zero.
def test_zero_reference_has_no_relative_deviation(monkeypatch) -> None:
    real = service.price_contract

    def _fake_price(request: PricingRequest):
        response = real(request)
        return replace(response, price=0.0, black_scholes=0.0)

    monkeypatch.setattr(service, "price_contract", _fake_price)
    report = compare_models(PricingRequest(steps=10), ("Binomial",), 1)
    rows = display_rows(report)
    assert rows[0]["Δ vs reference (€)"] == 0.0
    assert rows[0]["Δ vs reference (%)"] is None


# A failed reference cannot be silently substituted by another engine.
def test_failed_reference_not_silently_replaced(monkeypatch) -> None:
    real = service.price_contract

    def _fake_price(request: PricingRequest):
        if request.model == "Binomial":
            raise RuntimeError("Deliberate failure")
        return real(request)

    monkeypatch.setattr(service, "price_contract", _fake_price)
    report = compare_models(
        PricingRequest(steps=10),
        ("Binomial", "Trinomial · Allocation pruning"), 1,
        reference="Binomial",
    )
    assert len(report.successful) == 1
    assert report.reference_price is None
    assert display_rows(report)[1]["Δ vs reference (€)"] is None


# Exported results must preserve statuses and contract metadata.
def test_csv_contains_header_metadata_and_failure_status() -> None:
    request = PricingRequest(
        steps=20, dividends=(DividendInput(0.5, 3.0),),
    )
    report = compare_models(
        request, ("Binomial", "Trinomial · Allocation pruning"), 1
    )
    raw = comparison_csv(report)
    assert "Steps,Pruning threshold" in raw
    assert "unsupported" in raw
    records = list(csv.DictReader(io.StringIO(raw), skipinitialspace=True))
    assert all(row["Steps"] == "20" for row in records)
    assert math.isfinite(report.successful[0].result.price)
    assert len([row for row in records if row.get("Status") == "ok"]) == 1
