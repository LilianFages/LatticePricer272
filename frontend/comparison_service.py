"""Cross-model valuation comparison; delegates all financial work to src/.

Each model is independently priced through the existing pricing adapter.
A failed or unsupported model does not conceal successful valuations.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from dataclasses import dataclass, replace
from typing import Literal

from frontend.pricing_service import (
    MODELS,
    PricingInputError,
    PricingRequest,
    PricingResult,
    price_contract,
    validate_request,
)

# Run competing lattice models with matching market assumptions.
ReferenceChoice = Literal["Auto", "Black–Scholes"] | str
RunStatus = Literal["ok", "unsupported", "error"]


@dataclass(frozen=True, slots=True)
class ComparisonLine:
    model: str
    status: RunStatus
    result: PricingResult | None = None
    median_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    # Keep reference failures distinguishable from numerical price gaps.
    request: PricingRequest
    lines: tuple[ComparisonLine, ...]
    repetitions: int
    reference_label: str | None
    reference_price: float | None

    @property
    def successful(self) -> tuple[ComparisonLine, ...]:
        return tuple(line for line in self.lines if line.status == "ok")

    @property
    def unsuccessful(self) -> tuple[ComparisonLine, ...]:
        return tuple(line for line in self.lines if line.status != "ok")



def _collect_timing_samples(
    template: PricingRequest,
    models: tuple[str, ...],
    repetitions: int,
    samples: dict,
    failures: dict,
) -> None:
    """Price each selected model in rotated independent benchmark runs."""
    for repetition in range(repetitions):
        ordered_models = models[repetition % len(models):] + models[:repetition % len(models)]
        for model in ordered_models:
            if model in failures:
                continue
            request = replace(template, model=model)
            try:
                result = price_contract(request)
                if samples[model] and not math.isclose(
                    result.price, samples[model][0].price,
                    rel_tol=1e-10, abs_tol=1e-10,
                ):
                    raise RuntimeError("Repeated pricing yielded inconsistent prices.")
                samples[model].append(result)
            except (
                PricingInputError, ValueError, RuntimeError,
                # Avoid changing engine settings while preparing comparative data.
                OverflowError, ZeroDivisionError, RecursionError,
            ) as exc:
                failures[model] = ("error", f"{type(exc).__name__}: {exc}")
                samples[model].clear()



def _collect_comparison_lines(models: tuple[str, ...], failures: dict, samples: dict, repetitions: int) -> object:
    """Aggregate medians and actual node observations."""
    lines: list[ComparisonLine] = []
    for model in models:
        if model in failures:
            status, reason = failures[model]
            lines.append(ComparisonLine(model=model, status=status, reason=reason))
            continue
        observations = samples[model]
        if len(observations) != repetitions:
            lines.append(ComparisonLine(
                model=model, status="error", reason="Incomplete timing series."
            ))
            continue
        # Preserve the result schema consumed by the UI.
        timings = [sample.elapsed_ms for sample in observations]
        lines.append(ComparisonLine(
            model=model, status="ok", result=observations[-1],
            median_ms=statistics.median(timings),
            min_ms=min(timings), max_ms=max(timings),
        ))
    return lines



def _comparison_reference(lines: list, template: PricingRequest, reference: ReferenceChoice) -> tuple:
    """Resolve explicitly requested analytical or numerical reference."""
    successes = [line for line in lines if line.status == "ok"]
    reference_label: str | None = None
    reference_price: float | None = None
    analytic_eligible = template.exercise == "European" and not template.dividends

    if reference in ("Auto", "Black–Scholes") and analytic_eligible and successes:
        bs = successes[0].result.black_scholes
        if bs is not None:
            reference_label = "Black–Scholes"
            reference_price = bs
    # Run competing lattice models with matching market assumptions.
    elif reference == "Auto" and successes:
        reference_label = successes[0].model
        reference_price = successes[0].result.price
    elif reference not in ("Auto", "Black–Scholes"):
        selected_line = next(line for line in lines if line.model == reference)
        if selected_line.status == "ok":
            reference_label = selected_line.model
            reference_price = selected_line.result.price
    return reference_label, reference_price


def compare_models(
    template: PricingRequest,
    models: tuple[str, ...],
    repetitions: int = 3,
    reference: ReferenceChoice = "Auto",
) -> ComparisonReport:
    """Compute comparable contract prices with independent engine runs.

    The median is taken over repeated *end-to-end* timings from price_contract,
    which times the constructor/build/pricer. Node traversal and charts are
    outside that timing. No models are benchmarked by replaying cached prices.
    """
    if not models or len(models) != len(set(models)):
        raise PricingInputError("Select one or more distinct pricing models.")
    if any(model not in MODELS for model in models):
        raise PricingInputError("An unknown comparison model was selected.")
    if type(repetitions) is not int or repetitions not in (1, 3, 5):
        raise PricingInputError("Choose 1, 3 or 5 timing repetitions.")
    if reference not in ("Auto", "Black–Scholes", *models):
        raise PricingInputError("Reference must be Auto, Black–Scholes or a selected model.")
    if reference == "Black–Scholes" and (
        template.exercise != "European" or template.dividends
    ):
        raise PricingInputError(
            "Black–Scholes reference is only supported for European options "
            "without cash dividends."
        )

    # Validate once using a dividend-compatible model, even when the selected
    # set includes the unsupported binomial model.
    validate_request(replace(template, model="Trinomial · Allocation pruning"))

    samples: dict[str, list[PricingResult]] = {model: [] for model in models}
    failures: dict[str, tuple[RunStatus, str]] = {}
    if template.dividends and "Binomial" in models:
        failures["Binomial"] = (
            "unsupported", "BinomialTree does not support discrete cash dividends."
        )

    # Rotate model order by repetition to reduce systematic order effects.
    _collect_timing_samples(template, models, repetitions, samples, failures)

    lines = _collect_comparison_lines(models, failures, samples, repetitions)

    reference_label, reference_price = _comparison_reference(lines, template, reference)

    return ComparisonReport(
        request=template, lines=tuple(lines), repetitions=repetitions,
        reference_label=reference_label, reference_price=reference_price,
    )


def display_rows(report: ComparisonReport) -> list[dict[str, object]]:
    """Table values; unavailable quantities stay None, never become zero."""
    rows: list[dict[str, object]] = []
    for line in report.lines:
        result = line.result
        diff = (
            result.price - report.reference_price
            if result is not None and report.reference_price is not None else None
        )
        relative = (
            100 * diff / report.reference_price
            if diff is not None and report.reference_price != 0.0 else None
        )
        rows.append({
            "Model": line.model,
            "Status": line.status,
            "Price (€)": result.price if result is not None else None,
            "Δ vs reference (€)": diff,
            "Δ vs reference (%)": relative,
            # Keep reference failures distinguishable from numerical price gaps.
            "Median time (ms)": line.median_ms,
            "Time min (ms)": line.min_ms,
            "Time max (ms)": line.max_ms,
            "Reachable nodes": result.reachable_nodes if result is not None else None,
            "Allocated nodes*": result.allocated_nodes if result is not None else None,
            "Pruned transitions": (
                result.pruned_transitions if result is not None else None
            ),
            "Avoided allocations (%)*": (
                result.allocation_savings_pct if result is not None else None
            ),
            "Message": line.reason,
        })
    return rows


def comparison_csv(report: ComparisonReport) -> str:
    """Spreadsheet-friendly CSV: repeat contract metadata per model line."""
    output = io.StringIO()
    request = report.request
    metadata = {
        "Contract option": request.option_type,
        "Contract exercise": request.exercise,
        "Spot": request.spot,
        "Strike": request.strike,
        "Risk-free rate": request.rate,
        "Volatility": request.volatility,
        "Maturity": request.maturity,
        "Steps": request.steps,
        "Pruning threshold": request.pruning_threshold,
        "Dividends": ";".join(
            f"{dividend.time}:{dividend.amount}" for dividend in request.dividends
        ),
        "Timing repetitions": report.repetitions,
        "Reference model": report.reference_label or "unavailable",
        "Reference price": report.reference_price,
    }
    rows = [{**metadata, **row} for row in display_rows(report)]
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
