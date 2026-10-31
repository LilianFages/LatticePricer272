"""Pure desk-model configuration: UI families to existing pricing engines.

Keep the model family, pruning toggle and pruning implementation distinct.
No Streamlit dependency and no numerical pricing side effects.
"""

from __future__ import annotations

import math

from frontend.pricing_service import (
    DEFAULT_THRESHOLD, MODELS, ModelName, PricingInputError,
)

FAMILIES = ("Binomial", "Hybrid", "Trinomial")
PRUNING_METHODS = ("Allocation", "Branch")
PRUNING_THRESHOLDS = (1e-18, 1e-15, 1e-12)


def resolve_model_selection(
    family: str,
    pruning: bool = False,
    method: str = "Allocation",
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[ModelName, float]:
    """Resolve explicit desk controls into the legacy five-model contract.

    Full and pruned Trinomial remain distinct on the engine side. A pruning
    toggle is not silently ignored for a model that does not support it.
    """
    if family not in FAMILIES:
        raise PricingInputError("Unknown model family.")
    if type(pruning) is not bool:
        raise PricingInputError("Pruning selection must be a boolean.")
    if method not in PRUNING_METHODS:
        raise PricingInputError("Unknown trinomial pruning implementation.")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise PricingInputError("Invalid pruning threshold.")
    if not math.isfinite(threshold):
        raise PricingInputError("Pruning threshold must be finite.")
    if family != "Trinomial":
        if pruning:
            raise PricingInputError("Pruning is supported only for Trinomial.")
        if family == "Binomial":
            return "Binomial", DEFAULT_THRESHOLD
        return "Hybrid", DEFAULT_THRESHOLD
    if not pruning:
        # The Full engine forces pruning_threshold=0 internally. Keep the
        # request-level default unchanged for exact cross-workspace transfer.
        return "Trinomial · Full", DEFAULT_THRESHOLD
    return _pruned_trinomial_selection(method, threshold)


def _pruned_trinomial_selection(method: str, threshold: float) -> tuple[ModelName, float]:
    """Keep branch and allocation pruning as distinct engine configurations."""
    if threshold not in PRUNING_THRESHOLDS:
        raise PricingInputError("Choose a supported positive pruning threshold.")
    model = (
        "Trinomial · Allocation pruning"
        if method == "Allocation" else "Trinomial · Branch pruning"
    )
    if model not in MODELS:
        raise RuntimeError("Unrecognized trinomial configuration.")
    return model, float(threshold)
