"""Verify seven Greek calculations live in src/risk, without UI dependencies."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontend.greeks_service import (
    GreekBumps,
    calculate_greeks as frontend_greeks,
)
from frontend.pricing_service import (
    PricingInputError,
    PricingRequest,
    validate_request,
)
from frontend.trading_risk import (
    compute_desk_risk as frontend_desk,
    greek_table,
)
from src.pricing_errors import PricingInputError as CoreInputError
from src.risk.desk import compute_desk_risk as core_desk
from src.risk.greeks import (
    GreekBumps as CoreBumps,
    calculate_greeks as core_greeks,
)


CONTRACT = PricingRequest()


def polynomial_pricer(request: PricingRequest) -> SimpleNamespace:
    s, v, r, t = (
        request.spot, request.volatility, request.rate, request.maturity,
    )
    value = 4.0 + 0.5*s + 3.0*s*v + 7.0*v*v + 0.8*s*v*v + r + t
    return SimpleNamespace(price=value, elapsed_ms=0.0)


def test_pricing_input_exception_keeps_public_identity() -> None:
    assert PricingInputError is CoreInputError


def test_bump_dataclass_is_shared() -> None:
    assert GreekBumps is CoreBumps
    with pytest.raises(PricingInputError):
        GreekBumps(spot_pct=0.0).validate()


@pytest.mark.parametrize("vol", [0.0, 0.001, 0.20])
def test_five_greeks_match_old_adapter(vol: float) -> None:
    contract = replace(CONTRACT, volatility=vol)
    front = frontend_greeks(contract, pricer=polynomial_pricer)
    kernel = core_greeks(
        contract,
        pricer=polynomial_pricer,
        validate_request=validate_request,
    )
    assert front.measured() == pytest.approx(kernel.measured(), nan_ok=True)
    assert front.volatility_scheme == kernel.volatility_scheme
    assert front.scenario_prices == kernel.scenario_prices


@pytest.mark.parametrize("vol", [0.0, 0.20])
def test_seven_greeks_match_core(vol: float) -> None:
    contract = replace(CONTRACT, volatility=vol)
    front = frontend_desk(contract, pricer=polynomial_pricer)
    core = core_desk(
        contract,
        pricer=polynomial_pricer,
        validate_request=validate_request,
        calculate_greeks=frontend_greeks,
    )
    assert [front.value(name) for name in ("Delta", "Gamma", "Vega", "Vanna", "Vomma", "Theta", "Rho")] == pytest.approx(
        [core.value(name) for name in ("Delta", "Gamma", "Vega", "Vanna", "Vomma", "Theta", "Rho")]
    )
    assert len(greek_table(front)) == 7


def test_risk_kernel_is_independent_of_frontend_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "risk"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("frontend")
            elif isinstance(node, ast.Import):
                assert all(not item.name.startswith("frontend") for item in node.names)
