import pytest

from src.option import CallOption, PutOption


# Calls pay positive intrinsic value only above the strike.
@pytest.mark.parametrize(
    ("spot", "expected_payoff"),
    [
        (120.0, 20.0),
        (80.0, 0.0),
    ],
)
def test_call_payoff(
    spot: float,
    expected_payoff: float
) -> None:
    option = CallOption(strike=100.0)
    assert option.payoff(spot) == expected_payoff


# Puts pay positive intrinsic value only below the strike.
@pytest.mark.parametrize(
    ("spot", "expected_payoff"),
    [
        (80.0, 20.0),
        (120.0, 0.0),
    ],
)
def test_put_payoff(
    spot: float,
    expected_payoff: float
) -> None:
    option = PutOption(strike=100.0)
    assert option.payoff(spot) == expected_payoff


# Strike validation includes zero but rejects negative values.
def test_negative_strike_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Strike cannot be negative",
    ):
        CallOption(strike=-1.0)


def test_zero_strike_is_allowed() -> None:
    option = CallOption(strike=0.0)
    assert option.strike == 0.0


# European exercise always keeps the continuation value.
def test_european_option_keeps_hold_value() -> None:
    option = PutOption(
        strike=100.0,
        is_american=False,
    )
    value = option.value_at_node(
        spot=80.0,
        hold_value=15.0,
    )
    assert value == 15.0


# American exercise selects the best value between exercise and hold.
def test_american_option_can_exercise() -> None:
    option = PutOption(
        strike=100.0,
        is_american=True,
    )
    value = option.value_at_node(
        spot=80.0,
        hold_value=15.0,
    )
    assert value == 20.0


def test_american_option_can_hold() -> None:
    option = CallOption(
        strike=100.0,
        is_american=True,
    )
    value = option.value_at_node(
        spot=120.0,
        hold_value=25.0,
    )
    assert value == 25.0