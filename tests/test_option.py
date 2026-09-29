import pytest

from src.option import CallOption, PutOption


def test_call_payoff_in_the_money() -> None:
    option = CallOption(strike=100.0)

    assert option.payoff(120.0) == 20.0


def test_call_payoff_out_of_the_money() -> None:
    option = CallOption(strike=100.0)

    assert option.payoff(80.0) == 0.0


def test_put_payoff_in_the_money() -> None:
    option = PutOption(strike=100.0)

    assert option.payoff(80.0) == 20.0


def test_put_payoff_out_of_the_money() -> None:
    option = PutOption(strike=100.0)

    assert option.payoff(120.0) == 0.0


def test_invalid_strike_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Strike must be strictly positive",
    ):
        CallOption(strike=0.0)


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