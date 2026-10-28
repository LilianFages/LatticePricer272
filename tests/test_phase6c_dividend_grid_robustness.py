"""Phase 6C: scientific conventions, failure integrity and reproducible exports."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments import dividend_grid_robustness as study
from src.dividend_timing import dividend_grid_step


def _fake_price(request):
    """Deterministic pricing spy: no dependence on expensive lattice builds."""
    date = request.dividends[0].time
    forward_term = study.parity_target(date)
    put = 6.0 + 2.0 / request.steps + (date - .5) * 8.0
    price = put
    if request.option_type == "Call":
        price = put + forward_term
    elif request.exercise == "American":
        price = put + 0.4
    return SimpleNamespace(price=price)


def test_two_independent_axes_do_not_confuse_shift_with_grid() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(-1, 0, 1),
                                date_steps=40, pricer=_fake_price,
                                models=(study.MODELS[0],))
    assert len(rows) == 5
    fixed = [r for r in rows if r["panel"] == "grid"]
    shifted = [r for r in rows if r["panel"] == "date"]
    assert {r["dividend_time"] for r in fixed} == {0.5}
    assert {r["steps"] for r in shifted} == {40}
    assert sorted(r["date_shift_days"] for r in shifted) == [-1, 0, 1]
    assert all(r["status"] == "success" for r in rows)


def test_parity_premium_and_finest_sample_gap_are_defined() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(0,), date_steps=40,
                                pricer=_fake_price, models=(study.MODELS[0],))
    grid = {r["steps"]: r for r in rows if r["panel"] == "grid"}
    assert grid[25]["gap_to_finest_sample"] == pytest.approx(2/25 - 2/40)
    assert grid[40]["gap_to_finest_sample"] == pytest.approx(0.0)
    assert all(r["finest_reference_steps"] == 40 for r in grid.values())
    assert all(r["early_exercise_premium"] == pytest.approx(0.4) for r in rows)
    assert all(r["parity_residual"] == pytest.approx(0.0, abs=1e-12) for r in rows)
    assert rows[-1]["date_shift_put"] == pytest.approx(0.0)


def test_per_model_reference_does_not_mix_lattices() -> None:
    rows = study.run_experiment(steps=(25, 40), shifts=(0,), date_steps=40,
                                pricer=_fake_price)
    assert len(rows) == 6
    assert {r["model"] for r in rows} == set(study.MODELS)
    assert all(r["finest_reference_steps"] == 40
               for r in rows if r["panel"] == "grid")


def test_failed_triplet_is_explicit_and_never_imputed() -> None:
    def failing(request):
        if request.exercise == "American":
            raise RuntimeError("Injected pricer outage")
        return _fake_price(request)
    rows = study.run_experiment(steps=(25,), shifts=(0,), date_steps=25,
                                pricer=failing, models=(study.MODELS[0],))
    assert len(rows) == 2
    assert all(r["status"] == "failed" for r in rows)
    assert all(r["european_put"] is None for r in rows)
    assert all("Injected pricer outage" in r["failure"] for r in rows)
    assert all(r["gap_to_finest_sample"] is None for r in rows)


def test_finite_prices_required() -> None:
    def invalid(request):
        return SimpleNamespace(price=float("nan"))
    row = study.measure_case(study.MODELS[0], 25, 0.5, pricer=invalid)
    assert row["status"] == "failed"
    assert "Non-finite" in row["failure"]


@pytest.mark.parametrize("steps,days,date_steps", [
    ((), (0,), 100), ((25, 25), (0,), 100),
    ((0,), (0,), 100), ((25,), (1,), 100),
    ((25,), (0, 0), 100), ((25,), (0,), 0),
])
def test_invalid_designs_fail_loudly(steps, days, date_steps) -> None:
    with pytest.raises(ValueError):
        study.validate_design(steps, days, date_steps)


def test_exdate_boundary_convention_and_dividend_pv() -> None:
    assert dividend_grid_step(.5, 1.0, 4) == 2
    assert dividend_grid_step(.5 + 1e-8, 1.0, 4) == 3
    assert study.parity_target(.51) != study.parity_target(.5)
    assert study.parity_target(.5) == pytest.approx(
        100 - 3 * __import__("math").exp(-.02 * .5)
        - 100 * __import__("math").exp(-.02)
    )


def test_grid_and_date_panels_share_one_valuation_when_identical() -> None:
    calls = []
    def recording(request):
        calls.append(request)
        return _fake_price(request)
    study.run_experiment(steps=(40,), shifts=(0,), date_steps=40,
                         pricer=recording, models=(study.MODELS[0],))
    assert len(calls) == 3  # Not 6: identical contract triplet memoized.


def test_summary_names_reference_and_contract_change_explicitly() -> None:
    rows = study.run_experiment(steps=(25,), shifts=(0,), date_steps=25,
                                pricer=_fake_price, models=(study.MODELS[0],))
    summary = study.summarize(rows)
    assert "NOT exact error" in summary
    assert "DIFFERENT dividend contract" in summary
    assert "valid: 2; failed: 0" in summary


def test_quick_and_full_outputs_are_disjoint() -> None:
    assert set(study.output_paths(True)).isdisjoint(study.output_paths(False))
    assert "phase6c_quick" in str(study.output_paths(True)[0])


def test_csv_fieldnames_are_unique() -> None:
    assert len(study.FIELDS) == len(set(study.FIELDS))
