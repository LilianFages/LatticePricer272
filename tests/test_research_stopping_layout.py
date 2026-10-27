"""UI-only checks for the reorganized Optimal Stopping study."""
from __future__ import annotations

from types import SimpleNamespace
import inspect

import pytest

from frontend import exercise_boundary_views as views


def _study(with_dividends: bool):
    dates = (
        SimpleNamespace(step=0, time=0.0, boundary_spot=None,
                        first_exercise_probability=0.0),
        SimpleNamespace(step=1, time=0.5, boundary_spot=85.0,
                        first_exercise_probability=0.25, non_contiguous=False),
        SimpleNamespace(step=2, time=1.0, boundary_spot=90.0,
                        first_exercise_probability=0.5),
    )
    dividends = (SimpleNamespace(time=.5, amount=3.),) if with_dividends else ()
    return SimpleNamespace(dates=dates,
                           request=SimpleNamespace(dividends=dividends, exercise="American"),
                           premium=8.0 if with_dividends else 7.0)


def _pair():
    return SimpleNamespace(with_dividends=_study(True),
                           without_dividends=_study(False),
                           premium_difference=1.0)


@pytest.mark.parametrize("kind", ("boundary", "node_first_exercise"))
def test_dividend_chart_has_bottom_legend(kind):
    spec = views._dividend_comparison_chart(_pair(), kind).to_dict()
    main = spec["layer"][0]
    assert main["encoding"]["color"]["legend"]["orient"] == "bottom"
    assert main["encoding"]["color"]["legend"]["title"] is None
    assert spec["height"] >= 285
    assert spec["layer"][1]["mark"]["type"] == "rule"


def test_comparison_boundary_keeps_actual_defined_nodes():
    frame = views._dividend_comparison_frame(_pair())
    assert len(frame) == 2
    assert set(frame["scenario"]) == {"Cash dividends", "No cash dividends"}
    assert frame["time"].eq(.5).all()


def test_original_probabilities_chart_has_bottom_legend():
    primary = _study(True)
    dates = tuple(SimpleNamespace(time=d.time, region_probability=.1,
                                  first_exercise_probability=d.first_exercise_probability)
                  for d in primary.dates)
    spec = views._probabilities(SimpleNamespace(dates=dates)).to_dict()
    assert spec["encoding"]["color"]["legend"]["orient"] == "bottom"


def test_sections_keep_explicit_comparison_and_nodal_grid():
    entrypoint = inspect.getsource(views.render_exercise_boundary_tab)
    results = inspect.getsource(views._show_exercise_results)
    assert "_show_exercise_results()" in entrypoint
    assert "_render_dividend_comparison(pair)" in results
    assert "st.columns([1.10, 1.0]" in results
    assert "_render_exercise_secondary_details(study)" in results


def test_no_repricing_or_computation_added_to_layout_renderers():
    names = ("_render_exercise_summary", "_render_dividend_comparison",
             "_render_exercise_lattice_map", "_render_exercise_stopping_details",
             "_render_exercise_secondary_details")
    text = "\n".join(inspect.getsource(getattr(views, name)) for name in names)
    assert "run_exercise_study(" not in text
    assert "compare_dividend_exercise(" not in text
    assert "price_contract(" not in text
