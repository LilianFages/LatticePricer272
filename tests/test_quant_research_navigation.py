"""Lot 4A: three pages, four live investigations, preserved legacy navigation."""
from pathlib import Path

import pytest

from frontend.model_diagnostics import WORKSPACES, DIAGNOSTIC_MODES, migrate_navigation_state
from frontend.quant_research import STUDIES, ARCHIVE_CHOICES


def test_three_unique_workspaces():
    assert WORKSPACES == (
        "01 / Pricing terminal", "02 / Model diagnostics", "03 / Quant research"
    )
    assert len(STUDIES) == len(set(STUDIES)) == 4
    assert DIAGNOSTIC_MODES == ("Compare models", "Explore lattice")


@pytest.mark.parametrize("old, expected", [
    ("03 / Quant analytics", "03 / Quant research"),
    ("04 / Research lab", "03 / Quant research"),
    ("04 / Quant analytics", "03 / Quant research"),
    ("05 / Research lab", "03 / Quant research"),
    ("02 / Model comparison", "02 / Model diagnostics"),
    ("03 / Lattice explorer", "02 / Model diagnostics"),
])
def test_legacy_workspace_migrates_without_calculation(old, expected):
    state = {"lattice_workspace": old, "qa_conv_result": object(),
             "qa_efficiency_result": object(), "qa_exb_result": object(),
             "qa_stab_result": object(), "lattice_explorer_result": object()}
    before = {key: state[key] for key in state if key != "lattice_workspace"}
    migrate_navigation_state(state)
    assert state["lattice_workspace"] == expected
    for key, value in before.items():
        assert state[key] is value
    previous = state.copy()
    migrate_navigation_state(state)
    assert state == previous


def test_new_workspace_is_stable():
    state = {"lattice_workspace": "03 / Quant research"}
    migrate_navigation_state(state)
    assert state == {"lattice_workspace": "03 / Quant research"}


def test_research_has_no_duplicate_analytics_tabs():
    assert STUDIES == ("Numerical accuracy", "Computational efficiency",
                       "Optimal stopping", "Risk numerics")
    assert "Greeks snapshot" not in STUDIES
    assert "Market sensitivities" not in STUDIES
    assert "Data library" not in STUDIES
    assert "Data library" in ARCHIVE_CHOICES


def test_research_page_routes_to_existing_studies():
    from inspect import getsource
    from frontend.quant_research import render_quant_research_page
    source = getsource(render_quant_research_page)
    for action in ("render_convergence_tab", "render_efficiency_tab",
                   "render_exercise_boundary_tab", "render_greeks_stability_tab"):
        assert action in source
    assert "render_sensitivity_tab()" in source
    assert "_render_quant_greeks_tab" not in source
    assert "calculate_greeks(" not in source


def test_archive_selection_only_uses_existing_legacy_renderers():
    import frontend.research_lab as archive
    for name in ARCHIVE_CHOICES:
        assert name in archive.RESEARCH_ARCHIVE_HANDLERS
        assert callable(archive.RESEARCH_ARCHIVE_HANDLERS[name])
    assert len(archive.RESEARCH_ARCHIVE_HANDLERS) == len(ARCHIVE_CHOICES)


def test_app_has_only_new_workspace_route():
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    code = path.read_text(encoding="utf-8")
    assert 'if workspace == "03 / Quant research"' in code
    assert 'if workspace == "03 / Quant analytics"' not in code
    assert 'if workspace == "04 / Research lab"' not in code
