"""Lot 2A: navigation migration preserves workspaces and selected subviews."""

import pytest

from frontend.model_diagnostics import (
    DIAGNOSTIC_MODES, WORKSPACES, migrate_navigation_state,
)


@pytest.mark.parametrize("legacy,workspace,mode", [
    ("02 / Model comparison", "02 / Model diagnostics", "Compare models"),
    ("03 / Lattice explorer", "02 / Model diagnostics", "Explore lattice"),
    ("04 / Quant analytics", "03 / Quant analytics", None),
    ("05 / Research lab", "04 / Research lab", None),
])
def test_legacy_workspace_migration(legacy, workspace, mode):
    state = {"lattice_workspace": legacy}
    migrate_navigation_state(state)
    assert state["lattice_workspace"] == workspace
    if mode is None:
        assert "model_diagnostics_view" not in state
    else:
        assert state["model_diagnostics_view"] == mode


def test_migration_idempotent():
    state = {"lattice_workspace": "03 / Lattice explorer"}
    migrate_navigation_state(state)
    before = state.copy()
    migrate_navigation_state(state)
    assert state == before


def test_current_workspace_is_not_overwritten():
    state = {"lattice_workspace": "02 / Model diagnostics",
             "model_diagnostics_view": "Explore lattice"}
    migrate_navigation_state(state)
    assert state["model_diagnostics_view"] == "Explore lattice"


def test_navigation_has_unique_workspaces_and_modes():
    assert len(WORKSPACES) == 4
    assert len(set(WORKSPACES)) == len(WORKSPACES)
    assert DIAGNOSTIC_MODES == ("Compare models", "Explore lattice")
