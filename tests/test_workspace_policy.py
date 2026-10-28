"""Lot 2C.2 — shared editor policy and exact contract-transfer regressions."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from frontend.contract_workflow import assess_transfer
from frontend.diagnostics_contract import THRESHOLDS, plan_handoff
from frontend.exercise_boundary_service import MAX_BOUNDARY_STEPS as EXERCISE_CAP
from frontend.lattice_explorer_service import MAX_VISUAL_STEPS as EXPLORER_CAP
from frontend.pricing_service import DividendInput, PricingRequest
from frontend.workspace_policy import (
    ANALYTICS_THRESHOLDS,
    COMPARISON_THRESHOLDS,
    EXPLORER_THRESHOLDS,
    MAX_BOUNDARY_STEPS,
    MAX_VISUAL_STEPS,
)


REPO = Path(__file__).resolve().parents[1]


def test_policy_caps_are_reexported_for_existing_callers():
    assert MAX_VISUAL_STEPS == EXPLORER_CAP == 100
    assert MAX_BOUNDARY_STEPS == EXERCISE_CAP == 200


def test_ui_model_diagnostics_thresholds_share_policy():
    assert THRESHOLDS["Compare models"] is COMPARISON_THRESHOLDS
    assert THRESHOLDS["Explore lattice"] is EXPLORER_THRESHOLDS


@pytest.mark.parametrize("target,accepted,rejected", [
    ("model_comparison", COMPARISON_THRESHOLDS[2], 1e-5),
    ("lattice_explorer", EXPLORER_THRESHOLDS[2], 1e-12),
    ("quant_analytics", ANALYTICS_THRESHOLDS[-1], 1e-9),
    ("exercise_boundary", ANALYTICS_THRESHOLDS[-1], 1e-9),
])
def test_allowed_threshold_sets_remain_exact(target, accepted, rejected):
    base = PricingRequest(steps=5)
    assert assess_transfer(replace(base, pruning_threshold=accepted), target).exact
    bad = assess_transfer(replace(base, pruning_threshold=rejected), target)
    assert any("threshold" in issue.lower() for issue in bad.issues)


def test_visual_depth_limit_remains_exact_and_non_clamping():
    accepted = PricingRequest(steps=MAX_VISUAL_STEPS)
    assert assess_transfer(accepted, "lattice_explorer").exact
    over = replace(accepted, steps=MAX_VISUAL_STEPS + 1)
    adjusted, issues = plan_handoff(over, "Explore lattice")
    assert adjusted.steps == MAX_VISUAL_STEPS + 1
    assert any("N=" in issue for issue in issues)
    assert not assess_transfer(over, "lattice_explorer").exact


def test_exercise_limit_remains_exact_and_non_clamping():
    assert assess_transfer(PricingRequest(steps=MAX_BOUNDARY_STEPS),
                           "exercise_boundary").exact
    over = PricingRequest(steps=MAX_BOUNDARY_STEPS + 1)
    assert not assess_transfer(over, "exercise_boundary").exact


def test_multiple_dividends_still_not_silently_dropped():
    request = PricingRequest(steps=5, dividends=(
        DividendInput(0.25, 2.), DividendInput(0.75, 3.),
    ))
    assert not assess_transfer(request, "lattice_explorer").exact
    assert assess_transfer(request, "model_comparison").exact


def test_workflow_import_has_no_explorer_or_boundary_service_dependency():
    source = (REPO / "frontend" / "contract_workflow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert "frontend.lattice_explorer_service" not in imported
    assert "frontend.exercise_boundary_service" not in imported
    assert "frontend.workspace_policy" in imported


def test_shared_policy_is_pure_and_imports_no_engine_or_streamlit():
    source = (REPO / "frontend" / "workspace_policy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in tree.body if isinstance(node, ast.Import)
                for alias in node.names]
    assert not any(name and (name.startswith("src") or name.startswith("streamlit")
                             or name.startswith("frontend")) for name in imports)
