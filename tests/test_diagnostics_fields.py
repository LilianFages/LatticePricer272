"""Lot 2B.2 — shared market editor and controlled lattice depth."""
from dataclasses import replace

import pytest

from frontend import diagnostics_fields as fields
from frontend.contract_workflow import assess_transfer
from frontend.diagnostics_contract import plan_handoff
from frontend.lattice_explorer_service import MAX_VISUAL_STEPS
from frontend.pricing_service import PricingRequest


class FakeStreamlit:
    def __init__(self):
        self.inputs = []
        self.headings = []

    def columns(self, n):
        return [self] * n

    def markdown(self, text, **kwargs):
        self.headings.append(text)

    def selectbox(self, label, options, **kwargs):
        self.inputs.append(("select", label, kwargs["key"]))
        return options[kwargs["index"]]

    def number_input(self, label, **kwargs):
        self.inputs.append(("number", label, kwargs["key"]))
        return kwargs["value"]


def _defaults(field, fallback):
    return getattr(PricingRequest(spot=112., strike=103., volatility=.24,
                                  rate=-.01, maturity=1.5,
                                  option_type="Put", exercise="American"),
                   field, fallback)


def test_common_editor_keeps_both_existing_widget_names(monkeypatch):
    ui = FakeStreamlit()
    monkeypatch.setattr(fields, "st", ui)
    key = lambda name: "fresh_" + name
    pair = fields.render_option_pair(key, _defaults,
                                    widget_names=("cmp_kind", "cmp_exercise"))
    market = fields.render_market_fields(
        key, _defaults,
        widget_names=("cmp_spot", "cmp_strike", "cmp_vol", "cmp_rate", "cmp_maturity"),
        heading="02 · Market data", spot_max=1e7,
        strike_max=1e7, maturity_max=100.,
    )
    assert pair == ("Put", "American")
    assert market == pytest.approx((112., 103., 24., -1., 1.5))
    assert [entry[2] for entry in ui.inputs] == [
        "fresh_cmp_kind", "fresh_cmp_exercise", "fresh_cmp_spot",
        "fresh_cmp_strike", "fresh_cmp_vol", "fresh_cmp_rate",
        "fresh_cmp_maturity",
    ]
    assert len(ui.headings) == 1


def test_explorer_keeps_its_key_family_and_default_option(monkeypatch):
    ui = FakeStreamlit()
    monkeypatch.setattr(fields, "st", ui)
    fallback = lambda field, value: value
    pair = fields.render_option_pair(
        lambda name: "lx_" + name, fallback,
        widget_names=("kind", "exercise"),
        default_option="Put", default_exercise="American",
    )
    market = fields.render_market_fields(
        lambda name: "lx_" + name, fallback,
        widget_names=("spot", "strike", "volatility", "rate", "maturity"),
    )
    assert pair == ("Put", "American")
    assert market == (100., 100., 20., 2., 1.)
    assert [entry[2] for entry in ui.inputs] == [
        "lx_kind", "lx_exercise", "lx_spot", "lx_strike",
        "lx_volatility", "lx_rate", "lx_maturity",
    ]


def test_visual_depth_extended_but_still_bounded():
    assert MAX_VISUAL_STEPS == 100
    assert assess_transfer(PricingRequest(steps=100), "lattice_explorer").exact
    assert not assess_transfer(PricingRequest(steps=101), "lattice_explorer").exact



def test_handoff_requires_explicit_change_above_new_cap():
    request = PricingRequest(steps=101)
    original, issues = plan_handoff(request, "Explore lattice")
    assert original.steps == 101
    assert issues
    adjusted, issues = plan_handoff(request, "Explore lattice", steps=100)
    assert not issues
    assert adjusted.steps == 100
    assert request.steps == 101



def test_live_20_step_geometry_sums_probability_mass():
    from frontend.lattice_explorer_service import snapshot_priced_tree
    request = PricingRequest(steps=20, model="Trinomial · Allocation pruning")
    tree = snapshot_priced_tree(request)
    assert len(tree.column_masses) == 21
    assert all(abs(total - 1.) < 1e-9 for total in tree.column_masses)
    assert tree.displayed_nodes > 0
    assert tree.nodes[0].option_value == pytest.approx(tree.model_price)


class ReferenceImportUI:
    """Small Streamlit double: rendering never invokes callbacks."""

    def __init__(self):
        self.session_state = {}
        self.buttons = []
        self.messages = []

    def markdown(self, text, **kwargs):
        self.messages.append(("heading", text))

    def caption(self, text):
        self.messages.append(("caption", text))

    def success(self, text):
        self.messages.append(("success", text))

    def warning(self, text):
        self.messages.append(("warning", text))

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False


@pytest.mark.parametrize("mode,key,feedback_key", [
    ("Compare models", "cmp_import_contract", "cmp_import_feedback"),
    ("Explore lattice", "lx_import_reference", "explorer_import_feedback"),
])
def test_shared_reference_import_keeps_keys_and_does_not_price(
    monkeypatch, mode, key, feedback_key,
):
    ui = ReferenceImportUI()
    monkeypatch.setattr(fields, "st", ui)
    calls = []
    monkeypatch.setattr(fields, "import_saved_reference",
                        lambda *args: calls.append(args))
    fields.render_reference_import(mode)
    assert not calls
    assert len(ui.buttons) == 1
    label, options = ui.buttons[0]
    assert label == "Import reference contract"
    assert options["key"] == key
    assert options["disabled"] is True
    assert options["args"] == (mode,)
    assert options["on_click"] is fields.on_reference_import
    assert options["width"] == "stretch"
    assert any(kind == "caption" and "Pricing Terminal" in text
               for kind, text in ui.messages)
    ui.session_state[feedback_key] = ("warning", "Import unavailable")
    fields.render_reference_import(mode)
    assert ("warning", "Import unavailable") in ui.messages


@pytest.mark.parametrize("mode,feedback_key", [
    ("Compare models", "cmp_import_feedback"),
    ("Explore lattice", "explorer_import_feedback"),
])
def test_shared_import_feedback_keeps_failure_atomic(monkeypatch, mode, feedback_key):
    ui = ReferenceImportUI()
    ui.session_state["other_result"] = object()
    previous_result = ui.session_state["other_result"]
    monkeypatch.setattr(fields, "st", ui)
    calls = []

    def reject(state, chosen_mode):
        calls.append((state, chosen_mode))
        return False, ("Threshold not offered",)

    monkeypatch.setattr(fields, "import_saved_reference", reject)
    fields.on_reference_import(mode)
    assert calls == [(ui.session_state, mode)]
    assert ui.session_state[feedback_key] == (
        "warning", "Import unavailable: Threshold not offered",
    )
    assert ui.session_state["other_result"] is previous_result


def test_shared_import_displays_success_without_new_pricing(monkeypatch):
    ui = ReferenceImportUI()
    monkeypatch.setattr(fields, "st", ui)
    monkeypatch.setattr(fields, "import_saved_reference",
                        lambda state, mode: (True, ()))
    fields.on_reference_import("Compare models")
    assert ui.session_state["cmp_import_feedback"][0] == "success"
    fields.render_reference_import("Compare models")
    assert any(kind == "success" for kind, _ in ui.messages)


def test_unknown_diagnostic_import_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown diagnostic import mode"):
        fields._reference_import_settings("unknown")
