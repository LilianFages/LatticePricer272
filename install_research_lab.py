"""Safely register 05 / Research lab in an existing Lattice 272 Streamlit app.

Run at repo root: python install_research_lab.py
Idempotent, does not touch the pricing engines or rewrite unrelated app logic.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "frontend" / "app.py"


class PatchError(RuntimeError):
    pass


def apply_patch(original: str) -> tuple[str, bool]:
    """Return (patched source, whether a modification was necessary)."""
    if (
        '"05 / Research lab"' in original
        and 'from frontend.research_lab import render_research_lab_page' in original
    ):
        return original, False

    navigation_old = '("01 / Pricing terminal", "02 / Model comparison", "03 / Lattice explorer", "04 / Quant analytics"),'
    navigation_new = '("01 / Pricing terminal", "02 / Model comparison", "03 / Lattice explorer", "04 / Quant analytics", "05 / Research lab"),'
    upcoming_old = 'st.markdown("**UPCOMING**")\n        st.caption("05 / Research lab")'
    upcoming_new = 'st.markdown("**RESEARCH**")\n        st.caption("Archived experiments · offline analysis")'
    routing_old = (
        '    if workspace == "04 / Quant analytics":\n'
        '        from frontend.quant_analytics import render_quant_analytics_page\n'
        '        render_quant_analytics_page()\n'
        '        return\n'
    )
    routing_new = routing_old + (
        '\n    if workspace == "05 / Research lab":\n'
        '        from frontend.research_lab import render_research_lab_page\n'
        '        render_research_lab_page()\n'
        '        return\n'
    )
    for anchor, name in [(navigation_old, "workspace navigation"),
                         (upcoming_old, "sidebar placeholder"),
                         (routing_old, "Quant Analytics router")]:
        if original.count(anchor) != 1:
            raise PatchError(
                f"Cannot safely locate {name}. Existing frontend/app.py may have changed; "
                "no files were modified."
            )
    result = original.replace(navigation_old, navigation_new)
    result = result.replace(upcoming_old, upcoming_new)
    result = result.replace(routing_old, routing_new)
    compile(result, str(APP), "exec")
    return result, True


def main() -> None:
    if not APP.is_file():
        raise SystemExit("frontend/app.py not found. Run this from a LatticePricer272 checkout.")
    original = APP.read_text(encoding="utf-8")
    try:
        patched, changed = apply_patch(original)
    except PatchError as exc:
        raise SystemExit(str(exc)) from exc
    if changed:
        APP.write_text(patched, encoding="utf-8")
        print("Research Lab registered in frontend/app.py (no pricing engines changed).")
    else:
        print("Research Lab already registered; no changes made.")


if __name__ == "__main__":
    main()
