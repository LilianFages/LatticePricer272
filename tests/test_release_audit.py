"""Release-level evidence tests: fast, deterministic, no lattice construction."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts.release_audit import (
    DATE_SHIFTS, FIELDS, GRID_STEPS, MODELS,
    ReleaseAuditError, audit_dividend_grid_csv, audit_local_imports,
    audit_release,
)


def _put(model: str, n: int, exdate: float) -> float:
    return 8.0 + (0.01 if model == MODELS[0] else 0.0) + 1.5 / n + .2 * (exdate - .5)


def _record(panel: str, model: str, n: int, days: int) -> dict:
    date = .5 if panel == "grid" else .5 + days / 365.0
    put = _put(model, n, date)
    step = math.ceil(date * n)
    parity_target = 100.0 - 3.0 * math.exp(-.02 * date) - 100.0 * math.exp(-.02)
    return dict(panel=panel, model=model, steps=n, dividend_time=date,
                date_shift_days=days, dividend_step=step,
                within_step_fraction=(date - (step-1)/n) * n,
                status="success", european_put=put, american_put=put+.2,
                european_call=put+parity_target, early_exercise_premium=.2,
                parity_residual=0.0, gap_to_finest_sample="",
                finest_reference_steps="", date_shift_put="", failure="")


def _complete_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        for n in GRID_STEPS:
            record = _record("grid", model, n, 0)
            record["gap_to_finest_sample"] = _put(model, n, .5) - _put(model, 200, .5)
            record["finest_reference_steps"] = 200
            rows.append(record)
        for day in DATE_SHIFTS:
            record = _record("date", model, 100, day)
            record["date_shift_put"] = (_put(model, 100, .5 + day/365)
                                        - _put(model, 100, .5))
            rows.append(record)
    return rows


def _export(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_full_scientific_grid_and_invariants_are_accepted(tmp_path: Path) -> None:
    data = _export(tmp_path / "evidence.csv", _complete_rows())
    assert audit_dividend_grid_csv(data) == {"cases": 26, "models": 2, "passed": 26}


@pytest.mark.parametrize("field,value,fragment", [
    ("parity_residual", .5, "parity"),
    ("early_exercise_premium", -.2, "early premium"),
    ("european_put", float("inf"), "non-finite"),
    ("within_step_fraction", 1.5, "ex-date"),
    ("finest_reference_steps", 125, "reference N"),
])
def test_corrupted_scientific_observation_is_rejected(
    tmp_path: Path, field: str, value: float, fragment: str,
) -> None:
    rows = _complete_rows()
    rows[0][field] = value
    with pytest.raises(ReleaseAuditError, match=fragment):
        audit_dividend_grid_csv(_export(tmp_path / "bad.csv", rows))


def test_duplicate_and_missing_coordinates_are_rejected(tmp_path: Path) -> None:
    rows = _complete_rows()
    with pytest.raises(ReleaseAuditError, match="Duplicate"):
        audit_dividend_grid_csv(_export(tmp_path / "duplicate.csv", rows + [rows[0]]))
    with pytest.raises(ReleaseAuditError, match="Incomplete"):
        audit_dividend_grid_csv(_export(tmp_path / "missing.csv", rows[:-1]))


def test_failed_valuation_cannot_be_declared_a_release_success(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows[5]["status"] = "failed"
    rows[5]["failure"] = "Not priced"
    with pytest.raises(ReleaseAuditError, match="Unpriced"):
        audit_dividend_grid_csv(_export(tmp_path / "failed.csv", rows))


def test_release_checks_all_artifacts_read_only(tmp_path: Path) -> None:
    root = tmp_path
    data = _export(root / "results/data/dividend_grid_robustness.csv", _complete_rows())
    names = ["README.md", "docs/Dividend_Grid_Robustness_Protocol.md",
             "docs/Quant_Research_Phase6.md", "docs/Release_Protocol.md"]
    for name in names:
        filename = root / name
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(name, encoding="utf-8")
    summary = root / "results/dividend_grid_robustness_summary.txt"
    summary.write_text("Configurations: 26; valid: 26; failed: 0.\n", encoding="utf-8")
    for name in ("dividend_grid_prices.png", "dividend_exdate_shifts.png"):
        path = root / "results/figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"Z" * 200)
    before = data.read_bytes()
    report = audit_release(root)
    assert report["cases"] == 26
    assert data.read_bytes() == before
    assert report["missing_imports"] == 0


def test_static_import_audit_catches_missing_internal_module(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "example.py").write_text("from src.module_that_does_not_exist import f\n")
    with pytest.raises(ReleaseAuditError, match="Unresolved"):
        audit_local_imports(tmp_path)
