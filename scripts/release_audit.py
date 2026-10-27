"""Read-only scientific and structural audit for the LatticePricer272 release.

Run ``python -m scripts.release_audit`` at repository root. This command
never constructs a lattice, alters an export or regenerates historical data.
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path

MODELS = ("Hybrid", "Trinomial · Allocation pruning")
GRID_STEPS = (25, 40, 50, 75, 100, 125, 150, 200)
DATE_SHIFTS = (-2, -1, 0, 1, 2)
GRID_DATE = .5
DATE_STEPS = 100
FIELDS = (
    "panel", "model", "steps", "dividend_time", "date_shift_days",
    "dividend_step", "within_step_fraction", "status", "european_put",
    "american_put", "european_call", "early_exercise_premium",
    "parity_residual", "gap_to_finest_sample", "finest_reference_steps",
    "date_shift_put", "failure",
)
ROOT = Path(__file__).resolve().parents[1]


class ReleaseAuditError(ValueError):
    """The recorded release does not satisfy its declared evidence contract."""


def _float(row: dict[str, str], field: str, case: str) -> float:
    value = row.get(field)
    if not value:
        raise ReleaseAuditError(f"{case}: missing {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseAuditError(f"{case}: invalid {field}") from exc
    if not math.isfinite(parsed):
        raise ReleaseAuditError(f"{case}: non-finite {field}")
    return parsed


def _close(actual: float, expected: float, case: str, field: str,
           tolerance: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ReleaseAuditError(f"{case}: inconsistent {field} ({actual} vs {expected})")


def audit_dividend_grid_csv(filename: Path) -> dict[str, object]:
    """Verify full factorial coverage and independently recompute each invariant."""
    try:
        with filename.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            if len(columns) != len(set(columns)) or set(columns) != set(FIELDS):
                raise ReleaseAuditError("Unexpected or duplicate Phase 6C CSV columns")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseAuditError(f"Cannot load Phase 6C data: {exc}") from exc

    expected = ({("grid", model, n, 0) for model in MODELS for n in GRID_STEPS}
                | {("date", model, DATE_STEPS, shift)
                   for model in MODELS for shift in DATE_SHIFTS})
    observed: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in rows:
        case = f"{row.get('panel')} / {row.get('model')} / {row.get('steps')}"
        steps_float = _float(row, "steps", case)
        shift_float = _float(row, "date_shift_days", case)
        if not steps_float.is_integer() or not shift_float.is_integer():
            raise ReleaseAuditError(f"{case}: non-integer grid coordinate")
        key = (row["panel"], row["model"], int(steps_float), int(shift_float))
        if key in observed:
            raise ReleaseAuditError(f"Duplicate recorded case: {key}")
        observed[key] = row
        if row["status"] != "success":
            raise ReleaseAuditError(f"Unpriced full-study case: {key}: {row['failure']}")
        exdate = _float(row, "dividend_time", case)
        expected_date = GRID_DATE if key[0] == "grid" else GRID_DATE + key[3] / 365.0
        _close(exdate, expected_date, case, "actual dividend ex-date", 1e-12)
        event_step = _float(row, "dividend_step", case)
        if not event_step.is_integer() or not 1 <= event_step <= key[2]:
            raise ReleaseAuditError(f"{case}: invalid dividend step")
        fraction = _float(row, "within_step_fraction", case)
        if fraction < -1e-10 or fraction > 1 + 1e-10:
            raise ReleaseAuditError(f"{case}: ex-date not in recorded step")
        european_put = _float(row, "european_put", case)
        american_put = _float(row, "american_put", case)
        european_call = _float(row, "european_call", case)
        if min(european_put, american_put, european_call) < -1e-10:
            raise ReleaseAuditError(f"{case}: negative option premium")
        early_premium = _float(row, "early_exercise_premium", case)
        _close(early_premium, american_put - european_put, case, "early premium")
        if early_premium < -1e-7:
            raise ReleaseAuditError(f"{case}: American put below European put")
        parity = (european_call - european_put - 100.0
                  + 3.0 * math.exp(-.02 * exdate)
                  + 100.0 * math.exp(-.02))
        _close(_float(row, "parity_residual", case), parity, case, "parity residual")
        if abs(parity) > 1e-7:
            raise ReleaseAuditError(f"{case}: put-call parity check failed")

    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        raise ReleaseAuditError(f"Incomplete or unexpected Phase 6C grid: "
                                f"missing={missing[:5]} unexpected={unexpected[:5]}")

    for model in MODELS:
        largest = observed[("grid", model, max(GRID_STEPS), 0)]
        ref = _float(largest, "european_put", model)
        for n in GRID_STEPS:
            case = f"grid/{model}/{n}"
            row = observed[("grid", model, n, 0)]
            _close(_float(row, "gap_to_finest_sample", case),
                   _float(row, "european_put", case) - ref,
                   case, "finite-N reference gap")
            _close(_float(row, "finest_reference_steps", case),
                   max(GRID_STEPS), case, "finest reference N", 0.0)
        baseline = _float(observed[("date", model, DATE_STEPS, 0)],
                          "european_put", model)
        for shift in DATE_SHIFTS:
            case = f"date/{model}/{shift}"
            row = observed[("date", model, DATE_STEPS, shift)]
            _close(_float(row, "date_shift_put", case),
                   _float(row, "european_put", case) - baseline,
                   case, "ex-date contract price difference")
    return {"cases": len(rows), "models": len(MODELS), "passed": len(observed)}


def audit_local_imports(root: Path) -> dict[str, int]:
    """Statically inspect first-party imports. This is not a dead-code proof."""
    pyfiles = [file for prefix in ("src", "frontend", "experiments", "scripts")
               for file in (root / prefix).rglob("*.py")]
    missing: set[str] = set()
    for path in pyfiles:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ReleaseAuditError(f"Unparseable source {path}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = (node.module or "",)
            else:
                continue
            for name in modules:
                tokens = name.split(".")
                if tokens[0] not in ("src", "frontend", "experiments", "scripts"):
                    continue
                module_path = root.joinpath(*tokens)
                if not (module_path.with_suffix(".py").is_file()
                        or (module_path / "__init__.py").is_file()):
                    missing.add(f"{path.relative_to(root)} -> {name}")
    if missing:
        raise ReleaseAuditError("Unresolved static first-party imports: "
                                + ", ".join(sorted(missing)[:12]))
    return {"modules_parsed": len(pyfiles), "missing_imports": 0}


def audit_release(root: Path) -> dict[str, object]:
    """Check versioned evidence without running models or writing any files."""
    required = (
        "README.md", "docs/Dividend_Grid_Robustness_Protocol.md",
        "docs/Quant_Research_Phase6.md", "docs/Release_Protocol.md",
        "results/dividend_grid_robustness_summary.txt",
        "results/figures/dividend_grid_prices.png",
        "results/figures/dividend_exdate_shifts.png",
    )
    missing = [str(p) for p in required if not (root / p).is_file()]
    if missing:
        raise ReleaseAuditError("Missing release artifacts: " + ", ".join(missing))
    data = audit_dividend_grid_csv(root / "results/data/dividend_grid_robustness.csv")
    summary = (root / "results/dividend_grid_robustness_summary.txt").read_text(
        encoding="utf-8")
    if not re.search(r"Configurations:\s*26;\s*valid:\s*26;\s*failed:\s*0", summary):
        raise ReleaseAuditError("Saved summary does not match complete 26-case CSV")
    for name in ("dividend_grid_prices.png", "dividend_exdate_shifts.png"):
        picture = root / "results/figures" / name
        if picture.stat().st_size < 100 or picture.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ReleaseAuditError(f"Invalid or empty figure: {name}")
    imports = audit_local_imports(root)
    return {**data, **imports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Repository root (default: script parent)")
    args = parser.parse_args(argv)
    try:
        report = audit_release(args.root.resolve())
    except ReleaseAuditError as exc:
        print(f"[RELEASE AUDIT FAILED] {exc}")
        return 1
    print("[RELEASE AUDIT OK] " + ", ".join(f"{k}={v}" for k, v in report.items()))
    print("No pricing run and no outputs modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
