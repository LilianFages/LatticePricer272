"""Phase 6C: dividend ex-date grid robustness, not a claim of exact convergence.

Two independent axes:
  grid: fixed actual ex-date, varying lattice step count;
  date: fixed lattice step count, shifting the *actual* ex-date in days.

No pricing engine is reimplemented or modified. Each observation reprices a
fresh European put, American put and European call for the same dividend.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from frontend.pricing_service import DividendInput, PricingRequest, price_contract
from src.contracts.dividend_timing import dividend_grid_step

ROOT = Path(__file__).resolve().parents[2]
SPOT = 100.0
STRIKE = 100.0
RATE = 0.02
VOLATILITY = 0.20
MATURITY = 1.0
DIVIDEND_TIME = 0.5
DIVIDEND_AMOUNT = 3.0
MODELS = ("Hybrid", "Trinomial · Allocation pruning")
FULL_STEPS = (25, 40, 50, 75, 100, 125, 150, 200)
QUICK_STEPS = (25, 40, 50)
FULL_SHIFTS = (-2, -1, 0, 1, 2)
QUICK_SHIFTS = (-1, 0, 1)
DEFAULT_DATE_STEPS = 100
FIELDS = (
    # Identify the independent experimental axis and numerical configuration.
    "panel", "model", "steps", "dividend_time", "date_shift_days",
    # Distinguish contract state, grid location, and pricing outcomes.
    "dividend_step", "within_step_fraction", "status",
    "european_put", "american_put", "european_call",
    "early_exercise_premium", "parity_residual",
    # Keep sampled-grid errors separate from actual ex-date sensitivity.
    "gap_to_finest_sample", "finest_reference_steps", "date_shift_put",
    "failure",
)


def validate_design(steps: Sequence[int], shifts: Sequence[int], date_steps: int) -> None:
    """Require a controlled, finite experiment; never silently drop a case."""
    if (not steps or any(type(n) is not int or n < 1 or n > 350 for n in steps)
            or len(set(steps)) != len(steps)):
        raise ValueError("Grid N must consist of distinct integers in [1, 350].")
    if type(date_steps) is not int or not 1 <= date_steps <= 350:
        raise ValueError("Date-sweep N must belong to [1, 350].")
    if (not shifts or 0 not in shifts
            or any(type(d) is not int or abs(d) > 60 for d in shifts)
            or len(set(shifts)) != len(shifts)):
        raise ValueError("Date shifts must be distinct integer days, including zero.")
    if any(not 0.0 < DIVIDEND_TIME + d / 365.0 <= MATURITY for d in shifts):
        raise ValueError("Shifted ex-date falls outside the contract maturity.")


def parity_target(ex_date: float) -> float:
    """Model-independent European C-P under deterministic cash payments."""
    return (SPOT - DIVIDEND_AMOUNT * math.exp(-RATE * ex_date)
            - STRIKE * math.exp(-RATE * MATURITY))


def _case_request(model: str, steps: int, dividend_time: float) -> PricingRequest:
    """Validate and construct the matched European-put reference contract."""
    # Reject invalid configurations before constructing the lattice.
    if model not in MODELS:
        raise ValueError(f"Unsupported study model: {model}")
    if type(steps) is not int or not 1 <= steps <= 350:
        raise ValueError("Invalid number of lattice steps.")
    if not math.isfinite(dividend_time) or not 0 < dividend_time <= MATURITY:
        raise ValueError("Invalid dividend ex-date.")

    # All three payoffs share the exact same dividend event.
    dividend = (DividendInput(time=dividend_time, amount=DIVIDEND_AMOUNT),)
    return PricingRequest(
        spot=SPOT, strike=STRIKE, rate=RATE, volatility=VOLATILITY,
        maturity=MATURITY, steps=steps, model=model, dividends=dividend,
        option_type="Put", exercise="European", pruning_threshold=1e-18,
    )



def _price_triplet(
    request: PricingRequest, pricer: Callable
) -> tuple[float, float, float]:
    """Value matched European put, American put and European call."""
    eur_put = float(pricer(request).price)
    amer_put = float(pricer(replace(request, exercise="American")).price)
    eur_call = float(pricer(replace(request, option_type="Call")).price)
    if not all(map(math.isfinite, (eur_put, amer_put, eur_call))):
        raise ArithmeticError("Non-finite option premium")
    if min(eur_put, amer_put, eur_call) < -1e-10:
        raise ArithmeticError("Negative option premium beyond tolerance")
    return eur_put, amer_put, eur_call


def measure_case(
    model: str, steps: int, dividend_time: float,
    pricer: Callable = price_contract,
) -> dict[str, object]:
    """Price one contract triplet atomically; disclose any failed valuation."""
    request = _case_request(model, steps, dividend_time)
    step = dividend_grid_step(dividend_time, MATURITY, steps)
    dt = MATURITY / steps
    position = (dividend_time - (step - 1) * dt) / dt
    data: dict[str, object] = {
        "model": model, "steps": steps, "dividend_time": dividend_time,
        "dividend_step": step, "within_step_fraction": position,
        "status": "failed", "european_put": None, "american_put": None,
        "european_call": None, "early_exercise_premium": None,
        "parity_residual": None, "failure": "",
    }
    # Measure put exercise premium and European parity from one market.
    try:
        eur_put, amer_put, eur_call = _price_triplet(request, pricer)
    except (ValueError, ArithmeticError, RuntimeError, OverflowError, TypeError) as exc:
        data["failure"] = f"{type(exc).__name__}: {exc}"
        return data

    # A successful observation records three finite premiums.
    data.update(
        status="success", european_put=eur_put, american_put=amer_put,
        european_call=eur_call, early_exercise_premium=amer_put - eur_put,
        parity_residual=eur_call - eur_put - parity_target(dividend_time),
    )
    return data


def _grid_panel(
    model: str, steps: Sequence[int],
    observe: Callable[[str, int, float], dict[str, object]]
) -> list[dict[str, object]]:
    """Vary N for a fixed cash-dividend ex-date."""
    fixed_rows: list[dict[str, object]] = []
    for n in sorted(steps):
        row = dict(observe(model, n, DIVIDEND_TIME), panel="grid",
                   date_shift_days=0, gap_to_finest_sample=None,
                   finest_reference_steps=None, date_shift_put=None)
        fixed_rows.append(row)
    # The reference is the finest successful observation, not an exact price.
    successes = [r for r in fixed_rows if r["status"] == "success"]
    if successes:
        finest = max(successes, key=lambda r: int(r["steps"]))
        for row in successes:
            row["gap_to_finest_sample"] = (
                float(row["european_put"]) - float(finest["european_put"])
            )
            row["finest_reference_steps"] = finest["steps"]
    return fixed_rows


def _date_panel(
    model: str, shifts: Sequence[int], date_steps: int,
    observe: Callable[[str, int, float], dict[str, object]]
) -> list[dict[str, object]]:
    """Vary the actual ex-date while holding N fixed."""
    records: list[dict[str, object]] = []
    baseline = observe(model, date_steps, DIVIDEND_TIME)
    for days in sorted(shifts):
        t = DIVIDEND_TIME + days / 365.0
        row = dict(observe(model, date_steps, t), panel="date",
                   date_shift_days=days, gap_to_finest_sample=None,
                   finest_reference_steps=None, date_shift_put=None)
        if row["status"] == "success" and baseline["status"] == "success":
            row["date_shift_put"] = (
                float(row["european_put"]) - float(baseline["european_put"])
            )
        records.append(row)
    return records


def run_experiment(
    steps: Sequence[int] = FULL_STEPS,
    shifts: Sequence[int] = FULL_SHIFTS,
    date_steps: int = DEFAULT_DATE_STEPS,
    pricer: Callable = price_contract,
    models: Sequence[str] = MODELS,
) -> list[dict[str, object]]:
    """Keep contract-fixed grid and ex-date-fixed-grid contrasts distinct."""
    validate_design(steps, shifts, date_steps)
    if not models or len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError("Models must be a unique nonempty subset of the study models.")

    # Reuse duplicated baseline requests within the controlled sweep.
    records: list[dict[str, object]] = []
    cache: dict[tuple[str, int, float], dict[str, object]] = {}

    def observe(model: str, n: int, t: float) -> dict[str, object]:
        key = (model, n, t)
        if key not in cache:
            cache[key] = measure_case(model, n, t, pricer=pricer)
        return cache[key]

    for model in models:
        records.extend(_grid_panel(model, steps, observe))
        records.extend(_date_panel(model, shifts, date_steps, observe))
    return records


def output_paths(quick: bool) -> tuple[Path, Path, Path, Path]:
    """Never overwrite established experiment outputs with a quick run."""
    if quick:
        root = ROOT / "results" / "phase6c_quick"
        return (root / "dividend_grid_robustness.csv", root / "summary.txt",
                root / "grid_prices.png", root / "exdate_shifts.png")
    return (ROOT / "results" / "data" / "dividend_grid_robustness.csv",
            ROOT / "results" / "dividend_grid_robustness_summary.txt",
            ROOT / "results" / "figures" / "dividend_grid_prices.png",
            ROOT / "results" / "figures" / "dividend_exdate_shifts.png")


def summarize(rows: Sequence[dict[str, object]]) -> str:
    """Describe observations and limitations without imputation."""
    good = [r for r in rows if r["status"] == "success"]
    failed = [r for r in rows if r["status"] != "success"]
    lines = [
        "LatticePricer272 / Phase 6C / Dividend ex-date grid robustness",
        f"Configurations: {len(rows)}; valid: {len(good)}; failed: {len(failed)}.",
        f"Contracts: S0={SPOT}, K={STRIKE}, r={RATE}, sigma={VOLATILITY}, T={MATURITY}.",
        f"Dividend: cash D={DIVIDEND_AMOUNT} at base ex-date t={DIVIDEND_TIME} years.",
        "GRID cohort: hold real contract fixed and vary lattice depth N.",
        "DATE cohort: hold N fixed and change the true contractual ex-date.",
        "European parity: C-P=S0-PV(dividend)-K*exp(-r*T).",
        "Gap-to-finest: difference to largest successful sampled N per model, NOT exact error.",
        "Date-shift put: finite price change under a DIFFERENT dividend contract, NOT error.",
        "Early premium: American put minus European put, same model and N.",
        "Interpret across-grid oscillations as observations, NOT proved convergence rates.",
        "Measured values depend on model and pruning convention; no market calibration.",
    ]
    for model in dict.fromkeys(str(r["model"]) for r in rows):
        sample = [r for r in good if r["model"] == model]
        if sample:
            parity = max(abs(float(r["parity_residual"])) for r in sample)
            premium = min(float(r["early_exercise_premium"]) for r in sample)
            lines.append(f"{model}: largest |parity residual|={parity:.9g} EUR; "
                         f"smallest American premium={premium:.9g} EUR.")
        else:
            lines.append(f"{model}: no successful triplets.")
    # Record every failed valuation explicitly; do not impute missing prices.
    for row in failed:
        lines.append(f"FAILURE / {row['panel']} / {row['model']} / N={row['steps']} "
                     f"/ day-shift={row['date_shift_days']}: {row['failure']}")
    return "\n".join(lines) + "\n"


def plot_results(rows: Sequence[dict[str, object]], grid_path: Path,
                 date_path: Path) -> None:
    """Static figures show observations only; lines are visual guides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for panel, path, xkey, ykey, xlabel, ylabel in (
        ("grid", grid_path, "steps", "european_put", "Lattice depth N",
         "European put value (€)"),
        ("date", date_path, "date_shift_days", "date_shift_put",
         "Actual ex-date shift (calendar days)", "Put price change (€)"),
    ):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for model in MODELS:
            subset = sorted((r for r in rows if r["panel"] == panel
                             and r["model"] == model and r["status"] == "success"
                             and r[ykey] is not None), key=lambda r: float(r[xkey]))
            if subset:
                ax.plot([float(r[xkey]) for r in subset],
                        [float(r[ykey]) for r in subset], marker="o", label=model)
        if panel == "date":
            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        if ax.has_data():
            ax.legend()
        # Figures are derived only from recorded model outputs.
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)


def write_results(rows: Sequence[dict[str, object]], quick: bool,
                  overwrite: bool = False, plots: bool = True) -> tuple[Path, ...]:
    csv_path, summary_path, grid_path, date_path = output_paths(quick)
    written = (csv_path, summary_path, grid_path, date_path) if plots else (csv_path, summary_path)
    if not overwrite and any(p.exists() for p in written):
        raise FileExistsError("Phase 6C output already exists; use --overwrite explicitly.")
    for path in written:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the fixed schema before any optional plot rendering.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(summarize(rows), encoding="utf-8")
    if plots:
        plot_results(rows, grid_path, date_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small, separately saved smoke run")
    parser.add_argument("--overwrite", action="store_true", help="Replace this study's own outputs")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV and summary only")
    args = parser.parse_args(argv)
    paths = output_paths(args.quick)
    chosen = paths[:2] if args.no_plots else paths
    if not args.overwrite and any(p.exists() for p in chosen):
        parser.error("Study outputs already exist; use --overwrite explicitly.")
    # Keep quick diagnostics separate from the reference full study.
    steps = QUICK_STEPS if args.quick else FULL_STEPS
    shifts = QUICK_SHIFTS if args.quick else FULL_SHIFTS
    rows = run_experiment(steps=steps, shifts=shifts)
    out = write_results(rows, quick=args.quick, overwrite=args.overwrite,
                        plots=not args.no_plots)
    print(f"[6C] {sum(r['status'] == 'success' for r in rows)}/{len(rows)} "
          f"triplets successful; each successful triplet means 3 fresh valuations.")
    for file in out:
        print(f"[6C OUTPUT] {file.relative_to(ROOT)}")
    return int(any(r["status"] != "success" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
