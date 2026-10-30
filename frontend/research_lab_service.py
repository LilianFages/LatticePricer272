"""Research Lab data access and reproducible summaries.

The Lab *reads* existing experiment exports, never performs pricing or changes src/.
Every supported path is part of a fixed, repository-relative catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.accuracy_metrics import nondominated_indices

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    label: str
    path: str
    description: str
    required: tuple[str, ...]


# Read only whitelisted experiment outputs from the repository.
CATALOG: dict[str, Dataset] = {
    "Robustness": Dataset(
        "Pruning robustness", "results/data/allocation_pruning_robustness.csv",
        "Independent branch/allocation statuses across contracts and dividend schedules.",
        ("scenario", "nb_steps", "strike", "kind", "exercise", "status",
         "branch_status", "allocation_status", "node_savings_pct", "absolute_gap"),
    ),
    "Pruning benchmark": Dataset(
        "Pruning performance", "results/data/allocation_pruning_benchmark.csv",
        "Node allocation, pricing/build time and peak Python memory by lattice depth.",
        ("nb_steps", "model", "status", "total_seconds", "allocated_nodes",
         "peak_python_mib"),
    ),
    "Accuracy/cost": Dataset(
        "Worst-strike accuracy and cost", "results/data/accuracy_cost_multistrike_aggregate.csv",
        "Worst Black–Scholes error across three observed strikes, measured per valuation.",
        ("model", "nb_steps", "worst_absolute_error",
         # Validate required columns before building research summaries.
         "max_median_runtime_seconds", "allocated_nodes_per_tree"),
    ),
    "Recorded Pareto": Dataset(
        "Previously saved Pareto frontier", "results/data/accuracy_cost_multistrike_pareto.csv",
        "Archived frontier for the original full model set, not automatically updated by filters.",
        ("model", "nb_steps", "worst_absolute_error", "max_median_runtime_seconds"),
    ),
    "Target thresholds": Dataset(
        "Achieved accuracy thresholds", "results/data/accuracy_cost_multistrike_targets.csv",
        "Fastest observed configurations satisfying specified worst-strike error targets.",
        ("target_eur", "status", "model", "nb_steps", "worst_absolute_error",
         "max_median_runtime_seconds"),
    ),
    "Phase 1 contracts": Dataset(
        "Individual price/cost observations", "results/data/accuracy_cost_benchmark.csv",
        "Contract-level benchmark and Black–Scholes error by model, strike and step count.",
        ("nb_steps", "strike", "option_kind", "model", "absolute_error",
         # Restrict Pareto comparisons to observed finite data points.
         "median_runtime_seconds"),
    ),
    "Temporal profiles": Dataset(
        "Discounted value and exercise probabilities", "results/data/option_by_dates_profiles.csv",
        "Time profiles for three equal-forward comparison contracts at N=100.",
        ("scenario", "step", "time", "discounted_mean_option_value",
         "exercise_region_probability", "first_exercise_probability"),
    ),
    "Temporal sensitivity": Dataset(
        "Temporal study across tree depths", "results/data/option_by_dates_sensitivity_summary.csv",
        "N=50,100,200 sensitivity of option premium and exercise timing.",
        ("nb_steps", "scenario", "initial_option_price", "early_exercise_probability",
         "first_exercise_at_dividend"),
    ),
    "Runtime scaling": Dataset(
        "Recursive vs backward induction runtime", "results/data/runtime_scaling.csv",
        "Recorded execution times in seconds for two traversal strategies.",
        # Distinguish successful validation from reference-unavailable cases.
        ("nb_steps", "recursive_seconds", "backward_seconds"),
    ),
    "Pruning thresholds": Dataset(
        "Pruning threshold sensitivity", "results/data/pruning_sensitivity.csv",
        "Observed price gap and runtime as a function of pruning threshold.",
        ("nb_steps", "threshold", "status", "price_gap", "runtime"),
    ),
    "Dividend impact": Dataset(
        "Dividend impact", "results/data/dividend_impact.csv",
        "Study of cash-dividend option prices.", ("dividend", "call_price", "put_price"),
    ),
}


class DatasetError(ValueError):
    """Dataset absent or with a schema incompatible with the published experiment."""


def load_dataset(name: str, root: Path | None = None) -> pd.DataFrame:
    """Load a pre-approved CSV from the project, preserving unavailable cells as NaN."""
    if name not in CATALOG:
        raise DatasetError(f"Unknown research dataset: {name}")
    spec = CATALOG[name]
    project = ROOT if root is None else Path(root)
    filename = project / spec.path
    if not filename.is_file():
        raise DatasetError(f"Missing experiment export: {spec.path}")
    try:
        frame = pd.read_csv(filename)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DatasetError(f"Cannot read {spec.path}: {exc}") from exc
    absent = set(spec.required).difference(frame.columns)
    if absent:
        raise DatasetError(f"Unexpected columns in {spec.path}: {', '.join(sorted(absent))}")
    return frame


def coverage_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Status breakdown; allocation-only successes are NOT price comparisons."""
    required = {"status", "allocation_status", "branch_status"}
    if not required.issubset(frame.columns):
        raise DatasetError("Missing robustness status fields")
    counts = frame["status"].value_counts()
    return {
        "total": len(frame),
        "compared": int(counts.get("passed", 0)),
        "allocation_only": int(counts.get("allocation_only", 0)),
        "failed": int(len(frame) - counts.get("passed", 0) - counts.get("allocation_only", 0)),
    }


def comparable_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Only runs with both engines successful and explicitly comparable."""
    mandatory = {"status", "branch_status", "allocation_status"}
    if not mandatory.issubset(frame.columns):
        raise DatasetError("Missing comparison status fields")
    return frame.loc[
        frame["status"].eq("passed")
        & frame["branch_status"].eq("success")
        & frame["allocation_status"].eq("success")
    ].copy()


def _valid_cost_observations(
    frame: pd.DataFrame,
    time_col: str,
    error_col: str,
    *,
    missing_message: str,
) -> pd.DataFrame:
    """Filter observed nonnegative errors and strictly positive runtimes.

    Missing and nonnumeric values remain excluded, never imputed as zero.
    The source DataFrame and its schema are not modified.
    """
    if time_col not in frame.columns or error_col not in frame.columns:
        raise DatasetError(missing_message)
    data = frame.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[error_col] = pd.to_numeric(data[error_col], errors="coerce")
    data = data.dropna(subset=[time_col, error_col])
    return data.loc[data[time_col].gt(0) & data[error_col].ge(0)].copy()


def observed_pareto(
    frame: pd.DataFrame,
    time_col: str = "max_median_runtime_seconds",
    error_col: str = "worst_absolute_error",
) -> pd.DataFrame:
    """Pareto over valid observed rows, retaining equivalent coordinates."""
    data = _valid_cost_observations(
        frame, time_col, error_col, missing_message="Missing Pareto axes",
    )
    if data.empty:
        return data
    pairs = list(zip(data[time_col].to_numpy(), data[error_col].to_numpy()))
    keep = nondominated_indices(pairs)
    return data.iloc[list(keep)].sort_values([time_col, error_col]).reset_index(drop=True)

def frontier_thresholds(frame: pd.DataFrame, targets: tuple[float, ...]) -> pd.DataFrame:
    """Fastest OBSERVED point achieving each selected error threshold."""
    valid = _valid_cost_observations(
        frame, "max_median_runtime_seconds", "worst_absolute_error",
        missing_message="Missing observed pricing cost/error columns",
    )
    records: list[dict] = []
    for target in targets:
        matched = valid.loc[valid["worst_absolute_error"] <= target]
        if matched.empty:
            records.append({"target_eur": target, "status": "not reached",
                            "model": None, "nb_steps": None, "error_eur": None,
                            "runtime_ms": None})
            continue
        best = matched.sort_values(["max_median_runtime_seconds", "worst_absolute_error"]).iloc[0]
        records.append({"target_eur": target, "status": "reached",
                        "model": best.get("model"), "nb_steps": best.get("nb_steps"),
                        "error_eur": float(best["worst_absolute_error"]),
                        "runtime_ms": float(best["max_median_runtime_seconds"]) * 1000})
    return pd.DataFrame.from_records(records)


def robustness_by_step(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate savings for comparable cases only (never fill absent gap with 0)."""
    matched = comparable_cases(frame)
    if matched.empty:
        return pd.DataFrame(columns=["scenario", "nb_steps", "mean_savings_pct", "cases"])
    matched["node_savings_pct"] = pd.to_numeric(matched["node_savings_pct"], errors="coerce")
    return matched.groupby(["scenario", "nb_steps"], as_index=False).agg(
        mean_savings_pct=("node_savings_pct", "mean"),
        cases=("status", "size"),
    )


def mass_consistency(frame: pd.DataFrame) -> float:
    """Maximum absolute risk-neutral probability-mass deviation from 1."""
    if "unconditional_probability_mass" not in frame.columns:
        raise DatasetError("Missing probability mass column")
    mass = pd.to_numeric(frame["unconditional_probability_mass"], errors="coerce").dropna()
    if mass.empty:
        raise DatasetError("No mass observations")
    return float((mass - 1.0).abs().max())



def missingness_diagnostics(name: str, frame: pd.DataFrame) -> dict[str, int]:
    """Explain empty CSV fields without treating unavailable prices as zero.

    The robustness export has two distinct, documented causes of blank fields:
    empty error messages for runs without that error, and comparison-only
    metrics for an allocation-only outcome (the branch baseline failed).
    Any other NaNs remain explicitly unclassified.
    """
    total = int(frame.isna().sum().sum())
    if name != "Robustness":
        return {"total": total, "empty_error_fields": 0,
                "unavailable_comparisons": 0, "unclassified": total}
    error_cols = ("branch_error", "allocation_error", "error")
    comparison_cols = ("branch_price", "absolute_gap", "relative_gap",
                       "branch_nodes", "node_savings_pct",
                       "branch_runtime_seconds", "branch_mass_error")
    empty_errors = sum(int(frame[col].isna().sum()) for col in error_cols if col in frame)
    if "status" not in frame:
        raise DatasetError("Missing robustness status field")
    allocation_only = frame["status"].eq("allocation_only")
    unavailable = sum(int(frame.loc[allocation_only, col].isna().sum())
                      for col in comparison_cols if col in frame)
    return {
        "total": total,
        "empty_error_fields": empty_errors,
        "unavailable_comparisons": unavailable,
        "unclassified": total - empty_errors - unavailable,
    }


def missingness_by_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Report real missing-cell counts for selected columns; no imputation."""
    missing = frame.isna().sum()
    result = missing[missing.gt(0)].sort_values(ascending=False)
    return pd.DataFrame({"column": result.index.tolist(),
                         "missing_cells": result.astype(int).tolist(),
                         "share_pct": (100 * result / max(len(frame), 1)).round(1).tolist()})


def as_csv(frame: pd.DataFrame) -> bytes:
    """Export current filter, with no index, to an Excel-compatible UTF-8 CSV."""
    return frame.to_csv(index=False).encode("utf-8-sig")
