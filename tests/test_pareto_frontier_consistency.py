"""Phase 6B: identical two-axis dominance in all research consumers."""
from __future__ import annotations

import pandas as pd

from experiments.accuracy_cost_multistrike import Aggregate, pareto_multi
from experiments.accuracy_vs_cost import Result, pareto_front
from frontend.research_lab_service import observed_pareto
from src.accuracy_metrics import nondominated_indices


def test_pure_frontier_retains_ties_zero_errors_and_tradeoffs() -> None:
    points = [(1., .2), (2., .1), (3., .15), (1., .2), (4., 0.)]
    assert nondominated_indices(points) == (0, 1, 3, 4)
    assert nondominated_indices([]) == ()
    assert nondominated_indices([(1, 0), (1, 0)]) == (0, 1)


def test_live_archive_accept_same_observations_and_duplicate_coordinates() -> None:
    frame = pd.DataFrame({
        'model': ['A', 'B', 'C', 'D', 'E'],
        'max_median_runtime_seconds': [1., 2., 3., 1., 4.],
        'worst_absolute_error': [.2, .1, .15, .2, 0.],
    })
    assert observed_pareto(frame)['model'].tolist() == ['A', 'D', 'B', 'E']


def test_phase_one_frontier_keeps_observation_order_on_equal_runtimes() -> None:
    points = []
    for steps, runtime, gap in [(20, 1., .2), (40, 2., .1),
                                (60, 3., .15), (80, 1., .2)]:
        points.append(Result(
            nb_steps=steps, strike=100., option_kind='call', model='binomial',
            reference_price=10., price=10. + gap, absolute_error=gap,
            relative_error=gap/10., median_runtime_seconds=runtime,
            min_runtime_seconds=runtime, max_runtime_seconds=runtime,
            allocated_nodes=100, status='success', failure_reason='',
        ))
    assert [r.nb_steps for r in pareto_front(points)] == [20, 80, 40]


def test_phase_two_frontier_retains_original_secondary_error_sort() -> None:
    rows = []
    for steps, runtime, gap in [(20, 1., .2), (40, 2., .1),
                                (60, 3., .15), (80, 1., .2)]:
        rows.append(Aggregate(
            model='binomial', nb_steps=steps, worst_absolute_error=gap,
            mean_absolute_error=gap, root_mean_square_error=gap,
            worst_strike=100., max_median_runtime_seconds=runtime,
            sum_median_runtime_seconds=runtime * 3,
            allocated_nodes_per_tree=100, source='phase1',
        ))
    assert [r.nb_steps for r in pareto_multi(rows)] == [20, 80, 40]
