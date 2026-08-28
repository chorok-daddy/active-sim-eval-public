"""Claim-aligned metrics for the independent RankSplit confirmation."""

from __future__ import annotations

import itertools
import statistics


def relation(left: float, right: float) -> int:
    return (left > right) - (left < right)


def weak_order_signature(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(
        relation(values[first], values[second])
        for first, second in itertools.combinations(range(len(values)), 2)
    )


def top_set(values: tuple[float, ...]) -> set[int]:
    maximum = max(values)
    return {index for index, value in enumerate(values) if value == maximum}


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def evaluate(
    observed: dict[tuple[int, int], list[int]],
    truth_rates: tuple[tuple[float, ...], ...],
) -> dict[str, float]:
    task_count = len(truth_rates)
    policy_count = len(truth_rates[0])
    estimated = tuple(
        tuple(statistics.fmean(observed[(task, policy)]) for policy in range(policy_count))
        for task in range(task_count)
    )
    weak_exact: list[float] = []
    pair_accuracy: list[float] = []
    top_scores: list[float] = []
    gap_errors: list[float] = []
    regrets: list[float] = []
    for task in range(task_count):
        weak_exact.append(
            float(
                weak_order_signature(estimated[task])
                == weak_order_signature(truth_rates[task])
            )
        )
        top_scores.append(jaccard(top_set(estimated[task]), top_set(truth_rates[task])))
        for first, second in itertools.combinations(range(policy_count), 2):
            truth_gap = truth_rates[task][first] - truth_rates[task][second]
            estimated_gap = estimated[task][first] - estimated[task][second]
            truth_relation = relation(truth_gap, 0.0)
            estimated_relation = relation(estimated_gap, 0.0)
            if truth_relation == 0:
                pair_accuracy.append(float(estimated_relation == 0))
            elif estimated_relation == 0:
                pair_accuracy.append(0.5)
            else:
                pair_accuracy.append(float(estimated_relation == truth_relation))
            gap_errors.append(abs(estimated_gap - truth_gap))
            if truth_relation == estimated_relation:
                regrets.append(0.0)
            elif estimated_relation == 0:
                regrets.append(0.5 * abs(truth_gap))
            else:
                regrets.append(abs(truth_gap))
    return {
        "task_weak_order_exact_rate": statistics.fmean(weak_exact),
        "task_pair_relation_accuracy": statistics.fmean(pair_accuracy),
        "task_top_set_jaccard": statistics.fmean(top_scores),
        "mean_task_gap_mae": statistics.fmean(gap_errors),
        "mean_decision_regret": statistics.fmean(regrets),
    }
