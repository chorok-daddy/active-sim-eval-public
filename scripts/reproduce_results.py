"""Replay the compact public outcome tensor and check reported metrics.

The default run replays one deterministic repetition and checks its primary
budget rows.  ``--full`` replays all 200 repetitions at the three primary
budgets and checks the reported mean metrics and primary-budget AUBC.  The
replay is CPU-only and uses the two public scoring modules directly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys

if __package__:
    from .fixed_preference import BetaPosterior, beta_bernoulli_information, ranking_pd_eig
    from .ranksplit import ranksplit_score
else:
    from fixed_preference import BetaPosterior, beta_bernoulli_information, ranking_pd_eig
    from ranksplit import ranksplit_score


ROOT = Path(__file__).resolve().parents[1]
TENSOR_PATH = ROOT / "data" / "confirmation_tensor.json"
REFERENCE_PATH = ROOT / "data" / "reported_results.json"


def stable_seed(master: int, *labels: object) -> int:
    payload = "|".join(map(str, (master, *labels))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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


def load_inputs() -> tuple[dict[str, object], dict[str, object]]:
    tensor = json.loads(TENSOR_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if tensor.get("status") != "pass":
        raise ValueError("public tensor is not complete")
    if tensor.get("policy_order") != reference["policy_order"]:
        raise ValueError("tensor and reference policy order differ")
    if int(tensor.get("scenario_count", -1)) != int(reference["scenario_count"]):
        raise ValueError("tensor and reference scenario counts differ")
    if len(tensor.get("scenario_records", [])) != int(reference["scenario_count"]):
        raise ValueError("tensor row count does not match its header")
    return tensor, reference


def prepare_tensor(
    tensor: dict[str, object],
) -> tuple[tuple[str, ...], dict[int, list[int]], dict[tuple[int, int], int], tuple[tuple[float, ...], ...]]:
    policies = tuple(str(value) for value in tensor["policy_order"])
    scenarios = list(tensor["scenario_records"])
    if len({str(row["scenario_id"]) for row in scenarios}) != len(scenarios):
        raise ValueError("tensor contains duplicate scenario identifiers")
    tasks = tuple(sorted({str(row["env_id"]) for row in scenarios}))
    task_lookup = {name: index for index, name in enumerate(tasks)}
    task_groups = {index: [] for index in range(len(tasks))}
    outcomes: dict[tuple[int, int], int] = {}
    for scenario_index, scenario in enumerate(scenarios):
        task = task_lookup[str(scenario["env_id"])]
        task_groups[task].append(scenario_index)
        for policy, name in enumerate(policies):
            outcome = int(scenario["policies"][name]["outcome"])
            if outcome not in (0, 1):
                raise ValueError("outcomes must be binary")
            outcomes[(scenario_index, policy)] = outcome
    truth_rates = tuple(
        tuple(
            statistics.fmean(
                outcomes[(scenario, policy)] for scenario in task_groups[task]
            )
            for policy in range(len(policies))
        )
        for task in task_groups
    )
    return policies, task_groups, outcomes, truth_rates


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


def acquisition_score(
    mode: str,
    states: tuple[BetaPosterior, ...],
    policy: int,
    preference_lambda: float,
    cache: dict[tuple[object, ...], float],
) -> float:
    key = (mode, states, policy, preference_lambda)
    if key in cache:
        return cache[key]
    if mode == "conventional_eig":
        value = beta_bernoulli_information(states[policy])
    elif mode == "fixed_preference":
        value = ranking_pd_eig(
            states,
            observed_policy=policy,
            preference_lambda=preference_lambda,
        ).score
    elif mode == "ranksplit":
        value = ranksplit_score(
            states,
            observed_policy=policy,
            preference_lambda=preference_lambda,
        )
    else:
        raise ValueError(f"unknown program mode: {mode}")
    cache[key] = value
    return value


def run_program(
    *,
    mode: str,
    preference_lambda: float,
    scenario_orders: dict[tuple[int, int], list[int]],
    task_groups: dict[int, list[int]],
    outcomes: dict[tuple[int, int], int],
    truth_rates: tuple[tuple[float, ...], ...],
    budgets: tuple[int, ...],
    repetition: int,
    seed: int,
    score_cache: dict[tuple[object, ...], float],
) -> dict[str, dict[str, float]]:
    policy_count = len(truth_rates[0])
    observed = {
        (task, policy): []
        for task in task_groups
        for policy in range(policy_count)
    }
    positions = {key: 0 for key in observed}
    alpha = [[1 for _ in task_groups] for _ in range(policy_count)]
    beta = [[1 for _ in task_groups] for _ in range(policy_count)]
    for task in task_groups:
        for policy in range(policy_count):
            scenario = scenario_orders[(task, policy)][0]
            outcome = outcomes[(scenario, policy)]
            observed[(task, policy)].append(outcome)
            positions[(task, policy)] = 1
            alpha[policy][task] += outcome
            beta[policy][task] += 1 - outcome
    acquired = sum(map(len, observed.values()))
    rows: dict[str, dict[str, float]] = {}
    while True:
        if acquired in budgets:
            rows[str(acquired)] = evaluate(observed, truth_rates)
        if acquired >= max(budgets):
            return rows
        candidates = []
        for task in task_groups:
            states = tuple(
                BetaPosterior(alpha[policy][task], beta[policy][task])
                for policy in range(policy_count)
            )
            for policy in range(policy_count):
                position = positions[(task, policy)]
                if position >= len(scenario_orders[(task, policy)]):
                    continue
                score = acquisition_score(
                    mode,
                    states,
                    policy,
                    preference_lambda,
                    score_cache,
                )
                tie = stable_seed(seed, "tie", repetition, acquired, task, policy)
                candidates.append((score, tie, task, policy))
        if not candidates:
            raise RuntimeError("candidate set exhausted before the final budget")
        _, _, task, policy = max(candidates)
        scenario = scenario_orders[(task, policy)][positions[(task, policy)]]
        positions[(task, policy)] += 1
        outcome = outcomes[(scenario, policy)]
        observed[(task, policy)].append(outcome)
        alpha[policy][task] += outcome
        beta[policy][task] += 1 - outcome
        acquired += 1


def replay(
    *,
    tensor: dict[str, object],
    reference: dict[str, object],
    repetition_indices: list[int],
    budgets: tuple[int, ...],
) -> dict[str, dict[str, list[dict[str, float]]]]:
    _, task_groups, outcomes, truth_rates = prepare_tensor(tensor)
    seed = int(reference["seed"])
    score_cache: dict[tuple[object, ...], float] = {}
    rows = {
        program: {str(budget): [] for budget in budgets}
        for program in reference["programs"]
    }
    for position, repetition in enumerate(repetition_indices, start=1):
        scenario_orders = {}
        for task, indices in task_groups.items():
            for policy in range(3):
                rng = random.Random(
                    stable_seed(seed, "scenario-order", repetition, task, policy)
                )
                scenario_orders[(task, policy)] = rng.sample(indices, len(indices))
        for program, settings in reference["programs"].items():
            program_rows = run_program(
                mode=str(settings["mode"]),
                preference_lambda=float(settings["preference_lambda"]),
                scenario_orders=scenario_orders,
                task_groups=task_groups,
                outcomes=outcomes,
                truth_rates=truth_rates,
                budgets=budgets,
                repetition=repetition,
                seed=seed,
                score_cache=score_cache,
            )
            for budget in budgets:
                rows[program][str(budget)].append(program_rows[str(budget)])
        if len(repetition_indices) > 1 and (position == 1 or position % 10 == 0):
            print(
                f"replayed {position}/{len(repetition_indices)} repetitions",
                file=sys.stderr,
            )
    return rows


def compare(
    actual: float,
    expected: float,
    *,
    label: str,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise AssertionError(
            f"{label}: observed {actual:.12g}, expected {expected:.12g}"
        )


def check_reference_rows(
    rows: dict[str, dict[str, list[dict[str, float]]]],
    reference: dict[str, object],
) -> int:
    count = 0
    for program, expected_budgets in reference["reference_rows"].items():
        for budget, expected_metrics in expected_budgets.items():
            observed = rows[program][budget][0]
            for metric in reference["metrics"]:
                compare(
                    float(observed[metric]),
                    float(expected_metrics[metric]),
                    label=f"{program}/{budget}/{metric}",
                    relative_tolerance=float(reference["relative_tolerance"]),
                    absolute_tolerance=float(reference["absolute_tolerance"]),
                )
                count += 1
    return count


def summarize(
    rows: dict[str, dict[str, list[dict[str, float]]]],
    reference: dict[str, object],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    by_budget = {}
    primary_aubc = {}
    metrics = tuple(str(metric) for metric in reference["metrics"])
    for program in reference["programs"]:
        by_budget[program] = {}
        for budget in reference["primary_budgets"]:
            key = str(budget)
            by_budget[program][key] = {
                metric: statistics.fmean(row[metric] for row in rows[program][key])
                for metric in metrics
            }
        primary_aubc[program] = {
            metric: statistics.fmean(
                by_budget[program][str(budget)][metric]
                for budget in reference["primary_budgets"]
            )
            for metric in metrics
        }
    return by_budget, primary_aubc


def check_full_aggregate(
    rows: dict[str, dict[str, list[dict[str, float]]]],
    reference: dict[str, object],
) -> int:
    by_budget, primary_aubc = summarize(rows, reference)
    count = 0
    for program in reference["programs"]:
        for budget in reference["primary_budgets"]:
            for metric in reference["metrics"]:
                compare(
                    by_budget[program][str(budget)][metric],
                    float(reference["reported_mean_by_budget"][program][str(budget)][metric]),
                    label=f"mean/{program}/{budget}/{metric}",
                    relative_tolerance=float(reference["relative_tolerance"]),
                    absolute_tolerance=float(reference["absolute_tolerance"]),
                )
                count += 1
        for metric in reference["metrics"]:
            compare(
                primary_aubc[program][metric],
                float(reference["reported_primary_aubc"][program][metric]),
                label=f"aubc/{program}/{metric}",
                relative_tolerance=float(reference["relative_tolerance"]),
                absolute_tolerance=float(reference["absolute_tolerance"]),
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="replay all 200 repetitions and check aggregate reported metrics",
    )
    args = parser.parse_args()
    tensor, reference = load_inputs()
    expected_repetitions = int(reference["repetitions"])
    if args.full:
        indices = list(range(expected_repetitions))
    else:
        indices = [int(reference["reference_repetition"])]
    rows = replay(
        tensor=tensor,
        reference=reference,
        repetition_indices=indices,
        budgets=tuple(
            int(value)
            for value in reference["primary_budgets"]
        ),
    )
    if args.full:
        comparisons = check_full_aggregate(rows, reference)
        status = "FULL RESULT REPRODUCTION PASS"
        check_name = "200-repetition aggregate"
    else:
        comparisons = check_reference_rows(rows, reference)
        status = "REFERENCE RESULT REPRODUCTION PASS"
        check_name = "deterministic reference repetition"
    print(
        json.dumps(
            {
                "status": status,
                "check": check_name,
                "repetitions": len(indices),
                "comparisons": comparisons,
                "relative_tolerance": reference["relative_tolerance"],
                "absolute_tolerance": reference["absolute_tolerance"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
