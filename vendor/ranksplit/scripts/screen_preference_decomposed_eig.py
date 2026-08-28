"""Development-only screen for the frozen Preference-Decomposed EIG funnel.

This script replays already known finite policy outcomes.  It is intentionally
ineligible for paper claims or same-tensor confirmation.  Its job is to test
whether the exact information decomposition changes acquisition behavior and
to reduce the candidate set before a fresh holdout is frozen.
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
import time

from preference_decomposed_eig import (
    BetaPosterior,
    beta_bernoulli_information,
    pairwise_preference_probability,
    ranking_pd_eig,
    winner_pd_eig,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "calm-eig"
    / "2026-07-19-preference-decomposed-eig-development-screen.freeze.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifact"
    / "generated"
    / "2026-07-19-preference-decomposed-eig-development-screen.json"
)
METRICS = ("mean_task_gap_mae", "mean_decision_regret", "task_switches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(master: int, *labels: object) -> int:
    payload = "|".join(map(str, (master, *labels))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_freeze(path: Path) -> dict[str, object]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen-before-development-outcome-inspection":
        raise ValueError("development screen is not frozen")
    for relative_key, hash_key in (
        ("tensor", "tensor_sha256"),
        ("program_contract", "program_contract_sha256"),
        ("calculator", "calculator_sha256"),
        ("runner", "runner_sha256"),
    ):
        candidate = ROOT / str(freeze["sealed_inputs"][relative_key])
        if not candidate.is_file() or sha256(candidate) != freeze["sealed_inputs"][hash_key]:
            raise ValueError(f"sealed input mismatch: {relative_key}")
    if len(freeze["candidates"]) != 8:
        raise ValueError("the initial funnel must contain exactly eight candidate families")
    if list(map(float, freeze["preference_lambdas"])) != [0.25, 0.5, 0.75]:
        raise ValueError("the development lambda probes changed")
    return freeze


def prepare_tensor(freeze: dict[str, object]):
    tensor_path = ROOT / str(freeze["sealed_inputs"]["tensor"])
    tensor = json.loads(tensor_path.read_text(encoding="utf-8"))
    if tensor.get("status") != "pass" or len(tensor.get("policy_order", [])) != 3:
        raise ValueError("screen requires one complete three-policy tensor")
    policies = tuple(tensor["policy_order"])
    scenarios = list(tensor["scenario_records"])
    task_names = sorted({str(row["env_id"]) for row in scenarios})
    task_lookup = {name: index for index, name in enumerate(task_names)}
    task_groups = {index: [] for index in range(len(task_names))}
    outcomes: dict[tuple[int, int], int] = {}
    for scenario_index, scenario in enumerate(scenarios):
        task = task_lookup[str(scenario["env_id"])]
        task_groups[task].append(scenario_index)
        for policy, policy_name in enumerate(policies):
            values = list(map(int, scenario["policies"][policy_name]["environment_seed_outcomes"]))
            if len(values) != 1 or values[0] not in (0, 1):
                raise ValueError("each scenario-policy cell must contain one Boolean outcome")
            outcomes[(scenario_index, policy)] = values[0]
    pairs = tuple(itertools.combinations(range(3), 2))
    truth = {
        pair: {
            task: statistics.fmean(
                outcomes[(scenario, pair[0])] - outcomes[(scenario, pair[1])]
                for scenario in indices
            )
            for task, indices in task_groups.items()
        }
        for pair in pairs
    }
    return policies, task_names, task_groups, outcomes, pairs, truth


def beta_variance(state: BetaPosterior) -> float:
    total = state.alpha + state.beta
    return state.alpha * state.beta / (total * total * (total + 1.0))


def ambiguity_weight(states: tuple[BetaPosterior, ...], policy: int) -> float:
    values = []
    for competitor in range(len(states)):
        if competitor == policy:
            continue
        scale = math.sqrt(beta_variance(states[policy]) + beta_variance(states[competitor]))
        gap = abs(states[policy].mean - states[competitor].mean)
        values.append(scale / (gap + scale))
    return max(values)


def expected_pairwise_risk(states: tuple[BetaPosterior, ...]) -> float:
    risk = 0.0
    for first, second in itertools.combinations(range(len(states)), 2):
        probability = pairwise_preference_probability(states[first], states[second])
        risk += min(probability, 1.0 - probability)
    return risk


def direct_risk_reduction(states: tuple[BetaPosterior, ...], observed_policy: int) -> float:
    observed = states[observed_policy]
    success = list(states)
    failure = list(states)
    success[observed_policy] = observed.after(1)
    failure[observed_policy] = observed.after(0)
    posterior_risk = (
        observed.mean * expected_pairwise_risk(tuple(success))
        + (1.0 - observed.mean) * expected_pairwise_risk(tuple(failure))
    )
    reduction = expected_pairwise_risk(states) - posterior_risk
    if reduction < -1e-12:
        raise ArithmeticError(f"negative expected risk reduction: {reduction}")
    return max(0.0, reduction)


def smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def acquisition_score(
    family: str,
    *,
    states: tuple[BetaPosterior, ...],
    policy: int,
    preference_lambda: float,
    progress: float,
    switching: int,
    ratio_switch_penalty: float,
    lagrangian_switch_nats: float,
) -> float:
    if family == "B0":
        raw = beta_bernoulli_information(states[policy])
        weight = (1.0 - preference_lambda) + preference_lambda * ambiguity_weight(states, policy)
        return raw * weight
    if family == "B1-P":
        return ranking_pd_eig(
            states, observed_policy=policy, preference_lambda=preference_lambda
        ).score
    if family == "B1-W":
        return winner_pd_eig(
            states, observed_policy=policy, preference_lambda=preference_lambda
        ).score
    if family == "B2-L":
        effective = preference_lambda * progress
        return ranking_pd_eig(states, observed_policy=policy, preference_lambda=effective).score
    if family == "B2-S":
        effective = preference_lambda * smoothstep(progress)
        return ranking_pd_eig(states, observed_policy=policy, preference_lambda=effective).score
    if family in ("B5-R", "B5-L"):
        base = ranking_pd_eig(
            states, observed_policy=policy, preference_lambda=preference_lambda
        ).score
        if family == "B5-R":
            return base / (1.0 + ratio_switch_penalty * switching)
        return base - lagrangian_switch_nats * switching
    if family == "R4":
        return direct_risk_reduction(states, policy)
    raise ValueError(f"unknown candidate family: {family}")


def pair_metrics(observed, truth, pair, tasks) -> dict[str, float]:
    gaps = {
        task: statistics.fmean(observed[(task, pair[0])])
        - statistics.fmean(observed[(task, pair[1])])
        for task in tasks
    }
    errors = {task: abs(gaps[task] - truth[task]) for task in tasks}
    correct = {
        task: (gaps[task] > 0) - (gaps[task] < 0)
        == (truth[task] > 0) - (truth[task] < 0)
        for task in tasks
    }
    regret = {
        task: 0.0
        if correct[task]
        else abs(truth[task]) * (0.5 if gaps[task] == 0 else 1.0)
        for task in tasks
    }
    return {
        "mean_task_gap_mae": statistics.fmean(errors.values()),
        "mean_decision_regret": statistics.fmean(regret.values()),
    }


def program_specs(freeze: dict[str, object]) -> list[tuple[str, str, float]]:
    specs = []
    for candidate in freeze["candidates"]:
        family = str(candidate["id"])
        if family == "R4":
            specs.append((family, family, 0.5))
        else:
            for preference_lambda in map(float, freeze["preference_lambdas"]):
                specs.append(
                    (f"{family}:lambda-{preference_lambda:.2f}", family, preference_lambda)
                )
    return specs


def run_program(
    *,
    role: str,
    family: str,
    preference_lambda: float,
    scenario_orders,
    task_groups,
    outcomes,
    pairs,
    truth,
    budgets,
    repetition: int,
    freeze: dict[str, object],
):
    observed = {(task, policy): [] for task in task_groups for policy in range(3)}
    positions = {(task, policy): 0 for task in task_groups for policy in range(3)}
    alpha = [[1 for _ in task_groups] for _ in range(3)]
    beta = [[1 for _ in task_groups] for _ in range(3)]
    for task in task_groups:
        for policy in range(3):
            scenario = scenario_orders[(task, policy)][0]
            outcome = outcomes[(scenario, policy)]
            observed[(task, policy)].append(outcome)
            positions[(task, policy)] = 1
            alpha[policy][task] += outcome
            beta[policy][task] += 1 - outcome
    acquired = sum(map(len, observed.values()))
    initial_count = acquired
    maximum_budget = max(budgets)
    current_task = 0
    switches = 0
    trace = []
    rows = {}
    runtime_seconds = 0.0
    while True:
        if acquired in budgets:
            by_pair = [pair_metrics(observed, truth[pair], pair, task_groups) for pair in pairs]
            rows[str(acquired)] = {
                "mean_task_gap_mae": statistics.fmean(row["mean_task_gap_mae"] for row in by_pair),
                "mean_decision_regret": statistics.fmean(
                    row["mean_decision_regret"] for row in by_pair
                ),
                "task_switches": float(switches),
                "runtime_seconds": runtime_seconds,
            }
        if acquired >= maximum_budget:
            break
        progress = (acquired - initial_count) / (maximum_budget - initial_count)
        candidates = []
        start = time.perf_counter()
        for task in task_groups:
            states = tuple(BetaPosterior(alpha[policy][task], beta[policy][task]) for policy in range(3))
            for policy in range(3):
                if positions[(task, policy)] >= len(scenario_orders[(task, policy)]):
                    continue
                switching = int(task != current_task)
                score = acquisition_score(
                    family,
                    states=states,
                    policy=policy,
                    preference_lambda=preference_lambda,
                    progress=progress,
                    switching=switching,
                    ratio_switch_penalty=float(freeze["cost_operators"]["ratio_switch_penalty"]),
                    lagrangian_switch_nats=float(
                        freeze["cost_operators"]["lagrangian_switch_nats"]
                    ),
                )
                tie = stable_seed(
                    int(freeze["replay"]["simulation_seed"]),
                    "tie",
                    repetition,
                    acquired,
                    task,
                    policy,
                )
                candidates.append((score, tie, task, policy))
        runtime_seconds += time.perf_counter() - start
        if not candidates:
            raise RuntimeError("candidate set exhausted before maximum budget")
        _, _, task, policy = max(candidates)
        scenario = scenario_orders[(task, policy)][positions[(task, policy)]]
        positions[(task, policy)] += 1
        outcome = outcomes[(scenario, policy)]
        observed[(task, policy)].append(outcome)
        alpha[policy][task] += outcome
        beta[policy][task] += 1 - outcome
        switches += int(task != current_task)
        current_task = task
        acquired += 1
        trace.append((task, policy))
    trace_hash = hashlib.sha256(
        json.dumps(trace, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rows, trace, trace_hash


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(left[metric] <= right[metric] for metric in METRICS) and any(
        left[metric] < right[metric] for metric in METRICS
    )


def summarize(rows, traces, specs, budgets):
    summary = {}
    for role, _, _ in specs:
        summary[role] = {}
        for budget in budgets:
            records = rows[role][str(budget)]
            summary[role][str(budget)] = {
                key: statistics.fmean(float(record[key]) for record in records)
                for key in (*METRICS, "runtime_seconds")
            }
    pareto = {}
    for budget in budgets:
        points = {role: summary[role][str(budget)] for role, _, _ in specs}
        pareto[str(budget)] = sorted(
            role
            for role, point in points.items()
            if not any(other != role and dominates(points[other], point) for other in points)
        )
    divergence = {}
    for role, family, preference_lambda in specs:
        if family == "R4":
            baseline = "B0:lambda-0.50"
        else:
            baseline = f"B0:lambda-{preference_lambda:.2f}"
        rates = []
        for trace, reference in zip(traces[role], traces[baseline]):
            rates.append(
                statistics.fmean(left != right for left, right in zip(trace, reference))
            )
        divergence[role] = statistics.fmean(rates)
    return summary, pareto, divergence


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    freeze = load_freeze(args.config)
    policies, task_names, task_groups, outcomes, pairs, truth = prepare_tensor(freeze)
    specs = program_specs(freeze)
    budgets = list(map(int, freeze["replay"]["budgets"]))
    rows = {role: {str(budget): [] for budget in budgets} for role, _, _ in specs}
    traces = {role: [] for role, _, _ in specs}
    trace_hashes = {role: [] for role, _, _ in specs}
    master = int(freeze["replay"]["simulation_seed"])
    for repetition in range(int(freeze["replay"]["repetitions"])):
        scenario_orders = {}
        for task, indices in task_groups.items():
            for policy in range(3):
                rng = random.Random(stable_seed(master, "scenario-order", repetition, task, policy))
                scenario_orders[(task, policy)] = rng.sample(indices, len(indices))
        for role, family, preference_lambda in specs:
            program_rows, trace, trace_hash = run_program(
                role=role,
                family=family,
                preference_lambda=preference_lambda,
                scenario_orders=scenario_orders,
                task_groups=task_groups,
                outcomes=outcomes,
                pairs=pairs,
                truth=truth,
                budgets=budgets,
                repetition=repetition,
                freeze=freeze,
            )
            for budget in budgets:
                rows[role][str(budget)].append(program_rows[str(budget)])
            traces[role].append(trace)
            trace_hashes[role].append(trace_hash)
            if (
                time.perf_counter() - started
                > float(freeze["replay"]["maximum_wall_clock_seconds"])
            ):
                raise TimeoutError("development screen exceeded the frozen wall-clock limit")
    summary, pareto, divergence = summarize(rows, traces, specs, budgets)
    report = {
        "schema_version": 1,
        "status": "complete-development-only-not-confirmation",
        "evidence_class": freeze["evidence_class"],
        "claim_boundary": freeze["claim_boundary"],
        "source": {
            "freeze": str(args.config.relative_to(ROOT)).replace("\\", "/"),
            "freeze_sha256": sha256(args.config),
            "tensor_sha256": freeze["sealed_inputs"]["tensor_sha256"],
        },
        "policies": list(policies),
        "tasks": task_names,
        "budgets": budgets,
        "repetitions": int(freeze["replay"]["repetitions"]),
        "programs": [
            {"role": role, "family": family, "preference_lambda": preference_lambda}
            for role, family, preference_lambda in specs
        ],
        "rows": rows,
        "summary": summary,
        "pareto_roles": pareto,
        "action_divergence_from_matched_B0": divergence,
        "action_trace_sha256": trace_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "programs": len(specs),
                "repetitions": report["repetitions"],
                "budgets": budgets,
                "pareto_roles": pareto,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
