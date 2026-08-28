"""Replay the pre-registered global-U=1536 comparator study.

This is a separate replay path.  It shares only the benchmark estimator,
RankSplit implementation, and policy tensor format with the earlier study;
the amended Saad path uses the independent global-U=1536 allocator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from preference_decomposed_eig import BetaPosterior
import ranksplit_v2
from ranksplit_established_ranking_allocators import SRankSingletonState
from ranksplit_global_u1536_amended_comparator_contract import (
    POLICY_ORDER,
    PROGRAM_ORDER,
    load_protocol,
    sha256,
)
from ranksplit_global_u1536_saad_allocator import GlobalU1536SaadState, stable_seed
from ranksplit_ranking_metrics import evaluate
import screen_preference_decomposed_eig as BASE


def validate_tensor(path: Path, protocol_path: Path, protocol: dict[str, object]):
    tensor = json.loads(path.read_text(encoding="utf-8"))
    if (
        tensor.get("status") != "pass"
        or tuple(tensor.get("policy_order", ())) != POLICY_ORDER
        or int(tensor.get("global_union_events", -1)) != 1536
        or tensor.get("native_reference_available") is not False
    ):
        raise ValueError("amended tensor is incomplete or comparator identity changed")
    if int(tensor.get("scenario_count", -1)) != int(protocol["scenario_design"]["scenario_count"]):
        raise ValueError("amended tensor scenario count changed")
    sources = tensor.get("source_sha256", {})
    if (
        sources.get("frozen_protocol") != sha256(protocol_path)
        or sources.get("scenario_manifest")
        != protocol["sealed_inputs"]["scenario_manifest"]["sha256"]
    ):
        raise ValueError("amended tensor source identity changed")
    manifest_path = Path(protocol["sealed_inputs"]["scenario_manifest"]["path"])
    manifest = json.loads((Path(__file__).resolve().parents[1] / manifest_path).read_text(encoding="utf-8"))
    expected = {
        str(row["scenario_id"]): (row["env_id"], row["transform_id"], row["descriptor_sha256"])
        for row in manifest["scenarios"]
    }
    records = tensor.get("scenario_records", [])
    if len(records) != len(expected) or len(records) != len({row["scenario_id"] for row in records}):
        raise ValueError("amended tensor contains duplicate or incomplete scenarios")
    for row in records:
        scenario_id = str(row["scenario_id"])
        if scenario_id not in expected:
            raise ValueError("amended tensor contains an unexpected scenario")
        if (row.get("env_id"), row.get("transform_id"), row.get("descriptor_sha256")) != expected[scenario_id]:
            raise ValueError("amended tensor descriptor identity changed")
        if tuple(row["policies"]) != POLICY_ORDER:
            raise ValueError("amended tensor row policy order changed")
        for policy in POLICY_ORDER:
            if row["policies"][policy]["outcome"] not in (0, 1):
                raise ValueError("each amended scenario-policy cell requires one Boolean outcome")
    return tensor


def prepare_tensor(tensor: dict[str, object]):
    policies = tuple(tensor["policy_order"])
    scenarios = list(tensor["scenario_records"])
    tasks = sorted({str(row["env_id"]) for row in scenarios})
    task_lookup = {name: index for index, name in enumerate(tasks)}
    task_groups = {index: [] for index in range(len(tasks))}
    outcomes: dict[tuple[int, int], int] = {}
    for scenario_index, scenario in enumerate(scenarios):
        task = task_lookup[str(scenario["env_id"])]
        task_groups[task].append(scenario_index)
        for policy, name in enumerate(policies):
            outcomes[(scenario_index, policy)] = int(scenario["policies"][name]["outcome"])
    truth_rates = tuple(
        tuple(
            statistics.fmean(outcomes[(scenario, policy)] for scenario in task_groups[task])
            for policy in range(len(policies))
        )
        for task in task_groups
    )
    return policies, tasks, task_groups, outcomes, truth_rates


def program_specs(protocol: dict[str, object]):
    specs = tuple(
        (str(row["role"]), str(row["family"]), float(row["preference_lambda"]))
        for row in protocol["programs"]
    )
    if tuple(role for role, _, _ in specs) != PROGRAM_ORDER:
        raise ValueError("amended program order differs from the frozen contract")
    return specs


def scenario_orders_for_repetition(
    *,
    master_seed: int,
    repetition: int,
    task_groups: dict[int, list[int]],
    policy_count: int,
) -> dict[tuple[int, int], list[int]]:
    orders: dict[tuple[int, int], list[int]] = {}
    for task, indices in task_groups.items():
        for policy in range(policy_count):
            rng = random.Random(
                stable_seed(master_seed, "scenario-order", repetition, task, policy)
            )
            orders[(task, policy)] = rng.sample(indices, len(indices))
    return orders


def choose_outer_task(
    *,
    positions: dict[tuple[int, int], int],
    scenario_orders: dict[tuple[int, int], list[int]],
    task_groups: dict[int, list[int]],
    repetition: int,
    global_step: int,
    master_seed: int,
) -> int:
    available_tasks = [
        task
        for task in task_groups
        if any(
            positions[(task, policy)] < len(scenario_orders[(task, policy)])
            for policy in range(3)
        )
    ]
    if not available_tasks:
        raise RuntimeError("outer scheduler has no available task")
    cell_counts = {
        task: sum(positions[(task, policy)] for policy in range(3))
        for task in available_tasks
    }
    minimum = min(cell_counts.values())
    tied = [task for task in available_tasks if cell_counts[task] == minimum]
    return max(
        tied,
        key=lambda task: stable_seed(
            master_seed, "outer-task-tie", repetition, global_step, task
        ),
    )


def choose_scored_policy(
    *,
    family: str,
    preference_lambda: float,
    states: tuple[BetaPosterior, ...],
    positions: dict[tuple[int, int], int],
    scenario_orders: dict[tuple[int, int], list[int]],
    task: int,
    repetition: int,
    global_step: int,
    master_seed: int,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for policy, state in enumerate(states):
        if positions[(task, policy)] >= len(scenario_orders[(task, policy)]):
            continue
        if family == "matched-eig":
            score = BASE.acquisition_score(
                "B0",
                states=states,
                policy=policy,
                preference_lambda=0.0,
                progress=0.0,
                switching=0,
                ratio_switch_penalty=0.0,
                lagrangian_switch_nats=0.0,
            )
        elif family == "ranksplit-v2":
            score = ranksplit_v2.score(
                "clarity-continuation",
                states,
                observed_policy=policy,
                preference_lambda=preference_lambda,
            )
        else:
            raise ValueError(f"unknown scored family: {family}")
        tie = stable_seed(
            master_seed,
            "inner-policy-tie",
            repetition,
            global_step,
            task,
            policy,
        )
        candidates.append((float(score), tie, policy))
    if not candidates:
        raise RuntimeError("within-task scored selector has no available policy")
    return max(candidates)[2]


def run_program(
    *,
    family: str,
    preference_lambda: float,
    scenario_orders: dict[tuple[int, int], list[int]],
    task_groups: dict[int, list[int]],
    outcomes: dict[tuple[int, int], int],
    truth_rates: tuple[tuple[float, ...], ...],
    budgets: list[int],
    repetition: int,
    protocol: dict[str, object],
):
    policy_count = len(truth_rates[0])
    observed = {(task, policy): [] for task in task_groups for policy in range(policy_count)}
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

    capacities = {
        task: tuple(len(scenario_orders[(task, policy)]) for policy in range(policy_count))
        for task in task_groups
    }
    master_seed = int(protocol["replay"]["simulation_seed"])
    allocator_states: dict[int, object] = {}
    if family == "saad-taskwise-d1-global-u1536":
        config = protocol["allocators"]["saad_taskwise_d1_global_u1536"]
        for task in task_groups:
            allocator_states[task] = GlobalU1536SaadState(
                task=task,
                repetition=repetition,
                master_seed=master_seed,
                capacities=capacities[task],
                counts=[1] * policy_count,
                successes=[observed[(task, policy)][0] for policy in range(policy_count)],
                delta=float(config["delta"]),
                global_union_events=int(config["global_union_events"]),
            )
    elif family == "srank-singleton":
        config = protocol["allocators"]["srank_singleton"]
        for task in task_groups:
            allocator_states[task] = SRankSingletonState(
                task=task,
                repetition=repetition,
                master_seed=master_seed,
                capacities=capacities[task],
                counts=[1] * policy_count,
                successes=[observed[(task, policy)][0] for policy in range(policy_count)],
                horizon=int(config["horizon_per_task"]),
                rounds=int(config["rounds"]),
            )

    acquired = sum(map(len, observed.values()))
    maximum_budget = max(budgets)
    rows: dict[str, dict[str, float]] = {}
    trace: list[tuple[int, int]] = []
    runtime_seconds = 0.0
    while True:
        if acquired in budgets:
            row = evaluate(observed, truth_rates)
            row.update(runtime_seconds=runtime_seconds)
            rows[str(acquired)] = row
        if acquired >= maximum_budget:
            break
        task = choose_outer_task(
            positions=positions,
            scenario_orders=scenario_orders,
            task_groups=task_groups,
            repetition=repetition,
            global_step=acquired,
            master_seed=master_seed,
        )
        states = tuple(
            BetaPosterior(alpha[policy][task], beta[policy][task])
            for policy in range(policy_count)
        )
        started = time.perf_counter()
        if family in ("saad-taskwise-d1-global-u1536", "srank-singleton"):
            policy = allocator_states[task].choose_policy(global_step=acquired)  # type: ignore[union-attr]
        else:
            policy = choose_scored_policy(
                family=family,
                preference_lambda=preference_lambda,
                states=states,
                positions=positions,
                scenario_orders=scenario_orders,
                task=task,
                repetition=repetition,
                global_step=acquired,
                master_seed=master_seed,
            )
        runtime_seconds += time.perf_counter() - started
        position = positions[(task, policy)]
        if position >= len(scenario_orders[(task, policy)]):
            raise RuntimeError("selector returned an exhausted policy")
        scenario = scenario_orders[(task, policy)][position]
        positions[(task, policy)] += 1
        outcome = outcomes[(scenario, policy)]
        observed[(task, policy)].append(outcome)
        alpha[policy][task] += outcome
        beta[policy][task] += 1 - outcome
        if family in ("saad-taskwise-d1-global-u1536", "srank-singleton"):
            allocator_states[task].update(policy, outcome)  # type: ignore[union-attr]
        acquired += 1
        trace.append((task, policy))

    if acquired != sum(sum(capacity) for capacity in capacities.values()):
        raise RuntimeError("amended replay did not reach the complete endpoint")
    return rows, hashlib.sha256(
        json.dumps(trace, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def replay(
    protocol: dict[str, object], tensor: dict[str, object], repetition_indices: list[int]
) -> dict[str, object]:
    policies, tasks, task_groups, outcomes, truth_rates = prepare_tensor(tensor)
    specs = program_specs(protocol)
    budgets = list(map(int, protocol["sampling_design"]["budgets"]))
    rows = {role: {str(budget): [] for budget in budgets} for role, _, _ in specs}
    trace_hashes = {role: [] for role, _, _ in specs}
    for repetition in repetition_indices:
        scenario_orders = scenario_orders_for_repetition(
            master_seed=int(protocol["replay"]["simulation_seed"]),
            repetition=repetition,
            task_groups=task_groups,
            policy_count=len(policies),
        )
        for role, family, preference_lambda in specs:
            program_rows, trace_hash = run_program(
                family=family,
                preference_lambda=preference_lambda,
                scenario_orders=scenario_orders,
                task_groups=task_groups,
                outcomes=outcomes,
                truth_rates=truth_rates,
                budgets=budgets,
                repetition=repetition,
                protocol=protocol,
            )
            for budget in budgets:
                rows[role][str(budget)].append(program_rows[str(budget)])
            trace_hashes[role].append(trace_hash)
    return {
        "status": "complete-shard-no-evaluation",
        "repetitions": len(repetition_indices),
        "repetition_indices": list(repetition_indices),
        "budgets": budgets,
        "primary_budgets": list(map(int, protocol["sampling_design"]["primary_budgets"])),
        "policy_order": list(policies),
        "task_names": tasks,
        "programs": [
            {"role": role, "family": family, "preference_lambda": value}
            for role, family, value in specs
        ],
        "global_union_events": 1536,
        "native_reference_available": False,
        "rows": rows,
        "action_trace_sha256": trace_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    tensor = validate_tensor(args.tensor, args.protocol, protocol)
    total = int(protocol["replay"]["repetitions"])
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard specification")
    indices = list(range(args.shard_index, total, args.shard_count))
    report = replay(protocol, tensor, indices)
    report.update(
        evidence_class=protocol["evidence_class"],
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        source={
            "protocol_sha256": sha256(args.protocol),
            "tensor_sha256": sha256(args.tensor),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "repetitions": len(indices)}))


if __name__ == "__main__":
    main()
