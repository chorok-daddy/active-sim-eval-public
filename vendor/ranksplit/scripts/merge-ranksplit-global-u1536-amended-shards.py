"""Merge and evaluate the frozen global-U=1536 amended replay shards.

This is a post-outcome, fail-closed evaluator.  The protocol owns the
comparators, budgets, metric, safeguards, and result branches; this module
only verifies complete deterministic shard coverage and applies those frozen
rules.  It never retunes a policy or selects a claim after observing results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any

from ranksplit_global_u1536_amended_comparator_contract import (
    PROGRAM_ORDER,
    load_protocol,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts" / "replay-ranksplit-global-u1536-amended.py"
REPLAY_SPEC = importlib.util.spec_from_file_location(
    "global_u1536_amended_replay_for_merge", REPLAY_SCRIPT
)
if REPLAY_SPEC is None or REPLAY_SPEC.loader is None:
    raise RuntimeError(f"cannot load amended replay: {REPLAY_SCRIPT}")
REPLAY = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(REPLAY)


PRIMARY = "ranksplit-v2:lambda-0.50"
SAAD = "saad-taskwise-d1:global-u1536"
SRANK = "srank-singleton"
EIG = "matched-eig"
PRIMARY_METRIC = "task_weak_order_exact_rate"
SAFEGUARDS = ("mean_task_gap_mae", "mean_decision_regret")
IDENTITY_METRICS = (
    "task_weak_order_exact_rate",
    "task_pair_relation_accuracy",
    "task_top_set_jaccard",
    "mean_task_gap_mae",
    "mean_decision_regret",
)
ROW_METRICS = IDENTITY_METRICS + ("runtime_seconds",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, message: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    _require(math.isfinite(number), message)
    return number


def _expected_programs(protocol: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "role": str(row["role"]),
            "family": str(row["family"]),
            "preference_lambda": float(row["preference_lambda"]),
        }
        for row in protocol["programs"]
    ]


def _expected_source(
    protocol_path: Path, tensor_path: Path
) -> dict[str, str]:
    return {
        "protocol_sha256": sha256(protocol_path),
        "tensor_sha256": sha256(tensor_path),
    }


def _validate_shard(
    shard: dict[str, object],
    *,
    expected_source: dict[str, str],
    expected_programs: list[dict[str, object]],
    budgets: list[int],
    primary_budgets: list[int],
    policy_order: tuple[str, ...],
    task_names: list[str] | None,
    evidence_class: str,
    total: int,
) -> tuple[int, int]:
    _require(
        shard.get("status") == "complete-shard-no-evaluation",
        "a shard is not a completed amended replay",
    )
    _require(shard.get("source") == expected_source, "shard source identity changed")
    _require(shard.get("evidence_class") == evidence_class, "shard evidence class changed")
    _require(
        int(shard.get("global_union_events", -1)) == 1536
        and shard.get("native_reference_available") is False,
        "shard comparator identity changed",
    )
    _require(shard.get("programs") == expected_programs, "shard program order changed")
    _require(list(map(int, shard.get("budgets", ()))) == budgets, "shard budgets changed")
    _require(
        list(map(int, shard.get("primary_budgets", ()))) == primary_budgets,
        "shard primary budgets changed",
    )
    _require(
        tuple(shard.get("policy_order", ())) == policy_order,
        "shard policy order changed",
    )
    if task_names is not None:
        _require(list(shard.get("task_names", ())) == task_names, "shard task order changed")

    shard_index = int(shard.get("shard_index", -1))
    shard_count = int(shard.get("shard_count", -1))
    _require(shard_count > 0 and 0 <= shard_index < shard_count, "invalid shard metadata")
    expected_indices = list(range(shard_index, total, shard_count))
    indices = list(map(int, shard.get("repetition_indices", ())))
    _require(indices == expected_indices, "shard repetition coverage is not canonical")
    _require(
        int(shard.get("repetitions", -1)) == len(indices),
        "shard repetition count changed",
    )

    rows = shard.get("rows")
    traces = shard.get("action_trace_sha256")
    _require(
        isinstance(rows, dict) and isinstance(traces, dict),
        "shard payload is missing rows or traces",
    )
    _require(set(rows) == set(PROGRAM_ORDER), "shard row program roles changed")
    _require(set(traces) == set(PROGRAM_ORDER), "shard trace program roles changed")
    for role in PROGRAM_ORDER:
        _require(
            set(rows[role]) == {str(budget) for budget in budgets},
            f"shard budget rows changed: {role}",
        )
        _require(
            len(traces[role]) == len(indices),
            f"shard trace coverage changed: {role}",
        )
        for budget in budgets:
            values = rows[role][str(budget)]
            _require(
                len(values) == len(indices),
                f"shard row coverage changed: {role}/{budget}",
            )
            for row in values:
                _require(
                    set(row) == set(ROW_METRICS),
                    f"shard metric schema changed: {role}/{budget}",
                )
                for metric in ROW_METRICS:
                    _finite_number(
                        row[metric],
                        f"non-finite row metric: {role}/{budget}/{metric}",
                    )
    return shard_index, shard_count


def merge_payloads(
    *,
    protocol_path: Path,
    protocol: dict[str, object],
    tensor_path: Path,
    tensor: dict[str, object],
    shard_payloads: list[dict[str, object]],
    shard_paths: list[Path],
) -> dict[str, object]:
    _require(shard_payloads, "at least one amended replay shard is required")
    total = int(protocol["replay"]["repetitions"])
    budgets = list(map(int, protocol["sampling_design"]["budgets"]))
    primary_budgets = list(map(int, protocol["sampling_design"]["primary_budgets"]))
    expected_programs = _expected_programs(protocol)
    policy_order = tuple(tensor["policy_order"])
    task_names = list(tensor["task_names"]) if "task_names" in tensor else None
    source = _expected_source(protocol_path, tensor_path)

    by_index: dict[int, tuple[dict[str, object], int]] = {}
    shard_counts: set[int] = set()
    for shard in shard_payloads:
        shard_index, shard_count = _validate_shard(
            shard,
            expected_source=source,
            expected_programs=expected_programs,
            budgets=budgets,
            primary_budgets=primary_budgets,
            policy_order=policy_order,
            task_names=task_names,
            evidence_class=str(protocol["evidence_class"]),
            total=total,
        )
        shard_counts.add(shard_count)
        for local_index, repetition in enumerate(shard["repetition_indices"]):
            repetition = int(repetition)
            _require(repetition not in by_index, "duplicate repetition across shards")
            by_index[repetition] = (shard, local_index)

    _require(
        set(by_index) == set(range(total)),
        "shards do not cover the frozen repetition set exactly",
    )
    _require(len(shard_counts) == 1, "shards use different shard counts")

    rows = {
        role: {
            str(budget): [
                by_index[index][0]["rows"][role][str(budget)][by_index[index][1]]
                for index in range(total)
            ]
            for budget in budgets
        }
        for role in PROGRAM_ORDER
    }
    traces = {
        role: [
            by_index[index][0]["action_trace_sha256"][role][by_index[index][1]]
            for index in range(total)
        ]
        for role in PROGRAM_ORDER
    }
    return {
        "schema_version": 1,
        "status": "complete-global-u1536-amended-comparator-replay",
        "evidence_class": protocol["evidence_class"],
        "repetitions": total,
        "repetition_indices": list(range(total)),
        "budgets": budgets,
        "primary_budgets": primary_budgets,
        "policy_order": list(policy_order),
        "task_names": task_names,
        "programs": expected_programs,
        "global_union_events": 1536,
        "native_reference_available": False,
        "rows": rows,
        "action_trace_sha256": traces,
        "source": {
            **source,
            "replay_source_sha256": sha256(REPLAY_SCRIPT),
            "shards": {path.name: sha256(path) for path in shard_paths},
        },
    }


def paired_interval(values: list[float], critical: float) -> dict[str, float | int]:
    _require(len(values) >= 2, "paired interval requires at least two repetitions")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    half_width = critical * standard_error
    return {
        "n": len(values),
        "mean_delta": mean,
        "standard_error": standard_error,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "critical_value": critical,
    }


def aubc(report: dict[str, object], role: str, metric: str, repetition: int) -> float:
    return statistics.fmean(
        float(report["rows"][role][str(budget)][repetition][metric])
        for budget in report["primary_budgets"]
    )


def paired_deltas(
    report: dict[str, object], candidate: str, comparator: str, metric: str
) -> list[float]:
    return [
        aubc(report, candidate, metric, repetition)
        - aubc(report, comparator, metric, repetition)
        for repetition in report["repetition_indices"]
    ]


def endpoint_identity(report: dict[str, object]) -> dict[str, object]:
    endpoint = str(max(map(int, report["budgets"])))
    mismatches: list[dict[str, object]] = []
    reference = PROGRAM_ORDER[0]
    for role in PROGRAM_ORDER[1:]:
        for repetition, (left, right) in enumerate(
            zip(report["rows"][reference][endpoint], report["rows"][role][endpoint])
        ):
            for metric in IDENTITY_METRICS:
                if left[metric] != right[metric]:
                    mismatches.append(
                        {
                            "role": role,
                            "repetition": repetition,
                            "metric": metric,
                            "reference": left[metric],
                            "observed": right[metric],
                        }
                    )
                    if len(mismatches) >= 10:
                        return {"passes": False, "first_mismatches": mismatches}
    return {"passes": not mismatches, "first_mismatches": mismatches}


def _mean_by_budget(report: dict[str, object], role: str, metric: str) -> dict[str, float]:
    return {
        str(budget): statistics.fmean(
            float(row[metric]) for row in report["rows"][role][str(budget)]
        )
        for budget in report["budgets"]
    }


def evaluate(report: dict[str, object], protocol: dict[str, object]) -> dict[str, object]:
    total = int(protocol["replay"]["repetitions"])
    _require(
        report.get("status") == "complete-global-u1536-amended-comparator-replay",
        "merged amended replay is incomplete",
    )
    _require(int(report.get("repetitions", -1)) == total, "merged repetition count changed")
    _require(
        report.get("repetition_indices") == list(range(total)),
        "merged repetition order changed",
    )
    _require(
        int(report.get("global_union_events", -1)) == 1536
        and report.get("native_reference_available") is False,
        "merged comparator identity changed",
    )
    _require(
        tuple(row["role"] for row in report["programs"]) == PROGRAM_ORDER,
        "merged program order changed",
    )

    gate = protocol["gate"]
    critical = float(gate["multiple_comparison"]["one_sided_critical_value"])
    primary = {
        comparator: paired_interval(
            paired_deltas(report, PRIMARY, comparator, PRIMARY_METRIC), critical
        )
        for comparator in (SAAD, SRANK)
    }
    safeguards = {
        metric: paired_interval(
            paired_deltas(report, PRIMARY, EIG, metric), 1.959963984540054
        )
        for metric in SAFEGUARDS
    }
    endpoint = endpoint_identity(report)
    primary_pass = {
        comparator: primary[comparator]["ci_low"] > 0.0
        for comparator in (SAAD, SRANK)
    }
    safeguards_pass = all(
        safeguards[metric]["ci_high"]
        <= float(gate["safeguard_margins"][metric])
        for metric in SAFEGUARDS
    )
    if not endpoint["passes"]:
        status = "FAIL-INTEGRITY"
    elif not safeguards_pass:
        status = "NEGATIVE"
    elif all(primary_pass.values()):
        status = "PASS"
    elif primary_pass[SAAD] and not primary_pass[SRANK]:
        status = "SAAD-ONLY"
    elif primary_pass[SRANK] and not primary_pass[SAAD]:
        status = "SRANK-ONLY"
    else:
        status = "NEGATIVE"

    summaries = {
        role: {
            "task_weak_order_exact_rate": _mean_by_budget(
                report, role, "task_weak_order_exact_rate"
            ),
            "mean_task_gap_mae": _mean_by_budget(report, role, "mean_task_gap_mae"),
            "mean_decision_regret": _mean_by_budget(
                report, role, "mean_decision_regret"
            ),
            "primary_aubc": statistics.fmean(
                statistics.fmean(
                    float(row[PRIMARY_METRIC])
                    for row in report["rows"][role][str(budget)]
                )
                for budget in report["primary_budgets"]
            ),
        }
        for role in PROGRAM_ORDER
    }
    return {
        "schema_version": 1,
        "status": status,
        "evidence_class": protocol["evidence_class"],
        "primary_contrasts": {
            "candidate": PRIMARY,
            "metric": PRIMARY_METRIC,
            "summary": "AUBC over budgets 48, 96, 192",
            "comparisons": primary,
            "passes": primary_pass,
        },
        "eig_relative_safeguards": {
            "candidate": PRIMARY,
            "comparator": EIG,
            "comparisons": safeguards,
            "margins": gate["safeguard_margins"],
            "passes": safeguards_pass,
        },
        "endpoint_identity": endpoint,
        "program_summaries": summaries,
        "checks": {
            "primary_comparator_passes": primary_pass,
            "safeguards_pass": safeguards_pass,
            "endpoint_identity_pass": endpoint["passes"],
        },
        "interpretation": {
            "result_branch": status,
            "prospective_evidence": True,
            "no_same_block_retuning": True,
            "claim_boundary": protocol["claim_boundary"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    tensor_path = args.tensor.resolve()
    protocol = load_protocol(protocol_path)
    tensor = REPLAY.validate_tensor(tensor_path, protocol_path, protocol)
    shard_paths = [path.resolve() for path in args.shards]
    shard_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    report = merge_payloads(
        protocol_path=protocol_path,
        protocol=protocol,
        tensor_path=tensor_path,
        tensor=tensor,
        shard_payloads=shard_payloads,
        shard_paths=shard_paths,
    )
    report["evaluation"] = evaluate(report, protocol)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
