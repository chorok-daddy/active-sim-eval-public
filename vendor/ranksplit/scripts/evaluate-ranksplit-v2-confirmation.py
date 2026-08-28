"""Apply the frozen hierarchical RankSplit-v2 confirmation gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from ranksplit_v2_confirmation_contract import PROGRAM_ORDER, load_protocol


PRIMARY = "task_weak_order_exact_rate"
SAFEGUARDS = ("mean_task_gap_mae", "mean_decision_regret")
IDENTITY_METRICS = (
    "task_weak_order_exact_rate",
    "task_pair_relation_accuracy",
    "task_top_set_jaccard",
    "mean_task_gap_mae",
    "mean_decision_regret",
)
V2 = "ranksplit-v2:lambda-0.25"
V1 = "ranksplit-v1:lambda-0.25"
EIG = "source-eig"


def interval(values: list[float], critical: float) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("paired interval requires at least two repetitions")
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


def aubc(report, role: str, metric: str, repetition: int) -> float:
    return statistics.fmean(
        float(report["rows"][role][str(budget)][repetition][metric])
        for budget in report["primary_budgets"]
    )


def paired_deltas(report, candidate: str, comparator: str, metric: str):
    return [
        aubc(report, candidate, metric, repetition)
        - aubc(report, comparator, metric, repetition)
        for repetition in report["repetition_indices"]
    ]


def endpoint_identity(report) -> dict[str, object]:
    endpoint = str(max(map(int, report["budgets"])))
    mismatches = []
    for role in PROGRAM_ORDER[1:]:
        for repetition, (reference, observed) in enumerate(
            zip(report["rows"][PROGRAM_ORDER[0]][endpoint], report["rows"][role][endpoint])
        ):
            for metric in IDENTITY_METRICS:
                if reference[metric] != observed[metric]:
                    mismatches.append(
                        {
                            "role": role,
                            "repetition": repetition,
                            "metric": metric,
                            "reference": reference[metric],
                            "observed": observed[metric],
                        }
                    )
                    if len(mismatches) == 10:
                        return {"passes": False, "first_mismatches": mismatches}
    return {"passes": not mismatches, "first_mismatches": mismatches}


def validate_report(report, protocol) -> None:
    repetitions = int(protocol["replay"]["repetitions"])
    if (
        report.get("status") != "complete-ranksplit-v2-confirmation-replay"
        or report.get("evidence_class") != protocol["evidence_class"]
        or int(report.get("repetitions", -1)) != repetitions
        or report.get("repetition_indices") != list(range(repetitions))
        or tuple(row["role"] for row in report.get("programs", ())) != PROGRAM_ORDER
        or list(map(int, report.get("budgets", ())))
        != list(map(int, protocol["sampling_design"]["budgets"]))
        or list(map(int, report.get("primary_budgets", ())))
        != list(map(int, protocol["sampling_design"]["primary_budgets"]))
    ):
        raise ValueError("confirmation replay is incomplete or changed structure")
    for role in PROGRAM_ORDER:
        for budget in protocol["sampling_design"]["budgets"]:
            if len(report["rows"][role][str(budget)]) != repetitions:
                raise ValueError(f"incomplete rows for {role} at budget {budget}")


def evaluate(report, protocol) -> dict[str, object]:
    validate_report(report, protocol)
    gate = protocol["gate"]
    critical = float(gate["ordered_tests"]["one_sided_critical_value"])
    safeguard_critical = float(gate["safeguard_ci"]["one_sided_critical_value"])
    endpoint = endpoint_identity(report)
    primary = interval(paired_deltas(report, V2, V1, PRIMARY), critical)
    secondary = interval(paired_deltas(report, V2, EIG, PRIMARY), critical)
    safeguards = {
        metric: interval(
            paired_deltas(report, V2, EIG, metric), safeguard_critical
        )
        for metric in SAFEGUARDS
    }
    primary_pass = primary["ci_low"] > 0.0
    secondary_tested = primary_pass
    secondary_pass = secondary_tested and secondary["ci_low"] > 0.0
    safeguard_pass = all(
        safeguards[metric]["ci_high"]
        <= float(gate["safeguard_margins"][metric])
        for metric in SAFEGUARDS
    )
    if not endpoint["passes"]:
        status = "FAIL-INTEGRITY"
    elif not primary_pass:
        status = "FAIL-PRIMARY"
    elif not secondary_pass:
        status = "FAIL-END-TO-END"
    elif not safeguard_pass:
        status = "FAIL-SAFEGUARD"
    else:
        status = "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "evidence_class": protocol["evidence_class"],
        "primary_v2_vs_matched_v1": primary,
        "secondary_v2_vs_conventional_eig": {
            **secondary,
            "tested_by_hierarchy": secondary_tested,
        },
        "eig_relative_safeguards": safeguards,
        "endpoint_identity": endpoint,
        "checks": {
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "safeguards_pass": safeguard_pass,
            "endpoint_identity_pass": endpoint["passes"],
        },
        "manuscript_branch": protocol["result_branches"][status],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    report = json.loads(args.replay.resolve().read_text(encoding="utf-8"))
    result = evaluate(report, protocol)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
