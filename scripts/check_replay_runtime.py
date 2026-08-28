"""Check historical EIG rows and action traces without simulator dependencies.

This runs the unchanged EIG selector, not a replacement numerical formula.
Expected results are consulted only after all selected repetitions are computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

if __package__:
    from . import reproduce_results as replay
    from .fixed_preference import BetaPosterior, beta_bernoulli_information
else:
    import reproduce_results as replay
    from fixed_preference import BetaPosterior, beta_bernoulli_information


ROOT = Path(__file__).resolve().parents[1]
TENSOR_SHA256 = "0889b5fe291f7495c19d95ef99a53dbdbf4a34a8d1f6d9e73c36531a766d904c"
PROBE_REPETITIONS = (54, 99, 151)


class ReadLedger(dict):
    """Observe already-selected reads without changing their values or order."""

    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def compute(spec, tensor, repetition_indices):
    policies, task_groups, outcomes, truth = replay.prepare_tensor(tensor)
    task_for_scenario = {
        scenario: task for task, scenarios in task_groups.items() for scenario in scenarios
    }
    cache = {}
    result = {}
    for repetition in repetition_indices:
        orders = {}
        for task, scenarios in task_groups.items():
            for policy in range(len(policies)):
                rng = random.Random(replay.stable_seed(
                    spec["seed"], "scenario-order", repetition, task, policy
                ))
                orders[(task, policy)] = rng.sample(scenarios, len(scenarios))
        ledger = ReadLedger(outcomes)
        rows = replay.run_program(
            mode="conventional_eig", preference_lambda=0.0,
            scenario_orders=orders, task_groups=task_groups, outcomes=ledger,
            truth_rates=truth, budgets=tuple(spec["budgets"]),
            repetition=repetition, seed=spec["seed"], score_cache=cache,
        )
        if len(ledger.reads) != len(outcomes) or len(set(ledger.reads)) != len(outcomes):
            raise AssertionError("full endpoint did not reveal every cell exactly once")
        trace = [(task_for_scenario[scenario], policy)
                 for scenario, policy in ledger.reads[spec["initial_observations"]:]]
        result[repetition] = {
            "rows": rows,
            "action_trace_sha256": hashlib.sha256(json.dumps(
                trace, separators=(",", ":")
            ).encode()).hexdigest(),
        }
    return result


def compare(actual, reference):
    differences = []
    max_abs_error = 0.0
    indices = reference["repetition_indices"]
    for repetition, result in actual.items():
        position = indices.index(repetition)
        expected_trace = reference["action_trace_sha256"]["source-eig"][position]
        if result["action_trace_sha256"] != expected_trace:
            differences.append({"repetition": repetition, "field": "action_trace_sha256"})
        for budget, row in result["rows"].items():
            expected = reference["rows"]["source-eig"][budget][position]
            for metric in reference["scientific_metrics"]:
                error = abs(row[metric] - expected[metric])
                max_abs_error = max(max_abs_error, error)
                if error > 1e-12:
                    differences.append({"repetition": repetition, "budget": int(budget),
                                        "metric": metric, "actual": row[metric],
                                        "expected": expected[metric]})
    return differences, max_abs_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="check all 200 EIG repetitions")
    parser.add_argument("--output", type=Path, help="optional new JSON report path")
    args = parser.parse_args()
    spec = json.loads((ROOT / "data/run_specification.json").read_text())
    payload = (ROOT / "data/confirmation_tensor.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != TENSOR_SHA256:
        raise ValueError("paper tensor bytes changed")
    tensor = json.loads(payload)
    indices = list(range(spec["repetitions"])) if args.all else list(PROBE_REPETITIONS)
    started = time.perf_counter()
    actual = compute(spec, tensor, indices)
    reference = json.loads((ROOT / "data/primary_reference_rows.json").read_text())
    differences, max_error = compare(actual, reference)
    report = {
        "historical_eig_runtime_compatible": not differences,
        "python": sys.version, "platform": platform.platform(),
        "repetition_indices": indices, "budgets": spec["budgets"],
        "full_endpoint_replayed": True,
        "scope": "EIG only; not a complete RankSplit or full-paper replay",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scalar_fingerprint": {
            f"Beta({a},{b})": beta_bernoulli_information(BetaPosterior(a, b)).hex()
            for a, b in ((3, 7), (7, 3))
        },
        "max_metric_absolute_error": max_error,
        "difference_count": len(differences), "first_differences": differences[:20],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
