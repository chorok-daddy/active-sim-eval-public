"""Study-specific replay adapters and independent, repetition-level checks.

The numerical kernels under vendor/ are unchanged copies of the research
source. Expected results are read only by the checking/aggregation functions,
never by load_study(), run_case(), or an allocator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from fractions import Fraction

if __package__:
    from . import reproduce_results as primary
    from .fixed_preference import BetaPosterior, beta_bernoulli_information
else:
    import reproduce_results as primary
    from fixed_preference import BetaPosterior, beta_bernoulli_information

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/ranksplit/scripts"
METRICS = (
    "task_weak_order_exact_rate", "task_pair_relation_accuracy",
    "task_top_set_jaccard", "mean_task_gap_mae", "mean_decision_regret",
)
STUDIES = ("mechanism", "preference", "ranking")
_MODULES = {}
_SCORE_CACHE = {}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_json(entry, root=ROOT):
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("input path escapes source package")
    payload = path.read_bytes()
    if digest(payload) != entry["sha256"]:
        raise ValueError(f"source/data hash mismatch: {entry['path']}")
    return json.loads(payload)


def manifest(root=ROOT):
    return json.loads((root / "data/paper/manifest.json").read_text(encoding="utf-8"))


def frozen_module(filename):
    if filename not in _MODULES:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        name = "paper_original_" + filename.replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(name, VENDOR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[filename] = module
    return _MODULES[filename]


def runtime_identity():
    return {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(),
        "eig_fingerprint": {
            f"Beta({a},{b})": beta_bernoulli_information(BetaPosterior(a, b)).hex()
            for a, b in ((3, 7), (7, 3))
        },
    }


def validate_tensor(tensor, spec, scenarios):
    """Validate before the historical preparer coerces outcomes to int."""
    records = tensor.get("scenario_records", [])
    expected = scenarios["scenarios"]
    policies = spec["policy_order"]
    if tensor.get("status") != "pass" or tensor.get("policy_order") != policies:
        raise ValueError("incomplete tensor or changed policy order")
    if (type(tensor.get("scenario_count")) is not int
            or tensor["scenario_count"] != len(expected) or len(records) != len(expected)):
        raise ValueError("scenario coverage mismatch")
    # Row order is part of the random-permutation input, not just a set of IDs.
    for row, original in zip(records, expected, strict=True):
        for field in ("scenario_id", "env_id", "transform_id", "descriptor_sha256"):
            if row.get(field) != original[field]:
                raise ValueError(f"scenario order/descriptor mismatch: {field}")
        if list(row.get("policies", {})) != policies:
            raise ValueError("row policy order mismatch")
        for policy in policies:
            value = row["policies"][policy].get("outcome")
            if type(value) is not int or value not in (0, 1):
                raise ValueError("outcomes must be integer 0/1, not rounded or coerced")
    if len({r["scenario_id"] for r in records}) != len(records):
        raise ValueError("duplicate scenario identifier")


def load_study(study, custom_tensor=None):
    if study not in STUDIES:
        raise ValueError("unknown study")
    book = manifest()
    entry = book["studies"][study]
    for path, expected in book["frozen_files"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"historical source changed: {path}")
    spec = checked_json(entry["spec"])
    scenarios = checked_json(entry["scenarios"])
    if custom_tensor is None:
        tensor = checked_json(entry["tensor"])
        tensor_hash = entry["tensor"]["sha256"]
    else:
        payload = Path(custom_tensor).read_bytes()
        tensor_hash, tensor = digest(payload), json.loads(payload)
        provenance = tensor.get("acquisition", {})
        if provenance.get("scenario_manifest_sha256") != entry["scenarios"]["sha256"]:
            raise ValueError("custom tensor lacks matching acquisition provenance")
        acquisition_spec = provenance.get("spec", {})
        if provenance.get("spec_sha256") != digest(encoded(acquisition_spec)):
            raise ValueError("custom acquisition specification digest mismatch")
        if not provenance.get("raw_sha256") or len(provenance["raw_sha256"]) != len(spec["policy_order"]):
            raise ValueError("custom tensor lacks per-policy raw-source identities")
    validate_tensor(tensor, spec, scenarios)
    paths = ["scripts/paper_reproduction.py", "scripts/reproduce_results.py",
             "scripts/fixed_preference.py", "scripts/ranksplit.py"]
    identity = {
        "study": study, "spec_sha256": entry["spec"]["sha256"],
        "tensor_sha256": tensor_hash,
        "scenarios_sha256": entry["scenarios"]["sha256"],
        "source_sha256": digest(encoded({
            **book["frozen_files"],
            **{p: digest((ROOT / p).read_bytes()) for p in paths},
        })),
        "custom_tensor": custom_tensor is not None, "runtime": runtime_identity(),
    }
    return {"spec": spec, "tensor": tensor, "identity": identity}


class ReadLedger(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


class PrefixComplete(Exception):
    """Stop only after the original evaluator has emitted the requested row."""


def run_program(study, program, spec, groups, outcomes, truth, orders, repetition, through):
    ledger = ReadLedger(outcomes)
    rows = {}
    module = (frozen_module("replay-ranksplit-global-u1536-amended.py")
              if study == "ranking" else primary)
    original_evaluate = module.evaluate
    endpoint = len(outcomes)

    def record(observed, target):
        row = original_evaluate(observed, target)
        acquired = sum(map(len, observed.values()))
        rows[str(acquired)] = {m: row[m] for m in METRICS}
        if acquired == through and through < endpoint:
            raise PrefixComplete()
        return row

    module.evaluate = record
    try:
        common = dict(
            preference_lambda=float(program["preference_lambda"]),
            scenario_orders=orders, task_groups=groups, outcomes=ledger,
            truth_rates=truth, budgets=spec["sampling_design"]["budgets"],
            repetition=repetition,
        )
        if study == "ranking":
            # In particular, do not replace SRank's horizon by the prefix budget.
            module.run_program(family=program["family"], protocol=spec, **common)
        else:
            mode = {"source-eig": "conventional_eig", "ranksplit": "fixed_preference",
                    "clarity-continuation": "ranksplit"}[program["family"]]
            module.run_program(mode=mode, seed=spec["replay"]["simulation_seed"],
                               score_cache=_SCORE_CACHE, **common)
    except PrefixComplete:
        pass
    finally:
        module.evaluate = original_evaluate
    if len(ledger.reads) != through or len(set(ledger.reads)) != through:
        raise AssertionError("replay coverage/duplicate-read failure")
    expected_budgets = [str(b) for b in spec["sampling_design"]["budgets"] if b <= through]
    if list(rows) != expected_budgets:
        raise AssertionError("missing or extra checkpoint row")
    if through == endpoint:
        end = rows[str(endpoint)]
        if set(ledger.reads) != set(outcomes) or any(end[m] != 1.0 for m in METRICS[:3]) or any(end[m] != 0.0 for m in METRICS[3:]):
            raise AssertionError("independent full-endpoint oracle failed")
    task_for = {s: task for task, scenarios in groups.items() for s in scenarios}
    census = len(groups) * len(truth[0])
    trace = [(task_for[s], p) for s, p in ledger.reads[census:]]
    return {
        "rows": rows, "reads": len(ledger.reads), "unique_reads": len(set(ledger.reads)),
        "revealed_cells": ledger.reads,
        "action_trace_sha256": digest(json.dumps(trace, separators=(",", ":")).encode()),
        "trace_budget": through, "full_endpoint_replayed": through == endpoint,
    }


def observation_oracle(cases, inputs):
    """Count revealed outcomes independently, using rational arithmetic."""
    policies, groups, outcomes, _ = primary.prepare_tensor(inputs["tensor"])
    task_for = {s: t for t, scenarios in groups.items() for s in scenarios}
    truth = [[Fraction(sum(outcomes[s, p] for s in groups[t]), len(groups[t]))
              for p in range(len(policies))] for t in groups]
    pairs = [(a, b) for a in range(len(policies)) for b in range(a+1, len(policies))]
    checks = 0
    for case in cases:
        for result in case["programs"].values():
            cells = [tuple(cell) for cell in result["revealed_cells"]]
            if len(cells) != case["through"] or len(set(cells)) != len(cells) or any(cell not in outcomes for cell in cells):
                raise ValueError("revealed-cell ledger is incomplete/invalid")
            census = [(groups[t][0], p) for t in groups for p in range(len(policies))]
            expected_orders = {}
            for t, scenarios in groups.items():
                for p in range(len(policies)):
                    rng = random.Random(primary.stable_seed(inputs["spec"]["replay"]["simulation_seed"],
                        "scenario-order", case["repetition"], t, p))
                    expected_orders[t, p] = rng.sample(scenarios, len(scenarios))
            expected_census = [(expected_orders[t, p][0], p) for t in groups for p in range(len(policies))]
            if cells[:len(census)] != expected_census:
                raise ValueError("changed census")
            counts = {key: 0 for key in expected_orders}
            for s, p in cells:
                key = (task_for[s], p)
                if s != expected_orders[key][counts[key]]:
                    raise ValueError("changed scenario permutation")
                counts[key] += 1
            trace = [(task_for[s], p) for s, p in cells[len(census):]]
            if digest(json.dumps(trace, separators=(",", ":")).encode()) != result["action_trace_sha256"]:
                raise ValueError("action digest does not describe actual revealed cells")
            for b, actual in result["rows"].items():
                observed = {(t, p): [] for t in groups for p in range(len(policies))}
                for s, p in cells[:int(b)]:
                    observed[task_for[s], p].append(outcomes[s, p])
                estimated = [[Fraction(sum(observed[t, p]), len(observed[t, p])) for p in range(len(policies))] for t in groups]
                weak, pair, top, gap, regret = [], [], [], [], []
                for t in groups:
                    signs = lambda values: [(values[a] > values[z]) - (values[a] < values[z]) for a, z in pairs]
                    weak.append(Fraction(signs(estimated[t]) == signs(truth[t])))
                    et = {i for i, x in enumerate(estimated[t]) if x == max(estimated[t])}
                    tt = {i for i, x in enumerate(truth[t]) if x == max(truth[t])}
                    top.append(Fraction(len(et & tt), len(et | tt)))
                    for a, z in pairs:
                        target, estimate = truth[t][a]-truth[t][z], estimated[t][a]-estimated[t][z]
                        ts, es = (target > 0)-(target < 0), (estimate > 0)-(estimate < 0)
                        pair.append(Fraction(ts == es) if ts == 0 or es != 0 else Fraction(1, 2))
                        gap.append(abs(estimate-target))
                        regret.append(Fraction(0) if ts == es else abs(target)*(Fraction(1, 2) if es == 0 else 1))
                expected = [float(sum(v)/len(v)) for v in (weak, pair, top, gap, regret)]
                for m, value in zip(METRICS, expected):
                    if abs(actual[m]-value) > 1e-12:
                        raise ValueError(f"independent observation-count oracle failed: {m}")
                    checks += 1
    return checks


def run_case(inputs, repetition, through=192):
    spec, identity = inputs["spec"], inputs["identity"]
    if type(repetition) is not int or repetition not in range(spec["replay"]["repetitions"]):
        raise ValueError("invalid repetition ID")
    if through not in spec["sampling_design"]["budgets"]:
        raise ValueError("cutoff must be a study checkpoint")
    policies, groups, outcomes, truth = primary.prepare_tensor(inputs["tensor"])
    orders = {}
    for task, scenarios in groups.items():
        for policy in range(len(policies)):
            rng = random.Random(primary.stable_seed(
                spec["replay"]["simulation_seed"], "scenario-order", repetition, task, policy))
            orders[(task, policy)] = rng.sample(scenarios, len(scenarios))
    return {
        "identity": identity, "repetition": repetition, "through": through,
        "programs": {
            p["role"]: run_program(identity["study"], p, spec, groups, outcomes,
                                   truth, orders, repetition, through)
            for p in spec["programs"]
        },
    }


def validate_cases(cases, inputs, indices, through):
    if [c["repetition"] for c in cases] != indices or len(set(indices)) != len(indices):
        raise ValueError("missing, duplicate, or reordered repetition IDs")
    roles = [p["role"] for p in inputs["spec"]["programs"]]
    budgets = [str(b) for b in inputs["spec"]["sampling_design"]["budgets"] if b <= through]
    for case in cases:
        if case["identity"] != inputs["identity"] or case["through"] != through:
            raise ValueError("mixed source, runtime, tensor, study, or replay cutoff")
        if list(case["programs"]) != roles:
            raise ValueError("missing/reordered program roles")
        for result in case["programs"].values():
            if list(result["rows"]) != budgets or result["reads"] != through or result["unique_reads"] != through or result["trace_budget"] != through:
                raise ValueError("invalid checkpoint or read coverage")
            if result["full_endpoint_replayed"] != (through == inputs["spec"]["sampling_design"]["full_endpoint"]):
                raise ValueError("incorrect full-endpoint status")
            for row in result["rows"].values():
                if set(row) != set(METRICS) or any(type(row[m]) not in (int, float) or not math.isfinite(row[m]) for m in METRICS):
                    raise ValueError("invalid metric row")


def rows_from_cases(cases):
    first = cases[0]
    return {role: {b: [dict(c["programs"][role]["rows"][b]) for c in cases]
                   for b in result["rows"]}
            for role, result in first["programs"].items()}


def stored_rows(study):
    """Load original per-repetition outputs, never fabricate absent rows."""
    entry = manifest()["studies"][study]
    if "reference_programs" not in entry:
        raise ValueError(f"{study}: original per-repetition rows are unavailable")
    result = {}
    for role, source in entry["reference_programs"].items():
        reference = checked_json(source)
        if reference["repetition_indices"] != list(range(200)):
            raise ValueError("reference repetition identity/coverage mismatch")
        if any(len(rows) != 200 for rows in reference["rows"].values()):
            raise ValueError("reference row coverage mismatch")
        result[role] = reference["rows"]
    return result


def aubc_values(rows, role, metric, budgets=(48, 96, 192)):
    return [statistics.fmean(rows[role][str(b)][i][metric] for b in budgets)
            for i in range(len(rows[role][str(budgets[0])]))]


def summarize(rows, study):
    spec = checked_json(manifest()["studies"][study]["spec"])
    budgets = spec["sampling_design"]["primary_budgets"]
    means = {role: {b: {m: statistics.fmean(r[m] for r in values) for m in METRICS}
                    for b, values in by_budget.items()} for role, by_budget in rows.items()}
    aubc = {role: {m: statistics.fmean(aubc_values(rows, role, m, budgets)) for m in METRICS}
            for role in rows}
    result = {"budget_means": means, "aubc_mean": aubc, "paired_intervals": {}}
    if study == "preference":
        pairs = [(r, a) for r in rows if r != "source-eig"
                 for a in ("source-eig", "ranksplit-v2:lambda-0.25") if a in rows]
        critical = 1.6448536269514722
    elif study == "mechanism":
        pairs = [("ranksplit-v2:lambda-0.25", r) for r in rows if r != "ranksplit-v2:lambda-0.25"]
        critical = spec["gate"]["ordered_tests"]["one_sided_critical_value"]
    else:
        pairs = [("ranksplit-v2:lambda-0.50", r) for r in rows if r != "ranksplit-v2:lambda-0.50"]
        critical = spec["gate"]["multiple_comparison"]["one_sided_critical_value"]
    interval = frozen_module("evaluate-ranksplit-v2-confirmation.py").interval
    for candidate, comparator in pairs:
        stats = {}
        for metric in METRICS:
            delta = [a-b for a, b in zip(aubc_values(rows, candidate, metric, budgets),
                                         aubc_values(rows, comparator, metric, budgets), strict=True)]
            if len(delta) >= 2:
                c = (1.959963984540054 if study == "mechanism" and metric in METRICS[3:]
                     else critical)
                stats[metric] = interval(delta, c)
        result["paired_intervals"][f"{candidate} vs {comparator}"] = stats
    return result


def preference_summary(rows):
    result = {"preference_response": {}, "paired_comparisons": {}}
    reported_metrics = (METRICS[0], METRICS[3], METRICS[4], "task_switches")
    for role in rows:
        result["preference_response"][role] = {
            "budget_means": {m: {str(b): statistics.fmean(r[m] for r in rows[role][str(b)])
                                  for b in (48, 96, 192)} for m in reported_metrics},
            "aubc_mean": {m: statistics.fmean(aubc_values(rows, role, m)) for m in reported_metrics},
        }
        if role == "source-eig":
            continue
        result["paired_comparisons"][role] = {}
        for key, anchor in (("vs_eig", "source-eig"), ("vs_lambda_0.25", "ranksplit-v2:lambda-0.25")):
            metrics = {}
            for m in reported_metrics:
                values = [a-b for a, b in zip(aubc_values(rows, role, m), aubc_values(rows, anchor, m), strict=True)]
                mean = statistics.fmean(values)
                se = statistics.stdev(values) / len(values)**0.5
                metrics[m] = {"n": len(values), "mean_delta": mean, "median_delta": statistics.median(values),
                              "standard_error": se, "one_sided_95_lower_bound": mean-1.6448536269514722*se,
                              "one_sided_95_upper_bound": mean+1.6448536269514722*se,
                              "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
                              "zero": sum(v == 0 for v in values), "minimum": min(values), "maximum": max(values)}
            result["paired_comparisons"][role][key] = metrics
    return result


def preference_with_anchors(rows, indices):
    anchor_rows = stored_rows("mechanism")
    return {**{r: {str(b): [anchor_rows[r][str(b)][i] for i in indices] for b in (48, 96, 192)}
               for r in ("source-eig", "ranksplit-v2:lambda-0.25")}, **rows}


def preference_with_switches(cases, inputs, indices):
    rows = preference_with_anchors(rows_from_cases(cases), indices)
    for role, entry in manifest()["studies"]["mechanism"]["reference_programs"].items():
        if role not in rows:
            continue
        reference = checked_json(entry)
        for b, values in rows[role].items():
            for i, row in enumerate(values):
                row["task_switches"] = reference["task_switches"][b][indices[i]]
    _, groups, _, _ = primary.prepare_tensor(inputs["tensor"])
    task_for = {s: t for t, scenarios in groups.items() for s in scenarios}
    census = inputs["spec"]["sampling_design"]["expected_initial_count"]
    for i, case in enumerate(cases):
        for role, result in case["programs"].items():
            for b, row in rows[role].items():
                tasks = [0] + [task_for[s] for s, _ in result["revealed_cells"][census:int(b)]]
                row[i]["task_switches"] = float(sum(a != z for a, z in zip(tasks, tasks[1:])))
    return rows


def check_cases(cases, inputs, indices, through):
    validate_cases(cases, inputs, indices, through)
    observation_checks = observation_oracle(cases, inputs)
    if inputs["identity"]["custom_tensor"]:
        return {"reference_comparison": "not_applicable", "reason": "new outcomes, not historical paper data"}
    study = inputs["identity"]["study"]
    entry = manifest()["studies"][study]
    differences, errors, row_checks, trace_checks = [], [], 0, 0
    if study != "preference":
        for role, source in entry["reference_programs"].items():
            ref = checked_json(source)
            if ref["repetition_indices"] != list(range(200)):
                raise ValueError("reference ID order changed")
            for case in cases:
                rep, actual = case["repetition"], case["programs"][role]
                for b, row in actual["rows"].items():
                    for m in METRICS:
                        expected = ref["rows"][b][rep][m]
                        error = abs(row[m] - expected)
                        errors.append(error)
                        row_checks += 1
                        if (row[m] != expected if m in METRICS[:3] else error > 1e-12):
                            differences.append({"repetition": rep, "role": role, "budget": b, "metric": m,
                                                "actual": row[m], "expected": expected})
                if through == ref["trace_budget"]:
                    trace_checks += 1
                    if actual["action_trace_sha256"] != ref["action_trace_sha256"][rep]:
                        differences.append({"repetition": rep, "role": role, "field": "action_trace_sha256"})
    else:
        ref = checked_json(entry["reference"])
        for case in cases:
            for role, actual in case["programs"].items():
                if through == ref["trace_budget"]:
                    trace_checks += 1
                    if actual["action_trace_sha256"] != ref["action_trace_sha256"][role][case["repetition"]]:
                        differences.append({"repetition": case["repetition"], "role": role, "field": "action_trace_sha256"})
        if indices == list(range(200)):
            computed = preference_summary(preference_with_switches(cases, inputs, indices))
            def compare_tree(actual, expected, path):
                nonlocal row_checks
                if isinstance(actual, dict):
                    for key, value in actual.items():
                        # The old report also contains telemetry, not paper metrics.
                        compare_tree(value, expected[key], path + [key])
                else:
                    error = abs(actual - expected)
                    errors.append(error)
                    row_checks += 1
                    if error > 1e-12:
                        differences.append({"field": "/".join(path), "actual": actual, "expected": expected})
            compare_tree(computed, ref, [])
    return {
        "reference_comparison": "match" if not differences else "mismatch",
        "metric_checks": row_checks, "trace_checks": trace_checks,
        "independent_observation_metric_checks": observation_checks,
        "max_metric_absolute_error": max(errors, default=0.0),
        "difference_count": len(differences), "first_differences": differences[:30],
        "all_200_repetitions_checked": indices == list(range(200)),
        "full_endpoint_replayed": through == inputs["spec"]["sampling_design"]["full_endpoint"],
        "trace_note": "no historical prefix trace exists" if not trace_checks else "compared at identical trace budget",
        "tolerance": "discrete metrics exact; other arithmetic 1e-12 absolute; no changed-path allowance",
    }
