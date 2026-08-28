"""Reconstruct the 12-seed study, or execute its unchanged allocation driver.

Stored mode verifies raw outcomes, reconstructs all tensor content, and
independently rebuilds the original merged report. Fresh mode keeps
T1's task order, 50 paths per seed, B24/48/96, and original local-capacity Saad
adapter separate from the later global-U=1536 comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import zipfile

if __package__:
    from . import paper_reproduction as core
    from .reproduce_paper import write_json
else:
    import paper_reproduction as core
    from reproduce_paper import write_json

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
_STATE = None


def case_identity():
    p = json.loads((ROOT / "data/environment-seeds-provenance.json").read_text(encoding="utf-8"))
    return {"archive_sha256": p["sha256"], "wrapper_sha256": core.digest(Path(__file__).read_bytes()),
            "runtime": core.runtime_identity()}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def unpack(directory):
    provenance = json.loads((ROOT / "data/environment-seeds-provenance.json").read_text(encoding="utf-8"))
    archive_path = ROOT / provenance["path"]
    if core.digest(archive_path.read_bytes()) != provenance["sha256"]:
        raise ValueError("environment-seed archive hash mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or info.is_dir():
                raise ValueError("unsafe archive member")
            destination = directory.joinpath(*path.parts)
            payload = archive.read(info)
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError("existing extracted input differs; choose a new output directory")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
    return directory


def load(directory):
    scripts = directory / "source/scripts"
    sys.path.insert(0, str(scripts))
    verifier = module("t1_original_verifier", scripts / "verify-ranksplit-t1-environment-seed-transfer-release.py")
    contract, merger = verifier.load_release_modules(directory)
    protocol_path = directory / verifier.PROTOCOL_RELATIVE
    protocol = contract.load_protocol(protocol_path)
    tensor_path = directory / verifier.TENSOR_RELATIVE
    tensor = json.loads(tensor_path.read_text(encoding="utf-8"))
    contract.validate_tensor_payload(tensor, protocol=protocol, protocol_path=protocol_path)
    return verifier, contract, merger, protocol_path, protocol, tensor


def stored(directory):
    verifier, contract, merger, protocol_path, protocol, tensor = load(directory)
    receipt = verifier.verify_release(directory)
    rebuilt = contract.build_tensor_payload(protocol_path=protocol_path,
        raw_paths={r: directory / p for r, p in verifier.RAW_RELATIVES.items()})
    provenance = json.loads((ROOT / "data/environment-seeds-provenance.json").read_text(encoding="utf-8"))
    changes = {r["file"]:r for r in provenance["alterations"] if "original_sha256" in r}
    # Anonymized raw metadata changes file hashes, not observations. Validate
    # both hash identities explicitly; never relabel rebuilt hashes as originals.
    for role, relative in verifier.RAW_RELATIVES.items():
        if tensor["source_sha256"][role] != changes[relative]["original_sha256"] or rebuilt["source_sha256"][role] != changes[relative]["released_sha256"]:
            raise ValueError("raw provenance identity mismatch")
    for key in tensor:
        if key != "source_sha256" and rebuilt[key] != tensor[key]:
            raise ValueError("raw-to-tensor content comparison failed")
    for key in ("frozen_protocol", "scenario_manifest"):
        if rebuilt["source_sha256"][key] != tensor["source_sha256"][key]:
            raise ValueError("scientific input hash changed")
    if (directory / verifier.TENSOR_RELATIVE).read_bytes() != (directory / verifier.SECOND_TENSOR_RELATIVE).read_bytes():
        raise ValueError("original tensor copies differ")
    report = json.loads((directory / verifier.MERGED_RELATIVE).read_text(encoding="utf-8"))
    for tensor_replay in report["tensor_replays"]:
        for role, by_budget in tensor_replay["rows"].items():
            for row in by_budget["408"]:
                if [row[m] for m in core.METRICS] != [1., 1., 1., 0., 0.]:
                    raise ValueError("independent stored endpoint oracle failed")
    return {"mode": "stored-reconstruction", "fresh_allocation_executed": False,
            "fresh_simulation_executed": False, "verification": receipt,
            "raw_to_tensor_content_match": True,
            "raw_file_hashes": "original and released metadata-redacted hashes checked separately",
            "scientific_result": report["evaluation"]["status"],
            "coverage": {"seeds": 12, "paths_per_seed": 50, "programs": 5,
                         "budgets": [24,48,96,204,408], "metric_values": 75000, "full_trace_digests": 3000},
            "evaluation": report["evaluation"]}


def initialize(directory):
    global _STATE
    loaded = load(Path(directory))
    driver = module("t1_original_replay", Path(directory) / "source/scripts/replay-ranksplit-environment-seed-transfer.py")
    _STATE = loaded, driver


def compute(job):
    seed, repetition, through = job
    (_, contract, _, _, protocol, tensor), driver = _STATE
    view = driver.seed_view(tensor, seed)
    policies, tasks, groups, outcomes, truth = driver.prepare_tensor(view)
    orders = driver.scenario_orders_for_repetition(master_seed=protocol["replay"]["simulation_seed"],
        environment_seed=seed, repetition=repetition, task_groups=groups, policy_count=len(policies))
    task_for = {s: t for t, group in groups.items() for s in group}
    actual = {}
    for program in protocol["programs"]:
        ledger = core.ReadLedger(outcomes)
        rows = {}
        original = driver.evaluate
        def record(observed, target):
            row = original(observed, target)
            spent = sum(map(len, observed.values()))
            rows[str(spent)] = {m: row[m] for m in core.METRICS}
            if spent == through and through < 408:
                raise core.PrefixComplete()
            return row
        driver.evaluate = record
        try:
            driver.run_program(family=program["family"], preference_lambda=float(program["preference_lambda"]),
                scenario_orders=orders, task_groups=groups, outcomes=ledger, truth_rates=truth,
                budgets=protocol["sampling_design"]["budgets"], environment_seed=seed,
                repetition=repetition, protocol=protocol)
        except core.PrefixComplete:
            pass
        finally:
            driver.evaluate = original
        if len(ledger.reads) != through or len(set(ledger.reads)) != through:
            raise ValueError("T1 reveal coverage mismatch")
        if through == 408 and [rows["408"][m] for m in core.METRICS] != [1.,1.,1.,0.,0.]:
            raise ValueError("T1 independent endpoint oracle failed")
        trace = [(task_for[s], p) for s, p in ledger.reads[12:]]
        actual[program["role"]] = {"rows": rows, "reads": through,
            "action_trace_sha256": hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()}
    return {"environment_seed": seed, "repetition": repetition, "through": through,
            "identity": case_identity(), "programs": actual}


def validate_case(case, identity, seed, repetition, through, protocol):
    if case.get("identity") != identity or case.get("environment_seed") != seed or case.get("repetition") != repetition or case.get("through") != through:
        raise ValueError("T1 case source/runtime/seed/repetition/cutoff changed")
    if list(case.get("programs", {})) != [p["role"] for p in protocol["programs"]]:
        raise ValueError("T1 program coverage changed")
    budgets = [str(b) for b in protocol["sampling_design"]["budgets"] if b <= through]
    for result in case["programs"].values():
        if result["reads"] != through or list(result["rows"]) != budgets:
            raise ValueError("T1 read/checkpoint coverage changed")
        for row in result["rows"].values():
            if set(row) != set(core.METRICS) or any(type(v) not in (int,float) or not math.isfinite(v) for v in row.values()):
                raise ValueError("T1 invalid metric row")


def check_and_summarize(cases, directory, repetitions, seeds, through):
    verifier, contract, merger, _, protocol, _ = load(directory)
    if sorted((c["environment_seed"], c["repetition"]) for c in cases) != [(s,r) for s in seeds for r in range(repetitions)]:
        raise ValueError("T1 missing/duplicate seed or repetition")
    # Original rows are opened only after fresh selection has finished.
    references = {s: json.loads((directory / f"generated/replay-seed-{s}.json").read_text(encoding="utf-8")) for s in seeds}
    differences, checks, traces, errors = [], 0, 0, []
    for case in cases:
        validate_case(case,case_identity(),case["environment_seed"],case["repetition"],through,protocol)
        ref = references[case["environment_seed"]]
        for role, actual in case["programs"].items():
            for b, row in actual["rows"].items():
                for m in core.METRICS:
                    target = ref["rows"][role][b][case["repetition"]][m]
                    error = abs(row[m]-target)
                    errors.append(error)
                    checks += 1
                    if (row[m] != target if m in core.METRICS[:3] else error > 1e-12):
                        differences.append([case["environment_seed"],case["repetition"],role,b,m,row[m],target])
            if through == 408:
                traces += 1
                if actual["action_trace_sha256"] != ref["action_trace_sha256"][role][case["repetition"]]:
                    differences.append([case["environment_seed"],case["repetition"],role,"action_trace_sha256"])
    tensor_replays = []
    for seed in seeds:
        ordered = sorted((c for c in cases if c["environment_seed"] == seed), key=lambda c:c["repetition"])
        rows = {p["role"]: {str(b): [c["programs"][p["role"]]["rows"][str(b)] for c in ordered]
                 for b in contract.BUDGETS if b <= through} for p in protocol["programs"]}
        tensor_replays.append({"environment_seed": seed, "rows": rows})
    report = {"tensor_replays": tensor_replays}
    candidate = protocol["gate"]["primary_program"]
    intervals = {}
    if len(seeds) >= 2 and repetitions == 50:
        for other in protocol["gate"]["primary_comparators"]:
            intervals[other] = {m: merger.interval(merger.tensor_level_deltas(report, candidate, other, m), contract.CRITICAL_VALUE)
                                for m in core.METRICS}
    return {"mode": "fresh-replay", "fresh_simulation_executed": False,
            "runtime": core.runtime_identity(), "seeds": seeds, "paths_per_seed": repetitions,
            "through": through, "full_endpoint_replayed": through == 408,
            "full_historical_result_branch": "not_evaluated_for_prefix" if through < 408 else "see separate stored reconstruction",
            "reference_comparison": "mismatch" if differences else "match",
            "metric_checks": checks, "trace_checks": traces, "max_metric_absolute_error": max(errors,default=0.),
            "difference_count": len(differences), "first_differences": differences[:20],
            "all_seeds_and_paths_checked": seeds == list(range(100,112)) and repetitions == 50,
            "tensor_level_intervals": intervals, "inference_unit": "environment seed, not replay path"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("stored","replay"), default="stored")
    p.add_argument("--through", type=int, choices=(96,204,408), default=96)
    p.add_argument("--repetitions", type=int, choices=range(1,51), default=50)
    p.add_argument("--seeds", type=int, choices=range(1,13), default=12, help="first N of seeds 100..111")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, default=Path("rerun/environment-seeds"))
    args = p.parse_args()
    directory = unpack((args.output / "inputs").resolve())
    if args.mode == "stored":
        result = stored(directory)
    else:
        seeds = list(range(100,100+args.seeds))
        jobs = []
        cases = []
        protocol = load(directory)[4]
        identity = case_identity()
        for s in seeds:
            for r in range(args.repetitions):
                path = args.output / "cases" / f"{s}-{r:02}.json"
                if path.exists():
                    if not args.resume:
                        raise ValueError("T1 output contains cases; use --resume or a new directory")
                    case = json.loads(path.read_text(encoding="utf-8"))
                    validate_case(case,identity,s,r,args.through,protocol)
                    cases.append(case)
                else:
                    jobs.append((s,r,args.through))
        total = args.seeds*args.repetitions
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(directory),)) as pool:
            for future in as_completed([pool.submit(compute,j) for j in jobs]):
                case = future.result()
                cases.append(case)
                write_json(args.output / "cases" / f"{case['environment_seed']}-{case['repetition']:02}.json",case)
                progress = {"completed":len(cases),"total":total,"through":args.through}
                write_json(args.output / "progress.json",progress)
                if len(cases) <= 3 or len(cases) % 25 == 0 or len(cases) == total:
                    print(json.dumps(progress),flush=True)
        result = check_and_summarize(cases,directory,args.repetitions,seeds,args.through)
    write_json(args.output / "result.json", result)
    print(json.dumps({"mode":args.mode,"report":str(args.output / "result.json"),
                      "reference_comparison":result.get("reference_comparison"),
                      "scientific_result":result.get("scientific_result")}),flush=True)
    return int(result.get("reference_comparison") == "mismatch")


if __name__ == "__main__":
    raise SystemExit(main())
