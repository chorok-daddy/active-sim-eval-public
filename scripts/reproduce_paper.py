"""Reconstruct saved paper results or replay actual observations at paper budgets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

if __package__:
    from . import paper_reproduction as core
else:
    import paper_reproduction as core

_INPUTS = {}


def initialize(studies, custom):
    global _INPUTS
    _INPUTS = {s: core.load_study(s, custom) for s in studies}


def compute(job):
    study, repetition, through = job
    return study, core.run_case(_INPUTS[study], repetition, through)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
    temp.replace(path)


def stored(studies):
    result = {"mode": "stored", "fresh_replay": False, "studies": {}}
    for study in studies:
        entry = core.manifest()["studies"][study]
        if "reference_programs" not in entry:
            ref = core.checked_json(entry["reference"])
            result["studies"][study] = {
                "status": "original summary and trace archive; replay required to reconstruct absent rows",
                "original_repetition_rows_available": False,
                "preference_response": ref["preference_response"],
                "paired_comparisons": ref["paired_comparisons"],
            }
            continue
        rows = core.stored_rows(study)
        if study == "preference":
            rows = core.preference_with_anchors(rows, list(range(200)))
        result["studies"][study] = {
            "status": "recomputed from stored per-repetition rows",
            "repetitions": 200, "summary": core.summarize(rows, study),
        }
    return result


def replay(args, studies):
    inputs = {s: core.load_study(s, args.custom_tensor) for s in studies}
    indices = list(range(args.repetitions))
    cutoffs = {s: min(args.through, max(inputs[s]["spec"]["sampling_design"]["budgets"])) for s in studies}
    cases = {s: {} for s in studies}
    jobs = []
    for study in studies:
        for repetition in indices:
            path = args.output / "cases" / study / f"{repetition:03}.json"
            if path.exists():
                if not args.resume:
                    raise ValueError("output already contains cases; use --resume or a new output directory")
                case = json.loads(path.read_text(encoding="utf-8"))
                core.validate_cases([case], inputs[study], [repetition], cutoffs[study])
                cases[study][repetition] = case
            else:
                jobs.append((study, repetition, cutoffs[study]))
    started = time.perf_counter()
    total = len(studies) * args.repetitions
    def progress():
        state = {"completed": sum(map(len, cases.values())), "total": total,
                 "by_study": {s: len(values) for s, values in cases.items()},
                 "through": cutoffs, "elapsed_seconds": round(time.perf_counter()-started, 2)}
        write_json(args.output / "progress.json", state)
        print(json.dumps(state), flush=True)
    progress()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(studies, args.custom_tensor)) as pool:
        futures = {pool.submit(compute, job): job for job in jobs}
        for future in as_completed(futures):
            study, case = future.result()
            cases[study][case["repetition"]] = case
            write_json(args.output / "cases" / study / f"{case['repetition']:03}.json", case)
            progress()
    result = {"mode": "replay", "fresh_replay": True, "studies": {}}
    for study in studies:
        ordered = [cases[study][i] for i in indices]
        check = core.check_cases(ordered, inputs[study], indices, cutoffs[study])
        rows = core.rows_from_cases(ordered)
        if study == "preference" and not args.custom_tensor:
            rows = core.preference_with_anchors(rows, indices)
        result["studies"][study] = {
            "check": check, "summary": core.summarize(rows, study),
            "identity": inputs[study]["identity"], "repetition_indices": indices,
            "anchor_source": "original mechanism rows" if study == "preference" and not args.custom_tensor else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stored", "replay"), default="stored")
    parser.add_argument("--study", choices=("all", *core.STUDIES), default="all")
    parser.add_argument("--through", type=int, choices=(192, 408, 816), default=192)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=Path("rerun/paper"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--custom-tensor", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 200 or args.workers < 1:
        parser.error("repetitions must be 1..200 and workers positive")
    if args.custom_tensor and (args.mode != "replay" or args.study == "all"):
        parser.error("custom outcomes require replay mode and one named study")
    studies = list(core.STUDIES) if args.study == "all" else [args.study]
    result = stored(studies) if args.mode == "stored" else replay(args, studies)
    write_json(args.output / "result.json", result)
    print(f"Report: {args.output / 'result.json'}", flush=True)
    mismatches = [s for s, r in result["studies"].items()
                  if r.get("check", {}).get("reference_comparison") == "mismatch"]
    if mismatches:
        print("Historical result mismatch (actual output preserved): " + ", ".join(mismatches))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
