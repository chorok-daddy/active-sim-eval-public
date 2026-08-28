"""Run one fixed policy checkpoint over the frozen interstitial pose holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from interstitial_scenario_contract import validate_manifest_structure


POLICIES = ("octo-small", "octo-base", "rt-1-x")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.5))


def load_manifest(path: Path) -> tuple[list[dict[str, object]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_structure(payload)
    return list(payload["scenarios"]), sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-name", choices=POLICIES, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument(
        "--simpler-root", type=Path, default=Path(r"C:\Users\User\src\SimplerEnv-ms3")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--shutdown-server", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.policy_name == "rt-1-x":
        if args.policy_seed is not None:
            raise ValueError("RT-1-X has no policy-seed control")
        policy_seed = None
    else:
        policy_seed = 0 if args.policy_seed is None else args.policy_seed
    scenarios, manifest_hash = load_manifest(args.manifest)
    client = Path(__file__).with_name("run-octo-small-policy-episode.py")
    process_environment = dict(__import__("os").environ)
    process_environment["PYTHONPATH"] = str(args.simpler_root)
    results: list[dict[str, object]] = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        expected_metadata = {
            "policy_name": args.policy_name,
            "scenario_manifest_sha256": manifest_hash,
            "policy_seed": policy_seed,
        }
        for key, expected in expected_metadata.items():
            if previous.get(key) != expected:
                raise ValueError(
                    f"resume metadata mismatch for {key}: {previous.get(key)!r} != {expected!r}"
                )
        results = list(previous.get("results", []))
    completed_ids = {str(row["scenario_id"]) for row in results}
    if len(completed_ids) != len(results):
        raise ValueError("resume output contains duplicate scenario rows")

    def checkpoint(status: str = "in-progress") -> None:
        write_json_atomic(
            args.output,
            {
                "schema_version": 1,
                "probe_type": "fixed-checkpoint interstitial-pose holdout",
                "policy_name": args.policy_name,
                "policy_seed": policy_seed,
                "scenario_manifest_sha256": manifest_hash,
                "scenario_count": len(scenarios),
                "total_jobs": len(scenarios),
                "completed_jobs": len(results),
                "results": results,
                "status": status,
            },
        )

    for index, scenario in enumerate(scenarios):
        scenario_id = str(scenario["scenario_id"])
        if scenario_id in completed_ids:
            continue
        command = [
            sys.executable,
            str(client),
            "--env-id",
            str(scenario["env_id"]),
            "--seed",
            "0",
            "--episode-id",
            str(scenario["base_episode_id"]),
            "--scenario-manifest",
            str(args.manifest),
            "--scenario-id",
            scenario_id,
        ]
        if policy_seed is not None:
            command.extend(["--policy-seed", str(policy_seed)])
        if args.shutdown_server and index == len(scenarios) - 1:
            command.append("--shutdown-server")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=process_environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"episode failed for {args.policy_name} {scenario_id}:\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        if result.get("scenario_id") != scenario_id:
            raise ValueError("client returned a mismatched scenario id")
        if result.get("scenario_descriptor_sha256") != scenario["descriptor_sha256"]:
            raise ValueError("client returned a mismatched scenario descriptor hash")
        if result.get("scenario_manifest_sha256") != manifest_hash:
            raise ValueError("client returned a mismatched manifest hash")
        if args.policy_name == "rt-1-x":
            if result.get("policy_seed") is not None:
                raise ValueError("RT-1-X must report policy_seed=null")
        elif int(result["policy_seed"]) != policy_seed:
            raise ValueError("Octo result returned a mismatched fixed policy seed")
        results.append(result)
        completed_ids.add(scenario_id)
        if len(results) % args.checkpoint_every == 0 or len(results) == len(scenarios):
            checkpoint()
        print(
            f"completed {len(results)}/{len(scenarios)} {args.policy_name} {scenario_id} "
            f"success={result['final_success']}",
            file=sys.stderr,
            flush=True,
        )
    if len(results) != len(scenarios):
        raise AssertionError("batch did not complete all frozen scenarios")
    task_summaries = {}
    for env_id in sorted({str(row["env_id"]) for row in results}):
        task_rows = [row for row in results if str(row["env_id"]) == env_id]
        task_summaries[env_id] = {
            "episodes": len(task_rows),
            "successes": sum(bool(row["final_success"]) for row in task_rows),
            "success_rate": statistics.fmean(
                float(bool(row["final_success"])) for row in task_rows
            ),
        }
    report = {
        "schema_version": 1,
        "probe_type": "fixed-checkpoint interstitial-pose holdout",
        "policy_name": args.policy_name,
        "checkpoints": sorted({str(row["checkpoint"]) for row in results}),
        "policy_seed": policy_seed,
        "scenario_manifest_sha256": manifest_hash,
        "scenario_count": len(scenarios),
        "total_jobs": len(scenarios),
        "completed_jobs": len(results),
        "episodes": len(results),
        "task_summaries": task_summaries,
        "results": results,
        "status": "pass",
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "policy_name": args.policy_name,
                "episodes": len(results),
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
