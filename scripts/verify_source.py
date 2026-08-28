"""Run the dependency-free public-source verification path."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.test_fixed_preference",
    "tests.test_ranksplit",
    "tests.test_ranksplit_quickstart",
    "tests.test_ranksplit_source_imports",
    "tests.test_simulation_source",
    "tests.test_paper_reproduction",
    "tests.test_custom_simulation_tensor",
    "tests.test_environment_seed_reproduction",
    "tests.test_simulation_setup",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def main() -> int:
    run([sys.executable, "-m", "unittest", "-q", *TEST_MODULES])
    completed = run(
        [sys.executable, "examples/ranksplit_quickstart.py"]
    )
    payload = json.loads(completed.stdout)
    if payload.get("method") != "RankSplit":
        raise SystemExit("quickstart method marker is incorrect")
    if payload.get("selected_next_task_policy_arm") != "drawer/policy-2":
        raise SystemExit("quickstart selection changed")
    result = run([sys.executable, "scripts/reproduce_results.py"])
    result_payload = json.loads(result.stdout)
    if result_payload.get("status") != "REFERENCE RESULT REPRODUCTION PASS":
        raise SystemExit("reported-result reference check did not pass")
    print("PUBLIC SOURCE VERIFY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
