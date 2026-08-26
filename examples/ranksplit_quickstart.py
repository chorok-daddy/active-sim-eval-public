"""Minimal CPU-only RankSplit selection example.

Run from the repository root:

    python examples/ranksplit_quickstart.py

This example uses hand-written Beta posteriors and selects one next
task-policy arm. The reported-result check is a separate command so that this
small example remains quick to inspect.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from fixed_preference import BetaPosterior  # noqa: E402
from ranksplit import ranksplit_score  # noqa: E402


def main() -> None:
    tasks = {
        "pick": (
            BetaPosterior(6, 2),
            BetaPosterior(4, 4),
            BetaPosterior(5, 3),
        ),
        "drawer": (
            BetaPosterior(3, 5),
            BetaPosterior(5, 3),
            BetaPosterior(4, 4),
        ),
    }
    preference_lambda = 0.50
    scores = {
        f"{task}/policy-{policy}": ranksplit_score(
            posteriors,
            observed_policy=policy,
            preference_lambda=preference_lambda,
        )
        for task, posteriors in tasks.items()
        for policy in range(len(posteriors))
    }
    selected = max(scores, key=lambda key: (scores[key], key))
    print(
        json.dumps(
            {
                "method": "RankSplit",
                "preference_lambda": preference_lambda,
                "selected_next_task_policy_arm": selected,
                "scores": {key: round(value, 8) for key, value in scores.items()},
                "result_reproduction_command": "python3 scripts/reproduce_results.py",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
