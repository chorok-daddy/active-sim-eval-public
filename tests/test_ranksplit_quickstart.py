from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ranksplit_quickstart.py"


class RankSplitQuickstartTests(unittest.TestCase):
    def test_cpu_only_example_selects_an_arm(self):
        completed = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["method"], "RankSplit")
        self.assertEqual(
            payload["selected_next_task_policy_arm"],
            "drawer/policy-2",
        )
        self.assertEqual(
            payload["result_reproduction_command"],
            "python3 scripts/reproduce_results.py",
        )
        self.assertEqual(len(payload["scores"]), 6)


if __name__ == "__main__":
    unittest.main()
