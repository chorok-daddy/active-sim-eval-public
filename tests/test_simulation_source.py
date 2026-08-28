"""Source and input checks only: these tests never run a robot simulator."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "simulation"
SPEC = importlib.util.spec_from_file_location(
    "simulation_scenarios", SIMULATION / "interstitial_scenario_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
SCENARIOS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIOS)


class SimulationSourceTests(unittest.TestCase):
    def test_original_files_are_unchanged(self):
        provenance = json.loads((SIMULATION / "SOURCE_PROVENANCE.json").read_text())
        for relative, expected in provenance["files"].items():
            with self.subTest(file=relative):
                payload = (SIMULATION / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)

    def test_sources_parse_without_importing_gpu_dependencies(self):
        for source in SIMULATION.glob("*.py"):
            with self.subTest(file=source.name):
                ast.parse(source.read_text(), filename=str(source))

    def test_both_scenario_sets_are_complete_and_disjoint(self):
        sets = []
        for name in ("mechanism-comparison.json", "ranking-comparison.json"):
            payload = json.loads((SIMULATION / "scenarios" / name).read_text())
            SCENARIOS.validate_manifest_structure(payload)
            counts = {}
            identities = set()
            for row in payload["scenarios"]:
                counts[row["env_id"]] = counts.get(row["env_id"], 0) + 1
                # Compare physical poses, not IDs that differ by construction.
                identities.add(json.dumps(
                    {"env_id": row["env_id"], "objects": row["objects"]},
                    sort_keys=True,
                ))
            self.assertEqual(
                counts,
                {task: 2 * count for task, count in SCENARIOS.EXPECTED_OFFICIAL_COUNTS.items()},
            )
            sets.append(identities)
        self.assertFalse(sets[0] & sets[1])

    def test_modified_pose_is_rejected(self):
        payload = json.loads(
            (SIMULATION / "scenarios" / "mechanism-comparison.json").read_text()
        )
        corrupted = copy.deepcopy(payload["scenarios"][0])
        corrupted["objects"][0]["position"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "descriptor hash mismatch"):
            SCENARIOS.validate_descriptor(corrupted)


if __name__ == "__main__":
    unittest.main()
