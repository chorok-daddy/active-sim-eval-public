import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import paper_reproduction as core
from simulation.build_tensor import build_tensor


class CustomTensorTests(unittest.TestCase):
    def fixture(self, study):
        spec = json.loads((core.ROOT / f"simulation/{study}-acquisition-spec.json").read_text())
        manifest = core.checked_json(core.manifest()["studies"][study]["scenarios"])
        payloads, hashes = {}, {}
        for p in spec["policies"]:
            role = p["role"]
            payloads[role] = {
                "status": "pass", "policy_name": role, "scenario_count": 272,
                "total_jobs": 272, "completed_jobs": 272, "episodes": 272,
                "scenario_manifest_sha256": spec["scenario_manifest_sha256"],
                "policy_seed": p["policy_seed"], "checkpoints": [p["checkpoint"]],
                "results": [{"scenario_id": s["scenario_id"], "env_id": s["env_id"],
                    "scenario_descriptor_sha256": s["descriptor_sha256"],
                    "scenario_manifest_sha256": spec["scenario_manifest_sha256"],
                    "checkpoint": p["checkpoint"], "policy_seed": p["policy_seed"],
                    "final_success": bool(i % 2), "initial_observation_sha256": "0"*64,
                    "environment_action_sha256": "1"*64}
                    for i,s in enumerate(manifest["scenarios"])],
            }
            hashes[role] = core.digest(core.encoded(payloads[role]))
        return spec, payloads, hashes

    def test_both_raw_converters_preserve_outcomes_and_order(self):
        for study in ("mechanism", "ranking"):
            spec, raw, hashes = self.fixture(study)
            original = build_tensor(study,spec,raw,hashes)
            changed = copy.deepcopy(raw)
            for value in changed.values():
                value["results"].reverse()
                value["success_rate"] = 999  # Unused aggregate must not replace observations.
            self.assertEqual(original["scenario_records"], build_tensor(study,spec,changed,hashes)["scenario_records"])
            changed = copy.deepcopy(raw)
            changed["octo-small"]["results"][0]["final_success"] = True
            replacement = build_tensor(study,spec,changed,hashes)
            self.assertEqual(replacement["scenario_records"][0]["policies"]["octo-small"]["outcome"], 1)
            self.assertEqual(original["scenario_records"][1:], replacement["scenario_records"][1:])

    def test_invalid_raw_is_not_silently_repaired(self):
        for study in ("mechanism", "ranking"):
            spec, raw, hashes = self.fixture(study)
            mutations = [
                lambda p: p.update(status="incomplete"),
                lambda p: p.update(completed_jobs=271),
                lambda p: p["results"].pop(),
                lambda p: p["results"].append(p["results"][0]),
                lambda p: p.update(scenario_manifest_sha256="0"*64),
                lambda p: p.update(checkpoints=["different"]),
                lambda p: p["results"][0].update(final_success=0.9),
                lambda p: p["results"][0].update(final_success=1),
                lambda p: p["results"][0].update(scenario_descriptor_sha256="0"*64),
                lambda p: p["results"][0].update(env_id="different"),
                lambda p: p["results"][0].update(policy_seed=1),
            ]
            for mutation in mutations:
                changed = copy.deepcopy(raw)
                mutation(changed["octo-small"])
                with self.subTest(study=study, mutation=mutation), self.assertRaises(ValueError):
                    build_tensor(study,spec,changed,hashes)

    def test_custom_result_does_not_read_paper_expectations(self):
        spec, raw, hashes = self.fixture("mechanism")
        tensor = build_tensor("mechanism",spec,raw,hashes)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tensor.json"
            path.write_text(json.dumps(tensor))
            with mock.patch.object(core,"stored_rows",side_effect=AssertionError("expected rows accessed")):
                inputs = core.load_study("mechanism",path)
                case = core.run_case(inputs,0,192)
                result = core.check_cases([case],inputs,[0],192)
                self.assertEqual(result["reference_comparison"],"not_applicable")


if __name__ == "__main__":
    unittest.main()
