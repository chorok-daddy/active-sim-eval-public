import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from scripts import reproduce_environment_seeds as seed
from scripts import paper_reproduction as core


class EnvironmentSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = seed.unpack(Path(cls.temp.name) / "inputs")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_stored_reconstruction_retains_mixed_result(self):
        result = seed.stored(self.directory)
        self.assertEqual(result["scientific_result"],"PARTIAL")
        self.assertTrue(result["raw_to_tensor_content_match"])
        self.assertEqual(result["coverage"]["metric_values"],75000)
        self.assertEqual(len(result["evaluation"]["primary_contrasts"]["comparisons"]),4)
        self.assertFalse(result["fresh_allocation_executed"])

    def test_prefix_matches_unchanged_full_driver(self):
        seed.initialize(str(self.directory))
        loaded, driver = seed._STATE
        protocol = loaded[4]
        original_programs = protocol["programs"]
        # Pure EIG is sufficient to test the stop-hook; no long exact-rank run.
        protocol["programs"] = [original_programs[0]]
        try:
            full = seed.compute((100,0,408))
            prefix = seed.compute((100,0,96))
        finally:
            protocol["programs"] = original_programs
        role = original_programs[0]["role"]
        self.assertEqual(prefix["programs"][role]["rows"],
                         {b:r for b,r in full["programs"][role]["rows"].items() if int(b)<=96})
        self.assertEqual(prefix["programs"][role]["reads"],96)
        self.assertEqual(protocol["allocators"]["srank_singleton"]["horizon_per_task"],102)
        self.assertEqual(list(loaded[1].ENV_COUNTS.values()),[24,24,24,64])

    def test_resume_rejects_changed_source_or_cutoff(self):
        seed.initialize(str(self.directory))
        case = seed.compute((100,0,96))
        protocol = seed._STATE[0][4]
        seed.validate_case(case,seed.case_identity(),100,0,96,protocol)
        with self.assertRaises(ValueError):
            seed.validate_case(case,seed.case_identity(),100,0,204,protocol)
        changed = copy.deepcopy(case)
        changed["identity"]["archive_sha256"] = "changed"
        with self.assertRaises(ValueError):
            seed.validate_case(changed,seed.case_identity(),100,0,96,protocol)

    def test_changed_outcome_fails_even_when_public_raw_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = seed.unpack(Path(temp)/"inputs")
            verifier = seed.load(directory)[0]
            relative = verifier.RAW_RELATIVES["octo-small"]
            path = directory/relative
            raw = json.loads(path.read_text())
            raw["results"][0]["final_success"] = not raw["results"][0]["final_success"]
            path.write_text(json.dumps(raw,indent=2)+"\n")
            sha = core.digest(path.read_bytes())
            manifest_path = directory/"MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["files"]:
                if entry["archive_path"] == relative:
                    entry.update(sha256=sha,bytes=path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
            provenance_path = seed.ROOT/"data/environment-seeds-provenance.json"
            provenance = json.loads(provenance_path.read_text())
            for entry in provenance["alterations"]:
                if entry["file"] == relative:
                    entry["released_sha256"] = sha
            read = Path.read_text
            def replaced(p,*args,**kwargs):
                return json.dumps(provenance) if p == provenance_path else read(p,*args,**kwargs)
            # The inventory and original verifier alone still pass. The added
            # observation-content comparison must reject the changed bit.
            self.assertEqual(verifier.verify_release(directory)["status"],"pass")
            with mock.patch.object(Path,"read_text",replaced), self.assertRaisesRegex(ValueError,"content comparison"):
                seed.stored(directory)
        self.assertFalse(seed.type_sensitive_equal([1],[True]))
        self.assertFalse(seed.type_sensitive_equal([1],[1.0]))


if __name__ == "__main__":
    unittest.main()
