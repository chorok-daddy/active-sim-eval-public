import copy
import tempfile
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
