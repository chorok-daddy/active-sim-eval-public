import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from scripts import paper_reproduction as core


class PaperReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = core.load_study("mechanism")
        cls.ranking = core.load_study("ranking")

    def tiny(self, study="mechanism"):
        source = self.ranking if study == "ranking" else self.inputs
        spec = copy.deepcopy(source["spec"])
        spec["sampling_design"].update(budgets=[6, 12, 24], primary_budgets=[6, 12],
                                       expected_initial_count=6, full_endpoint=24)
        groups = {0: list(range(4)), 1: list(range(4, 8))}
        outcomes = {(s, p): int((s + 2*p) % 3 == 0) for s in range(8) for p in range(3)}
        truth = tuple(tuple(sum(outcomes[s, p] for s in group)/4 for p in range(3))
                      for group in groups.values())
        orders = {(t, p): list(group) for t, group in groups.items() for p in range(3)}
        return spec, groups, outcomes, truth, orders

    def test_metric_oracle(self):
        observed = {(0, 0): [1, 0], (0, 1): [1, 0], (0, 2): [0, 0]}
        expected = [0.0, 0.5, 0.5, 2/3, 1/6]
        for fn in (core.primary.evaluate,
                   core.frozen_module("replay-ranksplit-global-u1536-amended.py").evaluate):
            actual = fn(observed, ((1., 0., 0.),))
            for metric, value in zip(core.METRICS, expected):
                self.assertAlmostEqual(actual[metric], value)
            ties = {(0, 0): [1], (0, 1): [1, 1], (0, 2): [0, 0, 0]}
            self.assertEqual(fn(ties, ((1., 1., 0.),))[core.METRICS[0]], 1.)

    def test_all_program_prefixes_are_original_full_run_prefixes(self):
        for study in ("mechanism", "ranking"):
            spec, groups, outcomes, truth, orders = self.tiny(study)
            module = (core.primary if study == "mechanism" else
                      core.frozen_module("replay-ranksplit-global-u1536-amended.py"))
            original = module.evaluate
            for program in spec["programs"]:
                with self.subTest(study=study, role=program["role"]):
                    args = (study, program, spec, groups, outcomes, truth, orders, 0)
                    full = core.run_program(*args, 24)
                    prefix = core.run_program(*args, 12)
                    self.assertEqual(prefix["rows"], {b: r for b, r in full["rows"].items() if int(b) <= 12})
                    self.assertEqual(prefix["reads"], 12)
                    self.assertFalse(prefix["full_endpoint_replayed"])
                    self.assertTrue(full["full_endpoint_replayed"])
                    self.assertIs(module.evaluate, original)

    def test_expected_files_are_not_inputs_to_allocator(self):
        original = core.checked_json
        def forbid(entry, *args, **kwargs):
            if "-rows.json" in entry["path"] or "reference" in entry["path"]:
                raise AssertionError("allocator read expected results")
            return original(entry, *args, **kwargs)
        with mock.patch.object(core, "checked_json", side_effect=forbid):
            inputs = core.load_study("mechanism")
            spec, groups, outcomes, truth, orders = self.tiny()
            core.run_program("mechanism", inputs["spec"]["programs"][0], spec,
                             groups, outcomes, truth, orders, 0, 12)

    def test_unrevealed_outcomes_do_not_change_initial_choices(self):
        spec, groups, outcomes, truth, orders = self.tiny()
        p = spec["programs"][0]
        left = core.run_program("mechanism", p, spec, groups, outcomes, truth, orders, 0, 6)
        changed = {key: (value if key[0] in (0, 4) else 1-value) for key, value in outcomes.items()}
        right = core.run_program("mechanism", p, spec, groups, changed, truth, orders, 0, 6)
        self.assertEqual(left, right)

    def test_strict_binary_and_layout_validation(self):
        scenarios = core.checked_json(core.manifest()["studies"]["mechanism"]["scenarios"])
        for value in (0.9, "1", True, None, 2):
            tensor = copy.deepcopy(self.inputs["tensor"])
            tensor["scenario_records"][0]["policies"]["octo-small"]["outcome"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.validate_tensor(tensor, self.inputs["spec"], scenarios)
        tensor = copy.deepcopy(self.inputs["tensor"])
        tensor["scenario_records"][0], tensor["scenario_records"][1] = tensor["scenario_records"][1], tensor["scenario_records"][0]
        with self.assertRaises(ValueError):
            core.validate_tensor(tensor, self.inputs["spec"], scenarios)

    def test_original_means_are_recomputed_from_rows(self):
        rows = core.stored_rows("mechanism")
        actual = core.summarize(rows, "mechanism")
        self.assertAlmostEqual(actual["aubc_mean"]["source-eig"][core.METRICS[0]], 0.35)
        self.assertAlmostEqual(actual["aubc_mean"]["ranksplit-v2:lambda-0.25"][core.METRICS[0]], 0.385)
        self.assertEqual(len(actual["paired_intervals"]), 2)
        for pair in actual["paired_intervals"].values():
            self.assertEqual(pair[core.METRICS[0]]["n"], 200)

    def test_reference_comparison_rejects_reordered_cases_and_changes(self):
        spec, groups, outcomes, truth, orders = self.tiny()
        cases = [core.run_case(self.inputs, rep, 192) for rep in (0, 1)]
        self.assertEqual(core.check_cases(cases, self.inputs, [0, 1], 192)["reference_comparison"], "match")
        for broken in (cases[::-1], cases[:1], [cases[0], cases[0]]):
            with self.assertRaises(ValueError):
                core.check_cases(broken, self.inputs, [0, 1], 192)
        changed = copy.deepcopy(cases)
        changed[0]["programs"]["source-eig"]["rows"]["48"][core.METRICS[0]] += 0.25
        with self.assertRaises(ValueError):
            core.check_cases(changed, self.inputs, [0, 1], 192)
        changed = copy.deepcopy(cases)
        changed[0]["identity"]["tensor_sha256"] = "different"
        with self.assertRaises(ValueError):
            core.validate_cases(changed, self.inputs, [0, 1], 192)

    def test_exception_restores_original_evaluator(self):
        spec, groups, outcomes, truth, orders = self.tiny("ranking")
        del outcomes[(0, 0)]
        module = core.frozen_module("replay-ranksplit-global-u1536-amended.py")
        original = module.evaluate
        with self.assertRaises(KeyError):
            core.run_program("ranking", spec["programs"][0], spec, groups, outcomes, truth, orders, 0, 12)
        self.assertIs(module.evaluate, original)


if __name__ == "__main__":
    unittest.main()
