import unittest
from collections import Counter
from pathlib import Path

from bl_rctn.graph import validate_root_case
from bl_rctn.models import DemoConfig, read_demo_config
from bl_rctn.samples import build_curated_cases, build_random_cases


EXPECTED_BY_FAMILY = {
    "small_clique": {
        "status": "SMALL", "cutsets": 0, "full_sigma": 0,
        "elementary": 0, "tn": 0,
    },
    "high_clique": {
        "status": "HIGH", "cutsets": 0, "full_sigma": 0,
        "elementary": 0, "tn": 0,
    },
    "joined_cycle": {
        "status": "CROSSED", "cutsets": 9, "full_sigma": 9,
        "elementary": 9, "tn": 0,
    },
    "complete_bipartite": {
        "status": "SPLIT", "cutsets": 1, "full_sigma": 15,
        "elementary": 5, "tn": 5,
    },
    "two_glued_cliques": {
        "status": "SPLIT", "cutsets": 1, "full_sigma": 1,
        "elementary": 1, "tn": 1,
    },
    "three_glued_cliques": {
        "status": "SPLIT", "cutsets": 1, "full_sigma": 3,
        "elementary": 3, "tn": 3,
    },
}


class SampleTests(unittest.TestCase):
    def test_curated_matrix(self):
        cases = build_curated_cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual(
            Counter(case.k for case in cases), Counter({2: 6, 3: 6, 4: 6})
        )
        self.assertEqual(len({case.case_id for case in cases}), 18)
        for case in cases:
            with self.subTest(case=case.case_id):
                validate_root_case(case)
                self.assertEqual(dict(case.expected), EXPECTED_BY_FAMILY[case.family])

    def test_curated_subset_preserves_six_family_order(self):
        cases = build_curated_cases((3,))
        self.assertEqual(
            tuple(case.family for case in cases), tuple(EXPECTED_BY_FAMILY)
        )
        self.assertTrue(all(case.k == 3 for case in cases))

    def test_random_reproducibility(self):
        first = build_random_cases((2, 3), 2, 20260811)
        second = build_random_cases((2, 3), 2, 20260811)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({case.case_id for case in first}), 4)
        self.assertEqual(len({case.seed for case in first}), 4)
        for case in first:
            validate_root_case(case)
        self.assertNotEqual(first, build_random_cases((2, 3), 2, 20260812))

    def test_zero_random_cases_and_invalid_arguments(self):
        self.assertEqual(build_random_cases((2, 3), 0, 20260811), ())
        with self.assertRaises(ValueError):
            build_random_cases((2, 2), 1, 0)
        with self.assertRaises(ValueError):
            build_random_cases((True,), 1, 0)
        with self.assertRaises(ValueError):
            build_curated_cases((5,))

    def test_demo_config(self):
        config = read_demo_config(Path("configs/demo_suite.json"))
        self.assertEqual(
            config,
            DemoConfig(
                "bl-rctn-demo-config-v1",
                (2, 3, 4),
                20260811,
                0,
                4096,
                20,
                "STRUCTURE_ONLY",
            ),
        )


if __name__ == "__main__":
    unittest.main()
