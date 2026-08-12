import unittest
from unittest.mock import patch

from bl_rctn.graph import adjacency, support_graph_from_case
from bl_rctn.local import (
    analyze_local,
    analyze_case_local,
    compute_tn_aggregated,
)
from bl_rctn.models import CompletionLevel, Separation, SupportGraph, stable_id
from bl_rctn.samples import build_curated_cases
from bl_rctn.separations import are_nested, crosses


EXPECTED_BY_FAMILY = {
    "small_clique": ("SMALL", 0, 0, 0, 0),
    "high_clique": ("HIGH", 0, 0, 0, 0),
    "joined_cycle": ("CROSSED", 9, 9, 9, 0),
    "complete_bipartite": ("SPLIT", 1, 15, 5, 5),
    "two_glued_cliques": ("SPLIT", 1, 1, 1, 1),
    "three_glued_cliques": ("SPLIT", 1, 3, 3, 3),
}


def separation_id(separation: Separation) -> str:
    return stable_id(
        "separation",
        {
            "side_a": separation.side_a,
            "side_b": separation.side_b,
            "separator": separation.separator,
        },
    )


class LocalTests(unittest.TestCase):
    def test_analytic_matrix_and_three_way_equality(self):
        cases = build_curated_cases()
        self.assertEqual(len(cases), 18)
        totals = [0, 0, 0, 0]
        for case in cases:
            with self.subTest(case=case.case_id):
                result = analyze_case_local(case)
                expected = EXPECTED_BY_FAMILY[case.family]
                observed = (
                    result.status.value,
                    len(result.cuts),
                    len(result.full_separations),
                    len(result.elementary),
                    len(result.tn_full),
                )
                self.assertEqual(observed, expected)
                self.assertEqual(result.tn_full, result.tn_pairwise)
                self.assertEqual(result.tn_full, result.tn_aggregated)
                self.assertEqual(
                    result.completion_level, CompletionLevel.LOCAL_EXACT
                )
                for index, value in enumerate(observed[1:]):
                    totals[index] += value
        self.assertEqual(totals, [36, 84, 54, 27])

    def test_full_oracle_cap_is_exact_and_never_truncates(self):
        case = next(
            case
            for case in build_curated_cases((2,))
            if case.family == "complete_bipartite"
        )
        exact = analyze_case_local(case, max_full_separations=15)
        self.assertEqual(len(exact.full_separations), 15)
        with self.assertRaisesRegex(ValueError, "limit"):
            analyze_case_local(case, max_full_separations=14)

    def test_rejection_witnesses_are_real_and_lexically_least(self):
        for case in build_curated_cases():
            result = analyze_case_local(case)
            elementary_by_id = {
                separation_id(value): value for value in result.elementary
            }
            witnesses = dict(result.rejection_witnesses)
            rejected = set(result.elementary).difference(result.tn_pairwise)
            self.assertEqual(set(witnesses), {separation_id(s) for s in rejected})
            for candidate in rejected:
                candidate_id = separation_id(candidate)
                crossing_ids = sorted(
                    separation_id(other)
                    for other in result.elementary
                    if crosses(candidate, other)
                )
                self.assertTrue(crossing_ids)
                self.assertEqual(witnesses[candidate_id], crossing_ids[0])
                self.assertIn(witnesses[candidate_id], elementary_by_id)
                self.assertTrue(
                    crosses(candidate, elementary_by_id[witnesses[candidate_id]])
                )

    def test_nestedness_checks_all_four_orientations(self):
        isolate_zero = Separation((0,), (1, 2, 3), ())
        isolate_one = Separation((1,), (0, 2, 3), ())
        diagonal_a = Separation((0, 1), (2, 3), ())
        diagonal_b = Separation((0, 2), (1, 3), ())
        self.assertTrue(are_nested(isolate_zero, isolate_one))
        self.assertFalse(crosses(isolate_zero, isolate_one))
        self.assertFalse(are_nested(diagonal_a, diagonal_b))
        self.assertTrue(crosses(diagonal_a, diagonal_b))

    def test_two_component_candidate_checks_both_orientations(self):
        case = next(
            case
            for case in build_curated_cases((3,))
            if case.family == "two_glued_cliques"
        )
        graph = support_graph_from_case(case)
        exact = analyze_case_local(case)
        real_oracle = __import__(
            "bl_rctn.local", fromlist=["_aggregated_crossing_vector"]
        )._aggregated_crossing_vector
        calls = []

        def disagreeing_oracle(*args, **kwargs):
            calls.append(args[3])
            vector = real_oracle(*args, **kwargs)
            if len(calls) == 2:
                return tuple(not value for value in vector)
            return vector

        with patch(
            "bl_rctn.local._aggregated_crossing_vector",
            side_effect=disagreeing_oracle,
        ), self.assertRaisesRegex(ValueError, "orientation"):
            compute_tn_aggregated(
                adjacency(graph), exact.cuts, exact.elementary
            )
        self.assertEqual(len(calls), 2)

    def test_large_recursive_support_graph_must_be_k_connected(self):
        nonconsecutive_path = SupportGraph(
            (2, 5, 9, 12), ((2, 5), (5, 9), (9, 12))
        )
        with self.assertRaisesRegex(ValueError, "not 2-connected"):
            analyze_local(nonconsecutive_path, 2)

        small_child = SupportGraph((2, 5, 9), ((2, 5),))
        self.assertEqual(analyze_local(small_child, 2).status.value, "SMALL")


if __name__ == "__main__":
    unittest.main()
