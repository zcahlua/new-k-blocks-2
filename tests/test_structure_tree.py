import unittest
from dataclasses import replace

from bl_rctn.graph import support_graph_from_case
from bl_rctn.local import analyze_case_local
from bl_rctn.models import Separation, StructureTree, SupportGraph
from bl_rctn.samples import build_curated_cases
from bl_rctn.structure_tree import build_structure_tree, verify_structure_tree


EXPECTED_TREE_SIZE = {
    "small_clique": (1, 0),
    "high_clique": (1, 0),
    "joined_cycle": (1, 0),
    "complete_bipartite": (6, 5),
    "two_glued_cliques": (2, 1),
    "three_glued_cliques": (4, 3),
}


class StructureTreeTests(unittest.TestCase):
    def test_all_curated_trees_satisfy_size_and_tree_axioms(self):
        totals = [0, 0]
        for case in build_curated_cases():
            with self.subTest(case=case.case_id):
                graph = support_graph_from_case(case)
                local = analyze_case_local(case)
                tree = build_structure_tree(graph, local.tn_aggregated)
                observed = len(tree.nodes), len(tree.edges)
                self.assertEqual(observed, EXPECTED_TREE_SIZE[case.family])
                self.assertEqual(len(tree.nodes), len(local.tn_aggregated) + 1)
                self.assertEqual(len(tree.edges), len(local.tn_aggregated))
                self.assertEqual(
                    verify_structure_tree(graph, local.tn_aggregated, tree), ()
                )
                totals[0] += observed[0]
                totals[1] += observed[1]
        self.assertEqual(totals, [45, 27])

    def test_one_cut_and_shared_boundary_star(self):
        cases = build_curated_cases((3,))
        for family, counts in (
            ("two_glued_cliques", (2, 1)),
            ("three_glued_cliques", (4, 3)),
        ):
            case = next(case for case in cases if case.family == family)
            graph = support_graph_from_case(case)
            local = analyze_case_local(case)
            tree = build_structure_tree(graph, local.tn_aggregated)
            self.assertEqual((len(tree.nodes), len(tree.edges)), counts)
            if family == "three_glued_cliques":
                degrees = {node.node_id: 0 for node in tree.nodes}
                for edge in tree.edges:
                    degrees[edge.source_node_id] += 1
                    degrees[edge.target_node_id] += 1
                self.assertEqual(sorted(degrees.values()), [1, 1, 1, 3])
                self.assertEqual(
                    {edge.adhesion for edge in tree.edges},
                    {tuple(range(case.k))},
                )
                self.assertEqual(
                    len({edge.separation for edge in tree.edges}), 3
                )

    def test_zero_tn_has_one_full_graph_bag(self):
        case = next(
            case
            for case in build_curated_cases((4,))
            if case.family == "high_clique"
        )
        graph = support_graph_from_case(case)
        tree = build_structure_tree(graph, ())
        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(tree.nodes[0].orientation_signature, ())
        self.assertEqual(tree.nodes[0].bag_vertices, graph.vertices)
        self.assertEqual(tree.edges, ())

    def test_verifier_detects_missing_edge_and_wrong_bag(self):
        case = next(
            case
            for case in build_curated_cases((3,))
            if case.family == "three_glued_cliques"
        )
        graph = support_graph_from_case(case)
        tn = analyze_case_local(case).tn_aggregated
        tree = build_structure_tree(graph, tn)

        missing_edge = StructureTree(tree.nodes, tree.edges[:-1])
        missing_issues = verify_structure_tree(graph, tn, missing_edge)
        self.assertTrue(missing_issues)
        self.assertTrue(
            any("edge_count" in issue or "connected" in issue for issue in missing_issues)
        )

        victim = max(tree.nodes, key=lambda node: len(node.bag_vertices))
        wrong_node = replace(victim, bag_vertices=(victim.bag_vertices[0],))
        wrong_bag = StructureTree(
            tuple(wrong_node if node.node_id == victim.node_id else node for node in tree.nodes),
            tree.edges,
        )
        bag_issues = verify_structure_tree(graph, tn, wrong_bag)
        self.assertTrue(any("bag" in issue for issue in bag_issues))

    def test_nonessential_or_crossing_input_is_rejected(self):
        graph = SupportGraph((0, 1, 2, 3), ())
        diagonal_a = Separation((0, 1), (2, 3), ())
        diagonal_b = Separation((0, 2), (1, 3), ())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_structure_tree(graph, (diagonal_a, diagonal_a))
        with self.assertRaisesRegex(ValueError, "nested"):
            build_structure_tree(graph, (diagonal_a, diagonal_b))


if __name__ == "__main__":
    unittest.main()
