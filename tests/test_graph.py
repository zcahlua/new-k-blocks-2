import unittest

from bl_rctn.graph import (
    adjacency,
    components_after_deleting,
    is_k_connected,
    support_graph_from_case,
    validate_root_case,
    validate_support_graph,
)
from bl_rctn.models import GraphCase, SupportGraph


class GraphTests(unittest.TestCase):
    def test_cycle_is_two_connected(self):
        case = GraphCase(
            "c6",
            2,
            6,
            tuple((i, (i + 1) % 6) for i in range(6)),
            "cycle",
            (),
            1,
            (),
        )
        adj = adjacency(support_graph_from_case(case))
        self.assertTrue(is_k_connected(adj, 2))
        self.assertEqual(
            components_after_deleting(adj, frozenset({0, 3})),
            ((1, 2), (4, 5)),
        )

    def test_path_is_rejected(self):
        case = GraphCase(
            "p4", 2, 4, ((0, 1), (1, 2), (2, 3)), "path", (), 1, ()
        )
        with self.assertRaisesRegex(ValueError, "not 2-connected"):
            validate_root_case(case)

    def test_support_graph_preserves_nonconsecutive_root_ids(self):
        graph = SupportGraph((2, 5, 9), ((9, 2),))
        validate_support_graph(graph)
        self.assertEqual(graph.vertices, (2, 5, 9))
        self.assertEqual(graph.edges, ((2, 9),))
        self.assertEqual(
            adjacency(graph),
            {2: frozenset({9}), 5: frozenset(), 9: frozenset({2})},
        )
        self.assertEqual(
            components_after_deleting(adjacency(graph), frozenset()),
            ((2, 9), (5,)),
        )

    def test_k_connectivity_uses_all_smaller_deletion_sets(self):
        triangle = GraphCase(
            "triangle", 2, 3, ((0, 1), (1, 2), (0, 2)), "clique", (), 0, ()
        )
        self.assertTrue(
            is_k_connected(adjacency(support_graph_from_case(triangle)), 2)
        )
        too_small = SupportGraph((0, 1), ((0, 1),))
        self.assertFalse(is_k_connected(adjacency(too_small), 2))

    def test_root_gate_rejects_too_few_vertices(self):
        case = GraphCase("edge", 2, 2, ((0, 1),), "manual", (), 0, ())
        with self.assertRaisesRegex(ValueError, "not 2-connected"):
            validate_root_case(case)

    def test_deleted_vertices_must_belong_to_graph(self):
        graph = SupportGraph((0, 1), ((0, 1),))
        with self.assertRaisesRegex(ValueError, "unknown deleted vertex"):
            components_after_deleting(adjacency(graph), frozenset({2}))


if __name__ == "__main__":
    unittest.main()
