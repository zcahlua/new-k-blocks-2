import unittest
from dataclasses import replace

from bl_rctn.hierarchy import (
    HierarchyInvariantError,
    _require_recursive_progress,
    decompose_case,
    propagate_interface_ref,
    reconstruct_root,
)
from bl_rctn.models import (
    CompletionLevel,
    Coverage,
    DemoConfig,
    EdgeRecord,
    HierarchyResult,
    InterfaceRef,
)
from bl_rctn.samples import build_curated_cases


CONFIG = DemoConfig(
    "bl-rctn-demo-config-v1",
    (2, 3, 4),
    20260811,
    0,
    4096,
    20,
    "STRUCTURE_ONLY",
)


def _replace_record(hierarchy, replacement):
    return replace(
        hierarchy,
        records=tuple(
            replacement if record.record_id == replacement.record_id else record
            for record in hierarchy.records
        ),
    )


class HierarchyTests(unittest.TestCase):
    def test_all_curated_cases_reconstruct_from_terminal_records(self):
        for case in build_curated_cases():
            with self.subTest(case=case.case_id):
                hierarchy = decompose_case(case, CONFIG)
                self.assertEqual(
                    hierarchy.completion_level,
                    CompletionLevel.RECURSIVE_CANDIDATE,
                )
                self.assertEqual(
                    reconstruct_root(hierarchy),
                    (tuple(range(case.num_nodes)), case.edges),
                )

    def test_two_glued_cliques_have_two_high_terminals_and_one_interface(self):
        for case in (
            case
            for case in build_curated_cases()
            if case.family == "two_glued_cliques"
        ):
            with self.subTest(case=case.case_id):
                hierarchy = decompose_case(case, CONFIG)
                records = {record.record_id: record for record in hierarchy.records}
                terminals = [records[value] for value in hierarchy.terminal_record_ids]

                self.assertEqual(
                    sorted(record.status.value for record in terminals),
                    ["HIGH", "HIGH"],
                )
                self.assertEqual(len(hierarchy.interfaces), 1)
                interface = hierarchy.interfaces[0]
                self.assertEqual(interface.boundary, tuple(range(case.k)))
                self.assertEqual(len(interface.incidences), 2)
                left, right = interface.incidences
                self.assertEqual(left.opposite_incidence_id, right.incidence_id)
                self.assertEqual(right.opposite_incidence_id, left.incidence_id)

    def test_equal_boundaries_create_distinct_creator_interfaces(self):
        case = next(
            case
            for case in build_curated_cases((3,))
            if case.family == "three_glued_cliques"
        )
        hierarchy = decompose_case(case, CONFIG)

        self.assertEqual(len(hierarchy.interfaces), 3)
        self.assertEqual(
            {interface.boundary for interface in hierarchy.interfaces},
            {tuple(range(case.k))},
        )
        self.assertEqual(
            len({interface.interface_id for interface in hierarchy.interfaces}),
            3,
        )
        self.assertEqual(
            len(
                {
                    (interface.creator_record_id, interface.creator_tree_edge_id)
                    for interface in hierarchy.interfaces
                }
            ),
            3,
        )

    def test_root_real_and_virtual_roles_are_unioned_on_one_edge_record(self):
        case = next(
            case
            for case in build_curated_cases((3,))
            if case.family == "two_glued_cliques"
        )
        hierarchy = decompose_case(case, CONFIG)
        interface_id = hierarchy.interfaces[0].interface_id
        terminal_records = {
            record.record_id: record for record in hierarchy.records
        }

        for terminal_id in hierarchy.terminal_record_ids:
            record = terminal_records[terminal_id]
            boundary_edge = next(
                edge
                for edge in record.edge_records
                if edge.endpoints == (0, 1)
            )
            self.assertTrue(boundary_edge.is_root_real)
            self.assertIsNotNone(boundary_edge.root_edge_id)
            self.assertEqual(
                boundary_edge.virtual_interface_ids,
                (interface_id,),
            )

    def test_interface_fragment_propagation_is_monotone(self):
        full = InterfaceRef("interface", "incidence", Coverage.FULL, (0, 1, 2))

        partial = propagate_interface_ref(full, (0, 1, 2), (0, 1, 3))
        self.assertEqual(
            (partial.coverage, partial.local_boundary),
            (Coverage.PARTIAL, (0, 1)),
        )
        singleton = propagate_interface_ref(full, (0, 1, 2), (0, 3))
        self.assertEqual(
            (singleton.coverage, singleton.local_boundary),
            (Coverage.PARTIAL, (0,)),
        )
        self.assertIsNone(
            propagate_interface_ref(full, (0, 1, 2), (3, 4))
        )
        self.assertEqual(
            propagate_interface_ref(full, (0, 1, 2), (0, 1, 2)).coverage,
            Coverage.FULL,
        )

        inherited_partial = InterfaceRef(
            "interface", "incidence", Coverage.PARTIAL, (0, 1)
        )
        retained = propagate_interface_ref(
            inherited_partial,
            (0, 1, 2),
            (0, 1, 2),
        )
        self.assertEqual(
            (retained.coverage, retained.local_boundary),
            (Coverage.PARTIAL, (0, 1)),
        )

    def test_propagation_rejects_a_malformed_full_parent_fragment(self):
        malformed = InterfaceRef(
            "interface", "incidence", Coverage.FULL, (0, 1)
        )
        with self.assertRaisesRegex(
            HierarchyInvariantError, "FULL interface reference"
        ):
            propagate_interface_ref(
                malformed,
                (0, 1, 2),
                (0, 1),
            )

    def test_reconstruction_ignores_nonterminal_and_purely_virtual_edges(self):
        case = next(
            case
            for case in build_curated_cases((2,))
            if case.family == "two_glued_cliques"
        )
        hierarchy = decompose_case(case, CONFIG)
        root = next(
            record
            for record in hierarchy.records
            if record.record_id == hierarchy.root_record_id
        )
        forged = EdgeRecord((2, 4), True, "root-edge-forged", ())
        tampered_root = replace(root, edge_records=root.edge_records + (forged,))
        tampered = _replace_record(hierarchy, tampered_root)

        self.assertEqual(
            reconstruct_root(tampered),
            (tuple(range(case.num_nodes)), case.edges),
        )

        bipartite = next(
            case
            for case in build_curated_cases((2,))
            if case.family == "complete_bipartite"
        )
        bipartite_hierarchy = decompose_case(bipartite, CONFIG)
        self.assertTrue(
            any(
                edge.virtual_interface_ids and not edge.is_root_real
                for record in bipartite_hierarchy.records
                if record.record_id in bipartite_hierarchy.terminal_record_ids
                for edge in record.edge_records
            )
        )
        self.assertEqual(
            reconstruct_root(bipartite_hierarchy),
            (tuple(range(bipartite.num_nodes)), bipartite.edges),
        )

    def test_reconstruction_rejects_unresolved_terminal_interface_reference(self):
        case = next(
            case
            for case in build_curated_cases((2,))
            if case.family == "two_glued_cliques"
        )
        hierarchy = decompose_case(case, CONFIG)
        terminal_id = hierarchy.terminal_record_ids[0]
        terminal = next(
            record for record in hierarchy.records if record.record_id == terminal_id
        )
        unresolved = InterfaceRef(
            "missing-interface",
            "missing-incidence",
            Coverage.PARTIAL,
            (terminal.bag_vertices[0],),
        )
        tampered = _replace_record(
            hierarchy,
            replace(
                terminal,
                interface_refs=terminal.interface_refs + (unresolved,),
            ),
        )

        with self.assertRaisesRegex(
            HierarchyInvariantError, "unresolved interface reference"
        ):
            reconstruct_root(tampered)

    def test_recursive_guard_rejects_nonprogress_cycle_and_depth_overflow(self):
        _require_recursive_progress(
            parent_vertices=(0, 1, 2, 3),
            child_vertices=(0, 1, 2),
            child_depth=1,
            max_depth=1,
            semantic_state="child-state",
            path_states=frozenset({"root-state"}),
        )

        with self.assertRaisesRegex(HierarchyInvariantError, "proper child"):
            _require_recursive_progress(
                parent_vertices=(0, 1, 2),
                child_vertices=(0, 1, 2),
                child_depth=1,
                max_depth=2,
                semantic_state="child-state",
                path_states=frozenset({"root-state"}),
            )
        with self.assertRaisesRegex(HierarchyInvariantError, "depth"):
            _require_recursive_progress(
                parent_vertices=(0, 1, 2),
                child_vertices=(0, 1),
                child_depth=2,
                max_depth=1,
                semantic_state="child-state",
                path_states=frozenset({"root-state"}),
            )
        with self.assertRaisesRegex(
            HierarchyInvariantError, "repeated semantic state"
        ):
            _require_recursive_progress(
                parent_vertices=(0, 1, 2),
                child_vertices=(0, 1),
                child_depth=1,
                max_depth=2,
                semantic_state="cycle-state",
                path_states=frozenset({"cycle-state"}),
            )

    def test_hierarchy_result_has_closed_references_and_strict_progress(self):
        for case in build_curated_cases():
            with self.subTest(case=case.case_id):
                hierarchy = decompose_case(case, CONFIG)
                records = {record.record_id: record for record in hierarchy.records}
                interfaces = {
                    interface.interface_id: interface
                    for interface in hierarchy.interfaces
                }
                self.assertIn(hierarchy.root_record_id, records)
                self.assertEqual(records[hierarchy.root_record_id].depth, 0)

                for edge in hierarchy.hierarchy_edges:
                    parent = records[edge.parent_record_id]
                    child = records[edge.child_record_id]
                    self.assertEqual(child.parent_record_id, parent.record_id)
                    self.assertEqual(child.depth, parent.depth + 1)
                    self.assertLess(len(child.bag_vertices), len(parent.bag_vertices))
                    self.assertTrue(set(child.bag_vertices).issubset(parent.bag_vertices))

                for record in hierarchy.records:
                    self.assertEqual(
                        record.support_edges,
                        tuple(edge.endpoints for edge in record.edge_records),
                    )
                    for ref in record.interface_refs:
                        interface = interfaces[ref.interface_id]
                        incidences = {
                            value.incidence_id: value
                            for value in interface.incidences
                        }
                        self.assertIn(ref.incidence_id, incidences)
                        self.assertTrue(
                            set(ref.local_boundary).issubset(interface.boundary)
                        )


if __name__ == "__main__":
    unittest.main()
