import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from bl_rctn.models import (
    CompletionLevel,
    Coverage,
    CutRecord,
    DemoConfig,
    EdgeRecord,
    GraphCase,
    HierarchyEdge,
    HierarchyResult,
    InterfaceIncidence,
    InterfaceObject,
    InterfaceRef,
    LocalResult,
    LocalStatus,
    Separation,
    StructureTree,
    StructureTreeEdge,
    StructureTreeNode,
    TorsoRecord,
    VerificationReport,
    canonical_json_bytes,
    canonical_sha256,
    graph_case_from_dict,
    graph_case_to_dict,
    hierarchy_result_from_dict,
    hierarchy_result_to_dict,
    read_demo_config,
    read_graph_cases,
    read_hierarchy_result,
    read_verification_report,
    stable_id,
    verification_report_from_dict,
    verification_report_to_dict,
    write_demo_config_once,
    write_graph_cases_once,
    write_hierarchy_result_once,
    write_json_once,
    write_verification_report_once,
)


def sample_hierarchy() -> HierarchyResult:
    separation = Separation((0, 1), (1, 2), (1,))
    cut = CutRecord((1,), ((0,), (2,)))
    local = LocalResult(
        LocalStatus.SPLIT,
        (cut,),
        (separation,),
        (separation,),
        (separation,),
        (separation,),
        (separation,),
        (),
        CompletionLevel.LOCAL_EXACT,
    )
    left_node = StructureTreeNode("node-left", (), (0, 1))
    right_node = StructureTreeNode("node-right", (), (1, 2))
    tree_edge = StructureTreeEdge(
        "tree-edge", "node-left", "node-right", separation, (1,)
    )
    tree = StructureTree((left_node, right_node), (tree_edge,))
    incidence_left = InterfaceIncidence("inc-left", "node-left", "inc-right")
    incidence_right = InterfaceIncidence("inc-right", "node-right", "inc-left")
    interface = InterfaceObject(
        "interface-1",
        "record-root",
        "tree-edge",
        (1,),
        (incidence_left, incidence_right),
    )
    edge_record = EdgeRecord((0, 1), True, "root-edge-01", ())
    interface_ref = InterfaceRef("interface-1", "inc-left", Coverage.FULL, (1,))
    record = TorsoRecord(
        "record-root",
        None,
        0,
        (0, 1, 2),
        LocalStatus.SPLIT,
        ((0, 1), (1, 2)),
        (edge_record,),
        (interface_ref,),
        local,
        tree,
    )
    hierarchy_edge = HierarchyEdge("record-root", "record-child", "node-left")
    return HierarchyResult(
        "record-root",
        "a" * 64,
        (record,),
        (interface,),
        (hierarchy_edge,),
        ("record-root",),
        CompletionLevel.RECURSIVE_CANDIDATE,
    )


class ModelTests(unittest.TestCase):
    def scratch(self):
        root = Path.cwd() / "runs" / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_edges_are_normalized(self):
        case = GraphCase("x", 2, 4, ((3, 1), (0, 2)), "manual", (), 7, ())
        self.assertEqual(case.edges, ((0, 2), (1, 3)))

    def test_loop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self-loop"):
            GraphCase("x", 2, 3, ((1, 1),), "manual", (), 7, ())

    def test_write_once(self):
        case = GraphCase(
            "x", 2, 3, ((0, 1), (1, 2), (0, 2)), "clique", (), 7, ()
        )
        with self.scratch() as raw:
            path = Path(raw) / "cases.jsonl"
            write_graph_cases_once(path, (case,))
            self.assertEqual(read_graph_cases(path), (case,))
            with self.assertRaises(FileExistsError):
                write_graph_cases_once(path, (case,))

    def test_publish_scratch_symlink_is_rejected(self):
        with self.scratch() as raw:
            root = Path(raw) / "project"
            scratch_target = Path(raw) / "scratch-target"
            (root / "runs").mkdir(parents=True)
            scratch_target.mkdir()
            (root / "runs" / ".tmp").symlink_to(scratch_target, target_is_directory=True)
            with mock.patch("bl_rctn.models._PROJECT_ROOT", root):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    write_json_once(root / "result.json", {"ok": True})
            self.assertEqual(tuple(scratch_target.iterdir()), ())

    def test_stable_id_ignores_dict_key_order(self):
        self.assertEqual(
            stable_id("x", {"a": 1, "b": 2}),
            stable_id("x", {"b": 2, "a": 1}),
        )
        self.assertRegex(canonical_sha256({"x": 1}), r"^[0-9a-f]{64}$")
        self.assertEqual(
            canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}'
        )

    def test_graph_case_is_frozen_and_strict(self):
        case = GraphCase("x", 2, 3, ((0, 1),), "manual", (), 0, ())
        with self.assertRaises(FrozenInstanceError):
            case.case_id = "y"
        bad_values = (
            ("boolean k", ("x", True, 3, (), "manual", (), 0, ())),
            ("unsafe id", ("../x", 2, 3, (), "manual", (), 0, ())),
            ("duplicate edge", ("x", 2, 3, ((0, 1), (1, 0)), "manual", (), 0, ())),
            ("outside endpoint", ("x", 2, 3, ((0, 3),), "manual", (), 0, ())),
        )
        for label, args in bad_values:
            with self.subTest(label=label), self.assertRaises(ValueError):
                GraphCase(*args)

    def test_config_validation_and_round_trip(self):
        config = DemoConfig(
            "bl-rctn-demo-config-v1", (2, 3, 4), 20260811, 0, 4096, 20,
            "STRUCTURE_ONLY",
        )
        with self.scratch() as raw:
            path = Path(raw) / "config.json"
            write_demo_config_once(path, config)
            self.assertEqual(read_demo_config(path), config)
        with self.assertRaises(ValueError):
            DemoConfig(
                "wrong", (2,), 0, 0, 1, 1, "STRUCTURE_ONLY"
            )
        with self.assertRaises(ValueError):
            DemoConfig(
                "bl-rctn-demo-config-v1", (2, 2), 0, 0, 1, 1,
                "STRUCTURE_ONLY",
            )
        with self.assertRaises(ValueError):
            DemoConfig(
                "bl-rctn-demo-config-v1", (2,), 0, 0, 1, 1,
                "WITH_AMBIENT_BLOCKS",
            )

    def test_closed_schema_and_strict_jsonl(self):
        value = graph_case_to_dict(
            GraphCase("x", 2, 3, ((0, 1),), "manual", (), 0, ())
        )
        self.assertEqual(graph_case_from_dict(value).case_id, "x")
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            graph_case_from_dict({**value, "extra": 1})
        with self.scratch() as raw:
            root = Path(raw)
            duplicate = root / "duplicate.jsonl"
            line = canonical_json_bytes(value)
            duplicate.write_bytes(line + b"\n" + line + b"\n")
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                read_graph_cases(duplicate)
            blank = root / "blank.jsonl"
            blank.write_bytes(line + b"\n\n")
            with self.assertRaisesRegex(ValueError, "blank JSONL row"):
                read_graph_cases(blank)
            malformed = root / "malformed.jsonl"
            malformed.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                read_graph_cases(malformed)
            noncanonical = root / "noncanonical.jsonl"
            noncanonical.write_bytes(line)
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                read_graph_cases(noncanonical)

    def test_hierarchy_and_report_round_trip(self):
        hierarchy = sample_hierarchy()
        self.assertEqual(
            hierarchy_result_from_dict(hierarchy_result_to_dict(hierarchy)),
            hierarchy,
        )
        report = VerificationReport(
            "bl-rctn-verification-v1",
            "1.0.0",
            "case-1",
            "b" * 64,
            "c" * 64,
            "a" * 64,
            True,
            CompletionLevel.RECURSIVE_VERIFIED,
            (("schema", "PASS"),),
            (),
        )
        self.assertEqual(
            verification_report_from_dict(verification_report_to_dict(report)),
            report,
        )
        with self.scratch() as raw:
            root = Path(raw)
            hierarchy_path = root / "hierarchy.json"
            report_path = root / "report.json"
            write_hierarchy_result_once(hierarchy_path, hierarchy)
            write_verification_report_once(report_path, report)
            self.assertEqual(read_hierarchy_result(hierarchy_path), hierarchy)
            self.assertEqual(read_verification_report(report_path), report)

            noncanonical = root / "noncanonical.json"
            noncanonical.write_text(
                json.dumps(hierarchy_result_to_dict(hierarchy), indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                read_hierarchy_result(noncanonical)


if __name__ == "__main__":
    unittest.main()
