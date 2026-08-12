import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bl_rctn.export import export_case_bundle, export_run_summary
from bl_rctn.hierarchy import decompose_case
from bl_rctn.local import analyze_case_local
from bl_rctn.models import (
    CompletionLevel,
    DemoConfig,
    EdgeRecord,
    HierarchyResult,
    InterfaceIncidence,
    InterfaceObject,
    TorsoRecord,
    VerificationReport,
    canonical_sha256,
    demo_config_to_dict,
    graph_case_to_dict,
    hierarchy_result_to_dict,
    stable_id,
    write_demo_config_once,
    write_graph_cases_once,
    write_hierarchy_result_once,
    write_verification_report_once,
)
from bl_rctn.samples import build_curated_cases
from bl_rctn.verify import verify_case_result, verify_run


CONFIG = DemoConfig(
    "bl-rctn-demo-config-v1",
    (2, 3, 4),
    20260811,
    0,
    4096,
    20,
    "STRUCTURE_ONLY",
)
EXPECTED_FILES = {
    "gpt_visualization_bundle.json",
    "gpt_prompt.md",
    "root_graph.mmd",
    "hierarchy.mmd",
    "crossing_graph.mmd",
    "graph.dot",
}
SUMMARY_FIELDS = [
    "case_id",
    "k",
    "n",
    "m",
    "status",
    "cutset_count",
    "full_sigma_count",
    "elementary_count",
    "tn_count",
    "hierarchy_depth",
    "small_terminals",
    "high_terminals",
    "crossed_terminals",
    "verified",
    "bundle_path",
]


def fixture(family: str):
    case = next(
        case
        for case in build_curated_cases((2,))
        if case.family == family
    )
    local = analyze_case_local(case)
    interface_id = "interface-fixture"
    edge_records = tuple(
        EdgeRecord(
            edge,
            True,
            stable_id(
                "root-edge", {"case_id": case.case_id, "endpoints": edge}
            ),
            (interface_id,) if index == 0 else (),
        )
        for index, edge in enumerate(case.edges)
    )
    record = TorsoRecord(
        "record-root",
        None,
        0,
        tuple(range(case.num_nodes)),
        local.status,
        case.edges,
        edge_records,
        (),
        local,
        None,
    )
    left = InterfaceIncidence("incidence-left", "tree-side-left", "incidence-right")
    right = InterfaceIncidence("incidence-right", "tree-side-right", "incidence-left")
    interface = InterfaceObject(
        interface_id,
        record.record_id,
        "tree-edge-fixture",
        (0, 1),
        (left, right),
    )
    config_digest = canonical_sha256(demo_config_to_dict(CONFIG))
    hierarchy = HierarchyResult(
        record.record_id,
        config_digest,
        (record,),
        (interface,),
        (),
        (record.record_id,),
        CompletionLevel.RECURSIVE_CANDIDATE,
    )
    report = VerificationReport(
        "bl-rctn-verification-v1",
        "1.0.0",
        case.case_id,
        canonical_sha256(graph_case_to_dict(case)),
        canonical_sha256(hierarchy_result_to_dict(hierarchy)),
        config_digest,
        True,
        CompletionLevel.RECURSIVE_VERIFIED,
        (("fixture", "PASS"),),
        (),
    )
    return case, hierarchy, report


class ExportTests(unittest.TestCase):
    def scratch(self):
        root = Path.cwd() / "runs" / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_verified_only_exact_deterministic_six_file_bundle(self):
        case, hierarchy, report = fixture("high_clique")
        with self.scratch() as raw:
            root = Path(raw)
            first = export_case_bundle(
                case, hierarchy, report, CONFIG, root / "first"
            )
            second = export_case_bundle(
                case, hierarchy, report, CONFIG, root / "second"
            )
            self.assertEqual({path.name for path in first}, EXPECTED_FILES)
            self.assertEqual({path.name for path in second}, EXPECTED_FILES)
            first_bytes = {path.name: path.read_bytes() for path in first}
            second_bytes = {path.name: path.read_bytes() for path in second}
            self.assertEqual(first_bytes, second_bytes)
            self.assertTrue(
                all(str(root).encode("utf-8") not in payload for payload in first_bytes.values())
            )

            bundle = json.loads(first_bytes["gpt_visualization_bundle.json"])
            self.assertEqual(
                set(bundle),
                {
                    "schema_version",
                    "case",
                    "root_graph",
                    "cutsets",
                    "separations",
                    "crossing_edges",
                    "hierarchy",
                    "torsos",
                    "interfaces",
                    "ambient_blocks",
                    "block_localization",
                    "verification",
                    "style_hints",
                    "view_recipes",
                },
            )
            self.assertEqual(
                bundle["schema_version"], "bl-rctn-gpt-visualization-v1"
            )
            self.assertIsNone(bundle["ambient_blocks"])
            self.assertIsNone(bundle["block_localization"])
            self.assertEqual(
                bundle["style_hints"],
                {
                    "edge_roles": {
                        "ROOT_REAL": {"color": "black", "line_style": "solid"},
                        "VIRTUAL_INTERFACE": {
                            "color": "blue",
                            "line_style": "dashed",
                        },
                    },
                    "separator": {"color": "orange"},
                    "tn": {"color": "green"},
                    "rejected_or_crossing": {"color": "red"},
                    "statuses": {
                        "SMALL": "grey",
                        "HIGH": "green",
                        "CROSSED": "red",
                        "SPLIT": "blue",
                    },
                },
            )
            self.assertIn(
                ["ROOT_REAL", "VIRTUAL_INTERFACE:interface-fixture"],
                [edge["roles"] for edge in bundle["torsos"][0]["edges"]],
            )
            prompt = first_bytes["gpt_prompt.md"].decode("utf-8")
            for panel in (
                "root graph and separators",
                "elementary crossing graph",
                "recursive hierarchy",
                "interface provenance",
            ):
                self.assertIn(panel, prompt)
            self.assertIn("Do not invent nodes, edges, cuts, blocks, or labels", prompt)
            self.assertIn(
                "STRUCTURE_ONLY; ambient blocks not computed", prompt
            )
            dot = first_bytes["graph.dot"].decode("utf-8")
            self.assertIn('color="black", style="solid"', dot)
            self.assertIn('color="blue", style="dashed"', dot)

            with self.assertRaises(FileExistsError):
                export_case_bundle(case, hierarchy, report, CONFIG, root / "first")

    def test_invalid_bindings_fail_before_destination_creation(self):
        case, hierarchy, report = fixture("high_clique")
        tampered_hierarchy = replace(hierarchy, config_digest="0" * 64)
        rebound_to_tampered = replace(
            report,
            candidate_hierarchy_digest=canonical_sha256(
                hierarchy_result_to_dict(tampered_hierarchy)
            ),
        )
        attempts = (
            (hierarchy, replace(report, verified=False), "unverified"),
            (hierarchy, replace(report, case_id="wrong-case"), "case-id"),
            (hierarchy, replace(report, case_digest="0" * 64), "case-digest"),
            (
                hierarchy,
                replace(report, candidate_hierarchy_digest="0" * 64),
                "hierarchy-digest",
            ),
            (hierarchy, replace(report, config_digest="0" * 64), "config-digest"),
            (
                hierarchy,
                replace(report, completion_level=CompletionLevel.RECURSIVE_CANDIDATE),
                "completion",
            ),
            (tampered_hierarchy, rebound_to_tampered, "candidate-config"),
        )
        with self.scratch() as raw:
            root = Path(raw)
            for candidate, invalid_report, label in attempts:
                destination = root / label
                with self.subTest(label=label), self.assertRaises(ValueError):
                    export_case_bundle(
                        case,
                        candidate,
                        invalid_report,
                        CONFIG,
                        destination,
                    )
                self.assertFalse(destination.exists())

    def test_atomic_publication_cleans_destination_after_mid_write_failure(self):
        case, hierarchy, report = fixture("high_clique")
        with self.scratch() as raw:
            destination = Path(raw) / "failed-publication"
            with patch(
                "bl_rctn.export.os.fsync",
                # Publish the first artifact (file + directory fsync), then
                # fail while staging the second artifact.
                side_effect=(
                    None,
                    None,
                    OSError("simulated storage failure"),
                ),
            ), self.assertRaisesRegex(OSError, "simulated storage failure"):
                export_case_bundle(
                    case, hierarchy, report, CONFIG, destination
                )
            self.assertFalse(destination.exists())

    def test_rejected_elementary_separations_export_real_witnesses(self):
        case, hierarchy, report = fixture("joined_cycle")
        with self.scratch() as raw:
            paths = export_case_bundle(
                case, hierarchy, report, CONFIG, Path(raw) / "joined"
            )
            bundle_path = next(
                path
                for path in paths
                if path.name == "gpt_visualization_bundle.json"
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle["cutsets"]), 9)
            self.assertEqual(len(bundle["separations"]), 9)
            self.assertEqual(len(bundle["crossing_edges"]), 9)
            for separation in bundle["separations"]:
                self.assertEqual(
                    {
                        "side_a",
                        "side_b",
                        "separator",
                        "is_elementary",
                        "is_tn",
                        "rejection_witness_id",
                    }.difference(separation),
                    set(),
                )
                self.assertTrue(separation["is_elementary"])
                self.assertFalse(separation["is_tn"])
                self.assertIsInstance(separation["rejection_witness_id"], str)

    def test_run_summary_and_manifest_are_closed_and_relative(self):
        case, hierarchy, report = fixture("high_clique")
        with self.scratch() as raw:
            run_dir = Path(raw) / "run"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "engine").mkdir()
            (run_dir / "verification").mkdir()
            (run_dir / "gpt").mkdir()
            write_demo_config_once(run_dir / "input" / "config.json", CONFIG)
            write_graph_cases_once(run_dir / "input" / "cases.jsonl", (case,))
            write_hierarchy_result_once(
                run_dir / "engine" / f"{case.case_id}.json", hierarchy
            )
            write_verification_report_once(
                run_dir / "verification" / f"{case.case_id}.json", report
            )
            export_case_bundle(
                case,
                hierarchy,
                report,
                CONFIG,
                run_dir / "gpt" / case.case_id,
            )

            summary_path, manifest_path = export_run_summary(run_dir)
            self.assertEqual(
                (summary_path.name, manifest_path.name),
                ("summary.csv", "manifest.json"),
            )
            with summary_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, SUMMARY_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], case.case_id)
            self.assertEqual(rows[0]["status"], "HIGH")
            self.assertEqual(rows[0]["high_terminals"], "1")
            self.assertEqual(rows[0]["verified"], "true")
            self.assertEqual(
                rows[0]["bundle_path"],
                f"gpt/{case.case_id}/gpt_visualization_bundle.json",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "bl-rctn-run-manifest-v1")
            self.assertEqual(manifest["seed"], 20260811)
            self.assertEqual(manifest["case_count"], 1)
            self.assertTrue(manifest["all_verified"])
            self.assertEqual(
                manifest["config_digest"],
                canonical_sha256(demo_config_to_dict(CONFIG)),
            )
            listed_paths = [entry["path"] for entry in manifest["files"]]
            self.assertEqual(listed_paths, sorted(listed_paths))
            self.assertNotIn("manifest.json", listed_paths)
            self.assertIn("summary.csv", listed_paths)
            self.assertTrue(all(not Path(value).is_absolute() for value in listed_paths))
            for entry in manifest["files"]:
                payload = (run_dir / entry["path"]).read_bytes()
                self.assertEqual(entry["byte_size"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertNotIn(str(run_dir), manifest_path.read_text(encoding="utf-8"))

            with self.assertRaises(FileExistsError):
                export_run_summary(run_dir)

    def test_real_verified_split_hierarchy_exports_and_reaudits(self):
        case = next(
            case
            for case in build_curated_cases((3,))
            if case.family == "two_glued_cliques"
        )
        hierarchy = decompose_case(case, CONFIG)
        report = verify_case_result(case, hierarchy, CONFIG)
        self.assertTrue(report.verified, report.issues)
        self.assertEqual(
            report.completion_level, CompletionLevel.RECURSIVE_VERIFIED
        )

        with self.scratch() as raw:
            run_dir = Path(raw) / "real-split-run"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "engine").mkdir()
            (run_dir / "verification").mkdir()
            (run_dir / "gpt").mkdir()
            write_demo_config_once(run_dir / "input" / "config.json", CONFIG)
            write_graph_cases_once(run_dir / "input" / "cases.jsonl", (case,))
            write_hierarchy_result_once(
                run_dir / "engine" / f"{case.case_id}.json", hierarchy
            )
            write_verification_report_once(
                run_dir / "verification" / f"{case.case_id}.json", report
            )
            paths = export_case_bundle(
                case,
                hierarchy,
                report,
                CONFIG,
                run_dir / "gpt" / case.case_id,
            )
            export_run_summary(run_dir)

            bundle_path = next(
                path
                for path in paths
                if path.name == "gpt_visualization_bundle.json"
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle["interfaces"]), 1)
            self.assertEqual(len(bundle["hierarchy"]["nodes"]), 3)
            self.assertTrue(
                any(
                    any(role.startswith("VIRTUAL_INTERFACE:") for role in edge["roles"])
                    for torso in bundle["torsos"]
                    for edge in torso["edges"]
                )
            )
            audit = verify_run(run_dir)
            self.assertEqual(audit["case_count"], 1)
            self.assertTrue(audit["all_verified"])
            self.assertTrue(audit["gpt_artifacts_present"])


if __name__ == "__main__":
    unittest.main()
