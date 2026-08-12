import copy
import hashlib
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

# Import the Task 7 module first so the RED gate diagnoses this task, even while
# the independently assigned hierarchy module is still being completed.
from bl_rctn.verify import (
    publish_verification_reports_once,
    verify_case_result,
    verify_run,
)
from bl_rctn.export import export_case_bundle, export_run_summary
from bl_rctn.hierarchy import decompose_case
from bl_rctn.models import (
    CompletionLevel,
    Coverage,
    DemoConfig,
    EdgeRecord,
    GraphCase,
    HierarchyResult,
    InterfaceRef,
    Separation,
    canonical_json_bytes,
    read_verification_report,
    write_demo_config_once,
    write_graph_cases_once,
    write_hierarchy_result_once,
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


def unsafe_change(value, **changes):
    forged = copy.copy(value)
    for field, replacement in changes.items():
        object.__setattr__(forged, field, replacement)
    return forged


def replace_record(
    hierarchy: HierarchyResult, record_id: str, replacement
) -> HierarchyResult:
    records = tuple(
        replacement if record.record_id == record_id else record
        for record in hierarchy.records
    )
    return replace(hierarchy, records=records)


def map_vertices(vertices, inverse):
    return tuple(sorted(inverse[vertex] for vertex in vertices))


def map_edge(edge, inverse):
    return tuple(sorted((inverse[edge[0]], inverse[edge[1]])))


def map_separation(value: Separation, inverse):
    sides = sorted(
        (map_vertices(value.side_a, inverse), map_vertices(value.side_b, inverse))
    )
    return sides[0], sides[1], map_vertices(value.separator, inverse)


def transported_signature(hierarchy: HierarchyResult, inverse):
    record_keys = {
        record.record_id: (
            record.depth,
            map_vertices(record.bag_vertices, inverse),
            record.status.value,
        )
        for record in hierarchy.records
    }
    records = []
    for record in hierarchy.records:
        records.append(
            (
                record_keys[record.record_id],
                None
                if record.parent_record_id is None
                else record_keys[record.parent_record_id],
                tuple(sorted(map_edge(edge, inverse) for edge in record.support_edges)),
                tuple(
                    sorted(
                        (
                            map_edge(edge.endpoints, inverse),
                            edge.is_root_real,
                            len(edge.virtual_interface_ids),
                        )
                        for edge in record.edge_records
                    )
                ),
                tuple(
                    sorted(
                        (ref.coverage.value, map_vertices(ref.local_boundary, inverse))
                        for ref in record.interface_refs
                    )
                ),
                tuple(
                    sorted(
                        map_separation(separation, inverse)
                        for separation in record.local_result.tn_full
                    )
                ),
            )
        )
    interfaces = tuple(
        sorted(
            (
                map_vertices(interface.boundary, inverse),
                record_keys[interface.creator_record_id],
                len(interface.incidences),
            )
            for interface in hierarchy.interfaces
        )
    )
    hierarchy_edges = tuple(
        sorted(
            (
                record_keys[edge.parent_record_id],
                record_keys[edge.child_record_id],
            )
            for edge in hierarchy.hierarchy_edges
        )
    )
    terminals = tuple(sorted(record_keys[value] for value in hierarchy.terminal_record_ids))
    return tuple(sorted(records)), interfaces, hierarchy_edges, terminals


def relabel_case(case: GraphCase, round_index: int):
    permutation = list(range(case.num_nodes))
    random.Random(case.seed ^ (case.k << 20) ^ round_index).shuffle(permutation)
    if permutation == list(range(case.num_nodes)):
        permutation = permutation[1:] + permutation[:1]
    inverse = {new: old for old, new in enumerate(permutation)}
    edges = tuple(
        tuple(sorted((permutation[left], permutation[right])))
        for left, right in case.edges
    )
    relabeled = GraphCase(
        f"{case.case_id}-r{round_index}",
        case.k,
        case.num_nodes,
        edges,
        case.family,
        case.parameters,
        case.seed + round_index + 1,
        case.expected,
    )
    return relabeled, inverse


class VerifyTests(unittest.TestCase):
    def test_all_cases_and_config_binding(self):
        for case in build_curated_cases():
            with self.subTest(case=case.case_id):
                candidate = decompose_case(case, CONFIG)
                report = verify_case_result(case, candidate, CONFIG)
                self.assertTrue(report.verified, report.issues)
                self.assertEqual(
                    report.completion_level, CompletionLevel.RECURSIVE_VERIFIED
                )
                tampered = replace(candidate, config_digest="0" * 64)
                rejected = verify_case_result(case, tampered, CONFIG)
                self.assertFalse(rejected.verified)
                self.assertEqual(
                    rejected.completion_level,
                    CompletionLevel.RECURSIVE_CANDIDATE,
                )

    def test_forged_real_edge_is_rejected(self):
        case = next(
            value
            for value in build_curated_cases((2,))
            if value.family == "complete_bipartite"
        )
        candidate = decompose_case(case, CONFIG)
        target = next(
            record
            for record in candidate.records
            if any(not edge.is_root_real for edge in record.edge_records)
        )
        virtual = next(edge for edge in target.edge_records if not edge.is_root_real)
        forged = replace(
            virtual,
            is_root_real=True,
            root_edge_id="forged-root-edge",
        )
        changed = replace(
            target,
            edge_records=tuple(
                forged if edge.endpoints == virtual.endpoints else edge
                for edge in target.edge_records
            ),
        )
        tampered = replace_record(candidate, target.record_id, changed)
        self.assertFalse(verify_case_result(case, tampered, CONFIG).verified)

    def test_missing_incidence_and_upgraded_partial_are_rejected(self):
        case = next(
            value
            for value in build_curated_cases((3,))
            if value.family == "two_glued_cliques"
        )
        candidate = decompose_case(case, CONFIG)
        interface = candidate.interfaces[0]
        broken_interface = unsafe_change(interface, incidences=interface.incidences[:1])
        missing = unsafe_change(candidate, interfaces=(broken_interface,))
        self.assertFalse(verify_case_result(case, missing, CONFIG).verified)

        target = next(record for record in candidate.records if record.interface_refs)
        ref = target.interface_refs[0]
        self.assertGreaterEqual(len(ref.local_boundary), 2)
        upgraded = InterfaceRef(
            ref.interface_id,
            ref.incidence_id,
            Coverage.FULL,
            ref.local_boundary[:1],
        )
        changed = replace(
            target,
            interface_refs=tuple(
                upgraded if value == ref else value for value in target.interface_refs
            ),
        )
        tampered = replace_record(candidate, target.record_id, changed)
        self.assertFalse(verify_case_result(case, tampered, CONFIG).verified)

    def test_wrong_bag_omitted_tn_and_wrong_terminal_are_rejected(self):
        case = next(
            value
            for value in build_curated_cases((2,))
            if value.family == "two_glued_cliques"
        )
        candidate = decompose_case(case, CONFIG)
        terminal = next(
            record
            for record in candidate.records
            if record.record_id in candidate.terminal_record_ids
        )
        wrong_record = unsafe_change(
            terminal, bag_vertices=terminal.bag_vertices[:-1]
        )
        wrong_bag = unsafe_change(
            candidate,
            records=tuple(
                wrong_record if value.record_id == terminal.record_id else value
                for value in candidate.records
            ),
        )
        self.assertFalse(verify_case_result(case, wrong_bag, CONFIG).verified)

        root = next(
            record for record in candidate.records if record.record_id == candidate.root_record_id
        )
        self.assertTrue(root.local_result.tn_full)
        omitted_local = replace(
            root.local_result, tn_full=root.local_result.tn_full[1:]
        )
        omitted_record = replace(root, local_result=omitted_local)
        omitted = replace_record(candidate, root.record_id, omitted_record)
        self.assertFalse(verify_case_result(case, omitted, CONFIG).verified)

        wrong_terminal = replace(candidate, terminal_record_ids=(candidate.root_record_id,))
        self.assertFalse(verify_case_result(case, wrong_terminal, CONFIG).verified)

    def test_three_deterministic_relabelings_per_case(self):
        for case in build_curated_cases():
            original = decompose_case(case, CONFIG)
            expected = transported_signature(
                original, {vertex: vertex for vertex in range(case.num_nodes)}
            )
            for round_index in range(3):
                with self.subTest(case=case.case_id, round=round_index):
                    relabeled, inverse = relabel_case(case, round_index)
                    candidate = decompose_case(relabeled, CONFIG)
                    report = verify_case_result(relabeled, candidate, CONFIG)
                    self.assertTrue(report.verified, report.issues)
                    self.assertEqual(
                        transported_signature(candidate, inverse), expected
                    )

    def test_report_publication_is_write_once_and_verify_run_is_read_only(self):
        case = build_curated_cases((2,))[0]
        candidate = decompose_case(case, CONFIG)
        scratch = Path.cwd() / "runs" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            run_dir = Path(raw) / "run"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "engine").mkdir()
            write_demo_config_once(run_dir / "input" / "config.json", CONFIG)
            write_graph_cases_once(run_dir / "input" / "cases.jsonl", (case,))
            write_hierarchy_result_once(
                run_dir / "engine" / f"{case.case_id}.json", candidate
            )

            paths = publish_verification_reports_once(run_dir)
            self.assertEqual(
                paths,
                (run_dir / "verification" / f"{case.case_id}.json",),
            )
            before = paths[0].read_bytes()
            self.assertEqual(publish_verification_reports_once(run_dir), paths)
            self.assertEqual(paths[0].read_bytes(), before)
            audit = verify_run(run_dir)
            self.assertEqual(audit["case_count"], 1)
            self.assertTrue(audit["all_verified"])

    def test_full_export_audit_recomputes_bundle_and_manifest(self):
        case = next(
            value
            for value in build_curated_cases((2,))
            if value.family == "two_glued_cliques"
        )
        candidate = decompose_case(case, CONFIG)
        scratch = Path.cwd() / "runs" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            run_dir = Path(raw) / "run"
            (run_dir / "input").mkdir(parents=True)
            (run_dir / "engine").mkdir()
            (run_dir / "gpt").mkdir()
            write_demo_config_once(run_dir / "input" / "config.json", CONFIG)
            write_graph_cases_once(run_dir / "input" / "cases.jsonl", (case,))
            write_hierarchy_result_once(
                run_dir / "engine" / f"{case.case_id}.json", candidate
            )
            report_path = publish_verification_reports_once(run_dir)[0]
            report = read_verification_report(report_path)
            export_case_bundle(
                case,
                candidate,
                report,
                CONFIG,
                run_dir / "gpt" / case.case_id,
            )
            export_run_summary(run_dir)
            audit = verify_run(run_dir)
            self.assertTrue(audit["all_verified"])
            self.assertTrue(audit["gpt_artifacts_present"])

            # Forge one bundle and then update its manifest entry so every
            # listed size/hash remains self-consistent. An auditor that merely
            # trusts the manifest would accept this; exact recomputation must not.
            bundle_path = (
                run_dir / "gpt" / case.case_id / "gpt_visualization_bundle.json"
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["root_graph"]["nodes"][0]["label"] = "forged"
            forged = canonical_json_bytes(bundle)
            bundle_path.write_bytes(forged)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            relative = bundle_path.relative_to(run_dir).as_posix()
            entry = next(value for value in manifest["files"] if value["path"] == relative)
            entry["byte_size"] = len(forged)
            entry["sha256"] = hashlib.sha256(forged).hexdigest()
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaises(ValueError):
                verify_run(run_dir)


if __name__ == "__main__":
    unittest.main()
