from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import platform
import stat
from pathlib import Path

from .models import (
    CompletionLevel,
    DemoConfig,
    GraphCase,
    HierarchyResult,
    LocalStatus,
    Separation,
    VerificationReport,
    canonical_json_bytes,
    canonical_sha256,
    demo_config_to_dict,
    graph_case_to_dict,
    hierarchy_result_to_dict,
    interface_object_to_dict,
    interface_ref_to_dict,
    read_demo_config,
    read_graph_cases,
    read_hierarchy_result,
    read_verification_report,
    separation_to_dict,
    stable_id,
    structure_tree_to_dict,
    verification_report_to_dict,
)


_PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent))
_PUBLISH_COUNTER = itertools.count()
_BUNDLE_SCHEMA = "bl-rctn-gpt-visualization-v1"
_MANIFEST_SCHEMA = "bl-rctn-run-manifest-v1"
_BUNDLE_FILENAMES = (
    "gpt_visualization_bundle.json",
    "gpt_prompt.md",
    "root_graph.mmd",
    "hierarchy.mmd",
    "crossing_graph.mmd",
    "graph.dot",
)
_SUMMARY_FIELDS = (
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
)
_STYLE_HINTS = {
    "edge_roles": {
        "ROOT_REAL": {"color": "black", "line_style": "solid"},
        "VIRTUAL_INTERFACE": {"color": "blue", "line_style": "dashed"},
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
}


def _project_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("path must remain inside the algorithm directory") from exc
    current = _PROJECT_ROOT
    for component in relative.parts:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("symlink path components are forbidden")
    return absolute


def _write_bytes_once(path: Path, payload: bytes) -> None:
    target = _project_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {target.parent}")
    temporary: Path | None = None
    descriptor: int | None = None
    linked = False
    durable = False
    try:
        while descriptor is None:
            candidate = target.parent / (
                f".{target.name}.publish-{os.getpid()}-"
                f"{next(_PUBLISH_COUNTER)}.tmp"
            )
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                temporary = candidate
            except FileExistsError:
                continue
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        linked = True
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        durable = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if linked and not durable:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _publish_files_once(
    directory: Path, artifacts: dict[str, bytes]
) -> tuple[Path, ...]:
    ordered_names = tuple(artifacts)
    targets = tuple(directory / name for name in ordered_names)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    published: list[Path] = []
    try:
        for target, name in zip(targets, ordered_names, strict=True):
            _write_bytes_once(target, artifacts[name])
            published.append(target)
    except BaseException:
        for target in reversed(published):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise
    return targets


def _validate_bindings(
    case: GraphCase,
    hierarchy: HierarchyResult,
    report: VerificationReport,
    config: DemoConfig,
) -> None:
    if type(case) is not GraphCase:
        raise TypeError("case must be GraphCase")
    if type(hierarchy) is not HierarchyResult:
        raise TypeError("hierarchy must be HierarchyResult")
    if type(report) is not VerificationReport:
        raise TypeError("report must be VerificationReport")
    if type(config) is not DemoConfig:
        raise TypeError("config must be DemoConfig")
    config_digest = canonical_sha256(demo_config_to_dict(config))
    if config.mode != "STRUCTURE_ONLY":
        raise ValueError("only STRUCTURE_ONLY export is supported")
    if hierarchy.config_digest != config_digest:
        raise ValueError("candidate hierarchy is not bound to the exact config")
    if hierarchy.completion_level != CompletionLevel.RECURSIVE_CANDIDATE:
        raise ValueError("candidate hierarchy must remain RECURSIVE_CANDIDATE")
    if not report.verified or report.issues:
        raise ValueError("only an issue-free verified report may be exported")
    if report.completion_level != CompletionLevel.RECURSIVE_VERIFIED:
        raise ValueError("verification report is not RECURSIVE_VERIFIED")
    if report.case_id != case.case_id:
        raise ValueError("verification report case_id does not match the case")
    if report.case_digest != canonical_sha256(graph_case_to_dict(case)):
        raise ValueError("verification report case digest does not match")
    if report.candidate_hierarchy_digest != canonical_sha256(
        hierarchy_result_to_dict(hierarchy)
    ):
        raise ValueError("verification report hierarchy digest does not match")
    if report.config_digest != config_digest:
        raise ValueError("verification report config digest does not match")
    record_ids = {record.record_id for record in hierarchy.records}
    if hierarchy.root_record_id not in record_ids:
        raise ValueError("hierarchy root record is unresolved")
    if not set(hierarchy.terminal_record_ids).issubset(record_ids):
        raise ValueError("hierarchy terminal record is unresolved")
    for edge in hierarchy.hierarchy_edges:
        if edge.parent_record_id not in record_ids or edge.child_record_id not in record_ids:
            raise ValueError("hierarchy edge references an unresolved record")


def _separation_id(value: Separation) -> str:
    return stable_id("separation", separation_to_dict(value))


def _edge_roles(edge_record) -> list[str]:
    roles: list[str] = []
    if edge_record.is_root_real:
        roles.append("ROOT_REAL")
    roles.extend(
        f"VIRTUAL_INTERFACE:{identifier}"
        for identifier in edge_record.virtual_interface_ids
    )
    return roles


def _bundle(
    case: GraphCase,
    hierarchy: HierarchyResult,
    report: VerificationReport,
) -> dict[str, object]:
    terminal_ids = set(hierarchy.terminal_record_ids)
    cutsets: list[dict[str, object]] = []
    separations: list[dict[str, object]] = []
    crossing_edges: list[dict[str, object]] = []
    torsos: list[dict[str, object]] = []

    for record in hierarchy.records:
        local = record.local_result
        witnesses = dict(local.rejection_witnesses)
        elementary = set(local.elementary)
        tn = set(local.tn_aggregated)
        for cut in local.cuts:
            cutsets.append(
                {
                    "record_id": record.record_id,
                    "separator": list(cut.separator),
                    "components": [list(component) for component in cut.components],
                }
            )
        for value in local.full_separations:
            identifier = _separation_id(value)
            payload = separation_to_dict(value)
            payload.update(
                {
                    "record_id": record.record_id,
                    "separation_id": identifier,
                    "is_elementary": value in elementary,
                    "is_tn": value in tn,
                    "rejection_witness_id": witnesses.get(identifier),
                }
            )
            separations.append(payload)
        for candidate_id, witness_id in local.rejection_witnesses:
            crossing_edges.append(
                {
                    "record_id": record.record_id,
                    "source_separation_id": candidate_id,
                    "target_separation_id": witness_id,
                }
            )
        torso_edges = []
        for edge in record.edge_records:
            roles = _edge_roles(edge)
            torso_edges.append(
                {
                    "endpoints": list(edge.endpoints),
                    "roles": roles,
                    "root_edge_id": edge.root_edge_id,
                }
            )
        torsos.append(
            {
                "record_id": record.record_id,
                "parent_record_id": record.parent_record_id,
                "depth": record.depth,
                "bag_vertices": list(record.bag_vertices),
                "status": record.status.value,
                "support_edges": [list(edge) for edge in record.support_edges],
                "edges": torso_edges,
                "interface_refs": [
                    interface_ref_to_dict(value) for value in record.interface_refs
                ],
                "structure_tree": (
                    None
                    if record.structure_tree is None
                    else structure_tree_to_dict(record.structure_tree)
                ),
                "is_terminal": record.record_id in terminal_ids,
            }
        )

    cutsets.sort(key=lambda value: (value["record_id"], value["separator"]))
    separations.sort(
        key=lambda value: (
            value["record_id"],
            value["side_a"],
            value["side_b"],
        )
    )
    crossing_edges.sort(
        key=lambda value: (
            value["record_id"],
            value["source_separation_id"],
            value["target_separation_id"],
        )
    )
    torsos.sort(key=lambda value: value["record_id"])

    hierarchy_nodes = [
        {
            "record_id": record.record_id,
            "depth": record.depth,
            "status": record.status.value,
            "bag_vertices": list(record.bag_vertices),
            "is_terminal": record.record_id in terminal_ids,
        }
        for record in hierarchy.records
    ]
    hierarchy_edges = [
        {
            "parent_record_id": edge.parent_record_id,
            "child_record_id": edge.child_record_id,
            "local_tree_node_id": edge.local_tree_node_id,
        }
        for edge in hierarchy.hierarchy_edges
    ]
    root_nodes = [
        {"id": vertex, "label": str(vertex)}
        for vertex in range(case.num_nodes)
    ]
    root_edges = [
        {
            "endpoints": list(edge),
            "roles": ["ROOT_REAL"],
            "color": "black",
            "line_style": "solid",
        }
        for edge in case.edges
    ]
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "case": graph_case_to_dict(case),
        "root_graph": {"nodes": root_nodes, "edges": root_edges},
        "cutsets": cutsets,
        "separations": separations,
        "crossing_edges": crossing_edges,
        "hierarchy": {
            "root_record_id": hierarchy.root_record_id,
            "nodes": hierarchy_nodes,
            "edges": hierarchy_edges,
            "terminal_record_ids": list(hierarchy.terminal_record_ids),
        },
        "torsos": torsos,
        "interfaces": [
            interface_object_to_dict(value) for value in hierarchy.interfaces
        ],
        "ambient_blocks": None,
        "block_localization": None,
        "verification": verification_report_to_dict(report),
        "style_hints": _STYLE_HINTS,
        "view_recipes": [
            {"panel": "A", "title": "root graph and separators"},
            {"panel": "B", "title": "elementary crossing graph"},
            {"panel": "C", "title": "recursive hierarchy"},
            {"panel": "D", "title": "interface provenance"},
        ],
    }


def _mermaid_label(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _dot_label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_prompt(case: GraphCase) -> bytes:
    text = f"""# GPT visualization request: {case.case_id}

Create a deterministic four-panel academic figure using only the supplied bundle:

1. root graph and separators
2. elementary crossing graph
3. recursive hierarchy
4. interface provenance

Apply every color and line-style rule from `style_hints`. Preserve all identifiers and vertex labels exactly. Do not invent nodes, edges, cuts, blocks, or labels. Equal interface boundaries remain distinct when their interface IDs differ.

Display this warning verbatim: **STRUCTURE_ONLY; ambient blocks not computed**.
"""
    return text.encode("utf-8")


def _render_root_mermaid(bundle: dict[str, object]) -> bytes:
    lines = ["graph LR"]
    for node in bundle["root_graph"]["nodes"]:
        lines.append(f'  n{node["id"]}["{_mermaid_label(node["label"])}"]')
    for edge in bundle["root_graph"]["edges"]:
        left, right = edge["endpoints"]
        lines.append(f"  n{left} --- n{right}")
    separator_vertices = sorted(
        {
            vertex
            for cut in bundle["cutsets"]
            for vertex in cut["separator"]
        }
    )
    lines.append("  classDef separator fill:orange,stroke:orange")
    if separator_vertices:
        lines.append(
            "  class "
            + ",".join(f"n{vertex}" for vertex in separator_vertices)
            + " separator"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_hierarchy_mermaid(bundle: dict[str, object]) -> bytes:
    nodes = bundle["hierarchy"]["nodes"]
    index_by_id = {
        node["record_id"]: index
        for index, node in enumerate(nodes)
    }
    lines = ["graph TD"]
    for index, node in enumerate(nodes):
        label = _mermaid_label(
            f'{node["record_id"]}\n{node["status"]}\ndepth={node["depth"]}'
        )
        lines.append(f'  r{index}["{label}"]')
        lines.append(f'  class r{index} {node["status"]}')
    for edge in bundle["hierarchy"]["edges"]:
        parent = index_by_id[edge["parent_record_id"]]
        child = index_by_id[edge["child_record_id"]]
        label = _mermaid_label(edge["local_tree_node_id"])
        lines.append(f'  r{parent} -->|"{label}"| r{child}')
    for status, color in _STYLE_HINTS["statuses"].items():
        lines.append(f"  classDef {status} fill:{color},stroke:{color}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_crossing_mermaid(bundle: dict[str, object]) -> bytes:
    elementary = [
        value for value in bundle["separations"] if value["is_elementary"]
    ]
    key_to_index = {
        (value["record_id"], value["separation_id"]): index
        for index, value in enumerate(elementary)
    }
    lines = ["graph LR"]
    for index, value in enumerate(elementary):
        label = _mermaid_label(
            f'{value["record_id"]}:{value["separation_id"][-12:]}'
        )
        lines.append(f'  s{index}["{label}"]')
        lines.append(f'  class s{index} {"tn" if value["is_tn"] else "rejected"}')
    for edge in bundle["crossing_edges"]:
        source = key_to_index[(edge["record_id"], edge["source_separation_id"])]
        target = key_to_index[(edge["record_id"], edge["target_separation_id"])]
        lines.append(f"  s{source} ---|crosses| s{target}")
    lines.extend(
        (
            "  classDef tn fill:green,stroke:green",
            "  classDef rejected fill:red,stroke:red",
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_dot(case: GraphCase, hierarchy: HierarchyResult) -> bytes:
    lines = [
        "graph BL_RCTN {",
        f'  graph [label="{_dot_label(case.case_id)}"];',
        "  subgraph cluster_root {",
        '    label="root graph";',
    ]
    for vertex in range(case.num_nodes):
        lines.append(f'    root_{vertex} [label="{vertex}"];')
    for left, right in case.edges:
        lines.append(
            f'    root_{left} -- root_{right} [color="black", style="solid", label="ROOT_REAL"];'
        )
    lines.append("  }")
    for record_index, record in enumerate(hierarchy.records):
        lines.extend(
            (
                f"  subgraph cluster_torso_{record_index} {{",
                f'    label="{_dot_label(record.record_id)}";',
            )
        )
        for vertex in record.bag_vertices:
            lines.append(
                f'    torso_{record_index}_{vertex} [label="{vertex}"];'
            )
        for edge in record.edge_records:
            left, right = edge.endpoints
            if edge.is_root_real:
                lines.append(
                    f'    torso_{record_index}_{left} -- torso_{record_index}_{right} '
                    '[color="black", style="solid", label="ROOT_REAL"];'
                )
            for interface_id in edge.virtual_interface_ids:
                lines.append(
                    f'    torso_{record_index}_{left} -- torso_{record_index}_{right} '
                    f'[color="blue", style="dashed", label="VIRTUAL_INTERFACE:{_dot_label(interface_id)}"];'
                )
        lines.append("  }")
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def export_case_bundle(
    case: GraphCase,
    hierarchy: HierarchyResult,
    verification: VerificationReport,
    config: DemoConfig,
    output_dir: Path,
) -> tuple[Path, ...]:
    _validate_bindings(case, hierarchy, verification, config)
    destination = _project_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {destination.parent}"
        )

    bundle = _bundle(case, hierarchy, verification)
    artifacts = {
        "gpt_visualization_bundle.json": canonical_json_bytes(bundle),
        "gpt_prompt.md": _render_prompt(case),
        "root_graph.mmd": _render_root_mermaid(bundle),
        "hierarchy.mmd": _render_hierarchy_mermaid(bundle),
        "crossing_graph.mmd": _render_crossing_mermaid(bundle),
        "graph.dot": _render_dot(case, hierarchy),
    }
    if tuple(artifacts) != _BUNDLE_FILENAMES:
        raise AssertionError("internal bundle filename contract changed")

    destination.mkdir()
    try:
        return _publish_files_once(destination, artifacts)
    except BaseException:
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def _summary_bytes(
    cases: tuple[GraphCase, ...],
    hierarchies: dict[str, HierarchyResult],
    reports: dict[str, VerificationReport],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_SUMMARY_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for case in cases:
        hierarchy = hierarchies[case.case_id]
        report = reports[case.case_id]
        records = {record.record_id: record for record in hierarchy.records}
        root = records[hierarchy.root_record_id]
        terminals = [records[identifier] for identifier in hierarchy.terminal_record_ids]
        writer.writerow(
            {
                "case_id": case.case_id,
                "k": case.k,
                "n": case.num_nodes,
                "m": len(case.edges),
                "status": root.status.value,
                "cutset_count": len(root.local_result.cuts),
                "full_sigma_count": len(root.local_result.full_separations),
                "elementary_count": len(root.local_result.elementary),
                "tn_count": len(root.local_result.tn_aggregated),
                "hierarchy_depth": max(record.depth for record in hierarchy.records),
                "small_terminals": sum(
                    record.status == LocalStatus.SMALL for record in terminals
                ),
                "high_terminals": sum(
                    record.status == LocalStatus.HIGH for record in terminals
                ),
                "crossed_terminals": sum(
                    record.status == LocalStatus.CROSSED for record in terminals
                ),
                "verified": "true" if report.verified else "false",
                "bundle_path": (
                    f"gpt/{case.case_id}/gpt_visualization_bundle.json"
                ),
            }
        )
    return stream.getvalue().encode("utf-8")


def _read_closed_files(run_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for current, directory_names, file_names in os.walk(run_dir, followlinks=False):
        current_path = Path(current)
        for name in tuple(directory_names):
            path = current_path / name
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ValueError("run artifacts may not contain symlinks")
        for name in file_names:
            path = current_path / name
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("run artifacts must be regular files")
            relative = path.relative_to(run_dir).as_posix()
            if relative == "manifest.json":
                continue
            files[relative] = path.read_bytes()
    return files


def export_run_summary(run_dir: Path) -> tuple[Path, Path]:
    root = _project_path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    summary_path = root / "summary.csv"
    manifest_path = root / "manifest.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(summary_path)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)

    config = read_demo_config(root / "input" / "config.json")
    cases = read_graph_cases(root / "input" / "cases.jsonl")
    hierarchies: dict[str, HierarchyResult] = {}
    reports: dict[str, VerificationReport] = {}
    for case in cases:
        hierarchy = read_hierarchy_result(root / "engine" / f"{case.case_id}.json")
        report = read_verification_report(
            root / "verification" / f"{case.case_id}.json"
        )
        _validate_bindings(case, hierarchy, report, config)
        bundle_path = root / "gpt" / case.case_id / "gpt_visualization_bundle.json"
        raw_bundle = bundle_path.read_bytes()
        try:
            bundle = json.loads(raw_bundle.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("GPT bundle is not canonical UTF-8 JSON") from exc
        if raw_bundle != canonical_json_bytes(bundle):
            raise ValueError("GPT bundle is not canonical JSON")
        if bundle.get("case") != graph_case_to_dict(case):
            raise ValueError("GPT bundle case binding does not match")
        if bundle.get("verification") != verification_report_to_dict(report):
            raise ValueError("GPT bundle verification binding does not match")
        hierarchies[case.case_id] = hierarchy
        reports[case.case_id] = report

    summary = _summary_bytes(cases, hierarchies, reports)
    existing = _read_closed_files(root)
    existing["summary.csv"] = summary
    file_entries = [
        {
            "path": relative,
            "byte_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(existing.items())
    ]
    manifest = canonical_json_bytes(
        {
            "schema_version": _MANIFEST_SCHEMA,
            "seed": config.seed,
            "config_digest": canonical_sha256(demo_config_to_dict(config)),
            "case_count": len(cases),
            "all_verified": all(report.verified for report in reports.values()),
            "python_version": platform.python_version(),
            "files": file_entries,
        }
    )
    published = _publish_files_once(
        root,
        {"summary.csv": summary, "manifest.json": manifest},
    )
    return published[0], published[1]
