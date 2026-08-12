from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import stat
from collections import Counter
from itertools import combinations
from pathlib import Path

from .models import (
    CompletionLevel,
    Coverage,
    CutRecord,
    DemoConfig,
    Edge,
    EdgeRecord,
    GraphCase,
    HierarchyResult,
    InterfaceObject,
    InterfaceRef,
    LocalStatus,
    Separation,
    StructureTree,
    TorsoRecord,
    VerificationReport,
    VertexSet,
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
    write_verification_report_once,
)


_PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent))
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
_BUNDLE_FILENAMES = (
    "gpt_visualization_bundle.json",
    "gpt_prompt.md",
    "root_graph.mmd",
    "hierarchy.mmd",
    "crossing_graph.mmd",
    "graph.dot",
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


class VerificationFailure(ValueError):
    """A fail-closed verifier invariant violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def _validated_graph(
    vertices: VertexSet, edges: tuple[Edge, ...]
) -> dict[int, frozenset[int]]:
    _require(type(vertices) is tuple, "vertices must be a tuple")
    _require(
        all(type(vertex) is int and vertex >= 0 for vertex in vertices),
        "vertices must be nonnegative integers",
    )
    _require(
        len(set(vertices)) == len(vertices) and tuple(sorted(vertices)) == vertices,
        "vertices must be unique and sorted",
    )
    vertex_set = set(vertices)
    mutable = {vertex: set() for vertex in vertices}
    seen: set[Edge] = set()
    _require(type(edges) is tuple, "edges must be a tuple")
    for edge in edges:
        _require(
            type(edge) is tuple
            and len(edge) == 2
            and all(type(vertex) is int for vertex in edge),
            "each edge must be an integer pair",
        )
        left, right = edge
        _require(left < right, "edges must be normalized and loop-free")
        _require(
            left in vertex_set and right in vertex_set,
            "edge endpoint lies outside the graph",
        )
        _require(edge not in seen, "graph contains a duplicate edge")
        seen.add(edge)
        mutable[left].add(right)
        mutable[right].add(left)
    _require(tuple(sorted(edges)) == edges, "edges must be sorted")
    return {
        vertex: frozenset(mutable[vertex])
        for vertex in vertices
    }


def _components(
    adjacency: dict[int, frozenset[int]], deleted: frozenset[int]
) -> tuple[VertexSet, ...]:
    _require(
        deleted.issubset(adjacency), "deleted set lies outside the graph"
    )
    unseen = set(adjacency).difference(deleted)
    result: list[VertexSet] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in sorted(
                adjacency[vertex].difference(deleted).intersection(unseen),
                reverse=True,
            ):
                unseen.remove(neighbor)
                stack.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(sorted(result))


def _is_k_connected(
    adjacency: dict[int, frozenset[int]], k: int
) -> bool:
    vertices = tuple(adjacency)
    if len(vertices) <= k:
        return False
    for size in range(k):
        for deleted in combinations(vertices, size):
            if len(_components(adjacency, frozenset(deleted))) != 1:
                return False
    return True


def _separation_key(value: Separation) -> tuple[VertexSet, VertexSet]:
    return value.side_a, value.side_b


def _nested(left: Separation, right: Separation) -> bool:
    for away_left, toward_left in (
        (frozenset(left.side_a), frozenset(left.side_b)),
        (frozenset(left.side_b), frozenset(left.side_a)),
    ):
        for away_right, toward_right in (
            (frozenset(right.side_a), frozenset(right.side_b)),
            (frozenset(right.side_b), frozenset(right.side_a)),
        ):
            if away_left <= away_right and toward_left >= toward_right:
                return True
    return False


def _separation_id(value: Separation) -> str:
    return stable_id(
        "separation",
        {
            "side_a": value.side_a,
            "side_b": value.side_b,
            "separator": value.separator,
        },
    )


def _make_separation(
    separator: tuple[int, ...],
    left_wing: frozenset[int],
    right_wing: frozenset[int],
) -> Separation:
    separator_set = frozenset(separator)
    return Separation(
        tuple(sorted(separator_set | left_wing)),
        tuple(sorted(separator_set | right_wing)),
        separator,
    )


def _independent_local_oracle(
    vertices: VertexSet,
    edges: tuple[Edge, ...],
    k: int,
    limit: int,
) -> dict[str, object]:
    adjacency = _validated_graph(vertices, edges)
    _require(type(k) is int and k >= 2, "k must be an integer at least 2")
    _require(
        type(limit) is int and limit >= 1,
        "full separation budget must be positive",
    )
    if len(vertices) > k + 1:
        _require(
            _is_k_connected(adjacency, k),
            f"non-small support graph is not {k}-connected",
        )

    cuts: list[CutRecord] = []
    for separator in combinations(vertices, k):
        components = _components(adjacency, frozenset(separator))
        if len(components) >= 2:
            cuts.append(CutRecord(separator, components))

    # This deliberately does not reuse the engine cut table to enumerate the
    # full universe. It assigns every nonseparator vertex to one of two wings,
    # fixes the first vertex on the left to quotient side reversal, and retains
    # exactly assignments with no cross-wing edge.
    full: set[Separation] = set()
    for separator in combinations(vertices, k):
        separator_set = frozenset(separator)
        remaining = tuple(
            vertex for vertex in vertices if vertex not in separator_set
        )
        if len(remaining) < 2:
            continue
        first = remaining[0]
        tail = remaining[1:]
        for mask in range(1 << len(tail)):
            left = {first}
            for index, vertex in enumerate(tail):
                if mask & (1 << index):
                    left.add(vertex)
            right = set(remaining).difference(left)
            if not right:
                continue
            if any(
                (u in left and v in right) or (u in right and v in left)
                for u, v in edges
            ):
                continue
            full.add(
                _make_separation(
                    separator,
                    frozenset(left),
                    frozenset(right),
                )
            )
            if len(full) > limit:
                raise VerificationFailure(
                    f"full separation limit exceeded: need more than {limit}"
                )

    elementary: set[Separation] = set()
    vertex_set = frozenset(vertices)
    for cut in cuts:
        separator_set = frozenset(cut.separator)
        nonseparator = vertex_set.difference(separator_set)
        for component in cut.components:
            wing = frozenset(component)
            elementary.add(
                _make_separation(
                    cut.separator,
                    wing,
                    nonseparator.difference(wing),
                )
            )

    full_values = tuple(sorted(full, key=_separation_key))
    elementary_values = tuple(sorted(elementary, key=_separation_key))
    tn_full = tuple(
        value
        for value in full_values
        if all(_nested(value, other) for other in full_values)
    )

    pairwise: list[Separation] = []
    witnesses: list[tuple[str, str]] = []
    for value in elementary_values:
        crossing = tuple(
            other for other in elementary_values if not _nested(value, other)
        )
        if crossing:
            witnesses.append(
                (
                    _separation_id(value),
                    min(_separation_id(other) for other in crossing),
                )
            )
        else:
            pairwise.append(value)
    tn_pairwise = tuple(pairwise)
    _require(
        tn_full == tn_pairwise,
        "independent full and elementary TN universes disagree",
    )

    if len(vertices) <= k + 1:
        status = LocalStatus.SMALL
    elif not cuts:
        status = LocalStatus.HIGH
    elif not tn_full:
        status = LocalStatus.CROSSED
    else:
        status = LocalStatus.SPLIT

    return {
        "adjacency": adjacency,
        "cuts": tuple(cuts),
        "full": full_values,
        "elementary": elementary_values,
        "tn": tn_full,
        "witnesses": tuple(sorted(witnesses)),
        "status": status,
    }


def _root_record(hierarchy: HierarchyResult) -> TorsoRecord:
    matches = tuple(
        record
        for record in hierarchy.records
        if record.record_id == hierarchy.root_record_id
    )
    _require(len(matches) == 1, "root record ID must resolve exactly once")
    return matches[0]


def _verify_binding(
    case: GraphCase, hierarchy: HierarchyResult, config: DemoConfig
) -> None:
    expected_config = canonical_sha256(demo_config_to_dict(config))
    _require(
        hierarchy.config_digest == expected_config,
        "candidate is not bound to the exact frozen config",
    )
    _require(
        hierarchy.completion_level is CompletionLevel.RECURSIVE_CANDIDATE,
        "engine output must remain RECURSIVE_CANDIDATE",
    )
    _require(config.mode == "STRUCTURE_ONLY", "unsupported config mode")
    _require(case.k in config.ks, "case k is not enabled by the config")


def _verify_root_legality(case: GraphCase) -> None:
    _require(
        case.num_nodes > case.k,
        f"root graph is not {case.k}-connected",
    )
    expected_vertices = tuple(range(case.num_nodes))
    adjacency = _validated_graph(expected_vertices, case.edges)
    _require(
        _is_k_connected(adjacency, case.k),
        f"root graph is not {case.k}-connected",
    )


def _verify_all_local_results(
    case: GraphCase, hierarchy: HierarchyResult, config: DemoConfig
) -> None:
    _require(hierarchy.records, "hierarchy has no records")
    for record in hierarchy.records:
        oracle = _independent_local_oracle(
            record.bag_vertices,
            record.support_edges,
            case.k,
            config.max_full_separations,
        )
        local = record.local_result
        _require(
            local.completion_level is CompletionLevel.LOCAL_EXACT,
            f"record {record.record_id} local result is not LOCAL_EXACT",
        )
        expected_fields = (
            ("cuts", oracle["cuts"]),
            ("full_separations", oracle["full"]),
            ("elementary", oracle["elementary"]),
            ("tn_full", oracle["tn"]),
            ("tn_pairwise", oracle["tn"]),
            ("tn_aggregated", oracle["tn"]),
            ("rejection_witnesses", oracle["witnesses"]),
        )
        for field, expected in expected_fields:
            _require(
                getattr(local, field) == expected,
                f"record {record.record_id} has an inexact {field}",
            )
        _require(
            local.status is oracle["status"],
            f"record {record.record_id} has the wrong local status",
        )
        _require(
            record.status is local.status,
            f"record {record.record_id} status disagrees with local result",
        )


def _verify_fixture_expectations(
    case: GraphCase, hierarchy: HierarchyResult, config: DemoConfig
) -> None:
    if not case.expected:
        return
    expected = dict(case.expected)
    required = {
        "status",
        "cutsets",
        "full_sigma",
        "elementary",
        "tn",
    }
    _require(
        set(expected) == required,
        "fixture expectation schema is not exact",
    )
    for field in ("cutsets", "full_sigma", "elementary", "tn"):
        _require(
            type(expected[field]) is int and expected[field] >= 0,
            f"fixture numeric field {field} is invalid",
        )
    root = _root_record(hierarchy)
    local = root.local_result
    actual = {
        "cutsets": len(local.cuts),
        "full_sigma": len(local.full_separations),
        "elementary": len(local.elementary),
        "tn": len(local.tn_full),
    }
    for field, value in actual.items():
        _require(
            value == expected[field],
            f"fixture numeric field {field} disagrees with the oracle",
        )
    # Status is independently derived above; this checks only that the fixture
    # label agrees with that derivation and never uses family or case_id.
    _require(
        root.status.value == expected["status"],
        "fixture status label disagrees with the independent oracle",
    )


def _tree_neighbors(tree: StructureTree) -> dict[str, set[str]]:
    node_ids = [node.node_id for node in tree.nodes]
    _require(len(set(node_ids)) == len(node_ids), "tree has duplicate node IDs")
    neighbors = {node_id: set() for node_id in node_ids}
    endpoint_pairs: set[frozenset[str]] = set()
    for edge in tree.edges:
        _require(
            edge.source_node_id in neighbors and edge.target_node_id in neighbors,
            "tree edge references an unknown node",
        )
        pair = frozenset((edge.source_node_id, edge.target_node_id))
        _require(len(pair) == 2, "tree edge is a loop")
        _require(pair not in endpoint_pairs, "tree has parallel edges")
        endpoint_pairs.add(pair)
        neighbors[edge.source_node_id].add(edge.target_node_id)
        neighbors[edge.target_node_id].add(edge.source_node_id)
    return neighbors


def _reached(
    start: str, allowed: set[str], neighbors: dict[str, set[str]]
) -> set[str]:
    reached: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in reached or current not in allowed:
            continue
        reached.add(current)
        stack.extend(neighbors[current].intersection(allowed).difference(reached))
    return reached


def _verify_structure_tree(record: TorsoRecord) -> None:
    tree = record.structure_tree
    _require(tree is not None, "SPLIT record lacks a structure tree")
    tn = record.local_result.tn_full
    ordered = tuple(sorted(tn, key=_separation_key))
    _require(
        len(tree.nodes) == len(ordered) + 1,
        "structure tree node count is not separation_count + 1",
    )
    _require(
        len(tree.edges) == len(ordered),
        "structure tree edge count is not separation_count",
    )
    neighbors = _tree_neighbors(tree)
    node_by_id = {node.node_id: node for node in tree.nodes}

    if node_by_id:
        all_nodes = set(node_by_id)
        _require(
            _reached(next(iter(all_nodes)), all_nodes, neighbors) == all_nodes,
            "structure tree is disconnected",
        )
    _require(
        len(tree.edges) == max(0, len(tree.nodes) - 1),
        "structure tree is not acyclic",
    )

    base_by_id = {_separation_id(value): value for value in ordered}
    _require(
        len(base_by_id) == len(ordered),
        "TN separation IDs collide",
    )
    signatures: set[tuple[tuple[str, int], ...]] = set()
    bits_by_node: dict[str, dict[str, int]] = {}
    graph_vertices = set(record.bag_vertices)
    for node in tree.nodes:
        signature = node.orientation_signature
        _require(
            len(signature) == len(base_by_id)
            and len({item[0] for item in signature}) == len(signature),
            "tree node has an invalid orientation signature",
        )
        bits = dict(signature)
        _require(
            set(bits) == set(base_by_id)
            and all(value in (0, 1) for value in bits.values()),
            "tree node orientation signature is not complete",
        )
        _require(signature not in signatures, "tree has duplicate orientations")
        signatures.add(signature)
        bits_by_node[node.node_id] = bits
        expected_bag = set(record.bag_vertices)
        for base_id, separation in base_by_id.items():
            toward = separation.side_b if bits[base_id] == 0 else separation.side_a
            expected_bag.intersection_update(toward)
        _require(
            node.bag_vertices == tuple(sorted(expected_bag)),
            "tree node bag is not its pointed-side intersection",
        )
        _require(
            set(node.bag_vertices).issubset(graph_vertices),
            "tree bag lies outside its record",
        )

    label_counts: Counter[Separation] = Counter()
    for edge in tree.edges:
        _require(edge.separation in ordered, "tree edge has an unknown TN label")
        _require(
            edge.adhesion == edge.separation.separator,
            "tree edge adhesion disagrees with its separation",
        )
        label_counts[edge.separation] += 1
        source = node_by_id[edge.source_node_id]
        target = node_by_id[edge.target_node_id]
        intersection = tuple(
            sorted(set(source.bag_vertices).intersection(target.bag_vertices))
        )
        _require(
            intersection == edge.separation.separator,
            "tree edge endpoint-bag intersection is not its separator",
        )
        source_bits = bits_by_node[source.node_id]
        target_bits = bits_by_node[target.node_id]
        differing = tuple(
            base_id
            for base_id in base_by_id
            if source_bits[base_id] != target_bits[base_id]
        )
        _require(
            len(differing) == 1
            and base_by_id[differing[0]] == edge.separation,
            "tree edge is not the Hamming-one edge for its TN label",
        )
    _require(
        all(label_counts[value] == 1 for value in ordered),
        "tree-edge/TN labels are not bijective",
    )

    covered = set().union(
        *(set(node.bag_vertices) for node in tree.nodes)
    ) if tree.nodes else set()
    _require(covered == graph_vertices, "tree bags do not cover record vertices")
    for edge in record.support_edges:
        _require(
            any(set(edge).issubset(node.bag_vertices) for node in tree.nodes),
            f"tree bags do not cover support edge {edge}",
        )
    for vertex in record.bag_vertices:
        containing = {
            node.node_id for node in tree.nodes if vertex in node.bag_vertices
        }
        _require(containing, f"tree bags omit vertex {vertex}")
        _require(
            _reached(next(iter(containing)), containing, neighbors) == containing,
            f"bags containing vertex {vertex} are disconnected",
        )


def _verify_all_structure_trees(hierarchy: HierarchyResult) -> None:
    for record in hierarchy.records:
        if record.status is LocalStatus.SPLIT:
            _verify_structure_tree(record)
        else:
            _require(
                record.structure_tree is None,
                f"terminal record {record.record_id} unexpectedly has a tree",
            )


def _edge_role_signature(
    values: tuple[EdgeRecord, ...]
) -> tuple[tuple[Edge, bool, str | None, tuple[str, ...]], ...]:
    endpoints: set[Edge] = set()
    result: list[tuple[Edge, bool, str | None, tuple[str, ...]]] = []
    for value in values:
        _require(value.endpoints not in endpoints, "duplicate edge provenance record")
        endpoints.add(value.endpoints)
        _require(
            (value.is_root_real and value.root_edge_id is not None)
            or (not value.is_root_real and value.root_edge_id is None),
            "root-real flag and root edge ID disagree",
        )
        _require(
            len(set(value.virtual_interface_ids))
            == len(value.virtual_interface_ids),
            "edge role repeats a virtual interface ID",
        )
        result.append(
            (
                value.endpoints,
                value.is_root_real,
                value.root_edge_id,
                tuple(sorted(value.virtual_interface_ids)),
            )
        )
    return tuple(sorted(result))


def _propagated_ref(
    ref: InterfaceRef,
    interface: InterfaceObject,
    child_vertices: VertexSet,
) -> InterfaceRef | None:
    intersection = tuple(
        vertex for vertex in ref.local_boundary if vertex in child_vertices
    )
    if not intersection:
        return None
    coverage = (
        Coverage.FULL
        if ref.coverage is Coverage.FULL and intersection == interface.boundary
        else Coverage.PARTIAL
    )
    return InterfaceRef(
        ref.interface_id,
        ref.incidence_id,
        coverage,
        intersection,
    )


def _semantic_state(record: TorsoRecord) -> tuple:
    return (
        record.bag_vertices,
        record.support_edges,
        tuple(
            (
                value.endpoints,
                value.is_root_real,
                len(value.virtual_interface_ids),
            )
            for value in record.edge_records
        ),
        tuple(
            (value.coverage.value, value.local_boundary)
            for value in record.interface_refs
        ),
    )


def _verify_hierarchy_and_provenance(
    case: GraphCase, hierarchy: HierarchyResult, config: DemoConfig
) -> None:
    record_ids = [record.record_id for record in hierarchy.records]
    _require(len(set(record_ids)) == len(record_ids), "duplicate record ID")
    records = {record.record_id: record for record in hierarchy.records}
    root = _root_record(hierarchy)
    _require(
        root.parent_record_id is None
        and root.depth == 0
        and root.bag_vertices == tuple(range(case.num_nodes))
        and root.support_edges == case.edges,
        "root torso does not equal the root input graph",
    )

    interface_ids = [value.interface_id for value in hierarchy.interfaces]
    _require(
        len(set(interface_ids)) == len(interface_ids),
        "duplicate interface ID",
    )
    interfaces = {
        value.interface_id: value for value in hierarchy.interfaces
    }

    incoming: dict[str, object] = {}
    outgoing: dict[str, list] = {record_id: [] for record_id in records}
    for edge in hierarchy.hierarchy_edges:
        _require(
            edge.parent_record_id in records and edge.child_record_id in records,
            "hierarchy edge has an unresolved endpoint",
        )
        _require(
            edge.child_record_id not in incoming,
            "hierarchy child has multiple parents",
        )
        incoming[edge.child_record_id] = edge
        outgoing[edge.parent_record_id].append(edge)
    _require(
        set(incoming) == set(records).difference({hierarchy.root_record_id}),
        "non-root records do not each have one incoming hierarchy edge",
    )

    for record in hierarchy.records:
        _require(
            record.depth <= config.max_depth,
            f"record {record.record_id} exceeds max_depth",
        )
        actual_endpoints = tuple(
            value.endpoints for value in record.edge_records
        )
        _require(
            actual_endpoints == record.support_edges,
            f"record {record.record_id} support/provenance edges disagree",
        )
        _edge_role_signature(record.edge_records)

    root_roles = _edge_role_signature(root.edge_records)
    _require(
        tuple(value[0] for value in root_roles) == case.edges
        and all(value[1] and value[3] == () for value in root_roles),
        "root edge roles are not exactly the real root edges",
    )
    root_edge_by_id = {value[2]: value[0] for value in root_roles}
    _require(
        None not in root_edge_by_id
        and len(root_edge_by_id) == len(root_roles),
        "root edge IDs are not bijective",
    )

    for interface in hierarchy.interfaces:
        _require(
            len(interface.incidences) == 2,
            "interface does not have exactly two incidences",
        )
        left, right = interface.incidences
        _require(
            left.incidence_id != right.incidence_id
            and left.opposite_incidence_id == right.incidence_id
            and right.opposite_incidence_id == left.incidence_id,
            "interface incidences are not an opposite pair",
        )
        creator = records.get(interface.creator_record_id)
        _require(
            creator is not None and creator.structure_tree is not None,
            "interface creator record is unresolved",
        )
        matching = tuple(
            edge
            for edge in creator.structure_tree.edges
            if edge.tree_edge_id == interface.creator_tree_edge_id
        )
        _require(
            len(matching) == 1,
            "interface creator tree edge is unresolved",
        )
        creator_edge = matching[0]
        _require(
            interface.boundary == creator_edge.adhesion,
            "interface boundary differs from creator adhesion",
        )
        _require(
            {value.side_tree_node_id for value in interface.incidences}
            == {creator_edge.source_node_id, creator_edge.target_node_id},
            "interface incidence sides differ from creator tree edge",
        )

    expected_creator_pairs = {
        (record.record_id, edge.tree_edge_id)
        for record in hierarchy.records
        if record.structure_tree is not None
        for edge in record.structure_tree.edges
    }
    actual_creator_pairs = {
        (value.creator_record_id, value.creator_tree_edge_id)
        for value in hierarchy.interfaces
    }
    _require(
        expected_creator_pairs == actual_creator_pairs
        and len(actual_creator_pairs) == len(hierarchy.interfaces),
        "creator tree edges and immutable interfaces are not bijective",
    )

    for record in hierarchy.records:
        for edge_record in record.edge_records:
            for interface_id in edge_record.virtual_interface_ids:
                _require(
                    interface_id in interfaces,
                    "edge role references an unknown virtual interface",
                )
            if edge_record.is_root_real:
                _require(
                    edge_record.root_edge_id in root_edge_by_id
                    and root_edge_by_id[edge_record.root_edge_id]
                    == edge_record.endpoints,
                    "forged or inconsistent root-real edge provenance",
                )
        seen_refs: set[tuple[str, str]] = set()
        for ref in record.interface_refs:
            key = (ref.interface_id, ref.incidence_id)
            _require(key not in seen_refs, "duplicate interface reference")
            seen_refs.add(key)
            interface = interfaces.get(ref.interface_id)
            _require(interface is not None, "unresolved interface reference")
            incidence_ids = {
                value.incidence_id for value in interface.incidences
            }
            _require(
                ref.incidence_id in incidence_ids,
                "unresolved interface incidence reference",
            )
            _require(
                set(ref.local_boundary).issubset(interface.boundary)
                and set(ref.local_boundary).issubset(record.bag_vertices),
                "interface fragment lies outside its boundary or record bag",
            )
            _require(
                ref.coverage is not Coverage.FULL
                or ref.local_boundary == interface.boundary,
                "FULL reference does not contain the complete global boundary",
            )

    # Every SPLIT record creates one child per tree node. Non-SPLIT records
    # create none. Exact child torso support and provenance are recomputed from
    # the parent roles plus only the locally incident creator interfaces.
    for record in hierarchy.records:
        children = outgoing[record.record_id]
        if record.status is LocalStatus.SPLIT:
            _require(record.structure_tree is not None, "SPLIT record lacks a tree")
            node_by_id = {
                value.node_id: value for value in record.structure_tree.nodes
            }
            _require(
                {value.local_tree_node_id for value in children}
                == set(node_by_id)
                and len(children) == len(node_by_id),
                "SPLIT children do not biject with tree nodes",
            )
        else:
            _require(not children, "only SPLIT records may have children")
            continue

        local_interfaces = tuple(
            value
            for value in hierarchy.interfaces
            if value.creator_record_id == record.record_id
        )
        parent_roles = {
            value.endpoints: [
                value.is_root_real,
                value.root_edge_id,
                set(value.virtual_interface_ids),
            ]
            for value in record.edge_records
        }
        for hierarchy_edge in children:
            child = records[hierarchy_edge.child_record_id]
            node = node_by_id[hierarchy_edge.local_tree_node_id]
            _require(
                child.parent_record_id == record.record_id
                and child.depth == record.depth + 1
                and child.bag_vertices == node.bag_vertices,
                "child record disagrees with its hierarchy edge/tree node",
            )
            _require(
                set(child.bag_vertices) < set(record.bag_vertices),
                "recursive child is not a proper bag",
            )
            child_set = set(child.bag_vertices)
            expected_roles = {
                endpoints: [values[0], values[1], set(values[2])]
                for endpoints, values in parent_roles.items()
                if set(endpoints).issubset(child_set)
            }
            incident: list[tuple[InterfaceObject, object]] = []
            for interface in local_interfaces:
                for incidence in interface.incidences:
                    if incidence.side_tree_node_id == node.node_id:
                        incident.append((interface, incidence))
            for interface, _ in incident:
                _require(
                    set(interface.boundary).issubset(child_set),
                    "incident interface boundary lies outside its child",
                )
                for endpoints in combinations(interface.boundary, 2):
                    edge = tuple(sorted(endpoints))
                    if edge not in expected_roles:
                        expected_roles[edge] = [False, None, set()]
                    expected_roles[edge][2].add(interface.interface_id)
            expected_edge_signature = tuple(
                (
                    endpoints,
                    values[0],
                    values[1],
                    tuple(sorted(values[2])),
                )
                for endpoints, values in sorted(expected_roles.items())
            )
            _require(
                _edge_role_signature(child.edge_records)
                == expected_edge_signature,
                "child edge roles differ from induced support plus adhesion completion",
            )
            _require(
                child.support_edges
                == tuple(value[0] for value in expected_edge_signature),
                "child support graph differs from recomputed edge roles",
            )

            expected_refs: dict[tuple[str, str], InterfaceRef] = {}
            for ref in record.interface_refs:
                interface = interfaces[ref.interface_id]
                propagated = _propagated_ref(
                    ref, interface, child.bag_vertices
                )
                if propagated is not None:
                    expected_refs[
                        (propagated.interface_id, propagated.incidence_id)
                    ] = propagated
            for interface, incidence in incident:
                new_ref = InterfaceRef(
                    interface.interface_id,
                    incidence.incidence_id,
                    Coverage.FULL,
                    interface.boundary,
                )
                key = (new_ref.interface_id, new_ref.incidence_id)
                old = expected_refs.get(key)
                _require(
                    old is None or old == new_ref,
                    "creator and inherited interface fragments conflict",
                )
                expected_refs[key] = new_ref
            expected_ref_values = tuple(
                expected_refs[key] for key in sorted(expected_refs)
            )
            _require(
                child.interface_refs == expected_ref_values,
                "child interface refs differ from exact full/partial propagation",
            )

    expected_terminals = {
        record_id for record_id, values in outgoing.items() if not values
    }
    _require(
        len(set(hierarchy.terminal_record_ids))
        == len(hierarchy.terminal_record_ids)
        and set(hierarchy.terminal_record_ids) == expected_terminals,
        "terminal IDs do not exactly equal hierarchy leaves",
    )
    _require(
        all(records[value].status is not LocalStatus.SPLIT for value in expected_terminals),
        "a SPLIT record is incorrectly labelled terminal",
    )

    # Rooted reachability and repeated semantic-state checks are independent of
    # raw digest IDs.
    reached: set[str] = set()

    def visit(record_id: str, ancestors: frozenset[tuple]) -> None:
        _require(record_id not in reached, "hierarchy contains a cycle")
        reached.add(record_id)
        record = records[record_id]
        state = _semantic_state(record)
        _require(
            state not in ancestors,
            "recursive path repeats a semantic state",
        )
        for edge in outgoing[record_id]:
            visit(edge.child_record_id, ancestors | frozenset({state}))

    visit(hierarchy.root_record_id, frozenset())
    _require(reached == set(records), "hierarchy contains unreachable records")


def _verify_terminal_reconstruction(
    case: GraphCase, hierarchy: HierarchyResult
) -> None:
    records = {record.record_id: record for record in hierarchy.records}
    terminals = tuple(records[value] for value in hierarchy.terminal_record_ids)
    vertices = set().union(
        *(set(record.bag_vertices) for record in terminals)
    ) if terminals else set()
    _require(
        vertices == set(range(case.num_nodes)),
        "terminal-only reconstruction has the wrong vertex set",
    )

    root = _root_record(hierarchy)
    root_by_id = {
        value.root_edge_id: value.endpoints
        for value in root.edge_records
        if value.is_root_real
    }
    reconstructed: dict[str, Edge] = {}
    ids_by_edge: dict[Edge, str] = {}
    terminal_refs: set[tuple[str, str]] = set()
    for record in terminals:
        terminal_refs.update(
            (ref.interface_id, ref.incidence_id)
            for ref in record.interface_refs
        )
        for value in record.edge_records:
            if not value.is_root_real:
                continue
            _require(
                value.root_edge_id in root_by_id
                and root_by_id[value.root_edge_id] == value.endpoints,
                "terminal reconstruction contains a forged real edge",
            )
            previous = reconstructed.setdefault(
                value.root_edge_id, value.endpoints
            )
            _require(
                previous == value.endpoints,
                "root edge ID has inconsistent terminal endpoints",
            )
            previous_id = ids_by_edge.setdefault(
                value.endpoints, value.root_edge_id
            )
            _require(
                previous_id == value.root_edge_id,
                "root edge has inconsistent terminal IDs",
            )
    _require(
        tuple(sorted(ids_by_edge)) == case.edges
        and set(reconstructed) == set(root_by_id),
        "terminal-only reconstruction does not equal the root edge set",
    )
    expected_incidences = {
        (interface.interface_id, incidence.incidence_id)
        for interface in hierarchy.interfaces
        for incidence in interface.incidences
    }
    _require(
        expected_incidences.issubset(terminal_refs),
        "an interface incidence is unresolved at all terminals",
    )


def verify_case_result(
    case: GraphCase,
    hierarchy: HierarchyResult,
    config: DemoConfig,
) -> VerificationReport:
    if type(case) is not GraphCase:
        raise TypeError("case must be GraphCase")
    if type(hierarchy) is not HierarchyResult:
        raise TypeError("hierarchy must be HierarchyResult")
    if type(config) is not DemoConfig:
        raise TypeError("config must be DemoConfig")

    case_digest = canonical_sha256(graph_case_to_dict(case))
    candidate_digest = canonical_sha256(hierarchy_result_to_dict(hierarchy))
    config_digest = canonical_sha256(demo_config_to_dict(config))
    checks: list[tuple[str, str]] = []
    issues: list[tuple[str, str, str]] = []

    stages = (
        ("input_binding", lambda: _verify_binding(case, hierarchy, config)),
        ("root_legality", lambda: _verify_root_legality(case)),
        (
            "local_exactness",
            lambda: _verify_all_local_results(case, hierarchy, config),
        ),
        (
            "fixture_expectations",
            lambda: _verify_fixture_expectations(case, hierarchy, config),
        ),
        (
            "structure_trees",
            lambda: _verify_all_structure_trees(hierarchy),
        ),
        (
            "hierarchy_provenance",
            lambda: _verify_hierarchy_and_provenance(
                case, hierarchy, config
            ),
        ),
        (
            "terminal_reconstruction",
            lambda: _verify_terminal_reconstruction(case, hierarchy),
        ),
    )
    for name, operation in stages:
        try:
            operation()
        except Exception as exc:  # Fail closed; never promote a partial check.
            checks.append((name, "FAIL"))
            message = str(exc) or type(exc).__name__
            issues.append((name, "candidate", message))
        else:
            checks.append((name, "PASS"))

    verified = not issues
    return VerificationReport(
        "bl-rctn-verification-v1",
        "1.0.0",
        case.case_id,
        case_digest,
        candidate_digest,
        config_digest,
        verified,
        (
            CompletionLevel.RECURSIVE_VERIFIED
            if verified
            else CompletionLevel.RECURSIVE_CANDIDATE
        ),
        tuple(checks),
        tuple(issues),
    )


def _safe_run_dir(path: Path, *, must_exist: bool = True) -> Path:
    if not isinstance(path, Path):
        raise TypeError("run_dir must be pathlib.Path")
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("run_dir must remain inside the algorithm directory") from exc
    current = _PROJECT_ROOT
    for component in relative.parts:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("symlink run paths are forbidden")
    if must_exist and not absolute.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {absolute}")
    return absolute


def _regular_file_names(directory: Path) -> set[str]:
    _require(directory.is_dir(), f"required directory is missing: {directory}")
    result: set[str] = set()
    for entry in directory.iterdir():
        metadata = entry.lstat()
        _require(not stat.S_ISLNK(metadata.st_mode), "symlink artifacts are forbidden")
        _require(stat.S_ISREG(metadata.st_mode), "unexpected non-file artifact")
        result.add(entry.name)
    return result


def _load_run_inputs(
    run_dir: Path,
) -> tuple[DemoConfig, tuple[GraphCase, ...], dict[str, HierarchyResult]]:
    config = read_demo_config(run_dir / "input" / "config.json")
    cases = read_graph_cases(run_dir / "input" / "cases.jsonl")
    _require(cases, "run contains no graph cases")
    case_ids = {case.case_id for case in cases}
    _require(len(case_ids) == len(cases), "run contains duplicate case IDs")
    engine_dir = run_dir / "engine"
    expected_names = {f"{case_id}.json" for case_id in case_ids}
    _require(
        _regular_file_names(engine_dir) == expected_names,
        "engine artifact closure does not match input cases",
    )
    hierarchies = {
        case.case_id: read_hierarchy_result(
            engine_dir / f"{case.case_id}.json"
        )
        for case in cases
    }
    return config, cases, hierarchies


def publish_verification_reports_once(run_dir: Path) -> tuple[Path, ...]:
    root = _safe_run_dir(run_dir)
    config, cases, hierarchies = _load_run_inputs(root)
    reports = {
        case.case_id: verify_case_result(
            case, hierarchies[case.case_id], config
        )
        for case in cases
    }
    verification_dir = root / "verification"
    if verification_dir.exists():
        _require(
            verification_dir.is_dir() and not verification_dir.is_symlink(),
            "verification path is not a regular directory",
        )
        existing = _regular_file_names(verification_dir)
        _require(
            existing.issubset(
                {f"{case.case_id}.json" for case in cases}
            ),
            "verification directory contains unexpected artifacts",
        )
    else:
        verification_dir.mkdir()

    paths: list[Path] = []
    for case in cases:
        path = verification_dir / f"{case.case_id}.json"
        report = reports[case.case_id]
        expected_bytes = canonical_json_bytes(
            verification_report_to_dict(report)
        )
        if path.exists():
            read_verification_report(path)
            _require(
                path.read_bytes() == expected_bytes,
                f"existing verification report differs for {case.case_id}",
            )
        else:
            write_verification_report_once(path, report)
        paths.append(path)
    return tuple(paths)


def _load_bound_reports(
    run_dir: Path,
    config: DemoConfig,
    cases: tuple[GraphCase, ...],
    hierarchies: dict[str, HierarchyResult],
) -> dict[str, VerificationReport]:
    verification_dir = run_dir / "verification"
    expected_names = {f"{case.case_id}.json" for case in cases}
    _require(
        _regular_file_names(verification_dir) == expected_names,
        "verification artifact closure does not match input cases",
    )
    result: dict[str, VerificationReport] = {}
    for case in cases:
        path = verification_dir / f"{case.case_id}.json"
        actual = read_verification_report(path)
        expected = verify_case_result(case, hierarchies[case.case_id], config)
        _require(
            path.read_bytes()
            == canonical_json_bytes(verification_report_to_dict(expected)),
            f"verification report binding mismatch for {case.case_id}",
        )
        result[case.case_id] = actual
    return result


def _load_json_file(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationFailure(f"{path}: malformed UTF-8") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerificationFailure(f"{path}: malformed JSON") from exc
    _require(type(value) is dict, f"{path}: root must be an object")
    _require(
        raw == canonical_json_bytes(value),
        f"{path}: JSON is not canonical",
    )
    return value, raw


def _bundle_edge_roles(edge_record: EdgeRecord) -> list[str]:
    roles: list[str] = []
    if edge_record.is_root_real:
        roles.append("ROOT_REAL")
    roles.extend(
        f"VIRTUAL_INTERFACE:{identifier}"
        for identifier in edge_record.virtual_interface_ids
    )
    return roles


def _expected_bundle(
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
        torsos.append(
            {
                "record_id": record.record_id,
                "parent_record_id": record.parent_record_id,
                "depth": record.depth,
                "bag_vertices": list(record.bag_vertices),
                "status": record.status.value,
                "support_edges": [list(edge) for edge in record.support_edges],
                "edges": [
                    {
                        "endpoints": list(edge.endpoints),
                        "roles": _bundle_edge_roles(edge),
                        "root_edge_id": edge.root_edge_id,
                    }
                    for edge in record.edge_records
                ],
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
    return {
        "schema_version": "bl-rctn-gpt-visualization-v1",
        "case": graph_case_to_dict(case),
        "root_graph": {
            "nodes": [
                {"id": vertex, "label": str(vertex)}
                for vertex in range(case.num_nodes)
            ],
            "edges": [
                {
                    "endpoints": list(edge),
                    "roles": ["ROOT_REAL"],
                    "color": "black",
                    "line_style": "solid",
                }
                for edge in case.edges
            ],
        },
        "cutsets": cutsets,
        "separations": separations,
        "crossing_edges": crossing_edges,
        "hierarchy": {
            "root_record_id": hierarchy.root_record_id,
            "nodes": [
                {
                    "record_id": record.record_id,
                    "depth": record.depth,
                    "status": record.status.value,
                    "bag_vertices": list(record.bag_vertices),
                    "is_terminal": record.record_id in terminal_ids,
                }
                for record in hierarchy.records
            ],
            "edges": [
                {
                    "parent_record_id": edge.parent_record_id,
                    "child_record_id": edge.child_record_id,
                    "local_tree_node_id": edge.local_tree_node_id,
                }
                for edge in hierarchy.hierarchy_edges
            ],
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
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _expected_prompt(case: GraphCase) -> bytes:
    return f"""# GPT visualization request: {case.case_id}

Create a deterministic four-panel academic figure using only the supplied bundle:

1. root graph and separators
2. elementary crossing graph
3. recursive hierarchy
4. interface provenance

Apply every color and line-style rule from `style_hints`. Preserve all identifiers and vertex labels exactly. Do not invent nodes, edges, cuts, blocks, or labels. Equal interface boundaries remain distinct when their interface IDs differ.

Display this warning verbatim: **STRUCTURE_ONLY; ambient blocks not computed**.
""".encode("utf-8")


def _expected_root_mermaid(bundle: dict[str, object]) -> bytes:
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


def _expected_hierarchy_mermaid(bundle: dict[str, object]) -> bytes:
    nodes = bundle["hierarchy"]["nodes"]
    index_by_id = {
        node["record_id"]: index for index, node in enumerate(nodes)
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


def _expected_crossing_mermaid(bundle: dict[str, object]) -> bytes:
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
        lines.append(
            f'  class s{index} {"tn" if value["is_tn"] else "rejected"}'
        )
    for edge in bundle["crossing_edges"]:
        source = key_to_index[
            (edge["record_id"], edge["source_separation_id"])
        ]
        target = key_to_index[
            (edge["record_id"], edge["target_separation_id"])
        ]
        lines.append(f"  s{source} ---|crosses| s{target}")
    lines.extend(
        (
            "  classDef tn fill:green,stroke:green",
            "  classDef rejected fill:red,stroke:red",
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _expected_dot(case: GraphCase, hierarchy: HierarchyResult) -> bytes:
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


def _expected_case_artifacts(
    case: GraphCase,
    hierarchy: HierarchyResult,
    report: VerificationReport,
) -> dict[str, bytes]:
    bundle = _expected_bundle(case, hierarchy, report)
    return {
        "gpt_visualization_bundle.json": canonical_json_bytes(bundle),
        "gpt_prompt.md": _expected_prompt(case),
        "root_graph.mmd": _expected_root_mermaid(bundle),
        "hierarchy.mmd": _expected_hierarchy_mermaid(bundle),
        "crossing_graph.mmd": _expected_crossing_mermaid(bundle),
        "graph.dot": _expected_dot(case, hierarchy),
    }


def _expected_summary_row(
    case: GraphCase,
    hierarchy: HierarchyResult,
    report: VerificationReport,
) -> dict[str, object]:
    root = _root_record(hierarchy)
    terminals = {
        value: next(
            record for record in hierarchy.records if record.record_id == value
        )
        for value in hierarchy.terminal_record_ids
    }
    status_counts = Counter(value.status for value in terminals.values())
    return {
        "case_id": case.case_id,
        "k": case.k,
        "n": case.num_nodes,
        "m": len(case.edges),
        "status": root.status.value,
        "cutset_count": len(root.local_result.cuts),
        "full_sigma_count": len(root.local_result.full_separations),
        "elementary_count": len(root.local_result.elementary),
        "tn_count": len(root.local_result.tn_full),
        "hierarchy_depth": max(record.depth for record in hierarchy.records),
        "small_terminals": status_counts[LocalStatus.SMALL],
        "high_terminals": status_counts[LocalStatus.HIGH],
        "crossed_terminals": status_counts[LocalStatus.CROSSED],
        "verified": report.verified,
        "bundle_path": f"gpt/{case.case_id}/gpt_visualization_bundle.json",
    }


def _expected_summary_bytes(
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
        row = _expected_summary_row(
            case, hierarchies[case.case_id], reports[case.case_id]
        )
        row["verified"] = "true" if row["verified"] else "false"
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _audit_summary(
    path: Path,
    cases: tuple[GraphCase, ...],
    hierarchies: dict[str, HierarchyResult],
    reports: dict[str, VerificationReport],
) -> None:
    _require(
        path.read_bytes()
        == _expected_summary_bytes(cases, hierarchies, reports),
        "summary.csv differs from independent recomputation",
    )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationFailure("summary.csv is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    _require(
        tuple(reader.fieldnames or ()) == _SUMMARY_FIELDS,
        "summary.csv has the wrong columns",
    )
    rows = list(reader)
    _require(len(rows) == len(cases), "summary.csv has the wrong row count")
    by_id = {row["case_id"]: row for row in rows}
    _require(len(by_id) == len(rows), "summary.csv has duplicate case rows")
    for case in cases:
        expected = _expected_summary_row(
            case, hierarchies[case.case_id], reports[case.case_id]
        )
        actual = by_id.get(case.case_id)
        _require(actual is not None, f"summary row missing for {case.case_id}")
        for field, value in expected.items():
            raw = actual[field]
            if type(value) is int:
                try:
                    parsed: object = int(raw)
                except ValueError as exc:
                    raise VerificationFailure(
                        f"summary field {field} is not an integer"
                    ) from exc
            elif type(value) is bool:
                _require(
                    raw.lower() in ("true", "false"),
                    f"summary field {field} is not a boolean",
                )
                parsed = raw.lower() == "true"
            else:
                parsed = raw
            _require(
                parsed == value,
                f"summary field {field} disagrees for {case.case_id}",
            )


def _walk_artifacts(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda value: value.name):
            metadata = entry.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                "run artifacts must not contain symlinks",
            )
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry)
            elif stat.S_ISREG(metadata.st_mode):
                relative = entry.relative_to(root).as_posix()
                if relative != "manifest.json":
                    result[relative] = entry
            else:
                raise VerificationFailure("run contains a special filesystem object")

    visit(root)
    return result


def _manifest_entries(value: object) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    _require(type(value) is list, "manifest files must be an array")
    for row in value:
        _require(type(row) is dict, "manifest artifact entry must be an object")
        _require(
            set(row) == {"path", "byte_size", "sha256"},
            "manifest file entry schema is not exact",
        )
        path = row["path"]
        size = row["byte_size"]
        digest = row["sha256"]
        _require(
            type(path) is str
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts,
            "manifest artifact path is unsafe",
        )
        _require(
            type(size) is int and size >= 0,
            "manifest artifact size is invalid",
        )
        _require(
            type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "manifest artifact digest is invalid",
        )
        _require(path not in result, "manifest repeats an artifact path")
        result[path] = (size, digest)
    return result


def _audit_optional_exports(
    run_dir: Path,
    config: DemoConfig,
    cases: tuple[GraphCase, ...],
    hierarchies: dict[str, HierarchyResult],
    reports: dict[str, VerificationReport],
) -> bool:
    gpt_dir = run_dir / "gpt"
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.json"
    present = (gpt_dir.exists(), summary_path.exists(), manifest_path.exists())
    if not any(present):
        return False
    _require(all(present), "GPT export stage is only partially present")
    _require(
        gpt_dir.is_dir()
        and not gpt_dir.is_symlink()
        and summary_path.is_file()
        and manifest_path.is_file(),
        "GPT export stage contains invalid filesystem objects",
    )
    expected_case_dirs = {case.case_id for case in cases}
    actual_case_dirs = set()
    for entry in gpt_dir.iterdir():
        metadata = entry.lstat()
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode),
            "gpt directory contains an unexpected artifact",
        )
        actual_case_dirs.add(entry.name)
    _require(
        actual_case_dirs == expected_case_dirs,
        "GPT case-directory closure differs from input cases",
    )
    for case in cases:
        case_dir = gpt_dir / case.case_id
        _require(
            _regular_file_names(case_dir) == set(_BUNDLE_FILENAMES),
            f"GPT six-file closure differs for {case.case_id}",
        )
        expected_artifacts = _expected_case_artifacts(
            case, hierarchies[case.case_id], reports[case.case_id]
        )
        _require(
            tuple(expected_artifacts) == _BUNDLE_FILENAMES,
            "internal verifier bundle filename contract changed",
        )
        for name, expected in expected_artifacts.items():
            _require(
                (case_dir / name).read_bytes() == expected,
                f"GPT artifact {name} differs for {case.case_id}",
            )
    _audit_summary(
        summary_path, cases, hierarchies, reports
    )

    manifest, raw_manifest = _load_json_file(manifest_path)
    _require(
        set(manifest)
        == {
            "schema_version",
            "seed",
            "config_digest",
            "case_count",
            "all_verified",
            "python_version",
            "files",
        },
        "manifest top-level schema is not exact",
    )
    entries = _manifest_entries(manifest["files"])
    actual_files = _walk_artifacts(run_dir)
    _require(
        set(entries) == set(actual_files),
        "manifest relative-path closure differs from run artifacts",
    )
    expected_file_entries: list[dict[str, object]] = []
    for relative, path in sorted(actual_files.items()):
        raw = path.read_bytes()
        size, digest = entries[relative]
        _require(
            len(raw) == size
            and hashlib.sha256(raw).hexdigest() == digest,
            f"manifest size/digest mismatch for {relative}",
        )
        expected_file_entries.append(
            {
                "path": relative,
                "byte_size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    expected_manifest = {
        "schema_version": "bl-rctn-run-manifest-v1",
        "seed": config.seed,
        "config_digest": canonical_sha256(demo_config_to_dict(config)),
        "case_count": len(cases),
        "all_verified": all(report.verified for report in reports.values()),
        "python_version": platform.python_version(),
        "files": expected_file_entries,
    }
    _require(
        manifest == expected_manifest
        and raw_manifest == canonical_json_bytes(expected_manifest),
        "manifest differs from byte-for-byte independent recomputation",
    )
    return True


def verify_run(run_dir: Path) -> dict[str, object]:
    root = _safe_run_dir(run_dir)
    config, cases, hierarchies = _load_run_inputs(root)
    reports = _load_bound_reports(root, config, cases, hierarchies)
    exports_present = _audit_optional_exports(
        root, config, cases, hierarchies, reports
    )
    verified_count = sum(report.verified for report in reports.values())
    return {
        "case_count": len(cases),
        "verified_count": verified_count,
        "all_verified": verified_count == len(cases),
        "gpt_artifacts_present": exports_present,
    }
