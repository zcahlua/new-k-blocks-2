from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, TypeVar


Edge = tuple[int, int]
VertexSet = tuple[int, ...]
Adjacency = dict[int, frozenset[int]]

_CONFIG_SCHEMA = "bl-rctn-demo-config-v1"
_VERIFICATION_SCHEMA = "bl-rctn-verification-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent))
_PUBLISH_COUNTER = itertools.count()


def _value_error(field: str, message: str) -> ValueError:
    return ValueError(f"{field}: {message}")


def _require_string(value: object, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise _value_error(field, "must be a string")
    if nonempty and not value:
        raise _value_error(field, "must not be empty")
    return value


def _require_safe_id(value: object, field: str) -> str:
    text = _require_string(value, field)
    if _SAFE_ID.fullmatch(text) is None:
        raise _value_error(field, "unsafe ID")
    return text


def _require_digest(value: object, field: str) -> str:
    text = _require_string(value, field)
    if _HEX_DIGEST.fullmatch(text) is None:
        raise _value_error(field, "must be 64 lowercase hexadecimal characters")
    return text


def _require_int(
    value: object, field: str, *, minimum: int | None = None
) -> int:
    if type(value) is not int:
        raise _value_error(field, "must be an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        raise _value_error(field, f"must be at least {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise _value_error(field, "must be a boolean")
    return value


E = TypeVar("E", bound=StrEnum)


def _require_enum(value: object, enum_type: type[E], field: str) -> E:
    if type(value) is not enum_type:
        raise _value_error(field, f"must be {enum_type.__name__}")
    return value


def _vertex_set(value: object, field: str, *, allow_empty: bool = True) -> VertexSet:
    if type(value) is not tuple:
        raise _value_error(field, "must be an immutable tuple")
    vertices = tuple(
        _require_int(vertex, f"{field}[{index}]", minimum=0)
        for index, vertex in enumerate(value)
    )
    if len(set(vertices)) != len(vertices):
        raise _value_error(field, "contains duplicate vertices")
    if not allow_empty and not vertices:
        raise _value_error(field, "must not be empty")
    return tuple(sorted(vertices))


def _edges(
    value: object, field: str, *, allowed_vertices: frozenset[int] | None = None
) -> tuple[Edge, ...]:
    if type(value) is not tuple:
        raise _value_error(field, "must be an immutable tuple")
    normalized: list[Edge] = []
    seen: set[Edge] = set()
    for index, edge in enumerate(value):
        if type(edge) is not tuple or len(edge) != 2:
            raise _value_error(f"{field}[{index}]", "must be a two-item tuple")
        left = _require_int(edge[0], f"{field}[{index}][0]", minimum=0)
        right = _require_int(edge[1], f"{field}[{index}][1]", minimum=0)
        if left == right:
            raise _value_error(f"{field}[{index}]", "self-loop is forbidden")
        canonical = (min(left, right), max(left, right))
        if canonical in seen:
            raise _value_error(field, f"duplicate edge {canonical}")
        if allowed_vertices is not None and (
            canonical[0] not in allowed_vertices or canonical[1] not in allowed_vertices
        ):
            raise _value_error(field, f"endpoint outside the vertex set: {canonical}")
        seen.add(canonical)
        normalized.append(canonical)
    return tuple(sorted(normalized))


def _records(
    value: object,
    field: str,
    *,
    length: int,
    item: Callable[[object, str], object],
) -> tuple[tuple[object, ...], ...]:
    if type(value) is not tuple:
        raise _value_error(field, "must be an immutable tuple")
    result: list[tuple[object, ...]] = []
    for row_index, row in enumerate(value):
        if type(row) is not tuple or len(row) != length:
            raise _value_error(
                f"{field}[{row_index}]", f"must be a {length}-item tuple"
            )
        result.append(
            tuple(
                item(cell, f"{field}[{row_index}][{column_index}]")
                for column_index, cell in enumerate(row)
            )
        )
    return tuple(result)


def _instances(value: object, expected_type: type, field: str) -> tuple:
    if type(value) is not tuple:
        raise _value_error(field, "must be an immutable tuple")
    for index, item in enumerate(value):
        if type(item) is not expected_type:
            raise _value_error(f"{field}[{index}]", f"must be {expected_type.__name__}")
    return value


def _unique(values: tuple, key: Callable[[object], object], field: str) -> None:
    seen: set[object] = set()
    for item in values:
        identity = key(item)
        if identity in seen:
            raise _value_error(field, f"duplicate value {identity!r}")
        seen.add(identity)


class LocalStatus(StrEnum):
    SMALL = "SMALL"
    HIGH = "HIGH"
    CROSSED = "CROSSED"
    SPLIT = "SPLIT"


class CompletionLevel(StrEnum):
    LOCAL_EXACT = "LOCAL_EXACT"
    RECURSIVE_CANDIDATE = "RECURSIVE_CANDIDATE"
    RECURSIVE_VERIFIED = "RECURSIVE_VERIFIED"


class Coverage(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class DemoConfig:
    schema_version: str
    ks: tuple[int, ...]
    seed: int
    random_per_k: int
    max_full_separations: int
    max_depth: int
    mode: str

    def __post_init__(self) -> None:
        if self.schema_version != _CONFIG_SCHEMA:
            raise _value_error(
                "schema_version", f"must equal {_CONFIG_SCHEMA!r}"
            )
        if type(self.ks) is not tuple or not self.ks:
            raise _value_error("ks", "must be a nonempty immutable tuple")
        ks = tuple(
            _require_int(value, f"ks[{index}]", minimum=2)
            for index, value in enumerate(self.ks)
        )
        if len(set(ks)) != len(ks):
            raise _value_error("ks", "must contain unique values")
        object.__setattr__(self, "ks", ks)
        _require_int(self.seed, "seed", minimum=0)
        _require_int(self.random_per_k, "random_per_k", minimum=0)
        _require_int(
            self.max_full_separations, "max_full_separations", minimum=1
        )
        _require_int(self.max_depth, "max_depth", minimum=1)
        if self.mode != "STRUCTURE_ONLY":
            raise _value_error("mode", "only STRUCTURE_ONLY is supported")


@dataclass(frozen=True)
class GraphCase:
    case_id: str
    k: int
    num_nodes: int
    edges: tuple[Edge, ...]
    family: str
    parameters: tuple[tuple[str, int], ...]
    seed: int
    expected: tuple[tuple[str, int | str], ...]

    def __post_init__(self) -> None:
        _require_safe_id(self.case_id, "case_id")
        _require_int(self.k, "k", minimum=2)
        node_count = _require_int(self.num_nodes, "num_nodes", minimum=0)
        normalized_edges = _edges(
            self.edges, "edges", allowed_vertices=frozenset(range(node_count))
        )
        object.__setattr__(self, "edges", normalized_edges)
        _require_safe_id(self.family, "family")
        if type(self.parameters) is not tuple:
            raise _value_error("parameters", "must be an immutable tuple")
        parameters: list[tuple[str, int]] = []
        parameter_keys: set[str] = set()
        for index, row in enumerate(self.parameters):
            if type(row) is not tuple or len(row) != 2:
                raise _value_error(
                    f"parameters[{index}]", "must be a two-item tuple"
                )
            key = _require_safe_id(row[0], f"parameters[{index}][0]")
            value = _require_int(row[1], f"parameters[{index}][1]")
            if key in parameter_keys:
                raise _value_error("parameters", f"duplicate key {key!r}")
            parameter_keys.add(key)
            parameters.append((key, value))
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))
        _require_int(self.seed, "seed", minimum=0)
        if type(self.expected) is not tuple:
            raise _value_error("expected", "must be an immutable tuple")
        expected: list[tuple[str, int | str]] = []
        expected_keys: set[str] = set()
        for index, row in enumerate(self.expected):
            if type(row) is not tuple or len(row) != 2:
                raise _value_error(f"expected[{index}]", "must be a two-item tuple")
            key = _require_safe_id(row[0], f"expected[{index}][0]")
            value = row[1]
            if type(value) is int:
                pass
            elif type(value) is str and value:
                pass
            else:
                raise _value_error(
                    f"expected[{index}][1]", "must be an integer or nonempty string"
                )
            if key in expected_keys:
                raise _value_error("expected", f"duplicate key {key!r}")
            expected_keys.add(key)
            expected.append((key, value))
        object.__setattr__(self, "expected", tuple(sorted(expected)))


@dataclass(frozen=True)
class SupportGraph:
    vertices: VertexSet
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        vertices = _vertex_set(self.vertices, "vertices")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(
            self,
            "edges",
            _edges(self.edges, "edges", allowed_vertices=frozenset(vertices)),
        )


@dataclass(frozen=True)
class Separation:
    side_a: VertexSet
    side_b: VertexSet
    separator: VertexSet

    def __post_init__(self) -> None:
        side_a = _vertex_set(self.side_a, "side_a", allow_empty=False)
        side_b = _vertex_set(self.side_b, "side_b", allow_empty=False)
        separator = _vertex_set(self.separator, "separator")
        if frozenset(side_a).intersection(side_b) != frozenset(separator):
            raise _value_error(
                "separator", "must equal the intersection of the two sides"
            )
        if not frozenset(side_a).difference(separator) or not frozenset(
            side_b
        ).difference(separator):
            raise _value_error("sides", "both wings must be nonempty")
        if side_b < side_a:
            side_a, side_b = side_b, side_a
        object.__setattr__(self, "side_a", side_a)
        object.__setattr__(self, "side_b", side_b)
        object.__setattr__(self, "separator", separator)


@dataclass(frozen=True)
class OrientedSeparation:
    base_separation_id: str
    away_side: VertexSet
    toward_side: VertexSet

    def __post_init__(self) -> None:
        _require_safe_id(self.base_separation_id, "base_separation_id")
        object.__setattr__(
            self, "away_side", _vertex_set(self.away_side, "away_side")
        )
        object.__setattr__(
            self, "toward_side", _vertex_set(self.toward_side, "toward_side")
        )


@dataclass(frozen=True)
class CutRecord:
    separator: VertexSet
    components: tuple[VertexSet, ...]

    def __post_init__(self) -> None:
        separator = _vertex_set(self.separator, "separator")
        if type(self.components) is not tuple or len(self.components) < 2:
            raise _value_error("components", "must contain at least two components")
        components = tuple(
            _vertex_set(component, f"components[{index}]", allow_empty=False)
            for index, component in enumerate(self.components)
        )
        occupied = set(separator)
        for index, component in enumerate(components):
            if occupied.intersection(component):
                raise _value_error(
                    f"components[{index}]", "components and separator must be disjoint"
                )
            occupied.update(component)
        object.__setattr__(self, "separator", separator)
        object.__setattr__(self, "components", tuple(sorted(components)))


@dataclass(frozen=True)
class LocalResult:
    status: LocalStatus
    cuts: tuple[CutRecord, ...]
    full_separations: tuple[Separation, ...]
    elementary: tuple[Separation, ...]
    tn_full: tuple[Separation, ...]
    tn_pairwise: tuple[Separation, ...]
    tn_aggregated: tuple[Separation, ...]
    rejection_witnesses: tuple[tuple[str, str], ...]
    completion_level: CompletionLevel

    def __post_init__(self) -> None:
        _require_enum(self.status, LocalStatus, "status")
        cuts = _instances(self.cuts, CutRecord, "cuts")
        _unique(cuts, lambda value: value.separator, "cuts")
        object.__setattr__(
            self, "cuts", tuple(sorted(cuts, key=lambda value: value.separator))
        )
        for field in (
            "full_separations",
            "elementary",
            "tn_full",
            "tn_pairwise",
            "tn_aggregated",
        ):
            separations = _instances(getattr(self, field), Separation, field)
            _unique(
                separations,
                lambda value: (value.side_a, value.side_b),
                field,
            )
            object.__setattr__(
                self,
                field,
                tuple(
                    sorted(
                        separations, key=lambda value: (value.side_a, value.side_b)
                    )
                ),
            )
        witnesses = _records(
            self.rejection_witnesses,
            "rejection_witnesses",
            length=2,
            item=lambda value, field: _require_string(value, field),
        )
        object.__setattr__(self, "rejection_witnesses", tuple(sorted(witnesses)))
        _require_enum(self.completion_level, CompletionLevel, "completion_level")


@dataclass(frozen=True)
class StructureTreeNode:
    node_id: str
    orientation_signature: tuple[tuple[str, int], ...]
    bag_vertices: VertexSet

    def __post_init__(self) -> None:
        _require_safe_id(self.node_id, "node_id")
        if type(self.orientation_signature) is not tuple:
            raise _value_error(
                "orientation_signature", "must be an immutable tuple"
            )
        signature: list[tuple[str, int]] = []
        identifiers: set[str] = set()
        for index, row in enumerate(self.orientation_signature):
            if type(row) is not tuple or len(row) != 2:
                raise _value_error(
                    f"orientation_signature[{index}]", "must be a two-item tuple"
                )
            identifier = _require_safe_id(
                row[0], f"orientation_signature[{index}][0]"
            )
            orientation = _require_int(
                row[1], f"orientation_signature[{index}][1]"
            )
            if identifier in identifiers:
                raise _value_error(
                    "orientation_signature", f"duplicate ID {identifier!r}"
                )
            identifiers.add(identifier)
            signature.append((identifier, orientation))
        object.__setattr__(
            self, "orientation_signature", tuple(sorted(signature))
        )
        object.__setattr__(
            self,
            "bag_vertices",
            _vertex_set(self.bag_vertices, "bag_vertices", allow_empty=False),
        )


@dataclass(frozen=True)
class StructureTreeEdge:
    tree_edge_id: str
    source_node_id: str
    target_node_id: str
    separation: Separation
    adhesion: VertexSet

    def __post_init__(self) -> None:
        _require_safe_id(self.tree_edge_id, "tree_edge_id")
        source = _require_safe_id(self.source_node_id, "source_node_id")
        target = _require_safe_id(self.target_node_id, "target_node_id")
        if source == target:
            raise _value_error("target_node_id", "tree edge endpoints must differ")
        if type(self.separation) is not Separation:
            raise _value_error("separation", "must be Separation")
        adhesion = _vertex_set(self.adhesion, "adhesion")
        if adhesion != self.separation.separator:
            raise _value_error("adhesion", "must equal the separation separator")
        object.__setattr__(self, "adhesion", adhesion)


@dataclass(frozen=True)
class StructureTree:
    nodes: tuple[StructureTreeNode, ...]
    edges: tuple[StructureTreeEdge, ...]

    def __post_init__(self) -> None:
        nodes = _instances(self.nodes, StructureTreeNode, "nodes")
        edges = _instances(self.edges, StructureTreeEdge, "edges")
        _unique(nodes, lambda value: value.node_id, "nodes")
        _unique(edges, lambda value: value.tree_edge_id, "edges")
        node_ids = {node.node_id for node in nodes}
        for index, edge in enumerate(edges):
            if (
                edge.source_node_id not in node_ids
                or edge.target_node_id not in node_ids
            ):
                raise _value_error(
                    f"edges[{index}]", "references an unknown tree node"
                )
        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda x: x.node_id)))
        object.__setattr__(
            self, "edges", tuple(sorted(edges, key=lambda x: x.tree_edge_id))
        )


@dataclass(frozen=True)
class EdgeRecord:
    endpoints: Edge
    is_root_real: bool
    root_edge_id: str | None
    virtual_interface_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        endpoints = _edges((self.endpoints,), "endpoints")
        object.__setattr__(self, "endpoints", endpoints[0])
        is_root_real = _require_bool(self.is_root_real, "is_root_real")
        if is_root_real:
            _require_safe_id(self.root_edge_id, "root_edge_id")
        elif self.root_edge_id is not None:
            raise _value_error(
                "root_edge_id", "must be null when the edge is not root-real"
            )
        if type(self.virtual_interface_ids) is not tuple:
            raise _value_error(
                "virtual_interface_ids", "must be an immutable tuple"
            )
        identifiers = tuple(
            _require_safe_id(value, f"virtual_interface_ids[{index}]")
            for index, value in enumerate(self.virtual_interface_ids)
        )
        if len(set(identifiers)) != len(identifiers):
            raise _value_error("virtual_interface_ids", "contains duplicates")
        object.__setattr__(self, "virtual_interface_ids", tuple(sorted(identifiers)))


@dataclass(frozen=True)
class InterfaceIncidence:
    incidence_id: str
    side_tree_node_id: str
    opposite_incidence_id: str

    def __post_init__(self) -> None:
        identifier = _require_safe_id(self.incidence_id, "incidence_id")
        _require_safe_id(self.side_tree_node_id, "side_tree_node_id")
        opposite = _require_safe_id(
            self.opposite_incidence_id, "opposite_incidence_id"
        )
        if identifier == opposite:
            raise _value_error(
                "opposite_incidence_id", "must identify the opposite incidence"
            )


@dataclass(frozen=True)
class InterfaceObject:
    interface_id: str
    creator_record_id: str
    creator_tree_edge_id: str
    boundary: VertexSet
    incidences: tuple[InterfaceIncidence, InterfaceIncidence]

    def __post_init__(self) -> None:
        _require_safe_id(self.interface_id, "interface_id")
        _require_safe_id(self.creator_record_id, "creator_record_id")
        _require_safe_id(self.creator_tree_edge_id, "creator_tree_edge_id")
        object.__setattr__(
            self,
            "boundary",
            _vertex_set(self.boundary, "boundary", allow_empty=False),
        )
        incidences = _instances(
            self.incidences, InterfaceIncidence, "incidences"
        )
        if len(incidences) != 2:
            raise _value_error("incidences", "must contain exactly two incidences")
        left, right = incidences
        if (
            left.incidence_id == right.incidence_id
            or left.opposite_incidence_id != right.incidence_id
            or right.opposite_incidence_id != left.incidence_id
        ):
            raise _value_error("incidences", "must form one opposite pair")
        object.__setattr__(
            self, "incidences", tuple(sorted(incidences, key=lambda x: x.incidence_id))
        )


@dataclass(frozen=True)
class InterfaceRef:
    interface_id: str
    incidence_id: str
    coverage: Coverage
    local_boundary: VertexSet

    def __post_init__(self) -> None:
        _require_safe_id(self.interface_id, "interface_id")
        _require_safe_id(self.incidence_id, "incidence_id")
        _require_enum(self.coverage, Coverage, "coverage")
        object.__setattr__(
            self,
            "local_boundary",
            _vertex_set(
                self.local_boundary, "local_boundary", allow_empty=False
            ),
        )


@dataclass(frozen=True)
class TorsoRecord:
    record_id: str
    parent_record_id: str | None
    depth: int
    bag_vertices: VertexSet
    status: LocalStatus
    support_edges: tuple[Edge, ...]
    edge_records: tuple[EdgeRecord, ...]
    interface_refs: tuple[InterfaceRef, ...]
    local_result: LocalResult
    structure_tree: StructureTree | None

    def __post_init__(self) -> None:
        _require_safe_id(self.record_id, "record_id")
        if self.parent_record_id is not None:
            _require_safe_id(self.parent_record_id, "parent_record_id")
        _require_int(self.depth, "depth", minimum=0)
        bag = _vertex_set(self.bag_vertices, "bag_vertices", allow_empty=False)
        object.__setattr__(self, "bag_vertices", bag)
        _require_enum(self.status, LocalStatus, "status")
        object.__setattr__(
            self,
            "support_edges",
            _edges(
                self.support_edges,
                "support_edges",
                allowed_vertices=frozenset(bag),
            ),
        )
        edge_records = _instances(self.edge_records, EdgeRecord, "edge_records")
        _unique(edge_records, lambda value: value.endpoints, "edge_records")
        for index, edge_record in enumerate(edge_records):
            if not set(edge_record.endpoints).issubset(bag):
                raise _value_error(
                    f"edge_records[{index}]", "endpoint outside bag_vertices"
                )
        object.__setattr__(
            self,
            "edge_records",
            tuple(sorted(edge_records, key=lambda value: value.endpoints)),
        )
        interface_refs = _instances(
            self.interface_refs, InterfaceRef, "interface_refs"
        )
        _unique(
            interface_refs,
            lambda value: (value.interface_id, value.incidence_id),
            "interface_refs",
        )
        for index, ref in enumerate(interface_refs):
            if not set(ref.local_boundary).issubset(bag):
                raise _value_error(
                    f"interface_refs[{index}]", "local boundary outside bag_vertices"
                )
        object.__setattr__(
            self,
            "interface_refs",
            tuple(
                sorted(
                    interface_refs,
                    key=lambda value: (value.interface_id, value.incidence_id),
                )
            ),
        )
        if type(self.local_result) is not LocalResult:
            raise _value_error("local_result", "must be LocalResult")
        if self.structure_tree is not None and type(self.structure_tree) is not StructureTree:
            raise _value_error("structure_tree", "must be StructureTree or null")


@dataclass(frozen=True)
class HierarchyEdge:
    parent_record_id: str
    child_record_id: str
    local_tree_node_id: str

    def __post_init__(self) -> None:
        parent = _require_safe_id(self.parent_record_id, "parent_record_id")
        child = _require_safe_id(self.child_record_id, "child_record_id")
        _require_safe_id(self.local_tree_node_id, "local_tree_node_id")
        if parent == child:
            raise _value_error("child_record_id", "must differ from parent_record_id")


@dataclass(frozen=True)
class HierarchyResult:
    root_record_id: str
    config_digest: str
    records: tuple[TorsoRecord, ...]
    interfaces: tuple[InterfaceObject, ...]
    hierarchy_edges: tuple[HierarchyEdge, ...]
    terminal_record_ids: tuple[str, ...]
    completion_level: CompletionLevel

    def __post_init__(self) -> None:
        _require_safe_id(self.root_record_id, "root_record_id")
        _require_digest(self.config_digest, "config_digest")
        records = _instances(self.records, TorsoRecord, "records")
        interfaces = _instances(self.interfaces, InterfaceObject, "interfaces")
        hierarchy_edges = _instances(
            self.hierarchy_edges, HierarchyEdge, "hierarchy_edges"
        )
        _unique(records, lambda value: value.record_id, "records")
        _unique(interfaces, lambda value: value.interface_id, "interfaces")
        _unique(
            hierarchy_edges,
            lambda value: (
                value.parent_record_id,
                value.child_record_id,
                value.local_tree_node_id,
            ),
            "hierarchy_edges",
        )
        object.__setattr__(
            self, "records", tuple(sorted(records, key=lambda value: value.record_id))
        )
        object.__setattr__(
            self,
            "interfaces",
            tuple(sorted(interfaces, key=lambda value: value.interface_id)),
        )
        object.__setattr__(
            self,
            "hierarchy_edges",
            tuple(
                sorted(
                    hierarchy_edges,
                    key=lambda value: (
                        value.parent_record_id,
                        value.child_record_id,
                        value.local_tree_node_id,
                    ),
                )
            ),
        )
        if type(self.terminal_record_ids) is not tuple:
            raise _value_error(
                "terminal_record_ids", "must be an immutable tuple"
            )
        terminal_ids = tuple(
            _require_safe_id(value, f"terminal_record_ids[{index}]")
            for index, value in enumerate(self.terminal_record_ids)
        )
        if len(set(terminal_ids)) != len(terminal_ids):
            raise _value_error("terminal_record_ids", "contains duplicates")
        object.__setattr__(self, "terminal_record_ids", tuple(sorted(terminal_ids)))
        _require_enum(self.completion_level, CompletionLevel, "completion_level")


@dataclass(frozen=True)
class VerificationReport:
    schema_version: str
    verifier_version: str
    case_id: str
    case_digest: str
    candidate_hierarchy_digest: str
    config_digest: str
    verified: bool
    completion_level: CompletionLevel
    checks: tuple[tuple[str, str], ...]
    issues: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version != _VERIFICATION_SCHEMA:
            raise _value_error(
                "schema_version", f"must equal {_VERIFICATION_SCHEMA!r}"
            )
        _require_safe_id(self.verifier_version, "verifier_version")
        _require_safe_id(self.case_id, "case_id")
        _require_digest(self.case_digest, "case_digest")
        _require_digest(
            self.candidate_hierarchy_digest, "candidate_hierarchy_digest"
        )
        _require_digest(self.config_digest, "config_digest")
        _require_bool(self.verified, "verified")
        _require_enum(self.completion_level, CompletionLevel, "completion_level")
        checks = _records(
            self.checks,
            "checks",
            length=2,
            item=lambda value, field: _require_string(value, field),
        )
        issues = _records(
            self.issues,
            "issues",
            length=3,
            item=lambda value, field: _require_string(value, field),
        )
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "issues", issues)


def _closed_dict(
    value: object, expected_fields: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise _value_error(name, "must be a JSON object")
    keys = set(value)
    unknown = keys.difference(expected_fields)
    missing = expected_fields.difference(keys)
    if unknown:
        raise _value_error(name, f"unknown fields: {sorted(unknown)!r}")
    if missing:
        raise _value_error(name, f"missing fields: {sorted(missing)!r}")
    return value


def _json_list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise _value_error(field, "must be a JSON array")
    return value


def _json_rows(value: object, field: str, length: int) -> tuple[tuple, ...]:
    rows = _json_list(value, field)
    result: list[tuple] = []
    for index, row in enumerate(rows):
        if type(row) is not list or len(row) != length:
            raise _value_error(
                f"{field}[{index}]", f"must be a {length}-item JSON array"
            )
        result.append(tuple(row))
    return tuple(result)


def demo_config_to_dict(config: DemoConfig) -> dict[str, object]:
    if type(config) is not DemoConfig:
        raise _value_error("config", "must be DemoConfig")
    return {
        "schema_version": config.schema_version,
        "ks": list(config.ks),
        "seed": config.seed,
        "random_per_k": config.random_per_k,
        "max_full_separations": config.max_full_separations,
        "max_depth": config.max_depth,
        "mode": config.mode,
    }


def demo_config_from_dict(value: dict[str, object]) -> DemoConfig:
    data = _closed_dict(
        value,
        frozenset(
            {
                "schema_version",
                "ks",
                "seed",
                "random_per_k",
                "max_full_separations",
                "max_depth",
                "mode",
            }
        ),
        "DemoConfig",
    )
    return DemoConfig(
        data["schema_version"],
        tuple(_json_list(data["ks"], "DemoConfig.ks")),
        data["seed"],
        data["random_per_k"],
        data["max_full_separations"],
        data["max_depth"],
        data["mode"],
    )


def graph_case_to_dict(case: GraphCase) -> dict[str, object]:
    if type(case) is not GraphCase:
        raise _value_error("case", "must be GraphCase")
    return {
        "case_id": case.case_id,
        "k": case.k,
        "num_nodes": case.num_nodes,
        "edges": [list(edge) for edge in case.edges],
        "family": case.family,
        "parameters": [list(row) for row in case.parameters],
        "seed": case.seed,
        "expected": [list(row) for row in case.expected],
    }


def graph_case_from_dict(value: dict[str, object]) -> GraphCase:
    data = _closed_dict(
        value,
        frozenset(
            {
                "case_id",
                "k",
                "num_nodes",
                "edges",
                "family",
                "parameters",
                "seed",
                "expected",
            }
        ),
        "GraphCase",
    )
    return GraphCase(
        data["case_id"],
        data["k"],
        data["num_nodes"],
        _json_rows(data["edges"], "GraphCase.edges", 2),
        data["family"],
        _json_rows(data["parameters"], "GraphCase.parameters", 2),
        data["seed"],
        _json_rows(data["expected"], "GraphCase.expected", 2),
    )


def support_graph_to_dict(graph: SupportGraph) -> dict[str, object]:
    return {"vertices": list(graph.vertices), "edges": [list(e) for e in graph.edges]}


def support_graph_from_dict(value: dict[str, object]) -> SupportGraph:
    data = _closed_dict(
        value, frozenset({"vertices", "edges"}), "SupportGraph"
    )
    return SupportGraph(
        tuple(_json_list(data["vertices"], "SupportGraph.vertices")),
        _json_rows(data["edges"], "SupportGraph.edges", 2),
    )


def separation_to_dict(value: Separation) -> dict[str, object]:
    return {
        "side_a": list(value.side_a),
        "side_b": list(value.side_b),
        "separator": list(value.separator),
    }


def separation_from_dict(value: dict[str, object]) -> Separation:
    data = _closed_dict(
        value, frozenset({"side_a", "side_b", "separator"}), "Separation"
    )
    return Separation(
        tuple(_json_list(data["side_a"], "Separation.side_a")),
        tuple(_json_list(data["side_b"], "Separation.side_b")),
        tuple(_json_list(data["separator"], "Separation.separator")),
    )


def oriented_separation_to_dict(value: OrientedSeparation) -> dict[str, object]:
    return {
        "base_separation_id": value.base_separation_id,
        "away_side": list(value.away_side),
        "toward_side": list(value.toward_side),
    }


def oriented_separation_from_dict(
    value: dict[str, object],
) -> OrientedSeparation:
    data = _closed_dict(
        value,
        frozenset({"base_separation_id", "away_side", "toward_side"}),
        "OrientedSeparation",
    )
    return OrientedSeparation(
        data["base_separation_id"],
        tuple(_json_list(data["away_side"], "OrientedSeparation.away_side")),
        tuple(_json_list(data["toward_side"], "OrientedSeparation.toward_side")),
    )


def cut_record_to_dict(value: CutRecord) -> dict[str, object]:
    return {
        "separator": list(value.separator),
        "components": [list(component) for component in value.components],
    }


def cut_record_from_dict(value: dict[str, object]) -> CutRecord:
    data = _closed_dict(
        value, frozenset({"separator", "components"}), "CutRecord"
    )
    return CutRecord(
        tuple(_json_list(data["separator"], "CutRecord.separator")),
        tuple(
            tuple(_json_list(component, f"CutRecord.components[{index}]"))
            for index, component in enumerate(
                _json_list(data["components"], "CutRecord.components")
            )
        ),
    )


def local_result_to_dict(value: LocalResult) -> dict[str, object]:
    return {
        "status": value.status.value,
        "cuts": [cut_record_to_dict(item) for item in value.cuts],
        "full_separations": [
            separation_to_dict(item) for item in value.full_separations
        ],
        "elementary": [separation_to_dict(item) for item in value.elementary],
        "tn_full": [separation_to_dict(item) for item in value.tn_full],
        "tn_pairwise": [separation_to_dict(item) for item in value.tn_pairwise],
        "tn_aggregated": [
            separation_to_dict(item) for item in value.tn_aggregated
        ],
        "rejection_witnesses": [list(row) for row in value.rejection_witnesses],
        "completion_level": value.completion_level.value,
    }


def _enum_from_wire(enum_type: type[E], value: object, field: str) -> E:
    if type(value) is not str:
        raise _value_error(field, "must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _value_error(field, f"unknown value {value!r}") from exc


def local_result_from_dict(value: dict[str, object]) -> LocalResult:
    fields = frozenset(
        {
            "status",
            "cuts",
            "full_separations",
            "elementary",
            "tn_full",
            "tn_pairwise",
            "tn_aggregated",
            "rejection_witnesses",
            "completion_level",
        }
    )
    data = _closed_dict(value, fields, "LocalResult")

    def separations(field: str) -> tuple[Separation, ...]:
        return tuple(
            separation_from_dict(item)
            for item in _json_list(data[field], f"LocalResult.{field}")
        )

    return LocalResult(
        _enum_from_wire(LocalStatus, data["status"], "LocalResult.status"),
        tuple(
            cut_record_from_dict(item)
            for item in _json_list(data["cuts"], "LocalResult.cuts")
        ),
        separations("full_separations"),
        separations("elementary"),
        separations("tn_full"),
        separations("tn_pairwise"),
        separations("tn_aggregated"),
        _json_rows(
            data["rejection_witnesses"],
            "LocalResult.rejection_witnesses",
            2,
        ),
        _enum_from_wire(
            CompletionLevel,
            data["completion_level"],
            "LocalResult.completion_level",
        ),
    )


def structure_tree_node_to_dict(value: StructureTreeNode) -> dict[str, object]:
    return {
        "node_id": value.node_id,
        "orientation_signature": [list(row) for row in value.orientation_signature],
        "bag_vertices": list(value.bag_vertices),
    }


def structure_tree_node_from_dict(value: dict[str, object]) -> StructureTreeNode:
    data = _closed_dict(
        value,
        frozenset({"node_id", "orientation_signature", "bag_vertices"}),
        "StructureTreeNode",
    )
    return StructureTreeNode(
        data["node_id"],
        _json_rows(
            data["orientation_signature"],
            "StructureTreeNode.orientation_signature",
            2,
        ),
        tuple(
            _json_list(data["bag_vertices"], "StructureTreeNode.bag_vertices")
        ),
    )


def structure_tree_edge_to_dict(value: StructureTreeEdge) -> dict[str, object]:
    return {
        "tree_edge_id": value.tree_edge_id,
        "source_node_id": value.source_node_id,
        "target_node_id": value.target_node_id,
        "separation": separation_to_dict(value.separation),
        "adhesion": list(value.adhesion),
    }


def structure_tree_edge_from_dict(value: dict[str, object]) -> StructureTreeEdge:
    data = _closed_dict(
        value,
        frozenset(
            {
                "tree_edge_id",
                "source_node_id",
                "target_node_id",
                "separation",
                "adhesion",
            }
        ),
        "StructureTreeEdge",
    )
    return StructureTreeEdge(
        data["tree_edge_id"],
        data["source_node_id"],
        data["target_node_id"],
        separation_from_dict(data["separation"]),
        tuple(_json_list(data["adhesion"], "StructureTreeEdge.adhesion")),
    )


def structure_tree_to_dict(value: StructureTree) -> dict[str, object]:
    return {
        "nodes": [structure_tree_node_to_dict(item) for item in value.nodes],
        "edges": [structure_tree_edge_to_dict(item) for item in value.edges],
    }


def structure_tree_from_dict(value: dict[str, object]) -> StructureTree:
    data = _closed_dict(
        value, frozenset({"nodes", "edges"}), "StructureTree"
    )
    return StructureTree(
        tuple(
            structure_tree_node_from_dict(item)
            for item in _json_list(data["nodes"], "StructureTree.nodes")
        ),
        tuple(
            structure_tree_edge_from_dict(item)
            for item in _json_list(data["edges"], "StructureTree.edges")
        ),
    )


def edge_record_to_dict(value: EdgeRecord) -> dict[str, object]:
    return {
        "endpoints": list(value.endpoints),
        "is_root_real": value.is_root_real,
        "root_edge_id": value.root_edge_id,
        "virtual_interface_ids": list(value.virtual_interface_ids),
    }


def edge_record_from_dict(value: dict[str, object]) -> EdgeRecord:
    data = _closed_dict(
        value,
        frozenset(
            {
                "endpoints",
                "is_root_real",
                "root_edge_id",
                "virtual_interface_ids",
            }
        ),
        "EdgeRecord",
    )
    endpoints = _json_list(data["endpoints"], "EdgeRecord.endpoints")
    if len(endpoints) != 2:
        raise _value_error("EdgeRecord.endpoints", "must contain two items")
    return EdgeRecord(
        tuple(endpoints),
        data["is_root_real"],
        data["root_edge_id"],
        tuple(
            _json_list(
                data["virtual_interface_ids"], "EdgeRecord.virtual_interface_ids"
            )
        ),
    )


def interface_incidence_to_dict(value: InterfaceIncidence) -> dict[str, object]:
    return {
        "incidence_id": value.incidence_id,
        "side_tree_node_id": value.side_tree_node_id,
        "opposite_incidence_id": value.opposite_incidence_id,
    }


def interface_incidence_from_dict(
    value: dict[str, object],
) -> InterfaceIncidence:
    data = _closed_dict(
        value,
        frozenset(
            {"incidence_id", "side_tree_node_id", "opposite_incidence_id"}
        ),
        "InterfaceIncidence",
    )
    return InterfaceIncidence(
        data["incidence_id"],
        data["side_tree_node_id"],
        data["opposite_incidence_id"],
    )


def interface_object_to_dict(value: InterfaceObject) -> dict[str, object]:
    return {
        "interface_id": value.interface_id,
        "creator_record_id": value.creator_record_id,
        "creator_tree_edge_id": value.creator_tree_edge_id,
        "boundary": list(value.boundary),
        "incidences": [
            interface_incidence_to_dict(item) for item in value.incidences
        ],
    }


def interface_object_from_dict(value: dict[str, object]) -> InterfaceObject:
    data = _closed_dict(
        value,
        frozenset(
            {
                "interface_id",
                "creator_record_id",
                "creator_tree_edge_id",
                "boundary",
                "incidences",
            }
        ),
        "InterfaceObject",
    )
    return InterfaceObject(
        data["interface_id"],
        data["creator_record_id"],
        data["creator_tree_edge_id"],
        tuple(_json_list(data["boundary"], "InterfaceObject.boundary")),
        tuple(
            interface_incidence_from_dict(item)
            for item in _json_list(
                data["incidences"], "InterfaceObject.incidences"
            )
        ),
    )


def interface_ref_to_dict(value: InterfaceRef) -> dict[str, object]:
    return {
        "interface_id": value.interface_id,
        "incidence_id": value.incidence_id,
        "coverage": value.coverage.value,
        "local_boundary": list(value.local_boundary),
    }


def interface_ref_from_dict(value: dict[str, object]) -> InterfaceRef:
    data = _closed_dict(
        value,
        frozenset(
            {"interface_id", "incidence_id", "coverage", "local_boundary"}
        ),
        "InterfaceRef",
    )
    return InterfaceRef(
        data["interface_id"],
        data["incidence_id"],
        _enum_from_wire(Coverage, data["coverage"], "InterfaceRef.coverage"),
        tuple(
            _json_list(data["local_boundary"], "InterfaceRef.local_boundary")
        ),
    )


def torso_record_to_dict(value: TorsoRecord) -> dict[str, object]:
    return {
        "record_id": value.record_id,
        "parent_record_id": value.parent_record_id,
        "depth": value.depth,
        "bag_vertices": list(value.bag_vertices),
        "status": value.status.value,
        "support_edges": [list(edge) for edge in value.support_edges],
        "edge_records": [edge_record_to_dict(item) for item in value.edge_records],
        "interface_refs": [
            interface_ref_to_dict(item) for item in value.interface_refs
        ],
        "local_result": local_result_to_dict(value.local_result),
        "structure_tree": (
            None
            if value.structure_tree is None
            else structure_tree_to_dict(value.structure_tree)
        ),
    }


def torso_record_from_dict(value: dict[str, object]) -> TorsoRecord:
    data = _closed_dict(
        value,
        frozenset(
            {
                "record_id",
                "parent_record_id",
                "depth",
                "bag_vertices",
                "status",
                "support_edges",
                "edge_records",
                "interface_refs",
                "local_result",
                "structure_tree",
            }
        ),
        "TorsoRecord",
    )
    tree = data["structure_tree"]
    if tree is not None and type(tree) is not dict:
        raise _value_error(
            "TorsoRecord.structure_tree", "must be an object or null"
        )
    return TorsoRecord(
        data["record_id"],
        data["parent_record_id"],
        data["depth"],
        tuple(_json_list(data["bag_vertices"], "TorsoRecord.bag_vertices")),
        _enum_from_wire(LocalStatus, data["status"], "TorsoRecord.status"),
        _json_rows(data["support_edges"], "TorsoRecord.support_edges", 2),
        tuple(
            edge_record_from_dict(item)
            for item in _json_list(data["edge_records"], "TorsoRecord.edge_records")
        ),
        tuple(
            interface_ref_from_dict(item)
            for item in _json_list(
                data["interface_refs"], "TorsoRecord.interface_refs"
            )
        ),
        local_result_from_dict(data["local_result"]),
        None if tree is None else structure_tree_from_dict(tree),
    )


def hierarchy_edge_to_dict(value: HierarchyEdge) -> dict[str, object]:
    return {
        "parent_record_id": value.parent_record_id,
        "child_record_id": value.child_record_id,
        "local_tree_node_id": value.local_tree_node_id,
    }


def hierarchy_edge_from_dict(value: dict[str, object]) -> HierarchyEdge:
    data = _closed_dict(
        value,
        frozenset(
            {"parent_record_id", "child_record_id", "local_tree_node_id"}
        ),
        "HierarchyEdge",
    )
    return HierarchyEdge(
        data["parent_record_id"],
        data["child_record_id"],
        data["local_tree_node_id"],
    )


def hierarchy_result_to_dict(result: HierarchyResult) -> dict[str, object]:
    if type(result) is not HierarchyResult:
        raise _value_error("result", "must be HierarchyResult")
    return {
        "root_record_id": result.root_record_id,
        "config_digest": result.config_digest,
        "records": [torso_record_to_dict(item) for item in result.records],
        "interfaces": [
            interface_object_to_dict(item) for item in result.interfaces
        ],
        "hierarchy_edges": [
            hierarchy_edge_to_dict(item) for item in result.hierarchy_edges
        ],
        "terminal_record_ids": list(result.terminal_record_ids),
        "completion_level": result.completion_level.value,
    }


def hierarchy_result_from_dict(value: dict[str, object]) -> HierarchyResult:
    data = _closed_dict(
        value,
        frozenset(
            {
                "root_record_id",
                "config_digest",
                "records",
                "interfaces",
                "hierarchy_edges",
                "terminal_record_ids",
                "completion_level",
            }
        ),
        "HierarchyResult",
    )
    return HierarchyResult(
        data["root_record_id"],
        data["config_digest"],
        tuple(
            torso_record_from_dict(item)
            for item in _json_list(data["records"], "HierarchyResult.records")
        ),
        tuple(
            interface_object_from_dict(item)
            for item in _json_list(
                data["interfaces"], "HierarchyResult.interfaces"
            )
        ),
        tuple(
            hierarchy_edge_from_dict(item)
            for item in _json_list(
                data["hierarchy_edges"], "HierarchyResult.hierarchy_edges"
            )
        ),
        tuple(
            _json_list(
                data["terminal_record_ids"],
                "HierarchyResult.terminal_record_ids",
            )
        ),
        _enum_from_wire(
            CompletionLevel,
            data["completion_level"],
            "HierarchyResult.completion_level",
        ),
    )


def verification_report_to_dict(
    report: VerificationReport,
) -> dict[str, object]:
    if type(report) is not VerificationReport:
        raise _value_error("report", "must be VerificationReport")
    return {
        "schema_version": report.schema_version,
        "verifier_version": report.verifier_version,
        "case_id": report.case_id,
        "case_digest": report.case_digest,
        "candidate_hierarchy_digest": report.candidate_hierarchy_digest,
        "config_digest": report.config_digest,
        "verified": report.verified,
        "completion_level": report.completion_level.value,
        "checks": [list(row) for row in report.checks],
        "issues": [list(row) for row in report.issues],
    }


def verification_report_from_dict(
    value: dict[str, object],
) -> VerificationReport:
    data = _closed_dict(
        value,
        frozenset(
            {
                "schema_version",
                "verifier_version",
                "case_id",
                "case_digest",
                "candidate_hierarchy_digest",
                "config_digest",
                "verified",
                "completion_level",
                "checks",
                "issues",
            }
        ),
        "VerificationReport",
    )
    return VerificationReport(
        data["schema_version"],
        data["verifier_version"],
        data["case_id"],
        data["case_digest"],
        data["candidate_hierarchy_digest"],
        data["config_digest"],
        data["verified"],
        _enum_from_wire(
            CompletionLevel,
            data["completion_level"],
            "VerificationReport.completion_level",
        ),
        _json_rows(data["checks"], "VerificationReport.checks", 2),
        _json_rows(data["issues"], "VerificationReport.issues", 3),
    )


_WIRE_ENCODERS: dict[type, Callable[[object], dict[str, object]]] = {
    DemoConfig: demo_config_to_dict,
    GraphCase: graph_case_to_dict,
    SupportGraph: support_graph_to_dict,
    Separation: separation_to_dict,
    OrientedSeparation: oriented_separation_to_dict,
    CutRecord: cut_record_to_dict,
    LocalResult: local_result_to_dict,
    StructureTreeNode: structure_tree_node_to_dict,
    StructureTreeEdge: structure_tree_edge_to_dict,
    StructureTree: structure_tree_to_dict,
    EdgeRecord: edge_record_to_dict,
    InterfaceIncidence: interface_incidence_to_dict,
    InterfaceObject: interface_object_to_dict,
    InterfaceRef: interface_ref_to_dict,
    TorsoRecord: torso_record_to_dict,
    HierarchyEdge: hierarchy_edge_to_dict,
    HierarchyResult: hierarchy_result_to_dict,
    VerificationReport: verification_report_to_dict,
}


def _json_ready(value: object, field: str = "value") -> object:
    encoder = _WIRE_ENCODERS.get(type(value))
    if encoder is not None:
        return _json_ready(encoder(value), field)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _value_error(field, "non-finite floats are not canonical JSON")
        return value
    if isinstance(value, StrEnum):
        return value.value
    if type(value) in (tuple, list):
        return [
            _json_ready(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is frozenset:
        return [_json_ready(item, field) for item in sorted(value)]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise _value_error(field, "JSON object keys must be strings")
            result[key] = _json_ready(item, f"{field}.{key}")
        return result
    raise _value_error(field, f"unsupported canonical JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("value is not valid UTF-8 encodable Unicode") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_id(namespace: str, value: object) -> str:
    safe_namespace = _require_safe_id(namespace, "namespace")
    return f"{safe_namespace}-{canonical_sha256(value)}"


def _project_path(path: Path, *, for_write: bool) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("path must remain inside the algorithm directory") from exc
    current = _PROJECT_ROOT
    components = relative.parts[:-1] if for_write else relative.parts
    for component in components:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("symlink path components are forbidden")
    return absolute


def _write_bytes_once(path: Path, payload: bytes) -> None:
    target = _project_path(path, for_write=True)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {target.parent}")
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(target)

    runs_root = _PROJECT_ROOT / "runs"
    try:
        runs_root.mkdir()
    except FileExistsError:
        pass
    runs_root = _project_path(runs_root, for_write=False)
    if not runs_root.is_dir():
        raise ValueError("runs path must be a real directory")

    scratch = runs_root / ".tmp"
    try:
        scratch.mkdir()
    except FileExistsError:
        pass
    scratch = _project_path(scratch, for_write=False)
    if not scratch.is_dir():
        raise ValueError("publish scratch path must be a real directory")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        while descriptor is None:
            candidate = scratch / (
                f".publish-{os.getpid()}-{next(_PUBLISH_COUNTER)}.tmp"
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
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_json_once(path: Path, value: object) -> None:
    _write_bytes_once(path, canonical_json_bytes(value))


def _decode_json(raw: bytes, source: str) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: malformed UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source}: duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{source}: non-standard JSON constant {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: malformed JSON") from exc


def _read_bytes(path: Path) -> bytes:
    source = _project_path(path, for_write=False)
    try:
        return source.read_bytes()
    except UnicodeError as exc:
        raise ValueError(f"{source}: malformed UTF-8") from exc


def read_graph_cases(path: Path) -> tuple[GraphCase, ...]:
    raw = _read_bytes(path)
    if not raw:
        return ()
    rows = raw.split(b"\n")
    if rows[-1] == b"":
        rows.pop()
    if any(row == b"" for row in rows):
        raise ValueError("blank JSONL row")
    cases: list[GraphCase] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        value = _decode_json(row, f"{path}: row {index}")
        case = graph_case_from_dict(value)
        if canonical_json_bytes(graph_case_to_dict(case)) != row:
            raise ValueError(f"{path}: row {index} is noncanonical JSON")
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    result = tuple(cases)
    canonical = b"".join(
        canonical_json_bytes(graph_case_to_dict(case)) + b"\n"
        for case in result
    )
    if raw != canonical:
        raise ValueError(f"{path}: noncanonical JSONL")
    return result


def write_graph_cases_once(path: Path, cases: tuple[GraphCase, ...]) -> None:
    values = _instances(cases, GraphCase, "cases")
    _unique(values, lambda value: value.case_id, "cases")
    payload = b"".join(
        canonical_json_bytes(graph_case_to_dict(case)) + b"\n" for case in values
    )
    _write_bytes_once(path, payload)


def read_demo_config(path: Path) -> DemoConfig:
    return demo_config_from_dict(_decode_json(_read_bytes(path), str(path)))


def write_demo_config_once(path: Path, config: DemoConfig) -> None:
    write_json_once(path, demo_config_to_dict(config))


def read_hierarchy_result(path: Path) -> HierarchyResult:
    raw = _read_bytes(path)
    result = hierarchy_result_from_dict(_decode_json(raw, str(path)))
    if raw != canonical_json_bytes(hierarchy_result_to_dict(result)):
        raise ValueError(f"{path}: noncanonical hierarchy JSON")
    return result


def write_hierarchy_result_once(path: Path, result: HierarchyResult) -> None:
    write_json_once(path, hierarchy_result_to_dict(result))


def read_verification_report(path: Path) -> VerificationReport:
    raw = _read_bytes(path)
    report = verification_report_from_dict(_decode_json(raw, str(path)))
    if raw != canonical_json_bytes(verification_report_to_dict(report)):
        raise ValueError(f"{path}: noncanonical verification JSON")
    return report


def write_verification_report_once(
    path: Path, report: VerificationReport
) -> None:
    write_json_once(path, verification_report_to_dict(report))
