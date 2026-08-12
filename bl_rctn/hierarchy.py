from __future__ import annotations

from itertools import combinations

from .graph import (
    support_graph_from_case,
    validate_root_case,
    validate_support_graph,
)
from .local import analyze_local
from .models import (
    CompletionLevel,
    Coverage,
    DemoConfig,
    Edge,
    EdgeRecord,
    GraphCase,
    HierarchyEdge,
    HierarchyResult,
    InterfaceIncidence,
    InterfaceObject,
    InterfaceRef,
    LocalStatus,
    StructureTree,
    StructureTreeNode,
    SupportGraph,
    TorsoRecord,
    VertexSet,
    canonical_sha256,
    demo_config_to_dict,
    stable_id,
)
from .structure_tree import build_structure_tree, verify_structure_tree


class HierarchyInvariantError(ValueError):
    """Raised before publication when recursive hierarchy invariants fail."""


def _normalized_vertex_set(value: VertexSet, field: str) -> VertexSet:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    checked: list[int] = []
    for index, vertex in enumerate(value):
        if type(vertex) is not int or vertex < 0:
            raise ValueError(
                f"{field}[{index}] must be a nonnegative integer"
            )
        checked.append(vertex)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field} contains duplicate vertices")
    return tuple(sorted(checked))


def propagate_interface_ref(
    ref: InterfaceRef,
    global_boundary: VertexSet,
    child_vertices: VertexSet,
) -> InterfaceRef | None:
    if type(ref) is not InterfaceRef:
        raise TypeError("ref must be InterfaceRef")
    boundary = _normalized_vertex_set(global_boundary, "global_boundary")
    child = frozenset(
        _normalized_vertex_set(child_vertices, "child_vertices")
    )
    if not set(ref.local_boundary).issubset(boundary):
        raise HierarchyInvariantError(
            "interface fragment lies outside its global boundary"
        )
    if ref.coverage is Coverage.FULL and ref.local_boundary != boundary:
        raise HierarchyInvariantError(
            "FULL interface reference does not cover its global boundary"
        )

    local_boundary = tuple(
        vertex for vertex in ref.local_boundary if vertex in child
    )
    if not local_boundary:
        return None
    coverage = (
        Coverage.FULL
        if ref.coverage is Coverage.FULL and local_boundary == boundary
        else Coverage.PARTIAL
    )
    return InterfaceRef(
        ref.interface_id,
        ref.incidence_id,
        coverage,
        local_boundary,
    )


def _require_recursive_progress(
    *,
    parent_vertices: VertexSet,
    child_vertices: VertexSet,
    child_depth: int,
    max_depth: int,
    semantic_state: str,
    path_states: frozenset[str],
) -> None:
    parent = frozenset(
        _normalized_vertex_set(parent_vertices, "parent_vertices")
    )
    child = frozenset(
        _normalized_vertex_set(child_vertices, "child_vertices")
    )
    if not child or not child < parent:
        raise HierarchyInvariantError(
            "recursive structure-tree bag is not a proper child"
        )
    if type(child_depth) is not int or child_depth < 0:
        raise ValueError("child_depth must be a nonnegative integer")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if child_depth > max_depth:
        raise HierarchyInvariantError(
            f"recursive depth {child_depth} exceeds limit {max_depth}"
        )
    if type(semantic_state) is not str or not semantic_state:
        raise ValueError("semantic_state must be a nonempty string")
    if type(path_states) is not frozenset or any(
        type(value) is not str or not value for value in path_states
    ):
        raise ValueError("path_states must be a frozenset of nonempty strings")
    if semantic_state in path_states:
        raise HierarchyInvariantError(
            "recursive path contains a repeated semantic state"
        )


def _root_edge_id(case: GraphCase, endpoints: Edge) -> str:
    return stable_id(
        "root-edge",
        {"case_id": case.case_id, "endpoints": endpoints},
    )


def _root_edge_records(case: GraphCase) -> tuple[EdgeRecord, ...]:
    return tuple(
        EdgeRecord(edge, True, _root_edge_id(case, edge), ())
        for edge in case.edges
    )


def _root_record_id(case: GraphCase, graph: SupportGraph) -> str:
    return stable_id(
        "torso-record",
        {
            "case_id": case.case_id,
            "path": "root",
            "vertices": graph.vertices,
            "edges": graph.edges,
        },
    )


def _child_record_id(
    case: GraphCase,
    parent_record_id: str,
    local_tree_node_id: str,
) -> str:
    return stable_id(
        "torso-record",
        {
            "case_id": case.case_id,
            "parent_record_id": parent_record_id,
            "local_tree_node_id": local_tree_node_id,
        },
    )


def _interface_id(record_id: str, tree_edge_id: str) -> str:
    return stable_id(
        "interface",
        {
            "creator_record_id": record_id,
            "creator_tree_edge_id": tree_edge_id,
        },
    )


def _incidence_id(interface_id: str, side_tree_node_id: str) -> str:
    return stable_id(
        "incidence",
        {
            "interface_id": interface_id,
            "side_tree_node_id": side_tree_node_id,
        },
    )


def _make_local_interfaces(
    creator_record_id: str,
    tree: StructureTree,
) -> tuple[InterfaceObject, ...]:
    result: list[InterfaceObject] = []
    for tree_edge in tree.edges:
        interface_id = _interface_id(
            creator_record_id, tree_edge.tree_edge_id
        )
        source_id = _incidence_id(
            interface_id, tree_edge.source_node_id
        )
        target_id = _incidence_id(
            interface_id, tree_edge.target_node_id
        )
        result.append(
            InterfaceObject(
                interface_id,
                creator_record_id,
                tree_edge.tree_edge_id,
                tree_edge.adhesion,
                (
                    InterfaceIncidence(
                        source_id,
                        tree_edge.source_node_id,
                        target_id,
                    ),
                    InterfaceIncidence(
                        target_id,
                        tree_edge.target_node_id,
                        source_id,
                    ),
                ),
            )
        )
    return tuple(sorted(result, key=lambda value: value.interface_id))


def _incident_interfaces(
    interfaces: tuple[InterfaceObject, ...],
    tree_node_id: str,
) -> tuple[tuple[InterfaceObject, InterfaceIncidence], ...]:
    result: list[tuple[InterfaceObject, InterfaceIncidence]] = []
    for interface in interfaces:
        matches = tuple(
            incidence
            for incidence in interface.incidences
            if incidence.side_tree_node_id == tree_node_id
        )
        if len(matches) > 1:
            raise HierarchyInvariantError(
                "interface has duplicate incidences on one tree node"
            )
        if matches:
            result.append((interface, matches[0]))
    return tuple(sorted(result, key=lambda value: value[0].interface_id))


def _edge_record_map(
    edge_records: tuple[EdgeRecord, ...],
) -> dict[Edge, list[object]]:
    roles: dict[Edge, list[object]] = {}
    for edge_record in edge_records:
        if edge_record.endpoints in roles:
            raise HierarchyInvariantError(
                "edge provenance contains duplicate endpoint records"
            )
        roles[edge_record.endpoints] = [
            edge_record.is_root_real,
            edge_record.root_edge_id,
            set(edge_record.virtual_interface_ids),
        ]
    return roles


def _edge_records_from_map(
    roles: dict[Edge, list[object]],
) -> tuple[EdgeRecord, ...]:
    result: list[EdgeRecord] = []
    for endpoints in sorted(roles):
        is_root_real, root_edge_id, virtual_ids = roles[endpoints]
        if type(is_root_real) is not bool:
            raise HierarchyInvariantError("invalid root-real edge role")
        if root_edge_id is not None and type(root_edge_id) is not str:
            raise HierarchyInvariantError("invalid root edge ID")
        if type(virtual_ids) is not set:
            raise HierarchyInvariantError("invalid virtual edge role set")
        result.append(
            EdgeRecord(
                endpoints,
                is_root_real,
                root_edge_id,
                tuple(sorted(virtual_ids)),
            )
        )
    return tuple(result)


def _expected_child_edge_records(
    parent: TorsoRecord,
    child_vertices: VertexSet,
    incident: tuple[tuple[InterfaceObject, InterfaceIncidence], ...],
) -> tuple[EdgeRecord, ...]:
    child_set = frozenset(child_vertices)
    roles = {
        endpoints: values
        for endpoints, values in _edge_record_map(parent.edge_records).items()
        if set(endpoints).issubset(child_set)
    }
    for interface, _ in incident:
        if not set(interface.boundary).issubset(child_set):
            raise HierarchyInvariantError(
                "incident interface boundary lies outside its child torso"
            )
        for endpoints in combinations(interface.boundary, 2):
            edge = (min(endpoints), max(endpoints))
            if edge not in roles:
                roles[edge] = [False, None, set()]
            virtual_ids = roles[edge][2]
            if type(virtual_ids) is not set:
                raise HierarchyInvariantError("invalid virtual edge role set")
            virtual_ids.add(interface.interface_id)
    return _edge_records_from_map(roles)


def _merge_ref(
    refs: dict[tuple[str, str], InterfaceRef], ref: InterfaceRef
) -> None:
    key = (ref.interface_id, ref.incidence_id)
    existing = refs.get(key)
    if existing is not None and existing != ref:
        raise HierarchyInvariantError(
            "one interface incidence has conflicting propagated fragments"
        )
    refs[key] = ref


def _expected_child_refs(
    parent: TorsoRecord,
    child_vertices: VertexSet,
    incident: tuple[tuple[InterfaceObject, InterfaceIncidence], ...],
    interfaces_by_id: dict[str, InterfaceObject],
) -> tuple[InterfaceRef, ...]:
    refs: dict[tuple[str, str], InterfaceRef] = {}
    for ref in parent.interface_refs:
        interface = interfaces_by_id.get(ref.interface_id)
        if interface is None:
            raise HierarchyInvariantError(
                "parent contains an unresolved interface reference"
            )
        propagated = propagate_interface_ref(
            ref, interface.boundary, child_vertices
        )
        if propagated is not None:
            _merge_ref(refs, propagated)
    for interface, incidence in incident:
        _merge_ref(
            refs,
            InterfaceRef(
                interface.interface_id,
                incidence.incidence_id,
                Coverage.FULL,
                interface.boundary,
            ),
        )
    return tuple(
        refs[key]
        for key in sorted(refs)
    )


def _semantic_state(
    graph: SupportGraph,
    edge_records: tuple[EdgeRecord, ...],
    interface_refs: tuple[InterfaceRef, ...],
) -> str:
    return canonical_sha256(
        {
            "vertices": graph.vertices,
            "edges": graph.edges,
            "edge_roles": tuple(
                {
                    "endpoints": value.endpoints,
                    "is_root_real": value.is_root_real,
                    "virtual_role_count": len(value.virtual_interface_ids),
                }
                for value in edge_records
            ),
            "interface_fragments": tuple(
                {
                    "coverage": value.coverage.value,
                    "local_boundary": value.local_boundary,
                }
                for value in interface_refs
            ),
        }
    )


def _validate_record_edge_provenance(record: TorsoRecord) -> None:
    endpoints = tuple(value.endpoints for value in record.edge_records)
    if endpoints != record.support_edges:
        raise HierarchyInvariantError(
            f"record {record.record_id} support edges disagree with provenance"
        )


class _HierarchyBuilder:
    def __init__(self, case: GraphCase, config: DemoConfig) -> None:
        self.case = case
        self.config = config
        self.config_digest = canonical_sha256(demo_config_to_dict(config))
        self.records: list[TorsoRecord] = []
        self.interfaces: list[InterfaceObject] = []
        self.interfaces_by_id: dict[str, InterfaceObject] = {}
        self.hierarchy_edges: list[HierarchyEdge] = []
        self.terminal_ids: list[str] = []

    def build(self) -> HierarchyResult:
        validate_root_case(self.case)
        if self.case.k not in self.config.ks:
            raise HierarchyInvariantError(
                "case k is not enabled by the frozen configuration"
            )
        root_graph = support_graph_from_case(self.case)
        root_edge_records = _root_edge_records(self.case)
        root_record_id = _root_record_id(self.case, root_graph)
        root_state = _semantic_state(root_graph, root_edge_records, ())
        self._visit(
            record_id=root_record_id,
            parent_record_id=None,
            depth=0,
            graph=root_graph,
            edge_records=root_edge_records,
            interface_refs=(),
            path_states=frozenset({root_state}),
        )
        result = HierarchyResult(
            root_record_id,
            self.config_digest,
            tuple(self.records),
            tuple(self.interfaces),
            tuple(self.hierarchy_edges),
            tuple(self.terminal_ids),
            CompletionLevel.RECURSIVE_CANDIDATE,
        )
        _validate_candidate(result, root_graph, self.config)
        return result

    def _visit(
        self,
        *,
        record_id: str,
        parent_record_id: str | None,
        depth: int,
        graph: SupportGraph,
        edge_records: tuple[EdgeRecord, ...],
        interface_refs: tuple[InterfaceRef, ...],
        path_states: frozenset[str],
    ) -> None:
        try:
            validate_support_graph(graph)
            if tuple(value.endpoints for value in edge_records) != graph.edges:
                raise HierarchyInvariantError(
                    "support graph disagrees with edge provenance"
                )
            local_result = analyze_local(
                graph,
                self.case.k,
                self.config.max_full_separations,
            )
        except (TypeError, ValueError) as exc:
            if type(exc) is HierarchyInvariantError:
                raise
            raise HierarchyInvariantError(
                f"recursive local analysis failed at {record_id}: {exc}"
            ) from exc

        if local_result.completion_level is not CompletionLevel.LOCAL_EXACT:
            raise HierarchyInvariantError(
                "recursive local result is not LOCAL_EXACT"
            )

        tree: StructureTree | None = None
        if local_result.status is LocalStatus.SPLIT:
            try:
                tree = build_structure_tree(
                    graph, local_result.tn_aggregated
                )
                issues = verify_structure_tree(
                    graph, local_result.tn_aggregated, tree
                )
            except (TypeError, ValueError) as exc:
                raise HierarchyInvariantError(
                    f"structure-tree construction failed at {record_id}: {exc}"
                ) from exc
            if issues:
                raise HierarchyInvariantError(
                    "structure-tree invariant failed at "
                    f"{record_id}: {'; '.join(issues)}"
                )

        record = TorsoRecord(
            record_id,
            parent_record_id,
            depth,
            graph.vertices,
            local_result.status,
            graph.edges,
            edge_records,
            interface_refs,
            local_result,
            tree,
        )
        _validate_record_edge_provenance(record)
        self.records.append(record)

        if local_result.status is not LocalStatus.SPLIT:
            self.terminal_ids.append(record_id)
            return
        if tree is None:
            raise HierarchyInvariantError(
                "SPLIT record lacks a structure tree"
            )

        local_interfaces = _make_local_interfaces(record_id, tree)
        for interface in local_interfaces:
            if interface.interface_id in self.interfaces_by_id:
                raise HierarchyInvariantError("duplicate interface ID")
            self.interfaces.append(interface)
            self.interfaces_by_id[interface.interface_id] = interface

        for node in tree.nodes:
            incident = _incident_interfaces(local_interfaces, node.node_id)
            child_edge_records = _expected_child_edge_records(
                record, node.bag_vertices, incident
            )
            child_support = SupportGraph(
                node.bag_vertices,
                tuple(value.endpoints for value in child_edge_records),
            )
            child_refs = _expected_child_refs(
                record,
                node.bag_vertices,
                incident,
                self.interfaces_by_id,
            )
            child_depth = depth + 1
            child_state = _semantic_state(
                child_support, child_edge_records, child_refs
            )
            _require_recursive_progress(
                parent_vertices=graph.vertices,
                child_vertices=node.bag_vertices,
                child_depth=child_depth,
                max_depth=self.config.max_depth,
                semantic_state=child_state,
                path_states=path_states,
            )
            child_record_id = _child_record_id(
                self.case, record_id, node.node_id
            )
            self.hierarchy_edges.append(
                HierarchyEdge(record_id, child_record_id, node.node_id)
            )
            self._visit(
                record_id=child_record_id,
                parent_record_id=record_id,
                depth=child_depth,
                graph=child_support,
                edge_records=child_edge_records,
                interface_refs=child_refs,
                path_states=path_states | frozenset({child_state}),
            )


def _validate_interface_reference(
    record: TorsoRecord,
    ref: InterfaceRef,
    interfaces_by_id: dict[str, InterfaceObject],
) -> None:
    interface = interfaces_by_id.get(ref.interface_id)
    if interface is None:
        raise HierarchyInvariantError("unresolved interface reference")
    incidence_ids = {
        incidence.incidence_id for incidence in interface.incidences
    }
    if ref.incidence_id not in incidence_ids:
        raise HierarchyInvariantError("unresolved interface incidence reference")
    if not set(ref.local_boundary).issubset(interface.boundary):
        raise HierarchyInvariantError(
            "interface reference fragment lies outside global boundary"
        )
    if not set(ref.local_boundary).issubset(record.bag_vertices):
        raise HierarchyInvariantError(
            "interface reference fragment lies outside record bag"
        )
    if ref.coverage is Coverage.FULL and ref.local_boundary != interface.boundary:
        raise HierarchyInvariantError(
            "FULL interface reference does not cover the global boundary"
        )


def _validate_candidate(
    hierarchy: HierarchyResult,
    root_graph: SupportGraph,
    config: DemoConfig,
) -> None:
    records = {record.record_id: record for record in hierarchy.records}
    if hierarchy.root_record_id not in records:
        raise HierarchyInvariantError("root record ID is unresolved")
    root = records[hierarchy.root_record_id]
    if (
        root.parent_record_id is not None
        or root.depth != 0
        or root.bag_vertices != root_graph.vertices
        or root.support_edges != root_graph.edges
    ):
        raise HierarchyInvariantError("root torso does not match the root graph")
    if hierarchy.config_digest != canonical_sha256(
        demo_config_to_dict(config)
    ):
        raise HierarchyInvariantError("candidate config digest mismatch")
    if hierarchy.completion_level is not CompletionLevel.RECURSIVE_CANDIDATE:
        raise HierarchyInvariantError("engine result is not a recursive candidate")

    interfaces_by_id = {
        interface.interface_id: interface
        for interface in hierarchy.interfaces
    }
    incoming: dict[str, HierarchyEdge] = {}
    children: dict[str, list[HierarchyEdge]] = {
        record_id: [] for record_id in records
    }
    for hierarchy_edge in hierarchy.hierarchy_edges:
        if hierarchy_edge.parent_record_id not in records:
            raise HierarchyInvariantError("hierarchy edge has an unknown parent")
        if hierarchy_edge.child_record_id not in records:
            raise HierarchyInvariantError("hierarchy edge has an unknown child")
        if hierarchy_edge.child_record_id in incoming:
            raise HierarchyInvariantError("hierarchy child has multiple parents")
        incoming[hierarchy_edge.child_record_id] = hierarchy_edge
        children[hierarchy_edge.parent_record_id].append(hierarchy_edge)

    if set(incoming) != set(records).difference({hierarchy.root_record_id}):
        raise HierarchyInvariantError("non-root records lack a unique parent edge")

    root_real_by_id: dict[str, Edge] = {}
    root_real_by_endpoints: dict[Edge, str] = {}
    for record in hierarchy.records:
        _validate_record_edge_provenance(record)
        if record.status is not record.local_result.status:
            raise HierarchyInvariantError("record status disagrees with local result")
        if record.depth > config.max_depth:
            raise HierarchyInvariantError("record depth exceeds frozen config")
        if record.status is LocalStatus.SPLIT and record.structure_tree is None:
            raise HierarchyInvariantError("SPLIT record lacks structure tree")
        if record.status is not LocalStatus.SPLIT and record.structure_tree is not None:
            raise HierarchyInvariantError("terminal record unexpectedly has a tree")
        for edge_record in record.edge_records:
            for interface_id in edge_record.virtual_interface_ids:
                if interface_id not in interfaces_by_id:
                    raise HierarchyInvariantError(
                        "edge role has an unresolved virtual interface ID"
                    )
            if edge_record.is_root_real:
                root_edge_id = edge_record.root_edge_id
                if root_edge_id is None:
                    raise HierarchyInvariantError("root-real edge lacks an ID")
                old_endpoints = root_real_by_id.setdefault(
                    root_edge_id, edge_record.endpoints
                )
                if old_endpoints != edge_record.endpoints:
                    raise HierarchyInvariantError(
                        "one root edge ID has inconsistent endpoints"
                    )
                old_id = root_real_by_endpoints.setdefault(
                    edge_record.endpoints, root_edge_id
                )
                if old_id != root_edge_id:
                    raise HierarchyInvariantError(
                        "one root edge has inconsistent IDs"
                    )
        for ref in record.interface_refs:
            _validate_interface_reference(record, ref, interfaces_by_id)

    root_roles = {
        value.endpoints: value for value in root.edge_records
    }
    if set(root_roles) != set(root_graph.edges) or any(
        not value.is_root_real or value.virtual_interface_ids
        for value in root_roles.values()
    ):
        raise HierarchyInvariantError("root edge provenance is not exact")

    for interface in hierarchy.interfaces:
        creator = records.get(interface.creator_record_id)
        if creator is None or creator.structure_tree is None:
            raise HierarchyInvariantError("interface creator record is unresolved")
        tree_edges = {
            value.tree_edge_id: value for value in creator.structure_tree.edges
        }
        creator_edge = tree_edges.get(interface.creator_tree_edge_id)
        if creator_edge is None:
            raise HierarchyInvariantError("interface creator tree edge is unresolved")
        if interface.boundary != creator_edge.adhesion:
            raise HierarchyInvariantError("interface boundary disagrees with adhesion")
        if {
            value.side_tree_node_id for value in interface.incidences
        } != {creator_edge.source_node_id, creator_edge.target_node_id}:
            raise HierarchyInvariantError("interface incidences have wrong tree sides")

    terminal_ids = frozenset(hierarchy.terminal_record_ids)
    expected_terminals = frozenset(
        record_id for record_id, values in children.items() if not values
    )
    if terminal_ids != expected_terminals:
        raise HierarchyInvariantError("terminal record IDs do not equal the leaves")

    for record_id, record in records.items():
        outgoing = children[record_id]
        if record.status is LocalStatus.SPLIT:
            if record.structure_tree is None:
                raise HierarchyInvariantError("SPLIT record lacks a tree")
            tree_node_ids = {
                value.node_id for value in record.structure_tree.nodes
            }
            if {value.local_tree_node_id for value in outgoing} != tree_node_ids:
                raise HierarchyInvariantError(
                    "SPLIT record children do not match its tree nodes"
                )
        elif outgoing:
            raise HierarchyInvariantError("only SPLIT records may have children")

        local_interfaces = tuple(
            value
            for value in hierarchy.interfaces
            if value.creator_record_id == record_id
        )
        for edge in outgoing:
            child = records[edge.child_record_id]
            node_by_id = {
                value.node_id: value
                for value in record.structure_tree.nodes
            }
            node: StructureTreeNode = node_by_id[edge.local_tree_node_id]
            if (
                child.parent_record_id != record_id
                or child.depth != record.depth + 1
                or child.bag_vertices != node.bag_vertices
            ):
                raise HierarchyInvariantError(
                    "child record disagrees with its hierarchy edge"
                )
            if not set(child.bag_vertices) < set(record.bag_vertices):
                raise HierarchyInvariantError("child record is not strict progress")
            incident = _incident_interfaces(
                local_interfaces, node.node_id
            )
            expected_edges = _expected_child_edge_records(
                record, node.bag_vertices, incident
            )
            if child.edge_records != expected_edges:
                raise HierarchyInvariantError(
                    "child edge provenance differs from induced torso completion"
                )
            expected_refs = _expected_child_refs(
                record,
                node.bag_vertices,
                incident,
                interfaces_by_id,
            )
            if child.interface_refs != expected_refs:
                raise HierarchyInvariantError(
                    "child interface references differ from exact propagation"
                )

    for terminal_id in terminal_ids:
        if records[terminal_id].status is LocalStatus.SPLIT:
            raise HierarchyInvariantError("SPLIT record is labelled terminal")


def decompose_case(case: GraphCase, config: DemoConfig) -> HierarchyResult:
    if type(case) is not GraphCase:
        raise TypeError("case must be GraphCase")
    if type(config) is not DemoConfig:
        raise TypeError("config must be DemoConfig")
    return _HierarchyBuilder(case, config).build()


def reconstruct_root(
    hierarchy: HierarchyResult,
) -> tuple[VertexSet, tuple[Edge, ...]]:
    if type(hierarchy) is not HierarchyResult:
        raise TypeError("hierarchy must be HierarchyResult")
    records = {record.record_id: record for record in hierarchy.records}
    interfaces_by_id = {
        interface.interface_id: interface
        for interface in hierarchy.interfaces
    }

    for record in hierarchy.records:
        for ref in record.interface_refs:
            _validate_interface_reference(record, ref, interfaces_by_id)

    terminal_records: list[TorsoRecord] = []
    for terminal_id in hierarchy.terminal_record_ids:
        record = records.get(terminal_id)
        if record is None:
            raise HierarchyInvariantError("terminal record ID is unresolved")
        terminal_records.append(record)

    vertices: set[int] = set()
    root_edges_by_id: dict[str, Edge] = {}
    root_ids_by_edge: dict[Edge, str] = {}
    for record in terminal_records:
        vertices.update(record.bag_vertices)
        for edge_record in record.edge_records:
            if not edge_record.is_root_real:
                continue
            root_edge_id = edge_record.root_edge_id
            if root_edge_id is None:
                raise HierarchyInvariantError("root-real edge lacks an ID")
            previous_edge = root_edges_by_id.setdefault(
                root_edge_id, edge_record.endpoints
            )
            if previous_edge != edge_record.endpoints:
                raise HierarchyInvariantError(
                    "one root edge ID has inconsistent endpoints"
                )
            previous_id = root_ids_by_edge.setdefault(
                edge_record.endpoints, root_edge_id
            )
            if previous_id != root_edge_id:
                raise HierarchyInvariantError(
                    "one root edge has inconsistent IDs"
                )

    return tuple(sorted(vertices)), tuple(sorted(root_ids_by_edge))
