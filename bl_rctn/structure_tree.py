from __future__ import annotations

from collections import Counter
from itertools import product

from .graph import validate_support_graph
from .models import (
    OrientedSeparation,
    Separation,
    StructureTree,
    StructureTreeEdge,
    StructureTreeNode,
    SupportGraph,
    stable_id,
)
from .separations import are_nested, separation_sort_key


def _separation_payload(value: Separation) -> dict[str, object]:
    return {
        "side_a": value.side_a,
        "side_b": value.side_b,
        "separator": value.separator,
    }


def _separation_id(value: Separation) -> str:
    return stable_id("separation", _separation_payload(value))


def _orientation(
    value: Separation, base_id: str, bit: int
) -> OrientedSeparation:
    if bit == 0:
        return OrientedSeparation(base_id, value.side_a, value.side_b)
    if bit == 1:
        return OrientedSeparation(base_id, value.side_b, value.side_a)
    raise ValueError("orientation bit must be 0 or 1")


def reverse_orientation(value: OrientedSeparation) -> OrientedSeparation:
    if type(value) is not OrientedSeparation:
        raise TypeError("value must be OrientedSeparation")
    return OrientedSeparation(
        value.base_separation_id, value.toward_side, value.away_side
    )


def oriented_leq(
    left: OrientedSeparation, right: OrientedSeparation
) -> bool:
    if type(left) is not OrientedSeparation or type(right) is not OrientedSeparation:
        raise TypeError("oriented order requires OrientedSeparation values")
    return (
        frozenset(left.away_side) <= frozenset(right.away_side)
        and frozenset(left.toward_side) >= frozenset(right.toward_side)
    )


def oriented_lt(
    left: OrientedSeparation, right: OrientedSeparation
) -> bool:
    return left != right and oriented_leq(left, right)


def _validate_tn_family(
    graph: SupportGraph, tn: tuple[Separation, ...]
) -> tuple[Separation, ...]:
    validate_support_graph(graph)
    if type(tn) is not tuple:
        raise TypeError("tn must be a tuple")
    for value in tn:
        if type(value) is not Separation:
            raise TypeError("tn must contain Separation values")
    if len(set(tn)) != len(tn):
        raise ValueError("TN family contains a duplicate separation")

    vertices = frozenset(graph.vertices)
    for value in tn:
        side_a = frozenset(value.side_a)
        side_b = frozenset(value.side_b)
        separator = frozenset(value.separator)
        if value.side_b < value.side_a:
            raise ValueError("TN separation sides are not canonical")
        if side_a | side_b != vertices:
            raise ValueError("TN separation does not cover the support graph")
        if side_a & side_b != separator:
            raise ValueError("TN separation has the wrong separator")
        wing_a = side_a.difference(separator)
        wing_b = side_b.difference(separator)
        if not wing_a or not wing_b:
            raise ValueError("TN separation is not proper")
        if any(
            (left in wing_a and right in wing_b)
            or (left in wing_b and right in wing_a)
            for left, right in graph.edges
        ):
            raise ValueError("TN separation has a cross-wing support edge")

    ordered = tuple(sorted(tn, key=separation_sort_key))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if not are_nested(left, right):
                raise ValueError("TN family must be pairwise nested")

    base_ids = tuple(_separation_id(value) for value in ordered)
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("TN family has a duplicate separation ID")
    oriented = tuple(
        (
            _orientation(value, base_id, 0),
            _orientation(value, base_id, 1),
        )
        for value, base_id in zip(ordered, base_ids, strict=True)
    )
    flattened = tuple(item for pair in oriented for item in pair)
    if len(set(flattened)) != len(flattened):
        raise ValueError("TN family has a coterminal duplicate orientation")
    for candidate in flattened:
        for pair in oriented:
            if pair[0].base_separation_id == candidate.base_separation_id:
                continue
            if oriented_lt(candidate, pair[0]) and oriented_lt(candidate, pair[1]):
                raise ValueError("TN family contains a trivial or co-trivial orientation")
    return ordered


def _oriented_assignment(
    tn: tuple[Separation, ...],
    base_ids: tuple[str, ...],
    bits: tuple[int, ...],
) -> tuple[OrientedSeparation, ...]:
    return tuple(
        _orientation(value, base_id, bit)
        for value, base_id, bit in zip(tn, base_ids, bits, strict=True)
    )


def _is_consistent(assignment: tuple[OrientedSeparation, ...]) -> bool:
    for index, left in enumerate(assignment):
        reversed_left = reverse_orientation(left)
        for other_index, right in enumerate(assignment):
            if index == other_index:
                continue
            if oriented_lt(reversed_left, right):
                return False
    return True


def _consistent_assignments(
    tn: tuple[Separation, ...], base_ids: tuple[str, ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        bits
        for bits in product((0, 1), repeat=len(tn))
        if _is_consistent(_oriented_assignment(tn, base_ids, bits))
    )


def _hamming_edges(
    assignments: tuple[tuple[int, ...], ...]
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    assignment_set = set(assignments)
    edges: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for bits in assignments:
        for index, bit in enumerate(bits):
            if bit != 0:
                continue
            flipped = bits[:index] + (1,) + bits[index + 1 :]
            if flipped in assignment_set:
                edges.append((bits, flipped, index))
    return tuple(edges)


def _validate_essential_assignments(
    assignments: tuple[tuple[int, ...], ...], separation_count: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    if len(assignments) != separation_count + 1:
        raise ValueError(
            "non-essential TN family has the wrong consistent-orientation count"
        )
    for index in range(separation_count):
        if {bits[index] for bits in assignments} != {0, 1}:
            raise ValueError("non-essential TN orientation is forced")

    seen_columns: set[tuple[int, ...]] = set()
    for index in range(separation_count):
        column = tuple(bits[index] for bits in assignments)
        complement = tuple(1 - bit for bit in column)
        canonical = min(column, complement)
        if canonical in seen_columns:
            raise ValueError("non-essential TN family has coterminal duplicates")
        seen_columns.add(canonical)

    edges = _hamming_edges(assignments)
    labels = Counter(index for _, _, index in edges)
    if len(edges) != separation_count or any(
        labels[index] != 1 for index in range(separation_count)
    ):
        raise ValueError("each essential TN separation must own exactly one edge")

    if assignments:
        neighbors = {bits: set() for bits in assignments}
        for left, right, _ in edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        reached: set[tuple[int, ...]] = set()
        stack = [assignments[0]]
        while stack:
            current = stack.pop()
            if current in reached:
                continue
            reached.add(current)
            stack.extend(neighbors[current].difference(reached))
        if len(reached) != len(assignments):
            raise ValueError("consistent-orientation graph is not connected")
    return edges


def _signature(
    base_ids: tuple[str, ...], bits: tuple[int, ...]
) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(zip(base_ids, bits, strict=True)))


def _bag_for_assignment(
    graph: SupportGraph,
    tn: tuple[Separation, ...],
    base_ids: tuple[str, ...],
    bits: tuple[int, ...],
) -> tuple[int, ...]:
    if not tn:
        return graph.vertices
    oriented = _oriented_assignment(tn, base_ids, bits)
    bag = set(graph.vertices)
    for value in oriented:
        bag.intersection_update(value.toward_side)
    if not bag:
        raise ValueError("consistent orientation has an empty bag")
    return tuple(sorted(bag))


def _node_id(
    graph: SupportGraph, signature: tuple[tuple[str, int], ...]
) -> str:
    return stable_id(
        "tree-node",
        {
            "vertices": graph.vertices,
            "edges": graph.edges,
            "orientation_signature": signature,
        },
    )


def _tree_edge_id(
    source: str, target: str, separation: Separation
) -> str:
    return stable_id(
        "tree-edge",
        {
            "endpoints": tuple(sorted((source, target))),
            "separation": _separation_payload(separation),
        },
    )


def build_structure_tree(
    graph: SupportGraph, tn: tuple[Separation, ...]
) -> StructureTree:
    ordered = _validate_tn_family(graph, tn)
    base_ids = tuple(_separation_id(value) for value in ordered)
    assignments = _consistent_assignments(ordered, base_ids)
    hamming_edges = _validate_essential_assignments(assignments, len(ordered))

    nodes_by_bits: dict[tuple[int, ...], StructureTreeNode] = {}
    for bits in assignments:
        signature = _signature(base_ids, bits)
        node = StructureTreeNode(
            _node_id(graph, signature),
            signature,
            _bag_for_assignment(graph, ordered, base_ids, bits),
        )
        nodes_by_bits[bits] = node

    edges: list[StructureTreeEdge] = []
    for left_bits, right_bits, index in hamming_edges:
        left_id = nodes_by_bits[left_bits].node_id
        right_id = nodes_by_bits[right_bits].node_id
        source, target = sorted((left_id, right_id))
        separation = ordered[index]
        edges.append(
            StructureTreeEdge(
                _tree_edge_id(source, target, separation),
                source,
                target,
                separation,
                separation.separator,
            )
        )

    tree = StructureTree(tuple(nodes_by_bits.values()), tuple(edges))
    issues = verify_structure_tree(graph, ordered, tree)
    if issues:
        raise ValueError("invalid structure tree: " + "; ".join(issues))
    return tree


def _tree_connected_nodes(
    start: str,
    allowed: set[str],
    neighbors: dict[str, set[str]],
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


def verify_structure_tree(
    graph: SupportGraph,
    tn: tuple[Separation, ...],
    tree: StructureTree,
) -> tuple[str, ...]:
    if type(tree) is not StructureTree:
        raise TypeError("tree must be StructureTree")
    try:
        ordered = _validate_tn_family(graph, tn)
        base_ids = tuple(_separation_id(value) for value in ordered)
        assignments = _consistent_assignments(ordered, base_ids)
        _validate_essential_assignments(assignments, len(ordered))
    except (TypeError, ValueError) as exc:
        return (f"input_family: {exc}",)

    issues: list[str] = []
    expected_assignment_set = set(assignments)
    expected_base_ids = set(base_ids)
    node_by_id = {node.node_id: node for node in tree.nodes}
    if len(tree.nodes) != len(ordered) + 1:
        issues.append("node_count: expected separation_count + 1")
    if len(tree.edges) != len(ordered):
        issues.append("edge_count: expected separation_count")

    bits_by_node: dict[str, tuple[int, ...]] = {}
    represented: list[tuple[int, ...]] = []
    for node in tree.nodes:
        signature = dict(node.orientation_signature)
        if set(signature) != expected_base_ids or any(
            value not in (0, 1) for value in signature.values()
        ):
            issues.append(f"orientation_signature: invalid node {node.node_id}")
            continue
        bits = tuple(signature[base_id] for base_id in base_ids)
        bits_by_node[node.node_id] = bits
        represented.append(bits)
        if bits not in expected_assignment_set:
            issues.append(f"orientation_consistency: invalid node {node.node_id}")
            continue
        expected_signature = _signature(base_ids, bits)
        if node.node_id != _node_id(graph, expected_signature):
            issues.append(f"node_id: invalid node {node.node_id}")
        try:
            expected_bag = _bag_for_assignment(graph, ordered, base_ids, bits)
        except ValueError as exc:
            issues.append(f"bag: {exc}")
        else:
            if node.bag_vertices != expected_bag:
                issues.append(f"bag: wrong oriented intersection at {node.node_id}")

    if len(represented) != len(set(represented)):
        issues.append("orientation_signature: duplicate consistent orientation")
    if set(represented) != expected_assignment_set:
        issues.append("orientation_signature: missing consistent orientation")

    neighbors = {node_id: set() for node_id in node_by_id}
    label_counts: Counter[Separation] = Counter()
    endpoint_pairs: set[frozenset[str]] = set()
    cycle = False
    parent = {node_id: node_id for node_id in node_by_id}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for edge in tree.edges:
        pair = frozenset((edge.source_node_id, edge.target_node_id))
        if pair in endpoint_pairs:
            issues.append("edge_endpoints: duplicate tree edge")
        endpoint_pairs.add(pair)
        neighbors[edge.source_node_id].add(edge.target_node_id)
        neighbors[edge.target_node_id].add(edge.source_node_id)

        source_root = find(edge.source_node_id)
        target_root = find(edge.target_node_id)
        if source_root == target_root:
            cycle = True
        else:
            parent[source_root] = target_root

        if edge.separation not in ordered:
            issues.append(f"edge_label: unknown label {edge.tree_edge_id}")
            continue
        label_counts[edge.separation] += 1
        source_bits = bits_by_node.get(edge.source_node_id)
        target_bits = bits_by_node.get(edge.target_node_id)
        if source_bits is not None and target_bits is not None:
            differing = tuple(
                index
                for index, (left, right) in enumerate(
                    zip(source_bits, target_bits, strict=True)
                )
                if left != right
            )
            if len(differing) != 1:
                issues.append(f"hamming_distance: invalid edge {edge.tree_edge_id}")
            elif edge.separation != ordered[differing[0]]:
                issues.append(f"edge_label: wrong Hamming label {edge.tree_edge_id}")

        expected_id = _tree_edge_id(
            edge.source_node_id, edge.target_node_id, edge.separation
        )
        if edge.tree_edge_id != expected_id:
            issues.append(f"tree_edge_id: invalid edge {edge.tree_edge_id}")
        endpoint_intersection = tuple(
            sorted(
                set(node_by_id[edge.source_node_id].bag_vertices).intersection(
                    node_by_id[edge.target_node_id].bag_vertices
                )
            )
        )
        if endpoint_intersection != edge.separation.separator:
            issues.append(f"bag_intersection: invalid edge {edge.tree_edge_id}")

    if any(label_counts[value] != 1 for value in ordered):
        issues.append("edge_label_bijection: each TN separation must label one edge")
    if cycle:
        issues.append("acyclic: tree contains a cycle")
    if node_by_id:
        reached = _tree_connected_nodes(
            next(iter(node_by_id)), set(node_by_id), neighbors
        )
        if reached != set(node_by_id):
            issues.append("connected: tree is disconnected")

    vertices = set(graph.vertices)
    covered_vertices = set().union(
        *(set(node.bag_vertices) for node in tree.nodes)
    ) if tree.nodes else set()
    if covered_vertices != vertices:
        issues.append("bag_vertex_cover: bags do not cover exactly the graph vertices")
    for edge in graph.edges:
        if not any(set(edge) <= set(node.bag_vertices) for node in tree.nodes):
            issues.append(f"bag_edge_cover: uncovered support edge {edge}")
    for vertex in graph.vertices:
        containing = {
            node.node_id for node in tree.nodes if vertex in node.bag_vertices
        }
        if not containing:
            issues.append(f"bag_running_intersection: vertex {vertex} is uncovered")
            continue
        reached = _tree_connected_nodes(next(iter(containing)), containing, neighbors)
        if reached != containing:
            issues.append(
                f"bag_running_intersection: vertex {vertex} bags are disconnected"
            )
    return tuple(sorted(set(issues)))
