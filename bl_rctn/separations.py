from __future__ import annotations

from itertools import combinations

from .graph import (
    adjacency,
    components_after_deleting,
    validate_support_graph,
)
from .models import (
    Adjacency,
    CutRecord,
    Separation,
    SupportGraph,
    VertexSet,
)


def _validated_vertices(vertices: VertexSet) -> tuple[int, ...]:
    if type(vertices) is not tuple:
        raise TypeError("vertices must be a tuple")
    if any(type(vertex) is not int or vertex < 0 for vertex in vertices):
        raise ValueError("vertices must be nonnegative integers")
    if len(set(vertices)) != len(vertices):
        raise ValueError("vertices must be unique")
    if tuple(sorted(vertices)) != vertices:
        raise ValueError("vertices must be sorted")
    return vertices


def _validate_cut_partition(vertices: frozenset[int], cut: CutRecord) -> None:
    if type(cut) is not CutRecord:
        raise TypeError("cuts must contain CutRecord values")
    occupied = set(cut.separator)
    if not occupied.issubset(vertices):
        raise ValueError("cut separator lies outside the graph")
    for component in cut.components:
        if occupied.intersection(component):
            raise ValueError("cut components overlap")
        occupied.update(component)
    if occupied != set(vertices):
        raise ValueError("cut components do not partition the deleted graph")


def _canonical_separation(
    vertices: frozenset[int],
    separator: VertexSet,
    left_wing: frozenset[int],
    right_wing: frozenset[int],
) -> Separation:
    separator_set = frozenset(separator)
    if not left_wing or not right_wing:
        raise ValueError("a proper separation requires two nonempty wings")
    if left_wing.intersection(right_wing):
        raise ValueError("separation wings must be disjoint")
    if separator_set.intersection(left_wing | right_wing):
        raise ValueError("separator and wings must be disjoint")
    if separator_set | left_wing | right_wing != vertices:
        raise ValueError("separator and wings must cover the graph")
    return Separation(
        tuple(sorted(separator_set | left_wing)),
        tuple(sorted(separator_set | right_wing)),
        tuple(sorted(separator_set)),
    )


def list_order_k_cutsets(adj: Adjacency, k: int) -> tuple[CutRecord, ...]:
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer at least 2")
    # This validates the complete adjacency even when k exceeds the graph order.
    components_after_deleting(adj, frozenset())
    vertices = tuple(sorted(adj))
    cuts: list[CutRecord] = []
    for separator in combinations(vertices, k):
        components = components_after_deleting(adj, frozenset(separator))
        if len(components) >= 2:
            cuts.append(CutRecord(separator, components))
    return tuple(cuts)


def enumerate_full_separations(
    graph: SupportGraph,
    cuts: tuple[CutRecord, ...],
    limit: int,
) -> tuple[Separation, ...]:
    validate_support_graph(graph)
    if type(cuts) is not tuple:
        raise TypeError("cuts must be a tuple")
    if type(limit) is not int or limit < 1:
        raise ValueError("full separation limit must be a positive integer")

    vertices = frozenset(graph.vertices)
    graph_adj = adjacency(graph)
    required = 0
    for cut in cuts:
        _validate_cut_partition(vertices, cut)
        actual = components_after_deleting(graph_adj, frozenset(cut.separator))
        if actual != cut.components:
            raise ValueError("cut components do not match the support graph")
        required += (1 << (len(cut.components) - 1)) - 1
        if required > limit:
            raise ValueError(
                f"full separation limit exceeded: need more than {limit}"
            )

    separations: list[Separation] = []
    for cut in cuts:
        first = frozenset(cut.components[0])
        tail = cut.components[1:]
        # The all-one mask is omitted so the opposite wing remains nonempty.
        for mask in range((1 << len(tail)) - 1):
            left = set(first)
            right: set[int] = set()
            for index, component in enumerate(tail):
                target = left if mask & (1 << index) else right
                target.update(component)
            separations.append(
                _canonical_separation(
                    vertices,
                    cut.separator,
                    frozenset(left),
                    frozenset(right),
                )
            )

    if len(separations) != required or len(set(separations)) != required:
        raise ValueError("full separation enumeration produced duplicates")
    return tuple(sorted(separations, key=separation_sort_key))


def generate_elementary(
    vertices: VertexSet, cuts: tuple[CutRecord, ...]
) -> tuple[Separation, ...]:
    checked_vertices = _validated_vertices(vertices)
    if type(cuts) is not tuple:
        raise TypeError("cuts must be a tuple")
    vertex_set = frozenset(checked_vertices)
    elementary: set[Separation] = set()
    for cut in cuts:
        _validate_cut_partition(vertex_set, cut)
        separator = frozenset(cut.separator)
        nonseparator = vertex_set.difference(separator)
        for component in cut.components:
            wing = frozenset(component)
            elementary.add(
                _canonical_separation(
                    vertex_set,
                    cut.separator,
                    wing,
                    nonseparator.difference(wing),
                )
            )
    return tuple(sorted(elementary, key=separation_sort_key))


def separation_sort_key(value: Separation) -> tuple[VertexSet, VertexSet]:
    if type(value) is not Separation:
        raise TypeError("value must be Separation")
    return value.side_a, value.side_b


def are_nested(left: Separation, right: Separation) -> bool:
    if type(left) is not Separation or type(right) is not Separation:
        raise TypeError("nestedness requires Separation values")
    left_orientations = (
        (frozenset(left.side_a), frozenset(left.side_b)),
        (frozenset(left.side_b), frozenset(left.side_a)),
    )
    right_orientations = (
        (frozenset(right.side_a), frozenset(right.side_b)),
        (frozenset(right.side_b), frozenset(right.side_a)),
    )
    for away_left, toward_left in left_orientations:
        for away_right, toward_right in right_orientations:
            if away_left <= away_right and toward_left >= toward_right:
                return True
    return False


def crosses(left: Separation, right: Separation) -> bool:
    return not are_nested(left, right)
