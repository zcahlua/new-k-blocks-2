from __future__ import annotations

from itertools import combinations

from .models import Adjacency, GraphCase, SupportGraph, VertexSet


def validate_support_graph(graph: SupportGraph) -> None:
    if type(graph) is not SupportGraph:
        raise TypeError("graph must be SupportGraph")
    # Reconstructing re-applies the immutable model's complete simple-graph gate.
    SupportGraph(graph.vertices, graph.edges)


def support_graph_from_case(case: GraphCase) -> SupportGraph:
    if type(case) is not GraphCase:
        raise TypeError("case must be GraphCase")
    graph = SupportGraph(tuple(range(case.num_nodes)), case.edges)
    validate_support_graph(graph)
    return graph


def adjacency(graph: SupportGraph) -> Adjacency:
    validate_support_graph(graph)
    mutable = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        mutable[left].add(right)
        mutable[right].add(left)
    return {
        vertex: frozenset(mutable[vertex])
        for vertex in sorted(mutable)
    }


def _validate_adjacency(adj: Adjacency) -> None:
    if type(adj) is not dict:
        raise TypeError("adj must be a dict")
    vertices = set(adj)
    for vertex, neighbors in adj.items():
        if type(vertex) is not int or vertex < 0:
            raise ValueError("adjacency vertices must be nonnegative integers")
        if type(neighbors) is not frozenset:
            raise ValueError("adjacency neighborhoods must be frozensets")
        for neighbor in neighbors:
            if type(neighbor) is not int or neighbor < 0:
                raise ValueError("adjacency neighbors must be nonnegative integers")
            if neighbor == vertex:
                raise ValueError("adjacency contains a self-loop")
            if neighbor not in vertices:
                raise ValueError("adjacency references an unknown vertex")
            if vertex not in adj[neighbor]:
                raise ValueError("adjacency must be symmetric")


def components_after_deleting(
    adj: Adjacency, deleted: frozenset[int]
) -> tuple[VertexSet, ...]:
    _validate_adjacency(adj)
    if type(deleted) is not frozenset:
        raise TypeError("deleted must be a frozenset")
    for vertex in deleted:
        if type(vertex) is not int:
            raise ValueError("deleted vertices must be integers")
        if vertex not in adj:
            raise ValueError(f"unknown deleted vertex {vertex}")

    unseen = set(adj).difference(deleted)
    components: list[VertexSet] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            new_neighbors = sorted(
                (adj[vertex].difference(deleted)).intersection(unseen),
                reverse=True,
            )
            for neighbor in new_neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def is_k_connected(adj: Adjacency, k: int) -> bool:
    _validate_adjacency(adj)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer and not a boolean")
    vertices = tuple(sorted(adj))
    if len(vertices) <= k:
        return False
    for deleted_count in range(k):
        for deleted in combinations(vertices, deleted_count):
            if len(components_after_deleting(adj, frozenset(deleted))) != 1:
                return False
    return True


def validate_root_case(case: GraphCase) -> None:
    if type(case) is not GraphCase:
        raise TypeError("case must be GraphCase")
    graph = support_graph_from_case(case)
    if case.num_nodes <= case.k or not is_k_connected(adjacency(graph), case.k):
        raise ValueError(f"case {case.case_id!r} is not {case.k}-connected")
