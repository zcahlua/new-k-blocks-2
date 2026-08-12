from __future__ import annotations

import random
from itertools import combinations

from .graph import validate_root_case
from .models import Edge, GraphCase, canonical_sha256


_CURATED_SEED = 20260811
_FAMILIES = (
    "small_clique",
    "high_clique",
    "joined_cycle",
    "complete_bipartite",
    "two_glued_cliques",
    "three_glued_cliques",
)
_EXPECTED: dict[str, tuple[tuple[str, int | str], ...]] = {
    "small_clique": (
        ("status", "SMALL"),
        ("cutsets", 0),
        ("full_sigma", 0),
        ("elementary", 0),
        ("tn", 0),
    ),
    "high_clique": (
        ("status", "HIGH"),
        ("cutsets", 0),
        ("full_sigma", 0),
        ("elementary", 0),
        ("tn", 0),
    ),
    "joined_cycle": (
        ("status", "CROSSED"),
        ("cutsets", 9),
        ("full_sigma", 9),
        ("elementary", 9),
        ("tn", 0),
    ),
    "complete_bipartite": (
        ("status", "SPLIT"),
        ("cutsets", 1),
        ("full_sigma", 15),
        ("elementary", 5),
        ("tn", 5),
    ),
    "two_glued_cliques": (
        ("status", "SPLIT"),
        ("cutsets", 1),
        ("full_sigma", 1),
        ("elementary", 1),
        ("tn", 1),
    ),
    "three_glued_cliques": (
        ("status", "SPLIT"),
        ("cutsets", 1),
        ("full_sigma", 3),
        ("elementary", 3),
        ("tn", 3),
    ),
}


def _validated_ks(
    ks: tuple[int, ...], *, curated_only: bool
) -> tuple[int, ...]:
    if type(ks) is not tuple or not ks:
        raise ValueError("ks must be a nonempty tuple")
    checked: list[int] = []
    for index, k in enumerate(ks):
        if type(k) is not int or k < 2:
            raise ValueError(f"ks[{index}] must be an integer at least 2")
        if curated_only and k not in (2, 3, 4):
            raise ValueError("curated cases exist only for k in {2, 3, 4}")
        checked.append(k)
    if len(set(checked)) != len(checked):
        raise ValueError("ks must contain unique values")
    return tuple(checked)


def _clique_edges(vertices: tuple[int, ...]) -> set[Edge]:
    return {tuple(edge) for edge in combinations(vertices, 2)}


def _curated_shape(
    k: int, family: str
) -> tuple[int, tuple[Edge, ...], tuple[tuple[str, int], ...]]:
    if family == "small_clique":
        num_nodes = k + 1
        edges = _clique_edges(tuple(range(num_nodes)))
        parameters = (("order", num_nodes),)
    elif family == "high_clique":
        num_nodes = k + 2
        edges = _clique_edges(tuple(range(num_nodes)))
        parameters = (("order", num_nodes),)
    elif family == "joined_cycle":
        join_size = k - 2
        cycle = tuple(range(join_size, join_size + 6))
        num_nodes = join_size + 6
        join = tuple(range(join_size))
        edges = _clique_edges(join)
        edges.update(
            (min(cycle[index], cycle[(index + 1) % 6]),
             max(cycle[index], cycle[(index + 1) % 6]))
            for index in range(6)
        )
        edges.update((left, right) for left in join for right in cycle)
        parameters = (("cycle_size", 6), ("join_clique_size", join_size))
    elif family == "complete_bipartite":
        left = tuple(range(k))
        right = tuple(range(k, k + 5))
        num_nodes = k + 5
        edges = {(u, v) for u in left for v in right}
        parameters = (("left_size", k), ("right_size", 5))
    elif family in ("two_glued_cliques", "three_glued_cliques"):
        clique_count = 2 if family == "two_glued_cliques" else 3
        shared = tuple(range(k))
        num_nodes = k + 2 * clique_count
        edges = set()
        for index in range(clique_count):
            private = (k + 2 * index, k + 2 * index + 1)
            edges.update(_clique_edges(shared + private))
        parameters = (
            ("clique_count", clique_count),
            ("clique_size", k + 2),
            ("shared_size", k),
        )
    else:
        raise ValueError(f"unknown curated family {family!r}")
    return num_nodes, tuple(sorted(edges)), parameters


def build_curated_cases(
    ks: tuple[int, ...] = (2, 3, 4)
) -> tuple[GraphCase, ...]:
    checked_ks = _validated_ks(ks, curated_only=True)
    cases: list[GraphCase] = []
    for k in checked_ks:
        for family in _FAMILIES:
            num_nodes, edges, parameters = _curated_shape(k, family)
            case = GraphCase(
                f"curated-k{k}-{family}",
                k,
                num_nodes,
                edges,
                family,
                parameters,
                _CURATED_SEED,
                _EXPECTED[family],
            )
            validate_root_case(case)
            cases.append(case)
    return tuple(cases)


def build_random_cases(
    ks: tuple[int, ...], per_k: int, seed: int
) -> tuple[GraphCase, ...]:
    checked_ks = _validated_ks(ks, curated_only=False)
    if type(per_k) is not int or per_k < 0:
        raise ValueError("per_k must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    cases: list[GraphCase] = []
    for k in checked_ks:
        for index in range(per_k):
            digest = canonical_sha256({"seed": seed, "k": k, "index": index})
            case_seed = int(digest[:16], 16)
            generator = random.Random(case_seed)
            initial_order = k + 1
            added_vertices = 4
            num_nodes = initial_order + added_vertices
            edges = _clique_edges(tuple(range(initial_order)))

            for vertex in range(initial_order, num_nodes):
                neighbors = generator.sample(tuple(range(vertex)), k)
                edges.update(
                    (min(vertex, neighbor), max(vertex, neighbor))
                    for neighbor in neighbors
                )

            for left, right in combinations(range(num_nodes), 2):
                if (left, right) not in edges and generator.random() < 0.25:
                    edges.add((left, right))

            case = GraphCase(
                f"random-k{k}-i{index}-{digest[:16]}",
                k,
                num_nodes,
                tuple(sorted(edges)),
                "random_k_tree_plus_edges",
                (
                    ("index", index),
                    ("initial_order", initial_order),
                    ("added_vertices", added_vertices),
                    ("edge_probability_percent", 25),
                ),
                case_seed,
                (),
            )
            validate_root_case(case)
            cases.append(case)
    return tuple(cases)
