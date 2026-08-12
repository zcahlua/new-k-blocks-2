from __future__ import annotations

from .graph import (
    adjacency,
    components_after_deleting,
    is_k_connected,
    support_graph_from_case,
    validate_root_case,
    validate_support_graph,
)
from .models import (
    Adjacency,
    CompletionLevel,
    CutRecord,
    GraphCase,
    LocalResult,
    LocalStatus,
    Separation,
    SupportGraph,
    stable_id,
)
from .separations import (
    are_nested,
    crosses,
    enumerate_full_separations,
    generate_elementary,
    list_order_k_cutsets,
    separation_sort_key,
)


def _separation_id(value: Separation) -> str:
    return stable_id(
        "separation",
        {
            "side_a": value.side_a,
            "side_b": value.side_b,
            "separator": value.separator,
        },
    )


def _validated_separations(
    values: tuple[Separation, ...], field: str
) -> tuple[Separation, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    for value in values:
        if type(value) is not Separation:
            raise TypeError(f"{field} must contain Separation values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate separations")
    return tuple(sorted(values, key=separation_sort_key))


def compute_tn_full(
    full_separations: tuple[Separation, ...],
) -> tuple[Separation, ...]:
    universe = _validated_separations(full_separations, "full_separations")
    accepted = tuple(
        candidate
        for candidate in universe
        if all(are_nested(candidate, other) for other in universe)
    )
    return tuple(sorted(accepted, key=separation_sort_key))


def compute_tn_pairwise(
    elementary: tuple[Separation, ...],
) -> tuple[tuple[Separation, ...], tuple[tuple[str, str], ...]]:
    candidates = _validated_separations(elementary, "elementary")
    accepted: list[Separation] = []
    witnesses: list[tuple[str, str]] = []
    for candidate in candidates:
        crossing_ids = sorted(
            _separation_id(other)
            for other in candidates
            if crosses(candidate, other)
        )
        if crossing_ids:
            witnesses.append((_separation_id(candidate), crossing_ids[0]))
        else:
            accepted.append(candidate)
    return (
        tuple(sorted(accepted, key=separation_sort_key)),
        tuple(sorted(witnesses)),
    )


def _count_components_meeting(
    cut: CutRecord, vertices: frozenset[int]
) -> int:
    return sum(
        1 for component in cut.components if frozenset(component).intersection(vertices)
    )


def _aggregated_crossing_vector(
    adj: Adjacency,
    cuts: tuple[CutRecord, ...],
    candidate: Separation,
    component_wing: frozenset[int],
    remainder_wing: frozenset[int],
    source_separator: frozenset[int],
) -> tuple[bool, ...]:
    del adj  # CutRecord contains the complete component table needed by the oracle.
    vector: list[bool] = []
    for target in cuts:
        target_separator = frozenset(target.separator)
        meets_component = bool(target_separator.intersection(component_wing))
        meets_remainder = bool(target_separator.intersection(remainder_wing))
        if meets_component and meets_remainder:
            crossed = True
        elif not meets_component and meets_remainder:
            crossed = (
                _count_components_meeting(
                    target, component_wing | source_separator
                )
                >= 2
            )
        elif meets_component and not meets_remainder:
            crossed = (
                _count_components_meeting(
                    target, remainder_wing | source_separator
                )
                >= 2
            )
        else:
            if target_separator != source_separator:
                raise ValueError(
                    "aggregated oracle found a target outside all source wings"
                )
            crossed = False
        vector.append(crossed)
    return tuple(vector)


def _candidate_orientations(
    candidate: Separation, source_cut: CutRecord
) -> tuple[tuple[frozenset[int], frozenset[int]], ...]:
    separator = frozenset(source_cut.separator)
    wings = (
        frozenset(candidate.side_a).difference(separator),
        frozenset(candidate.side_b).difference(separator),
    )
    components = tuple(frozenset(component) for component in source_cut.components)
    if not wings[0] or not wings[1] or wings[0] | wings[1] != frozenset().union(
        *components
    ):
        raise ValueError("elementary candidate does not match its source cut")

    if len(components) == 2:
        if set(wings) != set(components):
            raise ValueError("two-component elementary candidate has invalid wings")
        first, second = sorted(components, key=lambda value: tuple(sorted(value)))
        return ((first, second), (second, first))

    isolated = [wing for wing in wings if wing in components]
    if len(isolated) != 1:
        raise ValueError("elementary candidate lacks a unique component wing")
    component_wing = isolated[0]
    remainder_wing = wings[1] if wings[0] == component_wing else wings[0]
    return ((component_wing, remainder_wing),)


def compute_tn_aggregated(
    adj: Adjacency,
    cuts: tuple[CutRecord, ...],
    elementary: tuple[Separation, ...],
) -> tuple[tuple[Separation, ...], tuple[tuple[str, str], ...]]:
    components_after_deleting(adj, frozenset())
    if type(cuts) is not tuple:
        raise TypeError("cuts must be a tuple")
    for cut in cuts:
        if type(cut) is not CutRecord:
            raise TypeError("cuts must contain CutRecord values")
        if components_after_deleting(adj, frozenset(cut.separator)) != cut.components:
            raise ValueError("cut table does not match adjacency")
    if len({cut.separator for cut in cuts}) != len(cuts):
        raise ValueError("cut table contains duplicate separators")
    candidates = _validated_separations(elementary, "elementary")
    cut_by_separator = {cut.separator: cut for cut in cuts}
    candidates_by_separator: dict[tuple[int, ...], list[Separation]] = {
        cut.separator: [] for cut in cuts
    }
    for candidate in candidates:
        if candidate.separator not in cut_by_separator:
            raise ValueError("elementary candidate has no source cut")
        candidates_by_separator[candidate.separator].append(candidate)

    accepted: list[Separation] = []
    witnesses: list[tuple[str, str]] = []
    for candidate in candidates:
        source_cut = cut_by_separator[candidate.separator]
        source_separator = frozenset(source_cut.separator)
        orientations = _candidate_orientations(candidate, source_cut)
        vectors = tuple(
            _aggregated_crossing_vector(
                adj,
                cuts,
                candidate,
                component_wing,
                remainder_wing,
                source_separator,
            )
            for component_wing, remainder_wing in orientations
        )
        if len(vectors) == 2 and vectors[0] != vectors[1]:
            raise ValueError(
                "two-component aggregated orientation checks disagree"
            )
        vector = vectors[0]

        actual_crossing_ids: list[str] = []
        for index, target in enumerate(cuts):
            actual = [
                other
                for other in candidates_by_separator[target.separator]
                if crosses(candidate, other)
            ]
            if vector[index] != bool(actual):
                raise ValueError(
                    "aggregated oracle disagrees with an elementary crossing witness"
                )
            actual_crossing_ids.extend(_separation_id(other) for other in actual)

        if actual_crossing_ids:
            witnesses.append(
                (_separation_id(candidate), min(actual_crossing_ids))
            )
        else:
            accepted.append(candidate)

    return (
        tuple(sorted(accepted, key=separation_sort_key)),
        tuple(sorted(witnesses)),
    )


def _validate_witnesses(
    elementary: tuple[Separation, ...],
    accepted: tuple[Separation, ...],
    witnesses: tuple[tuple[str, str], ...],
) -> None:
    by_id = {_separation_id(value): value for value in elementary}
    if len(by_id) != len(elementary):
        raise ValueError("separation ID collision")
    accepted_ids = {_separation_id(value) for value in accepted}
    expected_rejected = set(by_id).difference(accepted_ids)
    witness_by_candidate = dict(witnesses)
    if len(witness_by_candidate) != len(witnesses):
        raise ValueError("duplicate rejection witness record")
    if set(witness_by_candidate) != expected_rejected:
        raise ValueError("rejection witness coverage is incomplete")
    for candidate_id, witness_id in witness_by_candidate.items():
        if witness_id not in by_id:
            raise ValueError("rejection witness ID is unresolved")
        candidate = by_id[candidate_id]
        witness = by_id[witness_id]
        if not crosses(candidate, witness):
            raise ValueError("rejection witness does not cross its candidate")
        least = min(
            _separation_id(other)
            for other in elementary
            if crosses(candidate, other)
        )
        if witness_id != least:
            raise ValueError("rejection witness is not lexically least")


def analyze_local(
    graph: SupportGraph,
    k: int,
    max_full_separations: int = 4096,
) -> LocalResult:
    validate_support_graph(graph)
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer at least 2")
    if type(max_full_separations) is not int or max_full_separations < 1:
        raise ValueError("max_full_separations must be a positive integer")

    adj = adjacency(graph)
    if len(graph.vertices) > k + 1 and not is_k_connected(adj, k):
        raise ValueError(f"support graph is not {k}-connected")
    cuts = list_order_k_cutsets(adj, k)
    full_separations = enumerate_full_separations(
        graph, cuts, max_full_separations
    )
    elementary = generate_elementary(graph.vertices, cuts)
    tn_full = compute_tn_full(full_separations)
    tn_pairwise, pairwise_witnesses = compute_tn_pairwise(elementary)
    tn_aggregated, aggregated_witnesses = compute_tn_aggregated(
        adj, cuts, elementary
    )

    if tn_full != tn_pairwise or tn_full != tn_aggregated:
        raise ValueError("full, pairwise, and aggregated TN results disagree")
    if pairwise_witnesses != aggregated_witnesses:
        raise ValueError("pairwise and aggregated rejection witnesses disagree")
    _validate_witnesses(elementary, tn_pairwise, pairwise_witnesses)

    if len(graph.vertices) <= k + 1:
        status = LocalStatus.SMALL
    elif not cuts:
        status = LocalStatus.HIGH
    elif not tn_full:
        status = LocalStatus.CROSSED
    else:
        status = LocalStatus.SPLIT

    return LocalResult(
        status,
        cuts,
        full_separations,
        elementary,
        tn_full,
        tn_pairwise,
        tn_aggregated,
        pairwise_witnesses,
        CompletionLevel.LOCAL_EXACT,
    )


def analyze_case_local(
    case: GraphCase, max_full_separations: int = 4096
) -> LocalResult:
    validate_root_case(case)
    return analyze_local(
        support_graph_from_case(case), case.k, max_full_separations
    )
