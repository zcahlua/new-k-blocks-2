# BL-RCTN-V_k 小图运行器与 GPT 可视化包 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本目录内建立一个独立 Python 参考程序：确定性生成小图样本，运行 BL-RCTN-V_k 的 `STRUCTURE_ONLY` Elementary exact algorithm，独立验证结果，并输出可直接交给 GPT 生成图形的 JSON/CSV/Mermaid/DOT 包。

**Architecture:** 实现分为四条隔离通道：样本生成、Elementary XP 算法与 full-separation oracle、递归 structure-tree/provenance、fail-closed 独立 verifier。Engine 只产生 `RECURSIVE_CANDIDATE`；只有 verifier 全部通过才生成 `RECURSIVE_VERIFIED` 和 GPT 可视化包。

**Tech Stack:** Python 3.11+ standard library only：`argparse`、`dataclasses`、`enum`、`hashlib`、`itertools`、`json`、`pathlib`、`random`、`unittest`。

## Global Constraints

- 所有项目 source/data/artifact I/O 只允许位于 `/Users/zcahlua/Desktop/8月work/新方法/algorithm/`；Python 解释器读取其 standard-library runtime 属于允许的执行前提，但实现不得主动打开、扫描、导入或复制任何目录外项目/数据文件。
- 不初始化 Git，不提交，不访问网络，不安装依赖。
- 图必须是 finite undirected simple graph；顶点严格为连续整数 `0..n-1`；`k >= 2`。
- V1 只实现 `STRUCTURE_ONLY`。`ambient_blocks` 与 `block_localization` 必须序列化为 `null`；请求 `WITH_AMBIENT_BLOCKS` 必须明确失败。
- 主 cutset backend 为 exact `XP_SUBSET`：枚举全部 size-`k` vertex subsets。
- Full separation universe 仅作为 bounded differential oracle；demo 限制 `max_full_separations=4096`。
- 结果只有在 `verification.verified == true` 时才是成功输出。
- 所有 artifact writer 都是 write-once；不得覆盖已有文件。分阶段 CLI 只能新增本阶段缺失的 artifact，已有 artifact 必须在内存中重算并逐字节比较，完全相同则只读通过，不同则失败。
- 所有测试与 staging 临时目录固定在 `algorithm/runs/.tmp/` 下并在成功或失败后清理；禁止使用操作系统默认临时目录。
- 稳定 ID 使用 canonical JSON 的 SHA-256；relabeling 比较使用显式 vertex transport，不比较原始 digest 字符串。
- Demo seed 固定为 `20260811`。

---

## File map

~~~text
algorithm/
├── IMPLEMENTATION_PLAN.md
├── CODEX_EXECUTION_PROMPT.md
├── README.md
├── pyproject.toml
├── bl_rctn/
│   ├── __init__.py
│   ├── models.py
│   ├── graph.py
│   ├── samples.py
│   ├── separations.py
│   ├── local.py
│   ├── structure_tree.py
│   ├── hierarchy.py
│   ├── verify.py
│   ├── export.py
│   └── cli.py
├── tests/
│   ├── test_models.py
│   ├── test_graph.py
│   ├── test_samples.py
│   ├── test_local.py
│   ├── test_structure_tree.py
│   ├── test_hierarchy.py
│   ├── test_verify.py
│   ├── test_export.py
│   └── test_cli.py
├── configs/demo_suite.json
└── runs/<fresh-run-id>/
~~~

## Fixed interfaces

~~~python
# bl_rctn/models.py
Edge = tuple[int, int]
VertexSet = tuple[int, ...]
Adjacency = dict[int, frozenset[int]]

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

@dataclass(frozen=True)
class SupportGraph:
    vertices: VertexSet
    edges: tuple[Edge, ...]

@dataclass(frozen=True)
class Separation:
    side_a: VertexSet
    side_b: VertexSet
    separator: VertexSet

@dataclass(frozen=True)
class OrientedSeparation:
    base_separation_id: str
    away_side: VertexSet
    toward_side: VertexSet

@dataclass(frozen=True)
class CutRecord:
    separator: VertexSet
    components: tuple[VertexSet, ...]

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

@dataclass(frozen=True)
class StructureTreeNode:
    node_id: str
    orientation_signature: tuple[tuple[str, int], ...]
    bag_vertices: VertexSet

@dataclass(frozen=True)
class StructureTreeEdge:
    tree_edge_id: str
    source_node_id: str
    target_node_id: str
    separation: Separation
    adhesion: VertexSet

@dataclass(frozen=True)
class StructureTree:
    nodes: tuple[StructureTreeNode, ...]
    edges: tuple[StructureTreeEdge, ...]

@dataclass(frozen=True)
class EdgeRecord:
    endpoints: Edge
    is_root_real: bool
    root_edge_id: str | None
    virtual_interface_ids: tuple[str, ...]

@dataclass(frozen=True)
class InterfaceIncidence:
    incidence_id: str
    side_tree_node_id: str
    opposite_incidence_id: str

@dataclass(frozen=True)
class InterfaceObject:
    interface_id: str
    creator_record_id: str
    creator_tree_edge_id: str
    boundary: VertexSet
    incidences: tuple[InterfaceIncidence, InterfaceIncidence]

@dataclass(frozen=True)
class InterfaceRef:
    interface_id: str
    incidence_id: str
    coverage: Coverage
    local_boundary: VertexSet

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

@dataclass(frozen=True)
class HierarchyEdge:
    parent_record_id: str
    child_record_id: str
    local_tree_node_id: str

@dataclass(frozen=True)
class HierarchyResult:
    root_record_id: str
    config_digest: str
    records: tuple[TorsoRecord, ...]
    interfaces: tuple[InterfaceObject, ...]
    hierarchy_edges: tuple[HierarchyEdge, ...]
    terminal_record_ids: tuple[str, ...]
    completion_level: CompletionLevel

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

# The following lines are declarative signature contracts, not function bodies.
canonical_json_bytes(value: object) -> bytes
canonical_sha256(value: object) -> str
stable_id(namespace: str, value: object) -> str
write_json_once(path: Path, value: object) -> None
read_graph_cases(path: Path) -> tuple[GraphCase, ...]
write_graph_cases_once(path: Path, cases: tuple[GraphCase, ...]) -> None
read_demo_config(path: Path) -> DemoConfig
write_demo_config_once(path: Path, config: DemoConfig) -> None
graph_case_to_dict(case: GraphCase) -> dict[str, object]
graph_case_from_dict(value: dict[str, object]) -> GraphCase
hierarchy_result_to_dict(result: HierarchyResult) -> dict[str, object]
hierarchy_result_from_dict(value: dict[str, object]) -> HierarchyResult
write_hierarchy_result_once(path: Path, result: HierarchyResult) -> None
read_hierarchy_result(path: Path) -> HierarchyResult
verification_report_to_dict(report: VerificationReport) -> dict[str, object]
verification_report_from_dict(value: dict[str, object]) -> VerificationReport
write_verification_report_once(path: Path, report: VerificationReport) -> None
read_verification_report(path: Path) -> VerificationReport

# bl_rctn/graph.py
validate_root_case(case: GraphCase) -> None
support_graph_from_case(case: GraphCase) -> SupportGraph
validate_support_graph(graph: SupportGraph) -> None
adjacency(graph: SupportGraph) -> Adjacency
components_after_deleting(adj: Adjacency, deleted: frozenset[int]) -> tuple[VertexSet, ...]
is_k_connected(adj: Adjacency, k: int) -> bool

# bl_rctn/samples.py
build_curated_cases(ks: tuple[int, ...] = (2, 3, 4)) -> tuple[GraphCase, ...]
build_random_cases(ks: tuple[int, ...], per_k: int, seed: int) -> tuple[GraphCase, ...]

# bl_rctn/separations.py
list_order_k_cutsets(adj: Adjacency, k: int) -> tuple[CutRecord, ...]
enumerate_full_separations(graph: SupportGraph, cuts: tuple[CutRecord, ...], limit: int) -> tuple[Separation, ...]
generate_elementary(vertices: VertexSet, cuts: tuple[CutRecord, ...]) -> tuple[Separation, ...]
separation_sort_key(value: Separation) -> tuple[VertexSet, VertexSet]
are_nested(left: Separation, right: Separation) -> bool
crosses(left: Separation, right: Separation) -> bool

# bl_rctn/local.py
compute_tn_full(full_separations: tuple[Separation, ...]) -> tuple[Separation, ...]
compute_tn_pairwise(elementary: tuple[Separation, ...]) -> tuple[tuple[Separation, ...], tuple[tuple[str, str], ...]]
compute_tn_aggregated(adj: Adjacency, cuts: tuple[CutRecord, ...], elementary: tuple[Separation, ...]) -> tuple[tuple[Separation, ...], tuple[tuple[str, str], ...]]
analyze_local(graph: SupportGraph, k: int, max_full_separations: int = 4096) -> LocalResult
analyze_case_local(case: GraphCase, max_full_separations: int = 4096) -> LocalResult

# bl_rctn/structure_tree.py
build_structure_tree(graph: SupportGraph, tn: tuple[Separation, ...]) -> StructureTree
verify_structure_tree(graph: SupportGraph, tn, tree: StructureTree) -> tuple[str, ...]

# bl_rctn/hierarchy.py
decompose_case(case: GraphCase, config: DemoConfig) -> HierarchyResult
reconstruct_root(hierarchy: HierarchyResult) -> tuple[VertexSet, tuple[Edge, ...]]
propagate_interface_ref(ref: InterfaceRef, global_boundary: VertexSet, child_vertices: VertexSet) -> InterfaceRef | None

# bl_rctn/verify.py
verify_case_result(case: GraphCase, hierarchy: HierarchyResult, config: DemoConfig) -> VerificationReport
publish_verification_reports_once(run_dir: Path) -> tuple[Path, ...]
verify_run(run_dir: Path) -> dict[str, object]

# bl_rctn/export.py
export_case_bundle(case: GraphCase, hierarchy: HierarchyResult, verification: VerificationReport, config: DemoConfig, output_dir: Path) -> tuple[Path, ...]
export_run_summary(run_dir: Path) -> tuple[Path, Path]
~~~

---

### Task 1: Immutable schemas and write-once I/O

**Files:** Create `pyproject.toml`, `bl_rctn/__init__.py`, `bl_rctn/models.py`, `tests/test_models.py`.

**Produces:** All `models.py` interfaces above.

- [ ] **Step 1: Write failing tests**

~~~python
# tests/test_models.py
import tempfile, unittest
from pathlib import Path
from bl_rctn.models import GraphCase, stable_id, write_graph_cases_once

class ModelTests(unittest.TestCase):
    def test_edges_are_normalized(self):
        case = GraphCase("x", 2, 4, ((3, 1), (0, 2)), "manual", (), 7, ())
        self.assertEqual(case.edges, ((0, 2), (1, 3)))

    def test_loop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self-loop"):
            GraphCase("x", 2, 3, ((1, 1),), "manual", (), 7, ())

    def test_write_once(self):
        case = GraphCase("x", 2, 3, ((0, 1), (1, 2), (0, 2)), "clique", (), 7, ())
        scratch = Path.cwd() / "runs" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            path = Path(raw) / "cases.jsonl"
            write_graph_cases_once(path, (case,))
            with self.assertRaises(FileExistsError):
                write_graph_cases_once(path, (case,))

    def test_stable_id_ignores_dict_key_order(self):
        self.assertEqual(stable_id("x", {"a": 1, "b": 2}), stable_id("x", {"b": 2, "a": 1}))
~~~

- [ ] **Step 2: Run `python3 -m unittest tests.test_models -v` and confirm the import failure.**
- [ ] **Step 3: Implement strict dataclass validation, canonical UTF-8 JSON, SHA-256 IDs, strict JSONL parsing, and write-once atomic publication.** `canonical_sha256` returns exactly 64 lowercase hex characters; `stable_id` prefixes that digest with a safe namespace and is used for object IDs, never for report digest fields.
- [ ] **Step 4: Add `pyproject.toml` with project version `0.1.0`, Python `>=3.11`, and `dependencies=[]`.**
- [ ] **Step 5: Rerun Task 1 tests; expected result is zero failures.**

Validation must reject booleans as integers, unsafe IDs, loops, duplicate edges, endpoints outside `0..n-1`, duplicate case IDs, unknown fields, malformed UTF-8, blank JSONL rows, and overwrite attempts. `DemoConfig` must require its exact schema literal, unique `ks >= 2`, nonnegative seed/count, positive budgets, and `mode == "STRUCTURE_ONLY"`. Every wire dataclass must have a closed-schema dict encoder/decoder; the hierarchy reader must reject unknown fields and noncanonical JSON before the verifier sees the object.

---

### Task 2: Graph primitives and legal-root gate

**Files:** Create `bl_rctn/graph.py` and `tests/test_graph.py`.

- [ ] **Step 1: Write failing tests**

~~~python
# tests/test_graph.py
import unittest
from bl_rctn.graph import adjacency, components_after_deleting, is_k_connected, support_graph_from_case, validate_root_case
from bl_rctn.models import GraphCase

class GraphTests(unittest.TestCase):
    def test_cycle_is_two_connected(self):
        case = GraphCase("c6", 2, 6, tuple((i, (i + 1) % 6) for i in range(6)), "cycle", (), 1, ())
        adj = adjacency(support_graph_from_case(case))
        self.assertTrue(is_k_connected(adj, 2))
        self.assertEqual(components_after_deleting(adj, frozenset({0, 3})), ((1, 2), (4, 5)))

    def test_path_is_rejected(self):
        case = GraphCase("p4", 2, 4, ((0, 1), (1, 2), (2, 3)), "path", (), 1, ())
        with self.assertRaisesRegex(ValueError, "not 2-connected"):
            validate_root_case(case)
~~~

- [ ] **Step 2: Run the tests and observe the missing implementation.**
- [ ] **Step 3: Implement `SupportGraph`, root-to-support conversion, deterministic adjacency, components, deletion connectivity, and exact `k`-connectivity by testing every deletion set of size `0..k-1`.** `SupportGraph.vertices` may be a nonconsecutive subset of root IDs and may have size at most `k+1`; this is required for recursive torsos.
- [ ] **Step 4: Make `validate_root_case` enforce all root conditions before algorithm work.**
- [ ] **Step 5: Run `python3 -m unittest tests.test_graph -v`; expected result is PASS.**

Traversal order must always choose the least unseen vertex and return lexicographically sorted components.

---

### Task 3: Generate deterministic samples

**Files:** Create `bl_rctn/samples.py`, `configs/demo_suite.json`, `tests/test_samples.py`.

- [ ] **Step 1: Write failing tests requiring 18 valid curated cases and reproducible random cases.**
- [ ] **Red gate: Run `python3 -m unittest tests.test_samples -v`.** Expected failure is `ModuleNotFoundError: bl_rctn.samples`; a syntax error or unrelated failure is not an acceptable red gate.

~~~python
# tests/test_samples.py
import unittest
from collections import Counter
from bl_rctn.graph import validate_root_case
from bl_rctn.samples import build_curated_cases, build_random_cases

class SampleTests(unittest.TestCase):
    def test_curated_matrix(self):
        cases = build_curated_cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual(Counter(c.k for c in cases), Counter({2: 6, 3: 6, 4: 6}))
        for case in cases:
            validate_root_case(case)

    def test_random_reproducibility(self):
        self.assertEqual(build_random_cases((2, 3), 2, 20260811),
                         build_random_cases((2, 3), 2, 20260811))
~~~

- [ ] **Step 2: For each `k in {2,3,4}` implement these six analytic families.**

| Family | Construction | Status | cutsets | full Σ | elementary | TN |
|---|---|---:|---:|---:|---:|---:|
| `small_clique` | `K_(k+1)` | SMALL | 0 | 0 | 0 | 0 |
| `high_clique` | `K_(k+2)` | HIGH | 0 | 0 | 0 | 0 |
| `joined_cycle` | `K_(k-2) ∨ C6` | CROSSED | 9 | 9 | 9 | 0 |
| `complete_bipartite` | `K_(k,5)` | SPLIT | 1 | 15 | 5 | 5 |
| `two_glued_cliques` | two `K_(k+2)` sharing exactly `K_k` | SPLIT | 1 | 1 | 1 | 1 |
| `three_glued_cliques` | three `K_(k+2)` sharing exactly `K_k` | SPLIT | 1 | 3 | 3 | 3 |

- [ ] **Step 3: Implement optional random cases.** Start with `K_(k+1)`; every added vertex chooses exactly `k` existing neighbors, then each remaining edge is added with probability `0.25`. Derive each case seed from the first 16 hex characters of `canonical_sha256({"seed": seed, "k": k, "index": index})`; validate the final graph.
- [ ] **Step 4: Write this exact config.**

~~~json
{"schema_version":"bl-rctn-demo-config-v1","ks":[2,3,4],"seed":20260811,"random_per_k":0,"max_full_separations":4096,"max_depth":20,"mode":"STRUCTURE_ONLY"}
~~~

- [ ] **Step 5: Run `python3 -m unittest tests.test_samples -v`; expected result is PASS.**

---

### Task 4: Full oracle and Elementary local exactness

**Files:** Create `bl_rctn/separations.py`, `bl_rctn/local.py`, `tests/test_local.py`.

- [ ] **Step 1: Write a table-driven test over all 18 curated cases.** It must compare analytic counts and require:
- [ ] **Red gate: Run `python3 -m unittest tests.test_local -v`.** Expected failure is the missing `bl_rctn.local` or `bl_rctn.separations` module; a syntax error is not acceptable.

~~~text
TN(full Σ oracle) == TN(elementary pairwise) == TN(aggregated)
~~~

~~~python
# tests/test_local.py
import unittest
from bl_rctn.local import analyze_case_local
from bl_rctn.samples import build_curated_cases

class LocalTests(unittest.TestCase):
    def test_analytic_matrix_and_three_way_equality(self):
        for case in build_curated_cases():
            with self.subTest(case=case.case_id):
                result = analyze_case_local(case)
                expected = dict(case.expected)
                self.assertEqual(result.status.value, expected["status"])
                self.assertEqual(len(result.cuts), expected["cutsets"])
                self.assertEqual(len(result.full_separations), expected["full_sigma"])
                self.assertEqual(len(result.elementary), expected["elementary"])
                self.assertEqual(len(result.tn_full), expected["tn"])
                self.assertEqual(result.tn_full, result.tn_pairwise)
                self.assertEqual(result.tn_full, result.tn_aggregated)
~~~

Root tests call `analyze_case_local(case)`. Recursive code calls `analyze_local(SupportGraph, k)` so a SMALL child torso with nonconsecutive root IDs never has to masquerade as a root `GraphCase`.

- [ ] **Step 2: Implement canonical unoriented separations and exact nestedness.** Normalize the two sides lexicographically and use only `separation_sort_key`; `Separation` has no automatic ordering.

~~~python
def are_nested(left, right):
    for a, b in ((left.side_a, left.side_b), (left.side_b, left.side_a)):
        for c, d in ((right.side_a, right.side_b), (right.side_b, right.side_a)):
            if set(a) <= set(c) and set(b) >= set(d):
                return True
    return False
~~~

- [ ] **Step 3: Implement `XP_SUBSET` cut listing.** For every size-`k` subset `S`, compute `K-S` components and retain `S` iff at least two components remain.
- [ ] **Step 4: Implement both universes.** Full oracle fixes the first component on one side and enumerates all nontrivial component bipartitions; if the next object would exceed the budget, raise before returning anything and publish no `LocalResult`. Elementary generation isolates each component and deduplicates side-swapped copies.
- [ ] **Step 5: Implement and persist full TN.** `compute_tn_full` filters the complete full universe by exact nestedness against every full separation and stores the canonical sorted result in `LocalResult.tn_full`. It must never operate on a truncated universe.
- [ ] **Step 6: Implement pairwise TN.** Accept a candidate iff it is nested with every elementary candidate; save the least crossing witness ID for every rejection.
- [ ] **Step 7: Recover the aggregated orientation from the complete `CutRecord`.** When `K-S` has at least three components, the unique one-component wing is `C`. When it has exactly two components, choose the lexicographically smaller component as `C`, run the oracle once more with the other component as `C`, and assert that both orientations produce the same crossing-existence result before keeping the canonical result.
- [ ] **Step 8: Implement the aggregated four-case oracle.**

~~~python
if T & C and T & R:
    crossed = True
elif not (T & C) and T & R:
    crossed = count_components_meeting(T, C | S) >= 2
elif T & C and not (T & R):
    crossed = count_components_meeting(T, R | S) >= 2
else:
    assert T == S
    crossed = False
~~~

A false result for one `T` must continue to later cutsets. A true result stores a real elementary crossing witness.

- [ ] **Step 9: Implement statuses in this order:** `n <= k+1 -> SMALL`; empty cut table -> `HIGH`; nonempty cuts plus empty TN -> `CROSSED`; otherwise `SPLIT`.
- [ ] **Step 10: Return `LOCAL_EXACT` only after exact equality of `tn_full`, `tn_pairwise`, and `tn_aggregated`, both-orientation checks, and witness checks pass.**
- [ ] **Step 11: Run `python3 -m unittest tests.test_local -v`; all analytic counts and differential checks must pass.**

---

### Task 5: Structure tree from all accepted TN separations

**Files:** Create `bl_rctn/structure_tree.py` and `tests/test_structure_tree.py`.

- [ ] **Step 1: Write tests:** call `build_structure_tree(support_graph_from_case(case), local.tn_aggregated)`; one TN cut gives two nodes/one edge; three shared-boundary cuts give a four-node star with degrees `[1,1,1,3]`.
- [ ] **Red gate: Run `python3 -m unittest tests.test_structure_tree -v`.** Expected failure is the missing `bl_rctn.structure_tree` module, not a malformed test.

~~~python
# tests/test_structure_tree.py
import unittest
from bl_rctn.graph import support_graph_from_case
from bl_rctn.local import analyze_case_local
from bl_rctn.samples import build_curated_cases
from bl_rctn.structure_tree import build_structure_tree, verify_structure_tree

class StructureTreeTests(unittest.TestCase):
    def test_one_cut_and_shared_boundary_star(self):
        cases = build_curated_cases((3,))
        for family, counts in (("two_glued_cliques", (2, 1)),
                               ("three_glued_cliques", (4, 3))):
            case = next(c for c in cases if c.family == family)
            graph = support_graph_from_case(case)
            local = analyze_case_local(case)
            tree = build_structure_tree(graph, local.tn_aggregated)
            self.assertEqual((len(tree.nodes), len(tree.edges)), counts)
            self.assertEqual(verify_structure_tree(graph, local.tn_aggregated, tree), ())
            if family == "three_glued_cliques":
                degrees = {node.node_id: 0 for node in tree.nodes}
                for edge in tree.edges:
                    degrees[edge.source_node_id] += 1
                    degrees[edge.target_node_id] += 1
                self.assertEqual(sorted(degrees.values()), [1, 1, 1, 3])
~~~
- [ ] **Step 2: Validate the input family before construction.** Require canonical side normalization, no duplicates, every separation proper, and every pair nested. Reject any degenerate/coterminal duplicate that would make the oriented tree set non-essential; do not silently contract it because each accepted TN separation must own one tree edge.
- [ ] **Step 3: Represent orientation `(A,B)` as an `OrientedSeparation` pointing toward `B`.** Implement named predicates `oriented_leq(r,s)` for `r.away_side subset s.away_side` and `r.toward_side superset s.toward_side`, and `oriented_lt(r,s)` as `oriented_leq(r,s) and r != s`. Never use tuple ordering or dataclass `<` for separation theory.
- [ ] **Step 4: Enumerate every orientation assignment of the complete TN family and keep exactly consistent assignments:** no distinct `r,s` satisfy `oriented_lt(reverse(r), s)`.
- [ ] **Step 5: Make one node per consistent orientation.** Its bag is the intersection of all pointed-to sides; with zero TN cuts, the only bag is `graph.vertices`.
- [ ] **Step 6: Connect nodes with Hamming distance one and label the edge by the unique differing TN separation.**
- [ ] **Step 7: Verify before returning:**

~~~text
node_count = separation_count + 1
edge_count = separation_count
connected and acyclic
each TN separation labels exactly one edge
bags cover vertices and support edges
bags containing each vertex form a connected subtree
endpoint bag intersection equals the labelled separator
~~~

- [ ] **Step 8: Run `python3 -m unittest tests.test_structure_tree -v`; expected result is PASS.**

---

### Task 6: Recursive torsos, interfaces, and reconstruction

**Files:** Create `bl_rctn/hierarchy.py` and `tests/test_hierarchy.py`.

- [ ] **Step 1: Write tests:** two glued cliques must end in two HIGH terminals and exactly one creator interface; all 18 cases must reconstruct the original graph using terminal records only. Add direct tests for `propagate_interface_ref`: full boundary `(0,1,2)` intersecting child `(0,1,3)` becomes PARTIAL `(0,1)`; intersecting child `(0,3)` preserves a singleton PARTIAL `(0,)`; disjoint child returns `None`; a PARTIAL parent is never upgraded to FULL.
- [ ] **Red gate: Run `python3 -m unittest tests.test_hierarchy -v`.** Expected failure is the missing `bl_rctn.hierarchy` module, not a syntax/import error in an earlier completed module.

~~~python
# tests/test_hierarchy.py
import unittest
from bl_rctn.hierarchy import decompose_case, propagate_interface_ref, reconstruct_root
from bl_rctn.models import Coverage, DemoConfig, InterfaceRef
from bl_rctn.samples import build_curated_cases

CONFIG = DemoConfig("bl-rctn-demo-config-v1", (2, 3, 4), 20260811, 0, 4096, 20, "STRUCTURE_ONLY")

class HierarchyTests(unittest.TestCase):
    def test_reconstruction_and_two_clique_terminals(self):
        for case in build_curated_cases():
            result = decompose_case(case, CONFIG)
            self.assertEqual(reconstruct_root(result), (tuple(range(case.num_nodes)), case.edges))
            if case.family == "two_glued_cliques":
                terminals = [r for r in result.records if r.record_id in result.terminal_record_ids]
                self.assertEqual(sorted(r.status.value for r in terminals), ["HIGH", "HIGH"])
                self.assertEqual(len(result.interfaces), 1)

    def test_partial_reference_never_upgrades(self):
        ref = InterfaceRef("i", "inc", Coverage.FULL, (0, 1, 2))
        partial = propagate_interface_ref(ref, (0, 1, 2), (0, 1, 3))
        self.assertEqual((partial.coverage, partial.local_boundary), (Coverage.PARTIAL, (0, 1)))
        self.assertEqual(propagate_interface_ref(ref, (0, 1, 2), (0, 3)).local_boundary, (0,))
        self.assertIsNone(propagate_interface_ref(ref, (0, 1, 2), (3, 4)))
        parent_partial = InterfaceRef("i", "inc", Coverage.PARTIAL, (0, 1))
        self.assertEqual(propagate_interface_ref(parent_partial, (0, 1, 2), (0, 1, 2)).coverage,
                         Coverage.PARTIAL)
~~~
- [ ] **Step 2: Define each torso record with:**

~~~text
record_id, parent_record_id, depth, bag_vertices, status,
support_edges, edge_records, interface_refs, local_result, structure_tree
~~~

Each edge record has `endpoints`, `is_root_real`, `root_edge_id`, and `virtual_interface_ids`.

- [ ] **Step 3: For every local structure-tree edge create exactly one immutable InterfaceObject and two opposite incidences.** Equal boundary sets must not merge different creator interfaces.
- [ ] **Step 4: Construct each child torso from the induced parent support graph plus clique completion of incident adhesions.** Coalesce endpoint pairs by unioning roles; never overwrite a root-real role.
- [ ] **Step 5: Propagate inherited interface fragments.** Empty intersection is omitted; complete boundary remains `FULL` only if the parent was full; every other nonempty intersection is `PARTIAL`; singleton fragments remain record-level references.
- [ ] **Step 6: Recurse only on `SPLIT`.** Convert the root case to `SupportGraph` once; bind the candidate to the frozen config digest; construct every child as a `SupportGraph` over its root vertex IDs; call `analyze_local(child_graph, k)`. Require proper child bags, `depth <= 20`, and no repeated semantic state on one recursion path. Engine output remains `RECURSIVE_CANDIDATE`.
- [ ] **Step 7: Fail closed on every recursive invariant.** A nonproper child, depth overflow, repeated semantic state, invalid child support graph, structure-tree issue, or provenance issue raises a structured exception before `write_hierarchy_result_once` is called; no partial engine JSON may be published.
- [ ] **Step 8: Implement terminal-only reconstruction:** union terminal vertices, collect only root-real edges, require consistent edge ID/endpoints, ignore purely virtual edges, and resolve every interface reference. The function must not read `GraphCase.edges`.
- [ ] **Step 9: Run `python3 -m unittest tests.test_hierarchy -v`; expected result is PASS.**

---

### Task 7: Independent fail-closed verifier

**Files:** Create `bl_rctn/verify.py` and `tests/test_verify.py`.

- [ ] **Step 1: Write positive tests for all 18 cases and negative tampering tests for a forged real edge, missing interface incidence, upgraded partial reference, wrong bag, omitted TN cut, and wrong terminal label.**
- [ ] **Red gate: Run `python3 -m unittest tests.test_verify -v`.** Expected failure is the missing `bl_rctn.verify` module; all Tasks 1-6 tests must still pass.

~~~python
# tests/test_verify.py
import unittest
from dataclasses import replace
from bl_rctn.hierarchy import decompose_case
from bl_rctn.models import DemoConfig
from bl_rctn.samples import build_curated_cases
from bl_rctn.verify import verify_case_result

CONFIG = DemoConfig("bl-rctn-demo-config-v1", (2, 3, 4), 20260811, 0, 4096, 20, "STRUCTURE_ONLY")

class VerifyTests(unittest.TestCase):
    def test_all_cases_and_config_binding(self):
        for case in build_curated_cases():
            candidate = decompose_case(case, CONFIG)
            self.assertTrue(verify_case_result(case, candidate, CONFIG).verified)
            tampered = replace(candidate, config_digest="0" * 64)
            self.assertFalse(verify_case_result(case, tampered, CONFIG).verified)
~~~
- [ ] **Step 2: Independently recompute full separations without calling the engine's cut/candidate/TN helpers.** Enumerate assignments of nonseparator vertices to two nonempty wings and retain exactly assignments with no cross-wing edge.
- [ ] **Step 3: Verify root legality, local status, exact TN, tree-edge/TN bijection, tree/bag axioms, child torso equality, strict progress, interface pairing/multiplicity, role persistence, full/partial propagation, unresolved IDs, and terminal-only reconstruction.**
- [ ] **Step 4: Check fixture expectations from numeric fields only.** Never trust `family` or `case_id` as evidence.
- [ ] **Step 5: Add metamorphic tests:** three deterministic random relabelings per curated case. Map bags, separations, real edges, and boundaries back through the inverse permutation; compare the transported abstract hierarchy while ignoring raw digest strings.
- [ ] **Step 6: Bind the report to exact inputs before returning.** Compute the canonical case, candidate hierarchy, and frozen config digests independently. Return this fixed report shape; `checks` and `issues` are arrays of strict fixed-length records matching `VerificationReport`.

~~~json
{"schema_version":"bl-rctn-verification-v1","verifier_version":"1.0.0","case_id":"case-id","case_digest":"64-hex","candidate_hierarchy_digest":"64-hex","config_digest":"64-hex","verified":true,"completion_level":"RECURSIVE_VERIFIED","checks":[],"issues":[]}
~~~

When any issue exists, `verified` is false and completion remains `RECURSIVE_CANDIDATE`.

- [ ] **Step 7: Implement report publication and `verify_run`.** `publish_verification_reports_once` creates only missing reports atomically; for an existing report it recomputes canonical bytes and requires equality. `verify_run` is then a stage-aware read-only audit: always bind input config/cases/engine/reports; when GPT artifacts and manifest exist, also recompute every bundle, summary row, relative-path closure, byte size, and digest. Neither function overwrites an artifact.

- [ ] **Step 8: Run `python3 -m unittest tests.test_verify -v`; originals and relabelings must pass, tampered results must fail.**

---

### Task 8: GPT-ready exports

**Files:** Create `bl_rctn/export.py` and `tests/test_export.py`.

- [ ] **Step 1: Write tests requiring verified-only export and exact artifact names.** An unverified report, wrong `case_id`, mismatched case/hierarchy/config digest, or non-`RECURSIVE_VERIFIED` completion level must raise before creating the destination.
- [ ] **Red gate: Run `python3 -m unittest tests.test_export -v`.** Expected failure is the missing `bl_rctn.export` module; all earlier task tests must still pass.

~~~python
# tests/test_export.py
import tempfile, unittest
from dataclasses import replace
from pathlib import Path
from bl_rctn.export import export_case_bundle
from bl_rctn.hierarchy import decompose_case
from bl_rctn.models import DemoConfig
from bl_rctn.samples import build_curated_cases
from bl_rctn.verify import verify_case_result

class ExportTests(unittest.TestCase):
    def test_verified_only_and_exact_files(self):
        config = DemoConfig("bl-rctn-demo-config-v1", (2, 3, 4), 20260811, 0, 4096, 20, "STRUCTURE_ONLY")
        case = build_curated_cases((2,))[0]
        candidate = decompose_case(case, config)
        report = verify_case_result(case, candidate, config)
        scratch = Path.cwd() / "runs" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            paths = export_case_bundle(case, candidate, report, config, Path(raw) / "ok")
            self.assertEqual({p.name for p in paths}, {"gpt_visualization_bundle.json", "gpt_prompt.md", "root_graph.mmd", "hierarchy.mmd", "crossing_graph.mmd", "graph.dot"})
            with self.assertRaises(ValueError):
                export_case_bundle(case, candidate, replace(report, verified=False), config, Path(raw) / "bad")
~~~
- [ ] **Step 2: Before creating the output directory, recompute and compare all `VerificationReport` bindings.** Emit `gpt_visualization_bundle.json` only after the report is verified, bound to the exact case/candidate/config, and marked `RECURSIVE_VERIFIED`.

~~~json
{
  "schema_version":"bl-rctn-gpt-visualization-v1",
  "case":{},
  "root_graph":{"nodes":[],"edges":[]},
  "cutsets":[],
  "separations":[],
  "crossing_edges":[],
  "hierarchy":{"nodes":[],"edges":[]},
  "torsos":[],
  "interfaces":[],
  "ambient_blocks":null,
  "block_localization":null,
  "verification":{},
  "style_hints":{},
  "view_recipes":[]
}
~~~

Each separation stores sides, separator, `is_elementary`, `is_tn`, and `rejection_witness_id`. Each torso edge stores roles as `ROOT_REAL` and/or `VIRTUAL_INTERFACE:<id>`.

- [ ] **Step 3: Fix the color contract:** real edge black solid; virtual edge blue dashed; separator orange; TN green; rejected/crossing red; SMALL grey; HIGH green; CROSSED red; SPLIT blue.
- [ ] **Step 4: Emit `gpt_prompt.md`.** It must request a four-panel academic figure: root/separators, elementary crossing graph, recursive hierarchy, interface provenance. It must forbid invented nodes, edges, cuts, blocks, labels, and explicitly display `STRUCTURE_ONLY; ambient blocks not computed`.
- [ ] **Step 5: Emit deterministic `root_graph.mmd`, `hierarchy.mmd`, `crossing_graph.mmd`, and `graph.dot` with escaped labels and sorted objects.**
- [ ] **Step 6: Emit `summary.csv` with exactly:**

~~~text
case_id,k,n,m,status,cutset_count,full_sigma_count,elementary_count,tn_count,hierarchy_depth,small_terminals,high_terminals,crossed_terminals,verified,bundle_path
~~~

- [ ] **Step 7: Emit `manifest.json` with relative paths, byte sizes, SHA-256, seed, config digest, case count, and Python version. No absolute path may appear in generated artifacts.** Unit-test evidence belongs in the implementing Codex's final report because the demo process must not recursively invoke its own test suite.
- [ ] **Step 8: Run `python3 -m unittest tests.test_export -v`; expected result is PASS.**

---

### Task 9: CLI and one-command demo

**Files:** Create `bl_rctn/cli.py`, `tests/test_cli.py`, `README.md`.

- [ ] **Step 1: Write an end-to-end subprocess test that first creates `Path.cwd() / "runs" / ".tmp"`, then uses `tempfile.TemporaryDirectory(dir=that_path)`, runs a demo there, and asserts `case_count == 18` and `all_verified == true`.** Add negative tests for `../escape`, an absolute path lexically outside the project root, and any symlink path component; the symlink test may point to the inert text `../outside-sentinel` but must not resolve or access that target. Tests must never use the operating-system default temporary directory.
- [ ] **Red gate: Run `python3 -m unittest tests.test_cli -v`.** Expected failure is the missing `bl_rctn.cli` module; all Tasks 1-8 tests must still pass.

~~~python
# tests/test_cli.py
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

class CLITests(unittest.TestCase):
    def test_demo(self):
        scratch = Path.cwd() / "runs" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as raw:
            run_dir = Path(raw) / "demo"
            completed = subprocess.run(
                [sys.executable, "-m", "bl_rctn.cli", "demo", "--config",
                 "configs/demo_suite.json", "--run-dir", str(run_dir)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 18)
            self.assertTrue(manifest["all_verified"])
~~~
- [ ] **Step 2: Implement exactly these commands.**

~~~text
generate-samples --config CONFIG --output CASES_JSONL
run              --input CASES_JSONL --run-dir RUN_DIR --config CONFIG
verify           --run-dir RUN_DIR
export-gpt       --run-dir RUN_DIR
demo             --run-dir RUN_DIR --config CONFIG
~~~

- [ ] **Step 3: Implement the run-state machine.** `run` creates a fresh run root and atomically publishes only `input/` and `engine/`; `verify` atomically adds missing `verification/` reports, but when reports already exist it recomputes them in memory and requires canonical byte equality without writing; `export-gpt` adds only missing `gpt/`, `summary.csv`, and `manifest.json`, and audits rather than overwrites existing artifacts. Thus every file remains write-once while the stages can share one run root.
- [ ] **Step 4: Validate every input, config, output, and run path before opening it.** Normalize lexically against the fixed `algorithm/` root, reject any path outside that root, then use `lstat` only on existing components inside the root and reject every symlink rather than resolving it. This prevents escape without reading an external target. `demo` must build in `algorithm/runs/.tmp/`, run all stages, validate all manifests, and atomically rename to `RUN_DIR` only after 18/18 verification. Existing target means failure.
- [ ] **Step 5: Use this run layout.**

~~~text
runs/<run-id>/
├── input/config.json
├── input/cases.jsonl
├── engine/<case-id>.json
├── verification/<case-id>.json
├── gpt/<case-id>/{bundle,prompt,mermaid,dot files}
├── summary.csv
└── manifest.json
~~~

- [ ] **Step 6: Freeze `input/config.json` before engine execution and bind every candidate/report to its digest.** `manifest.json` lists and hashes every other run artifact but does not list or hash itself; the auditor recomputes the canonical manifest content from the closed artifact set and compares it byte-for-byte.
- [ ] **Step 7: README must state the exact scope:** small-graph XP reference, `STRUCTURE_ONLY`, no ambient blocks, no FPT claim, no SPQR equivalence, finite tests are not a proof.
- [ ] **Step 8: README must give these commands.**

~~~bash
cd /Users/zcahlua/Desktop/8月work/新方法/algorithm
python3 -m unittest discover -s tests -v
python3 -m bl_rctn.cli demo --config configs/demo_suite.json --run-dir runs/demo-20260811
~~~

- [ ] **Step 9: Run `python3 -m unittest tests.test_cli -v`; expected result is PASS.**

---

### Task 10: Real run and acceptance gate

**Files:** Generate only `runs/demo-20260811/**`.

- [ ] **Step 1: Run the complete suite.**

~~~bash
cd /Users/zcahlua/Desktop/8月work/新方法/algorithm
python3 -m unittest discover -s tests -v
~~~

Expected: zero failures and zero errors. Do not claim a test count before this command produces it.

- [ ] **Step 2: Run the immutable demo.**

~~~bash
python3 -m bl_rctn.cli demo --config configs/demo_suite.json --run-dir runs/demo-20260811
~~~

Expected: exit `0`, exactly 18 cases, 18 `verified=true` reports, and no run artifact outside `algorithm/runs/demo-20260811/`. Python bytecode caches may exist only below `algorithm/` and are excluded from the run manifest.

- [ ] **Step 3: Rerun the audit.**

~~~bash
python3 -m bl_rctn.cli verify --run-dir runs/demo-20260811
~~~

Expected: all file hashes, relative paths, case/result bindings, summaries, reconstruction checks, and GPT bundles validate.

- [ ] **Step 4: Inspect `summary.csv`.** For each `k` there must be one SMALL, one HIGH, one CROSSED, and three SPLIT cases. Counts must match the analytic table in Task 3.
- [ ] **Step 5: Read only generated JSON/Mermaid inside the run.** Inspect one case from each terminal class plus one shared-boundary split; confirm virtual roles, distinct equal-boundary interfaces, and the `STRUCTURE_ONLY` warning.
- [ ] **Step 6: Claim completion only if every line below is true.**

~~~text
full unittest suite passed
18/18 curated cases verified
all analytic expectations matched
terminal-only reconstruction matched every root graph
all relabeling metamorphic checks passed
all manifest hashes and bindings validated
GPT bundles were exported only from verified results
no project/data/artifact file was read from or written to outside algorithm/ (Python standard-library runtime excepted)
~~~

## Explicitly deferred

V1 must not claim or silently implement:

- `WITH_AMBIENT_BLOCKS`, ambient `(k+1)`-block listing, or localization;
- FPT minimum-cut listing or end-to-end FPT complexity;
- general-`k` SPQR/Tutte classification or S/P/Q/R normalization;
- benchmark-scale or near-linear performance;
- natural recursion-depth-at-least-two coverage from the 18-case demo; V1 verifies depth-one recursion plus targeted full/partial propagation unit tests and must record the maximum observed demo depth;
- canonical literal IDs across independently relabeled inputs;
- a general mathematical proof derived from finite examples.

## Plan self-review

Before implementation handoff, confirm:

- [ ] Every planned path is inside `algorithm/`.
- [ ] Every public signature is defined once and used consistently.
- [ ] No third-party import or external service appears.
- [ ] No unspecified implementation placeholder, silent fallback, or incomplete-success state appears.
- [ ] Full, pairwise, and aggregated TN are independently compared.
- [ ] Engine candidates cannot reach GPT export before verifier success.
- [ ] Every visualization bundle says ambient blocks were not computed.
