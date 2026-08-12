# BL-RCTN-V_k 小图参考运行器

本目录提供一个仅依赖 Python 标准库的、确定性的小图参考实现。它使用
`XP_SUBSET` 枚举全部 size-`k` 顶点子集，执行 `STRUCTURE_ONLY` 的
Elementary exact analysis，构造递归 torso/interface provenance，并由独立
verifier 核验后生成 GPT 可视化 JSON、CSV、Mermaid 与 DOT artifacts。

## 范围与限制

- 这是 small-graph XP reference，不是 benchmark-scale implementation。
- V1 仅支持 `STRUCTURE_ONLY`。
- `ambient_blocks` 与 `block_localization` 未计算，导出值固定为 `null`；不支持
  `WITH_AMBIENT_BLOCKS`。
- 本实现不声称具有 end-to-end FPT complexity，也不声称 near-linear
  performance。
- 本实现不等价于一般 `k` 的 SPQR/Tutte decomposition，也不提供
  S/P/Q/R normalization。
- 有限样本、unit tests 与 differential checks 不是一般数学证明。
- 默认 demo 固定使用 seed `20260811`、18 个 curated cases 和
  `max_full_separations=4096`。

## 如何运行程序

### 1. 检查 Python

项目要求 Python `>=3.11`，且不需要安装任何第三方依赖。本机已经使用
Python 3.13 验证：

```bash
python3.13 --version
```

如果你的 `python3 --version` 已经是 3.11 或更高版本，后续命令可以把
`python3.13` 替换为 `python3`。本机默认的 `python3` 是 3.9，因此以下示例
明确使用 `python3.13`。

### 2. 进入程序目录

```bash
cd /Users/zcahlua/Desktop/8月work/新方法/algorithm
```

可以先查看命令帮助：

```bash
python3.13 -m bl_rctn.cli --help
```

### 3. 运行测试

```bash
python3.13 -m unittest discover -s tests -v
```

成功时结尾应显示类似：

```text
Ran 57 tests
OK
```

### 4. 一键运行 18 个测试样本（推荐）

`demo` 会依次完成样本生成、算法运行、独立验证、GPT 可视化文件导出和
manifest 审计：

```bash
python3.13 -m bl_rctn.cli demo \
  --config configs/demo_suite.json \
  --run-dir runs/demo-local-01
```

成功时输出：

```json
{"all_verified":true,"case_count":18,"gpt_artifacts_present":true,"verified_count":18}
```

所有文件都是 write-once。`--run-dir` 必须是尚不存在的新目录；再次实验时请换
一个新名称，例如 `runs/demo-local-02`，不要复用已经生成的
`runs/demo-20260811`。

### 5. 查看结果

假设本次使用 `runs/demo-local-01`，主要结果位于：

```text
runs/demo-local-01/summary.csv
runs/demo-local-01/manifest.json
runs/demo-local-01/verification/
runs/demo-local-01/gpt/<case-id>/
```

每个 `gpt/<case-id>/` 目录都包含：

- `gpt_visualization_bundle.json`：供 GPT 读取的完整结构化数据；
- `gpt_prompt.md`：与 bundle 一起上传给 GPT 的绘图提示词；
- `root_graph.mmd`、`hierarchy.mmd`、`crossing_graph.mmd`：Mermaid 图；
- `graph.dot`：Graphviz DOT 图。

例如，将同一 case 目录中的 `gpt_visualization_bundle.json` 和
`gpt_prompt.md` 一起上传到 GPT，即可要求 GPT 根据真实算法结果生成学术可视图。

### 6. 再次审计已有结果

该命令不会覆盖文件，只会重新计算并核对 cases、hierarchy、verification、GPT
bundle、CSV 和 manifest：

```bash
python3.13 -m bl_rctn.cli verify --run-dir runs/demo-local-01
```

## 分阶段运行

```text
generate-samples --config CONFIG --output CASES_JSONL
run              --input CASES_JSONL --run-dir RUN_DIR --config CONFIG
verify           --run-dir RUN_DIR
export-gpt       --run-dir RUN_DIR
demo             --run-dir RUN_DIR --config CONFIG
```

完整调用需添加 `python3.13 -m bl_rctn.cli` 前缀。例如：

```bash
python3.13 -m bl_rctn.cli generate-samples \
  --config configs/demo_suite.json \
  --output runs/.tmp/cases-local-01.jsonl
python3.13 -m bl_rctn.cli run \
  --input runs/.tmp/cases-local-01.jsonl \
  --config configs/demo_suite.json \
  --run-dir runs/staged-local-01
python3.13 -m bl_rctn.cli verify --run-dir runs/staged-local-01
python3.13 -m bl_rctn.cli export-gpt --run-dir runs/staged-local-01
python3.13 -m bl_rctn.cli verify --run-dir runs/staged-local-01
```

`run` 只发布 frozen inputs 与 engine candidates；`verify` 只增加或逐字节核对
verification reports；`export-gpt` 只从 verified results 生成可视化 artifacts。
所有 artifacts 均 write-once：已有文件必须与内存重算结果逐字节相同，绝不覆盖。

## Run layout

```text
runs/<run-id>/
├── input/config.json
├── input/cases.jsonl
├── engine/<case-id>.json
├── verification/<case-id>.json
├── gpt/<case-id>/
│   ├── gpt_visualization_bundle.json
│   ├── gpt_prompt.md
│   ├── root_graph.mmd
│   ├── hierarchy.mmd
│   ├── crossing_graph.mmd
│   └── graph.dot
├── summary.csv
└── manifest.json
```

`manifest.json` 使用相对路径列出并哈希其他 run artifacts，但不列出或哈希自身。
CLI 仅接受 lexical containment 在本 `algorithm/` 目录内且不存在 symlink path
component 的路径。`demo` 在 `runs/.tmp/` 中完成 engine、verification、export 与
最终只读 audit，18/18 verified 后才原子发布目标目录。
