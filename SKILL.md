---
name: straggler-detector
description: Detect slow nodes (stragglers) in AI training/inference clusters from Ascend PyTorch Profiler .db files. Use when the user wants to run slow-node/亚健康 detection, parse ascend_pytorch_profiler_*.db files, identify slow computing cards (KERNEL_AICORE), slow communication domains (comm), slow CPU cards (cpu), NPU bubbles, or analyze compute/IO/communication metric breakdown across ranks. Performs KMeans + Z-score + elbow general anomaly detection over metric classes and produces a joint failure analysis report.
---

# straggler-detector

Slow Node Detection 算法，用于检测 AI 训练/推理集群中的慢节点（亚健康检测）。

本 skill 自带完整可运行代码包（本目录下的 `*.py`）：**Python 3.8+，仅标准库**
（`sqlite3` / `csv` / `json` / `os` / `logging` / `math` / `random`），无第三方依赖、不需要编译。

## 触发条件

用户通常用**自然语言**描述问题（例如「部分卡明显偏慢、step 耗时变高，看看是哪里拖慢了性能」），
**不需要用户自己写代码或敲命令**——由 agent 识别意图后按本文件的流程执行。

**走「检测流程」**（跑检测 / 重新检测 / 解析 `.db`）——用户需要：
- 执行慢节点 / 亚健康检测分析
- 解析 Ascend PyTorch Profiler 的 `.db` 文件
- 识别慢计算卡、慢通信域、慢 CPU 卡
- 看各卡在通信类 / 计算类 / IO 类指标下的耗时分布与异常

**走「结果分析层」**（不重跑检测，只解读已有结果）——用户说：
- 分析检测结果 / 为什么慢 / 根因是什么 / 谁慢

## 结果分析层（解读已有结果，不重跑检测）

- **触发词**：「分析检测结果 / 为什么慢 / 根因是什么 / 谁慢」→ 走本层；「重新检测 / 跑检测」→ 走检测流程。
- **输入**：只读 SQL 读取原始 `ascend_pytorch_profiler_*.db` / `analysis.db` / `cluster_analysis.db`（`diagnosis/evidence.py`），并复用检测产物 `straggler_detection_result.json`。
- **编排**：取证 → 规则 → 校验门 → 落盘只在 `diagnosis/pipeline.py` 一处；`main.py`（检测后自动诊断）与 `diagnose.py`（独立入口）共用，两条路径都会跑校验门。
- **权威顺序**：检测结果（`straggler_detection_result.json`，基于算子耗时 / 通信带宽）**优先**；分析层只做**辅助**，不改变 `verdict / primary / actions`；检测层无异常时才兜底，且置信度最多 `medium`。控制台会打印「结论来源」和「辅助发现（不改变检测结论）」。
- **硬约束**：只读、不写库、不改检测数据；正式结论**只由** `diagnosis/rules.py` 的规则产出；LLM 只能在报告第 9 节「推测与待验证假设」里下钻（标 `confidence=speculative`），校验门禁止其进入 `verdict/primary/actions`。
- **规则库**：A 覆盖/一致性、B 通信慢组、C 计算/搬运、D Host/气泡/PP、E 元规则（由旧分析层迁移，统一读 `evidence.json`）；F1 Host 阻塞（伪快卡）、F2 纯计算慢卡、F3 负载不均、F4 通信慢链路、F5 纯等待、F6 频率异常（新增，来自裸 db 证据）。
- **单独运行**（不重跑检测、不改检测代码）：

```bash
python diagnose.py path=<数据目录 | cluster_analysis.db> out=<输出目录>
# 可选阈值：compute=1.3 io=2.5 comm=1.3
# cluster 单文件默认跳过逐 op 等待分解；加 full-cluster 打开（较慢）
```

  `diagnose.py` 默认**逐项询问**阈值、是否清理已有数据、是否关闭 no-align（levelnone/level0）；非交互加 `no-ask`，并用 `clean=yes|no`、`no-align` 显式指定。
- **产物**（输出到 `<out>/`，默认 `<数据目录>/`，全部同级）：
  1. `evidence.json` — 确定性证据（`analysis.db` 优先 + 裸 profiler db 回退 + cluster db）
  2. `formal_result.json` — 正式结论（每条带 `rule_id` + `evidence_refs`，并含校验门结果 `validation`）
  3. `diagnosis_report.md` — 固定骨架报告（第 9 节为推测区）
- **详细规格**（规则库全表、输出契约、置信度、校验门、回归样例）见 `references/analysis_layer.md`；JSON schema 见 `references/output_schema.json`。

## 检测的 7 类指标

| 类别 | 指标列 | 检测方式 |
|---|---|---|
| `comm` | `{domain}_{opType}_{count}`（带宽） | 带宽聚类（min 方向，多 opType 交叉验证） |
| `pp_comm` | `PP_WaitGap`（点对点结束→下一次集合通信结束间隔） | 比较域（TP>CP>DP）内逐 rank，(组内最大−本值)/会话 > 5%；按 PP 通信组上报（取组内最大） |
| `KERNEL_AICORE` | `KERNEL_AICORE` | 检测组内 + 通用算法 |
| `kernel_aivec` | `KERNEL_AIVEC` | 检测组内 + 通用算法 |
| `memcpy_async` | `MEMCPY_ASYNC` | 检测组内 + 通用算法 |
| `npu_bubble` | `ZP_Bubble` | 单阈值 < 5000ns |
| `cpu` | `ZP_Host` | 节点对齐 + 通用算法 |

> 数据级别影响检测范围：`level0`/`level1` 做全部 7 类；`levelnone`（TASK 仅 `MsTx`，无计算/搬运算子）跳过 `KERNEL_AICORE`/`kernel_aivec`/`memcpy_async`；`cluster` 单文件仅 `comm`。

**数据级别**（自动识别，决定 `comm` 的口径与检测范围）：

| 级别 | 触发条件 | 检测范围 |
|------|----------|----------|
| `level0` | 目录；`ascend_pytorch_profiler_*.db` 的 `TASK.taskType` 含 `KERNEL_AICORE`/`KERNEL_AIVEC`/`MEMCPY_ASYNC` | 全部 7 类 |
| `levelnone` | 目录；`TASK.taskType` 仅 `MsTx`（无计算/搬运算子） | 跳过 `KERNEL_AICORE`/`kernel_aivec`/`memcpy_async`；其余照跑 |
| `level1` | profiler db 同目录有 `analysis.db` 且含 `CommAnalyzerBandwidth` 带宽 | 7 类，但 `comm` 用新两级带宽口径，且跳过 `pp_comm` |
| `cluster` | `path` 是单文件 db，含 `CommunicationGroupMapping` 与 `ClusterCommunicationBandwidth` | 仅 `comm`（新两级带宽口径） |

## 使用方法

### 正确运行目录

所有命令都应在**本 skill 目录**（含 `main.py` 的目录）内执行。agent 调用时使用本 skill 自身的目录（`SKILL.md` 所在目录），**不要写死任何绝对路径**。

### 命令行执行

```bash
python main.py path=/path/to/data compute=1.3 io=2.5 comm=1.3 clean=ask
# 可选：追加 no-align 关闭慢通信带宽的跨 rank 对齐
```

参数说明：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `path` | 数据目录，或 cluster db 单文件（文件名不限，含 `CommunicationGroupMapping`/`ClusterCommunicationBandwidth` 表即可）（必需） | - |
| `compute` | 计算类阈值（KERNEL_AICORE / kernel_aivec） | 1.3 |
| `io` | IO/CPU 类阈值（cpu / memcpy_async） | 2.5 |
| `comm` | 通信类阈值（comm / pp_comm） | 1.3 |
| `clean` | `yes`/`no`/`ask`：是否清理中间数据并重新解析 | `ask` |
| `no-align` | 关闭慢通信带宽的跨 rank 对齐（裸参，或 `no-align=true`） | 关（跨 rank 对齐） |

（`npu_bubble` 为固定硬阈值 `< 5000ns`，不询问。）

### 作为 Python 模块调用

```python
import main
result = main.run_detection("/path/to/data", compute=1.3, io=2.5, comm=1.3, clean="ask", no_align=False)
```

## 执行规则（重要）

### 必须在执行前询问用户

每次执行检测前，**必须依次询问以下参数**，等用户明确答复后再执行，**禁止跳过询问环节**：

1. **三组检测阈值**——**逐个询问**（一次只问一个阈值，禁止把多个阈值合成一个问题）：
   - 计算类阈值 `compute`（KERNEL_AICORE / kernel_aivec），固定候选：`1.3`（默认）
   - IO/CPU 类阈值 `io`（cpu / memcpy_async），固定候选：`2.5`（默认）
   - 通信类阈值 `comm`（comm / pp_comm），固定候选：`1.3`（默认）
   - `npu_bubble` 固定硬阈值 `< 5000ns`，不询问。
   - **候选项必须恒定**：同一阈值每次询问展示的候选项完全一致，不要临时增删变化。
   - **不要自行添加「其他」选项**：询问工具自带「输入其他内容」，自定义值走它即可。
   - 用户给出数值后，命令中加 `compute=<值> io=<值> comm=<值>`。
   - **按数据级别决定问哪几组**：`cluster`（单文件 db）只问 `comm`；`levelnone`（TASK 仅 MsTx）只问 `io` + `comm`（`cpu` 用 io、`comm`/`pp_comm` 用 comm，计算类已跳过）；`level0`/`level1` 问 `compute`+`io`+`comm`。
2. **是否删除已有数据**（op_metric 等中间文件）：
   - 用户选"删除" → 加 `clean=yes`
   - 用户选"保留" → 加 `clean=no`
   - **cluster db 单文件无需询问**：无解析步骤、无中间产物，输出文件（`straggler_detection_result.json` / `detection_report.log`）每次直接覆盖写，`clean` 对它无意义。
3. **是否关闭慢通信带宽的跨 rank 对齐**（`no-align`，默认**否**）——**按数据级别决定**：
   - **`levelnone` / `level0`（旧带宽口径）→ 需要问**：
     - 用户选"关闭"（用全体 rank 算子时长的扁平池算带宽，适用于跨 rank 对齐失败的数据）→ 加 `no-align`
     - 用户选"不关闭"（默认，跨 rank 对齐后取组内最短时长）→ 不加
   - **`level1` / `cluster`（新两级带宽口径）→ 无需问**（`no-align` 对它们无意义，直接跳过、不展示此项）。

### 必须在最终回答中原样列出产物绝对路径

检测/诊断结束后，程序会在 stdout 打印 5 行产物路径（**检测结果 / 检测报告 / 诊断报告 / 完整证据 / 诊断结论**），且**已经是绝对路径**。

**最终回答里必须原样带上这 5 行绝对路径**，禁止：
- 只写文件名（如 `straggler_detection_result.json`）或省略目录；
- 省略其中几个、或用"产物：…等"概括。

**原因**：产物目录随数据级别变化——`levelnone`/`level0`/`level1` 落在数据目录（多 job 场景为 job 输出目录），`cluster` 单文件落在 `<db 所在目录>/detection_output/`，`diagnose.py` 由 `out=` 参数决定。**只有绝对路径才能让用户直接找到文件**；用户读的是最终回答，不是 stdout。

### 清理范围

若用户选删除，会清理：`op_metric/`、`straggler_analysis_output/`、`straggler_detection_result.json`、`straggler_detection_result/`、`detection_report.log`、`evidence.json`、`formal_result.json`、`diagnosis_report.md`（以及旧布局遗留的 `analysis_result/`），然后从 `.db` 重新解析。

## 输入数据

`path` 可以是一个含多层子目录的父目录（`os.walk` 递归查找 `ascend_pytorch_profiler_*.db`），典型的：
```
<path>/
├── ascend_pytorch_profiler_0.db
├── ascend_pytorch_profiler_1.db
├── analysis.db            # 可选：含 CommAnalyzerBandwidth 带宽 → level1
└── ...
```
或常见 dump 结构：
```
<path>/
├── master_xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db
├── master_yyy_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db
└── ...
```

### verl colocate（训练 / rollout 混合采集，自动分离检测）

verl 混合部署时，一次采集会在同一节点产出**多组 worker*_ascend_pt 目录**（每组各自独立进程、一个 rank 一个 db）。此时训练（FSDP）与 rollout（vLLM/SGLang）的 rank 编号会重复（如都含 0~3），**绝不能混在一起检测**。

- 传入含同层 `worker*_ascend_pt` 的根目录，检测前会自动用 `classify_role.py` 分层判据（L1 backward 决定性 > L3 行为指纹 > L2 引擎词表）把 worker 分成训练 / rollout 两个"世界"，并做时间窗交叉验证。
- 每个世界**独立解析 + 独立检测**，结果分开输出：
  ```
  <path>/detection_output/training/    # 训练（FSDP）4 rank 完整检测
  <path>/detection_output/rollout/     # rollout（推理引擎）4 rank 完整检测
  ```
- 两世界各含自己的 `op_metric/`，避免 `global_rank_0.csv` 互相覆盖。
- 非 colocate 的普通数据目录不受影响，走原检测流程。

## 重要注意事项

- **禁止创建 `_db` / 软链接中转目录**：不要为了分类或去重创建任何中转目录；直接以原始数据目录作为输入。
- **输入与输出目录分离**：纯结果目录只放检测产物，绝不混入 db 原始数据。
- **空 / master db 处理**：部分 `master_*` 目录下有空的 master db（无 `STEP_TIME`/`PYTORCH_API`/`TASK` 表），应跳过，仅用含核心表的 rank db。
- **无通信域名时（情况 A，group_name 全空）的检测退化**：按 `HOST_INFO.hostUid` 物理节点分组（相同 hostUid 的 rank 为一组）作为检测组，检测计算/IO/通信单卡类指标；通信域组间指标（`comm`，`HasNamedDomain=False` 时）**直接跳过**（无域名无法解释对应 tp/ep）。Host 维持节点间拉齐，Bubble 维持固定阈值。
- **有命名域但未命中检测优先级（情况 B）**：检测组同样退化到物理节点分组，但通信域组间指标**仍然检测**（`HasNamedDomain=True`），检出慢通信组时可带域名。

## 输出

1. `op_metric/global_rank_*.csv` — 解析后的性能指标
2. `op_metric/group_info_*.json` — 并行域信息
3. `straggler_detection_result.json` — 检测结果（含全部类别）
4. `detection_report.log` — 可视化详情报告
5. `evidence.json` / `formal_result.json` / `diagnosis_report.md` — 结果分析层产物
6. **最终输出逐类别汇总表**（渲染到调用方 agent 的 stdout，不进任何 log 文件）——检测结束时在 stdout 打印一张 ASCII 框线表格，一行一个"有异常的类别"，列：`类别 | 异常卡 | 劣化指数 | 劣化阈值`（类别用 op_metric 指标列名，如 `KERNEL_AIVEC`/`MEMCPY_ASYNC`；劣化指数为 `项:值`，通信组如 `tp[0,1]:2.5`、**通信单卡（Level-1 组内）如 `ep[5]:2.5`**、PP 链路如 `0->4:2.5`）；无异常时打印含"无异常"提示的单行表。

## 算法说明

检测核心为 `kmeans_detector.py` 的 `general_anomaly_detection`（KMeans + Z-score + 肘部法 + 异常簇递归细分）。阈值分组：计算类（KERNEL_AICORE / kernel_aivec）= `compute`（默认 1.3），IO/CPU 类（cpu / memcpy_async）= `io`（默认 2.5），通信类（comm / pp_comm）= `comm`（默认 1.3）；`npu_bubble` 固定 `< 5000ns`。

**需要解释「为什么这么判」时，按需加载 `references/detection.md`**：七类指标总览、通用异常算法逐步流程、检测组优先级、`comm` 两套口径的完整公式、`cpu` 节点拉齐机制、`pp_comm` 的 `PP_WaitGap` 定义与比较域、实现文件一览。
