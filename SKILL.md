---
name: straggler-detector
description: Detect slow nodes (stragglers) in AI training/inference clusters from Ascend PyTorch Profiler .db files. Use when the user wants to run slow-node/亚健康 detection, parse ascend_pytorch_profiler_*.db files, identify slow computing cards (KERNEL_AICORE), slow communication domains (comm), slow CPU cards (cpu), NPU bubbles, or analyze compute/IO/communication metric breakdown across ranks. Performs KMeans + Z-score + elbow general anomaly detection over metric classes and produces a joint failure analysis report.
---

# straggler-detector

Slow Node Detection 算法的 Python 实现，用于检测 AI 训练/推理集群中的慢节点（亚健康检测）。

本 skill 自带完整可运行代码包（本目录下的 `*.py`）。

## 触发条件

当用户需要：
- 执行慢节点 / 亚健康检测分析
- 解析 Ascend PyTorch Profiler 的 `.db` 文件
- 识别慢计算卡、慢通信域、慢 CPU 卡
- 看各卡在通信类 / 计算类 / IO 类指标下的耗时分布与异常

## 检测的 7 类指标

| 类别 | 指标列 | 检测方式 |
|---|---|---|
| `comm` | `{domain}_{opType}_{count}`（带宽） | 带宽聚类（min 方向，多 opType 交叉验证） |
| `pp_comm` | `PP_Wait`（PP 接收后集合通信等待） | 按 PP 组求和后取差值占比（Δ/会话 > 5%） |
| `KERNEL_AICORE` | `KERNEL_AICORE` | 检测组内 + 通用算法 |
| `kernel_aivec` | `KERNEL_AIVEC` | 检测组内 + 通用算法 |
| `memcpy_async` | `MEMCPY_ASYNC` | 检测组内 + 通用算法 |
| `npu_bubble` | `ZP_Bubble` | 单阈值 < 5000ns |
| `cpu` | `ZP_Host` | 节点对齐 + 通用算法 |

## 使用方法

### 正确运行目录

所有命令都应在**本 skill 目录**（含 `main.py` 的目录）内执行：`cd "C:\Users\n30082019\.claude\skills\straggler-detector"`。

### 命令行执行

```bash
python main.py path=/path/to/data degradation=0.3 clean=ask
```

参数说明：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `path` | 数据目录路径（必需） | - |
| `degradation` | 劣化阈值 | 0.3 |
| `clean` | `yes`/`no`/`ask`：是否清理中间数据并重新解析 | `ask` |

### 作为 Python 模块调用

```python
import main
result = main.run_detection("/path/to/data", degradation=0.3, clean="ask")
```

## 执行规则（重要）

### 必须在执行前询问用户

每次执行检测前，**必须依次询问以下参数**，等用户明确答复后再执行，**禁止跳过询问环节**：

1. **劣化阈值 degradation**（默认 `0.3`，回车/不指定则用默认）：决定异常判定倍率——计算/IO/Host 类倍率 = `1+1×degradation`，内存搬运（memcpy_async）与慢 CPU（cpu）= `1+5×degradation`，慢通信（comm）用固定带宽比率阈值 `1.3`，`npu_bubble` 固定阈值 `< 5000ns`。用户给出数值后，命令中加 `degradation=<用户值>`。
2. **是否删除已有数据**（op_metric 等中间文件）：
   - 用户选"删除" → 加 `clean=yes`
   - 用户选"保留" → 加 `clean=no`

### 清理范围

若用户选删除，会清理：`op_metric/`、`straggler_analysis_output/`、`analysis_result/`、`straggler_detection_result.json`、`straggler_detection_result/`，然后从 `.db` 重新解析。

## 输入数据

`path` 可以是一个含多层子目录的父目录（`os.walk` 递归查找 `ascend_pytorch_profiler_*.db`），典型的：
```
<path>/
├── ascend_pytorch_profiler_0.db
├── ascend_pytorch_profiler_1.db
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
4. `joint_failure_analysis.log` — 故障联合分析报告
5. `analysis_result/detection_report.log` — 可视化详情报告
6. `analysis_result/detection_report.html` — 自包含 HTML 报告（内嵌 matplotlib 柱状图，异常卡标红；matplotlib 未安装时降级为纯表格，不报错）
7. **最终输出逐类别汇总表**（渲染到调用方 agent 的 stdout，不进任何 log 文件）——检测结束时在 stdout 打印一张 Unicode 框线表格，一行一个"有异常的类别"，列：`类别 | 异常卡 | 劣化指数 | 劣化阈值 | 数据要点`；无异常时打印含"无异常"提示的单行表。

## 算法说明

检测核心为 `kmeans_detector.py` 的 `general_anomaly_detection`：过滤 ≤0/-99999 → Z-score → 肘部法选 K → KMeans++ → 偏大方向异常簇（簇均值 > 基线×倍率）→ **异常簇递归细分**（对异常簇数据再次聚类，更深层异常替换父层、更深层无异常保持父层，减少误检；劣化指数统一用**第一次 KMeans（全数据）的基线簇均值**为分母，degradation = 异常值/第一次基线，同一刻度可比）。异常倍率由 `degradation` 决定：计算/IO/Host 类 = `1+1×degradation`，内存搬运（memcpy_async）与慢 CPU（cpu）= `1+5×degradation`，慢通信（comm）用固定带宽比率阈值 `1.3`；`npu_bubble` 用固定阈值 `< 5000ns`。检测组由 `nodelevel.get_cal_detection_group` 按优先级（tp→exp→ep→…→dp）选定，集群数据用完整分组、非集群按节点过滤，无命名域时退化按 hostUid 物理节点分组。

**PP 慢通信（`pp_comm`）另一方案**：PP 传输（Send/Recv）不在带宽白名单内，单独检测（只看 **Recv**，发送端不计）。解析后回填每卡「PP 接收(Recv)后**严格下一次**集合通信时长之和」（`PP_Wait`），同属一个 PP 组的卡求和 `d_g`；慢 PP 组接收方晚进集合通信 → `d_g` 偏小 → `Δ_g = max(d) − d_g` 最大；以 `Δ_g / 会话时长 > PP_WAIT_THRESHOLD`（默认 5%）判异常。

**PP 分组反推（关键）**：Ascend profiler 的 `parallel_group_info` 只登记集合通信域（tp/ep/mc2…），**不登记 `pp` 域**。因此 PP 分组由 `profilingdataparse.derive_pp_groups` 从各 rank 的 `COMMUNICATION_OP` 点对点 Send/Recv 算子反推：相邻 PP stage 的收发两端共享同一 PP 传输流号（`hcom_send__<flow>_*` / `hcom_receive__<flow>_*` 的 `<flow>`），按流号把 rank 分组即得 PP 组（如 tp4pp2 8 卡下流号 486/384/855/471 → `[[0,4],[1,5],[2,6],[3,7]]`）。该分组在 `get_cur_detection_info` 中注入 `parallels["pp"]`，供 `pp_comm` 检测使用。
