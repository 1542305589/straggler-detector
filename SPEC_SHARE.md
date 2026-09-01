# SPEC_SHARE — Slow Node Detection (亚健康检测) Skill 完整规格

> 本文件是该 Claude Code skill 的**唯一可分享规格**。外部助手应仅凭此文件还原（重建）整套 skill，
> 包括全部 Python 源码、算法与运行方式。规格以“当前实际代码行为”为准（已随代码演进更新）。

---

## 1. 用途概述

在 **AI 训练 / 推理集群** 上检测**慢节点（Straggler）**，即亚健康检测。
输入为 Ascend PyTorch Profiler 生成的 `.db`（SQLite）原始数据，解析后对以下 9 类指标做通用异常检测：

- 慢计算卡（`KERNEL_AICORE`）
- 慢通信域（`comm`，即 `{xp}_Duration`）
- 慢通信计数（`xp_count`，即 `{xp}_Count`）
- Step 总时长差异（`step_duration`）
- 矢量计算（`kernel_aivec`）
- 内存搬运（`memcpy_async`）
- NPU 空泡（`npu_bubble`，单阈值）
- Host 端耗时（`host_duration`）
- 慢 CPU 卡（`cpu`）

核心算法为 **KMeans + Z-score + 肘部法（elbow）** 的通用异常检测，配合通信域组间对比、节点对齐等方法。
最后输出 JSON 检测结果 + 故障联合分析日志 + Markdown 可视化报告。

**已删除指标（历史演进，勿恢复）**：`ai_core`、`SDMA`、卡级通信算子类型 `comm_{optype}`（如 `allReduce_Duration`），
以及“检出计算异常则跳过卡级通信检测”的 `has_compute_anomaly` 守卫。

---

## 2. 运行时/工程要求

- **纯 Python 标准库**实现，零第三方依赖：`sqlite3`、`csv`、`json`、`math`、`random`、`logging`、`os`、`sys`、`shutil`、`collections`、`dataclasses`、`typing`、`datetime`。
- **确定性 / 可复现**：随机种子固定 `seed=42`。
- Windows 控制台可能有 GBK 编码乱码，仅影响显示，不影响功能。
- 整体命名与注释使用中文，模块与函数命名对应一份 Go 参考实现（`nodelevel`、`nodelevel_data_handler`、`profilingdataparse`、`utils`、`joint_analysis`）。

---

## 3. 模块清单与职责

| 文件 | 职责 |
|---|---|
| `main.py` | 主入口：CLI `main()` 与 Python API `run_detection()`；多 job 处理 |
| `config.py` | 全局配置与运行时状态（阈值、标志、map） |
| `profilingdataparse.py` | `.db` → `op_metric/global_rank_*.csv` + `op_metric/group_info_*.json`；从 DB 各表计算指标 |
| `nodelevel_data_handler.py` | 读中间 CSV/JSON，构建 `parallels`（并行域）、`valid_ranks`、最新 step 快照；判定集群数据、节点回退分组 |
| `nodelevel.py` | 核心检测：`delimit_detection` + 各指标检测函数、检测组选择 |
| `kmeans_detector.py` | `general_anomaly_detection`：KMeans++/Z-score/肘部法/递归 |
| `joint_analysis.py` | 生成故障联合分析报告 `joint_failure_analysis.log` |
| `visualizer.py` / `markdown_viz.py` | 控制台实时反馈 + Markdown 报告 `analysis_result/detection_report.log` |
| `utils.py` | 结果 JSON 写出、清理、交互提问、批量结果辅助 |

依赖方向：`main → utils/config → (profilingdataparse, nodelevel_data_handler, nodelevel, visualizer, joint_analysis)`；
`nodelevel → (config, kmeans_detector, nodelevel_data_handler)`；`visualizer → markdown_viz`，`markdown_viz → (config, utils)`。

---

## 4. 输入数据格式

`path` 可以是含多层子目录的父目录（`os.walk` 递归查找）。要求的原始文件：

```
ascend_pytorch_profiler_<rank>.db    # 每张卡一个 SQLite 数据库；跳过 0 字节空文件
```

常见 dump 结构（可选，非必需）：
`<path>/master_xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_<rank>.db`。

**规则**：仅使用 `ascend_pytorch_profiler_*.db` 原始数据；**禁止**使用他人处理的中间数据
（如 `analysis.db`、`cluster_analysis.db`）；部分 `master_*` 目录下空 master db（无核心表）会被跳过。

### 4.1 使用的 DB 表（SQLite schema）

| 表 | 字段（用到的） | 用途 |
|---|---|---|
| `STEP_TIME` | `id, startNs, endNs` | step 起止时间 |
| `STRING_IDS` | `id, value` | 全局 id↔字符串名称表（groupName、算子名解析） |
| `COMMUNICATION_OP` | `opName, startNs, endNs, connectionId, count, groupName, _rowid_` | 通信算子明细（含域、次数） |
| `CANN_API` / `PYTORCH_API` | `startNs, endNs, connectionId` | Host 侧 API（取 Host 耗时） |
| `MSTX_EVENTS` | `message, startNs, endNs, connectionId` | Host 侧事件（取 Host 耗时、DataLoader） |
| `TASK` | `startNs, endNs, taskType, connectionId` | 设备侧任务；按名字求平均核时（KERNEL_AICORE/AIVEC/MEMCPY_ASYNC） |
| `HOST_INFO` | `hostUid, hostName` | 每卡仅一行；用于按物理节点分组 |
| `META_DATA` | `name, value` | 其中 `name='parallel_group_info'` 为并行域拓扑 |

注意：
- `TASK` 表行数通常比 `CANN_API`/`PYTORCH_API`/`COMMUNICATION_OP` 多很多，因为包含大量内部同步任务；核任务按名匹配。
- **hostUid 分组**：`HOST_INFO.hostUid` 相同的 rank 视为同一物理节点，用于“无命名通信域”时的检测组退化与 CPU 节点对齐。

---

## 5. 解析（profilingdataparse.py）→ 中间文件

### 5.1 单卡处理流程 `process_database`
1. 开库，`PRAGMA journal_mode=WAL`；建索引 `STRING_IDS(value)`、`TASK(startNs,endNs,taskType)`。
2. `detect_job_type(conn)` → 写 `config.JobType`（training/rollout）：查 `PYTORCH_API` 中优化器更新算子（`.step`/`.zero_grad`）判断。
3. 从文件名 `ascend_pytorch_profiler_<rank>.db` 提取 `global_rank`。
4. 读 `META_DATA` 中 `parallel_group_info` → 写 `op_metric/group_info_<rank>.json`。
5. `get_host_info(conn, rank)`：`SELECT hostUid FROM HOST_INFO LIMIT 1` → `config.set_host_rank_map(rank, hostUid)`（仅内存）。
6. `get_all_step_times()`：优先 `STEP_TIME`（`SELECT id,startNs,endNs FROM STEP_TIME ORDER BY id DESC`）；`STEP_TIME` 缺失则回退从 `TASK`/`MSTX_EVENTS` 推导。
7. 构造**聚合 step**：`start = 所有 step 最小 startNs`，`end = 所有 step 最大 endNs`，`id=0`。
8. `time_diff_for_step(...)` 计算所有指标 → `write_results_to_csv` 写 `op_metric/global_rank_<rank>.csv`（当前仅写这一行聚合 step）。

### 5.2 `time_diff_for_step` 指标计算（统一无效标记 `INVALID_MARKER = -99999`）

无通信域/无通信算子时的**回退**（保留 `zp_host`、`zp_kernel`，其余通信字段标 `-99999`）：

- 构造 `group_name → id` 映射（查 `STRING_IDS`）。
- `get_device_op_list`：`SELECT ... FROM COMMUNICATION_OP WHERE groupName IN (...) AND startNs>=? AND endNs<=? ORDER BY startNs`。
- Host 时间：按 `connectionId` 匹配 `CANN_API`，否则 `MSTX_EVENTS`。

逐项指标：
| 指标字段 | CSV 列 | 计算方式 |
|---|---|---|
| `step_duration` | `StepDuration` | 聚合 step 的 `endNs - startNs` |
| `zp_device` | `ZP_Device` | 非通信时间 = `step_duration − 合并后通信总时长` |
| `zp_duration` | `ZP_Duration` | 通信总时长 = `merge_intervals_simple(comm_intervals)`（区间合并） |
| `zp_host` | `ZP_Host` | `mean(所有 host 耗时)`，host 耗时 = 通信算子 Host 侧 + `get_kernel_host_durations`（可靠性兜底） |
| `zp_bubble` | `ZP_Bubble` | `mean(bubble)`，`bubble = op.start_ns − op.h_end_ns`（仅 `>0` 计入） |
| `zp_count` | `ZP_Count` | 字段默认 `0`（当前解析未赋值） |
| `zp_kernel` | `KERNEL_AICORE` | `get_avg_kernel_task_duration`：`SELECT AVG(t.endNs-t.startNs) FROM TASK t ...`（KERNEL_AICORE 核任务） |
| `memcpy_async` | `MEMCPY_ASYNC` | `get_avg_task_duration_by_name(...,"MEMCPY_ASYNC")` |
| `kernel_aivec` | `KERNEL_AIVEC` | `get_avg_task_duration_by_name(...,"KERNEL_AIVEC")` |
| `host_duration` | `HostDuration` | `mean(通信算子侧 host 耗时)`，不含 kernel（区别于 zp_host） |
| `data_loader` | `DataLoader` | `MSTX_EVENTS` 中 DataLoader 事件时长 |
| `durations[xp]` / `counts[xp]` | `{xp}_Duration` / `{xp}_Count` | 按 `domain_id→xp` 分组（合法 xp 集合见下），`calculate_mid_mean_pair` 求中位数-均值对 |

合法 xp（域）集合：`{"tp","ep","exp","pp","cp","tp_exp","dp_modulo_exp_cp","embd","mc2","dp"}`。

### 5.3 CSV schema（`op_metric/global_rank_<rank>.csv`）
表头固定：
```
StepIndex, StepDuration, ZP_Device, ZP_Duration, ZP_Host, ZP_Bubble, ZP_Count,
KERNEL_AICORE, MEMCPY_ASYNC, KERNEL_AIVEC, HostDuration, DataLoader,
<xp1>_Duration, <xp1>_Count, <xp2>_Duration, <xp2>_Count, ...
```
其中 `<xp>` 为该 rank 出现的所有合法域的并集（排序后）。当前仅写 1 行（聚合 step，`StepIndex=0`）。

### 5.4 `group_info_<rank>.json`
来自 `META_DATA.parallel_group_info` 的拓扑，形如：
```json
{
  "<domain_key>": {"group_name": "<名字>", "global_ranks": [0, 1, 4, 5]}
}
```
`group_name` 可能为**空字符串**（无命名通信域场景）。

---

## 6. 检测信息构建（nodelevel_data_handler.py）

### 6.1 `get_cur_detection_info(job_path)` → `(parallels, valid_ranks)`
1. 枚举 `op_metric/group_info_*.json` → `valid_ranks`。
2. 收集所有 `group_name`；**空域名不做分组推导**（无法确认 tp/ep/cp）。
3. 对每个**非空** `group_name`，用 `get_detection_job_parallel_info` 拼组（分组内卡数 `>1` 才保留）。
4. `config.set_has_named_domain(any(name for name in parallels))` —— 是否存在命名通信域。
5. 排序 `valid_ranks`。
6. `determine_cluster_data(...)` → 写 `config.IsClusterData`。

### 6.2 判定集群数据（`determine_cluster_data`）
三条件全满足才 `IsClusterData=True`：
1. 存在 `group_info_*.json`；
2. `group_info` 文件数 == `global_rank_*.csv` 文件数；
3. 由所有 group_info 拼出的完整通信域卡数（去 `-1` 占位）== group_info 文件数。

### 6.3 最新 step 快照（`get_cur_job_last_step_data`）
对每个指标列与每张卡的时间序列：长度 `>1` 取**倒数第二个**，长度 `==1` 取第一个，长度 `0` 忽略。
返回 `{metric: {rank: value}}` 单快照。

### 6.4 节点回退分组（`_build_node_fallback_groups`）
用 `config.HostRankMap`（rank→hostUid）把相同 hostUid 的 rank 归组；组内 `>=2` 卡才保留（单卡组丢弃），返回分组列表；无映射则 `[]`。

---

## 7. 检测组选择（nodelevel.py `get_cal_detection_group`）

并行域检测优先级（高→低）：
```
tp → exp → ep → tp_exp → cp → cp2 → cp_ulysses → cp_ring → dp → dp_cp → dp_modulo_exp_cp
```

- **命中优先级域**：
  - 集群数据（Case A）：返回该域的**完整集群分组**，不过滤节点。
  - 非集群数据（Case B）：用 `get_detection_groups(parallel_info, cur_npus)` 过滤到本地节点。
- **空域名 `""` 分支**（`parallels[""]` 由节点回退填出）：同样按集群/非集群处理。
- **未命中任何优先级域**（情况 B，有命名域但都不在优先级）：检测组退化=`_build_node_fallback_groups(cur_npus)` 节点分组（**不再短路为空**）。都失效才 `return "", []`。

**HasNamedDomain 语义**（`config.get_has_named_domain`）：
- `True` → 通信域组间指标（`comm`/`step_duration`/`xp_count`）正常检测。
- `False`（情况 A：全空域名）→ 通信域组间指标**直接跳过**（无法解释 tp/ep）；单卡指标在节点组内检测，Host 维持节点间拉齐，Bubble 维持固定阈值。

---

## 8. 主检测流程（nodelevel.py `delimit_detection`）

输入：`step_data`、`parallels`、`valid_ranks`。返回 `DegradationData`（`{category: {key: degradation}}`）。

执行顺序：
1. 取检测组 `cal_detection_group, cal_detection_group_name = get_cal_detection_group(...)`；无效则返回空。
2. **Bubble** `detection_zp_bubble_data`：`value < 5000` 计异常（排除 `-99999` 与 `<=0`）。
3. **KERNEL_AICORE** `get_slow_calculate_ranks`：在检测组逐个组内，对 `KERNEL_AICORE` 列做通用检测（max 方向）。
4. **通用检测** `GENERAL_METRIC_CATEGORIES = [(KERNEL_AICORE→KERNEL_AICORE),(kernel_aivec→KERNEL_AIVEC),(memcpy_async→MEMCPY_ASYNC)]`（KERNEL_AICORE 已在步骤 3 单独处理，跳过列名重复项）。
5. **通信域组间对比**（仅 `HasNamedDomain=True`）：`detection_all_communication_parallel`。
6. **Host/CPU**：`get_slow_host_ranks_by_homogenize`（`cpu`/`ZP_Host`）与 `_get_slow_host_metric_ranks`（`host_duration`/`HostDuration`），均先按物理节点组内均值（去首尾）拉齐再通用检测。

---

## 9. 通用检测算法（kmeans_detector.py `general_anomaly_detection`）

始终 **max 方向**（值偏大为异常）。参数与默认：
`anomaly_multiplier=2.0`、`max_k=10`、`max_iter=300`、`max_depth=10`、`convergence_eps=1e-9`、`seed=42`。

流程（递归）：
1. **过滤** `≤0` 及 `-99999`；过滤后 `<2` 个 → 无异常退出。
2. **Z-score 标准化**；标准差 `≈0` 时强制置 `1` 避免除零。
3. **肘部法选 K**：`K=2..min(n,MAX_K)` 跑 KMeans 得 inertia，取**二阶差分最大**的 K（退化用 2）。
4. **KMeans++ 初始化**：首质心 = `data[0]`，后续 D² 加权随机采样（`random.Random(seed)`）。
5. **Lloyd 迭代** ≤ `max_iter` 轮：最近质心分配 → 质心=簇均值；空簇质心放到离其分配质心最远的样本；收敛 = 质心位移 `<eps` 且无分配变化。
6. **识别异常簇**：按簇原始值均值降序，基线 = 最小均值簇；簇均值 `> 基线×倍率` 判定异常，首个不满足即停止。
7. 无异常簇 → 无异常退出。
8. **递归（剔除式）**：检出异常簇后**剔除**该簇，对**剩余数据**回到步骤 2 递归聚类（`depth+1 ≤ max_depth`）；逐层累积所有层的异常样本，直至剩余检不出异常或达到深度上限。
9. 返回异常 rank：累积所有层异常样本，`degradation = value / 最深层基线`（最深层 = 最后仍有异常那层的基线）；同 label 取较大值。

### 9.1 倍率（阈值）来源（config.py）
由运行时 `set_thresholds(degradation)` 设置：
- 计算/IO/Host 类：`multiplier_scale=1.0` → 倍率 = `1 + 1×degradation`（KERNEL_AICORE、kernel_aivec、memcpy_async、cpu、host_duration、主机拉齐）。
- 通信域类：`multiplier_scale=5.0` → 倍率 = `1 + 5×degradation`（comm、step_duration、xp_count 的组间对比）。
- `npu_bubble`：固定硬阈值 `value < 5000ns`（非倍率）。

### 9.2 组间对比（`detection_all_communication_parallel` / `_detect_comm_group_metric`）
- 对每个并行域，取各分组内该指标**值最小的有效卡**作为代表，跨组做通用检测；慢代表映射回整个通信域组（`add_group`）。
- `comm` ← `{name}_Duration`；`step_duration` ← `StepDuration`；`xp_count` ← `{name}_Count`。
- PP 域用 `cal_detection_group` 内比较并映射回 PP 域；`embd` 域跳过。
- 双保险：`HasNamedDomain=False` 时 `detection_all_communication_parallel` 直接返回，不检测。

### 9.3 其它检测辅助
- `process_cpu_data_by_node`：按物理节点分组，组内去首尾均值后覆盖整组（用于 Host/CPU 拉齐）。
- `minRanksInGroup = 2`（组内至少 2 卡才有聚类区分度）。

---

## 10. Config 参数（config.py）

| 参数 | 含义 | 默认 |
|---|---|---|
| `FilePath` | 输入数据目录 | `""` |
| `OutputPath` | 结果输出目录；空则回退 `FilePath`（单 job） | `""` |
| `Degradation` | 劣化阈值基数（运行时提问，回退 0.3） | `0.3` |
| `Utilization_ComputeMultiplier` | 计算类倍率 = `1+1×degradation` | `0`（运行时算） |
| `Utilization_CommMultiplier` | 通信类倍率 = `1+5×degradation` | `0`（运行时算） |
| `CALC_MULTIPLIER_BASE` / `COMM_MULTIPLIER_BASE` | 放缩基数 | `1.0` / `5.0` |
| `MAX_K` / `MAX_ITERATIONS` / `RECURSION_DEPTH` / `CONVERGENCE_EPS` | 算法上限 | `10`/`300`/`10`/`1e-9` |
| `IsClusterData` | 集群数据标志（检测时判定） | `False` |
| `HasNamedDomain` | 是否存在命名通信域（检测时判定） | `False` |
| `HostRankMap` | `{rank: hostUid}` 内存映射（解析阶段填充，不落盘） | `{}` |
| `JobType` | `training` / `rollout` / `unknown`（解析时判定） | `"unknown"` |

全局状态 API：`set_file_path` / `get_output_path`、`set_is_cluster_data`、`set_has_named_domain`、`set_host_rank_map` / `get_host_rank_map` / `reset_host_rank_map`、`set_job_type`、`set_thresholds`、`get_compute_multiplier` / `get_comm_multiplier`。

`DegradationData`（dict 子类）：
- `add_single(category, rank, degradation)`：键为单 rank 字符串。
- `add_group(category, ranks, degradation)`：键为排序后逗号连接的 rank 串；已存在时取较大值。

---

## 11. 数据类（dataclass）

```
StepTime(id:int, start_ns:int, end_ns:int)
CommunicationOp(start_ns, end_ns, connection_id, count=0, domain_id=0, op_stream_index=0, h_start_ns=0, h_end_ns=0)
HostOp(start_ns, end_ns)
OpStat(duration:int, count:int)
PerformanceMetrics:
    step_index, step_duration, zp_device, zp_duration, zp_host, zp_bubble, zp_count,
    zp_kernel, memcpy_async, kernel_aivec, host_duration, data_loader (int, 默认0)
    durations: Dict[str,int]; counts: Dict[str,int]
```

---

## 12. 运行入口与执行流程

### 12.1 CLI
```bash
python main.py path=/path/to/data degradation=0.3 clean=ask
```
参数：`path`（必需，数据目录）；`degradation`（劣化阈值，默认 0.3，`>0`）；`clean`（`yes`/`no`/`ask`）。

### 12.2 Python API
```python
import main
result = main.run_detection("/path/to/data", degradation=0.3, skip_parsing=False, clean="ask")
# 返回 {"KERNEL_AICORE": {"0": 1.5}, "comm": {"0,1": 1.8}, ...}
```

### 12.3 单 job 流程 `_process_single_job`
1. `config.set_file_path(job_path)`；可选 `set_output_path(output_path)`。
2. **总是交互提问 degradation**（`confirm_degradation`，回车用默认）→ `set_thresholds`。
3. 清理/解析（取决于 `clean`）：
   - `yes`：`clean_detection_outputs` + 重新解析 `data_parsing`。
   - `no`：跳过，直接用已有 `op_metric`。
   - `ask`：`confirm_clean` 交互询问后再决定。
4. `parallels, valid_ranks = get_cur_detection_info(job_path)`；空则失败返回。
5. `last_step_data = get_cur_job_last_step_data(valid_ranks)`。
6. `result = nodelevel.delimit_detection(last_step_data, parallels, valid_ranks)`。
7. `utils.write_result(result, parallels)` → JSON。
8. `joint_analysis.generate_joint_report(result, parallels, last_step_data, out)` → 联合分析。
9. `visualizer.run_visualization(...)` → 报告。

### 12.4 多 job（父目录含多个含 db 的子 job）
父目录含子 job（子目录直接含 db）且自身不含 db → 每个子 job 单独检测，结果统一输出到父目录 `detection_output/<job 名>/`（原始数据目录不写结果）。

### 12.5 清理范围（`clean_detection_outputs`）
`op_metric/`、`straggler_analysis_output/`、`analysis_result/`、`straggler_detection_result.json`、`straggler_detection_result/`。

### 12.6 强制执行规则（交互要求）
每次执行前**必须先询问用户**是否需要删除已有数据（`op_metric` 等中间文件），用户明确答复后再执行，禁止跳过询问。

---

## 13. 输出

1. `op_metric/global_rank_*.csv` — 解析后的性能指标（见 §5.3）。
2. `op_metric/group_info_*.json` — 并行域信息（见 §5.4）。
3. `straggler_detection_result.json` — 检测结果，恒包含 9 类空数组并集实际类别：
   ```json
   {
     "KERNEL_AICORE": [{"display_key": "5", "metric_value": 1.45, "is_abnormal": true}, ...],
     "comm":          [{"display_key": "tp[0, 1]", "metric_value": 1.8, "is_abnormal": true}, ...],
     ...
   }
   ```
   排序：非 bubble 升序（大值），bubble 降序（小值）；组键类别（comm/step_duration/xp_count）`display_key` 带域名 `domain[rank0, rank1]`。
4. `joint_failure_analysis.log` — 故障联合分析：按流水线阶段（计算→通信→等待）归类，根因定位到计算/搬运阶段的异常卡。
5. `analysis_result/detection_report.log` — Markdown 可视化详情报告。
6. 控制台实时输出（各指标柱状、通信域耗时分项/汇总、检测结果摘要）。

---

## 14. 9 类指标检测方式汇总

| 类别 | 指标列 | 检测方式 | 倍率 |
|---|---|---|---|
| `KERNEL_AICORE` | `KERNEL_AICORE` | 检测组内（优先级域/节点组）+ 通用算法 | `1+1×degradation` |
| `kernel_aivec` | `KERNEL_AIVEC` | 检测组内 + 通用算法 | `1+1×degradation` |
| `memcpy_async` | `MEMCPY_ASYNC` | 检测组内 + 通用算法 | `1+1×degradation` |
| `npu_bubble` | `ZP_Bubble` | 单阈值 `< 5000ns` | 硬阈值 |
| `host_duration` | `HostDuration` | 节点对齐（组均值拉齐）+ 通用算法 | `1+1×degradation` |
| `cpu` | `ZP_Host` | 节点对齐（组均值拉齐）+ 通用算法 | `1+1×degradation` |
| `comm` | `{xp}_Duration` | 通信域组间对比（代表卡跨组聚类） | `1+5×degradation` |
| `step_duration` | `StepDuration` | 通信域组间对比 | `1+5×degradation` |
| `xp_count` | `{xp}_Count` | 通信域组间对比 | `1+5×degradation` |

通信域组间指标仅在 `HasNamedDomain=True` 时检测；`False` 时直接跳过（情况 A 无域名退化为节点分组）。

---

## 15. 重要设计备注（避免重建时走偏）

- `hostUid` 节点分组用于两处：（a）无命名域时〔检测组〕退化；（b）Host/CPU 的〔节点均值拉齐〕。
- 空 `group_name` 会导致 `STRING_IDS IN ('')` 匹配失败 → 通信列断链为 `-99999`/空。本 skill 只处理“无域名时的检测语义”，**不修**空域名断链本身。
- 集群数据（Case A）用完整集群分组；非集群（Case B）按节点过滤；未命中优先级再回退节点分组但保留通信域检测。
- 递归检测为**剔除式**：每层剔除异常簇后对剩余数据再聚类，累积所有层异常，degradation 统一用最深层（最后有异常那层）基线重算。
- 解析仅写聚合 step 一行；检测取“准最新”点（时间序列倒数第二个）。
- 维护“技能目录”与“工程镜像目录”两份源码副本，需保持同步（`diff -q` 校验）。
