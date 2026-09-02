# straggler-detector 方案设计说明书（SPEC）

> 本文档逐模块阐述 `straggler-detector` skill 的完整方案：架构、数据流、核心算法、各指标的定义与生成方式、检测逻辑、结果输出与故障联合分析。
> 代码版本：Python 移植版，检测核心为 `kmeans_detector.py` 的通用检测算法（KMeans + Z-score + 肘部法 + 逐轮剥离）。

---

## 1. 目标与定位

检测 AI 训练集群中的慢节点（straggler / 亚健康）。基于**单次快照**（single snapshot）的多卡性能数据，识别：

- 慢计算卡（`KERNEL_AICORE`）
- 慢矢量/搬运（`kernel_aivec` / `memcpy_async`）
- 慢通信域（`comm` / `step_duration`）
- 慢 CPU 卡（`cpu` / `host_duration`）
- NPU 空泡（`npu_bubble`）

并基于硬件流水线的因果顺序，推断故障传播链、判定根因卡，生成整体联合分析报告。

**输入**：Ascend PyTorch Profiler 生成的 `.db` 文件（每 NPU 一个）。
**输出**：
- `op_metric/global_rank_{N}.csv`（单快照性能指标）
- `op_metric/group_info_{N}.json`（并行域拓扑）
- `straggler_detection_result.json`（检测结果）
- `joint_failure_analysis.log`（故障联合分析报告，log 格式）
- `analysis_result/detection_report.log`（可视化详情报告）
- **最终输出逐类别汇总表**（Unicode 框线表格，渲染到调用方 agent 的 stdout，**不进任何 log 文件**）

---

## 2. 目录结构与模块职责

```
straggler-detector/
├── __init__.py                 # 包初始化（版本号）
├── config.py                   # 全局配置、阈值、劣化数据容器、节点映射、域名标志
├── utils.py                    # 结果写入、清理、交互式询问、通用工具
├── kmeans_detector.py          # 通用检测算法（KMeans + Z-score + 肘部法 + 逐轮剥离）
├── profilingdataparse.py       # .db(SQLite) → op_metric CSV/JSON
├── nodelevel.py                # 各维度慢节点检测核心逻辑
├── nodelevel_data_handler.py   # 读取 op_metric CSV + group_info JSON，构建快照/并行域/节点组
├── markdown_viz.py             # 生成文本格式检测报告
├── visualizer.py               # 可视化编排（控制台 + 详情报告）
├── joint_analysis.py           # 故障联合分析（传播链 + 根因 + log 报告）
├── main.py                     # 主入口 / skill 调用入口
├── skill.md                    # skill 使用说明（与 SKILL.md 同步）
└── README.md / SPEC.md         # 使用说明 / 本文档
```

---

## 3. 整体数据流

```
ascend_pytorch_profiler_{N}.db（每 NPU 一个）
        │
        ▼
[profilingdataparse.data_parsing]
  └── start_process → process_database（逐库）
        ├── META_DATA.parallel_group_info → group_info_{N}.json
        ├── HOST_INFO.hostUid → config.HostRankMap（内存，不落盘）
        ├── 聚合 STEP_TIME → 单条 aggregated step
        ├── time_diff_for_step() 计算各指标
        └── global_rank_{N}.csv（仅 1 条数据行）
        │
        ▼
[nodelevel_data_handler.get_cur_detection_info]
  ├── 遍历 group_info_*.json → parallels {域名: [[rank组]...]}
  ├── 设置 config.IsClusterData / config.HasNamedDomain
  └── 无命名域时 parallels[""] = _build_node_fallback_groups（按 hostUid 节点分组）
        │
        ▼
[nodelevel_data_handler.get_cur_job_last_step_data]
  └── 读 global_rank_{N}.csv → 单快照 {metric: {rank: value}}
      （多行取倒数第二行；单行取自身；跳过 StepIndex）
        │
        ▼
[nodelevel.delimit_detection]
  ├── detection_zp_bubble_data()             → npu_bubble
  ├── get_slow_calculate_ranks()             → KERNEL_AICORE
  ├── get_slow_metric_ranks() ×2             → kernel_aivec / memcpy_async
  ├── detection_all_communication_parallel() → comm / step_duration（HasNamedDomain 时）
  ├── get_slow_host_ranks_by_homogenize()    → cpu
  └── _get_slow_host_metric_ranks()          → host_duration
        │
        ▼
[utils.write_result]                        → straggler_detection_result.json
[joint_analysis.generate_joint_report]      → joint_failure_analysis.log（含条形图）
[visualizer.run_visualization]              → analysis_result/detection_report.log
```

---

## 4. 配置模块（config.py）

| 全局 | 作用 |
|------|------|
| `FilePath` / `OutputPath` | 输入数据目录 / 输出目录（多 job 场景独立；为空回退到 FilePath） |
| `Degradation` | 劣化阈值基础值，默认 0.3（运行时 `confirm_degradation` 询问） |
| `Utilization_ComputeMultiplier` | 计算/IO/Host 类倍率 = `1 + degradation` |
| `Utilization_CommMultiplier` | 通信域类倍率 = `1 + 5×degradation` |
| `CALC_MULTIPLIER_BASE` / `COMM_MULTIPLIER_BASE` | 放缩基数，1.0 / 5.0 |
| `MAX_K` / `MAX_ITERATIONS` / `RECURSION_DEPTH` / `CONVERGENCE_EPS` | 算法参数：10 / 300 / 10 / 1e-9 |
| `ZP_BUBBLE_ABNORMAL_BOUNDARY` | desc 保留，实际 bubble 用硬编码 5000ns |
| `IsClusterData` | 集群数据标志（Case A 集群 / Case B 非集群） |
| `HasNamedDomain` | 是否有命名通信域标志（决定通信域组间指标是否检测） |
| `HostRankMap` | `{rank: hostName/hostUid}`，解析阶段内存填充，**不生成文件** |
| `JobType` | Job 类型（training/rollout，由优化器更新算子判断） |

`set_thresholds(degradation)` 根据 degradation 计算两个倍率并写入全局。`DegradationData`（继承 dict）：
- 结构 `{category: {key: value}}`，key 为单卡 `"0"` 或组 `"0,1,2"`。
- `add_single(category, rank, degradation)`：单卡。
- `add_group(category, ranks, degradation)`：组，按排序后 rank 集去重，保留最大劣化值。

---

## 5. 通用检测算法（kmeans_detector.py）—— 唯一异常检测算法

`general_anomaly_detection(ranks, values, anomaly_multiplier, ...)`，始终 **max 方向**（值偏大为异常）。返回 `(异常 rank 列表, 各异常 degradation 列表)`。

### 5.1 核心流程

1. 过滤 ≤0 及 `-99999`；过滤后不足 2 个 → 无异常退出。
2. Z-score 标准化；标准差 ≈ 0 → 强制置 1 避免除零。
3. **肘部法选最优 K**：K=2..min(n,MAX_K)，算 inertia（簇内平方和），取二阶差分最大者；退化为 2。
4. **KMeans++ 初始化质心**：首质心 = data[0]，后续 D² 加权随机采样（`random.Random(seed=42)` 保复现）。
5. **Lloyd 迭代** ≤MAX_ITERATIONS 轮：最近质心分配 → 质心 = 簇均值；空簇质心放到离其分配质心最远的样本；收敛 = 质心位移 < eps 且无分配变化。
6. **识别异常簇**：按原始值均值降序，基线 = 最小均值簇；簇均值 > 基线×倍率 → 该簇异常；从大到小遍历，遇第一个不满足即停止。
7. 无异常簇 → 无异常退出。
8. **逐轮剥离**：把本轮异常簇数据**剔除**，对**剩余数据**回到步骤 2（轮数 ≤ max_depth）。
9. 各轮按**当轮基线**判断是否异常（剥掉大值后基线单调不增，逐轮可检出更细微的离群点）；返回全部轮的异常簇（映射回 rank）。劣化指数统一用**最后一次得到的基线簇**（最严格地板）作为分母，`degradation = 值 / 最后基线`，使所有轮检出的异常劣化在同一刻度上可比。

> 与旧版差异：旧 homogeneous（spacedetector）是递归二分，且无剥离子集——新版每次**剔除**异常簇而非对异常数据继续聚类。

### 5.2 异常倍率由 degradation 决定

- 计算/IO/Host 类（`KERNEL_AICORE`, `kernel_aivec`, `memcpy_async`, `cpu`, `host_duration`）→ 倍率 = `1 + degradation`
- 通信域类（`step_duration`, `comm`）→ 倍率 = `1 + 5×degradation`
- `npu_bubble` → 固定硬阈值 `< 5000ns`
- `cpu` 沿用 `cpuDegradationPercent = 2.0`（实际由 config 倍率覆盖）

---

## 6. Profiling 数据解析（profilingdataparse.py）

### 6.1 表结构访问

| 表 | 用途 |
|----|------|
| `META_DATA` | `name='parallel_group_info'` 存 JSON 拓扑 |
| `STEP_TIME` | step 起止时间 |
| `TASK` | 算子执行，`taskType` → `STRING_IDS.value` |
| `STRING_IDS` | id ↔ 名称映射 |
| `COMMUNICATION_OP` | 通信算子（含 groupName、connectionId） |
| `CANN_API` / `PYTORCH_API` / `MSTX_EVENTS` | Host 端算子 |
| `HOST_INFO` | `hostUid` / `hostName`，节点归属（仅一行） |

### 6.2 聚合 step（关键设计）

`process_database` 将所有 step 合并为**单条聚合 step**：`start = min(startNs)`，`end = max(endNs)`。因此每张卡 CSV **只有 1 条数据行**。

### 6.3 指标生成（time_diff_for_step）

无数据/无通信域时用 `-99999` 标记或 `0`：

| 指标（CSV 列） | 生成方式 |
|----------------|----------|
| `StepDuration` | DB 数据总时间间隔 |
| `ZP_Device` | step_duration − 通信总耗时（非通信时长） |
| `ZP_Duration` | 通信算子区间合并总时长 |
| `ZP_Host` | 通信算子 Host 耗时 + `KERNEL_AICORE` Host 耗时的均值（无通信算子时仍取 Kernel Host） |
| `ZP_Bubble` | `op.startNs − op.h_endNs`（>0）的均值 |
| `KERNEL_AICORE` | `AVG(endNs−startNs)`，`taskType='KERNEL_AICORE'` |
| `MEMCPY_ASYNC` / `KERNEL_AIVEC` | `AVG(endNs−startNs)`，`taskType=对应名`（参照 KERNEL_AICORE） |
| `HostDuration` | Host 端执行耗时均值 |
| `DataLoader` | `MSTX_EVENTS` 中 dataloader 事件的 `endNs−startNs` |
| `{xp}_Duration` / `{xp}_Count` | 每个并行域算子 `endNs−startNs` 的均值 / `count` 均值 |

> 算子类型（KERNEL_AICORE / KERNEL_AIVEC / MEMCPY_ASYNC）只能从 `TASK.taskType → STRING_IDS` 推断；CANN_API 等 Host 端算子是 API 函数名，无法归类为 kernel 类型。

### 6.4 CSV 落盘（write_results_to_csv）

表头：`StepIndex, StepDuration, ZP_Device, ZP_Duration, ZP_Host, ZP_Bubble, ZP_Count, KERNEL_AICORE, MEMCPY_ASYNC, KERNEL_AIVEC, HostDuration, DataLoader` + 各域 `{xp}_Duration, {xp}_Count`。

### 6.5 节点信息（get_host_info）

从 `HOST_INFO` 表取 **`hostUid`**（仅一行），调用 `config.set_host_rank_map(rank, host_uid)` 存入**内存**，供检测阶段按物理节点分组。不生成文件。`data_parsing` 开始时 `config.reset_host_rank_map()` 清空。

---

## 7. 数据读取与快照构建（nodelevel_data_handler.py）

- `get_cur_detection_info(job_path)`：
  - 遍历 `group_info_*.json`，收集 `valid_ranks` 与每卡拓扑。
  - 聚合所有 `group_name`，构建 `parallels {域名: [[rank组]...]}`，仅保留存在**多卡组**的域。
  - 设置 `config.set_is_cluster_data(...)` 与 `config.set_has_named_domain(any(name ...))`。
  - **无命名域**（`not get_has_named_domain()`）时：`parallels[""] = _build_node_fallback_groups(valid_ranks)`，不再用 `get_detection_job_parallel_info` 推导的空域名分组。
- `_build_node_fallback_groups(ranks)`：按 `config.HostRankMap`（hostUid）将同节点 rank 分为一组；组内 ≥2 卡才保留，单卡组丢弃/拆分；无法分组→`{}`。
- `get_cur_job_last_step_data(ranks)`：
  - 读每张卡 CSV，跳过 `StepIndex` 列。
  - 每个 `(metric, rank)` 时间序列：长度 >1 取**倒数第二个**（n-2），=1 取第一个。
  - 返回 `{metric: {rank: value}}` 单快照。

---

## 8. 检测逻辑（nodelevel.py）

常量列名：`ZP_Device`、`KERNEL_AICORE`（原 `ZP_Kernel`）、`ZP_Duration`、`ZP_Host`、`ZP_Bubble`、`DataLoader`、`MEMCPY_ASYNC`、`KERNEL_AIVEC`、`StepDuration`、`HostDuration`；`minRanksInGroup=2`。

### 8.1 检测组选择（get_cal_detection_group）

并行域优先级：`tp → exp → ep → tp_exp → cp → cp2 → cp_ulysses → cp_ring → dp → dp_cp → dp_modulo_exp_cp`。
- 命中优先级域：集群数据（Case A）→ 完整集群分组；非集群（Case B）→ `get_detection_groups` 过滤为本地节点卡。
- 空域名 `""` 分支：用 `parallels[""]`（节点回退分组）。
- **未命中任何优先级域（情况 B）**：不再 `return "", []` 短路，回退 `_build_node_fallback_groups` 节点分组，保证单卡指标仍能检测。

### 8.2 慢计算卡 KERNEL_AICORE（get_slow_calculate_ranks / det_cal_for_one_group）

对每个检测组用 `KERNEL_AICORE`（方向 max），进入通用算法，结果写入 `KERNEL_AICORE`。

### 8.3 KERNEL_AIVEC / MEMCPY_ASYNC（get_slow_metric_ranks / det_metric_for_one_group）

`GENERAL_METRIC_CATEGORIES`：`(KERNEL_AICORE, KERNEL_AICORE)`、`(KERNEL_AIVEC, kernel_aivec)`、`(MEMCPY_ASYNC, memcpy_async)`。KERNEL_AICORE 已单独检测跳过，其余两列用检测组 + 通用算法，方向 max。

### 8.4 NPU 空泡 npu_bubble（detection_zp_bubble_data）

排除 -99999 与 ≤0；`value < 5000`（ns，硬编码）记异常，写入 `npu_bubble`（小值异常）。

### 8.5 通信域组间对比（detection_all_communication_parallel）

覆盖指标 1/2：
- 指标2：各域 `{xp}_Duration` → `comm`（沿用旧类别名）
- 指标1：`StepDuration` → `step_duration`

**守卫**（双保险，[nodelevel.py:97](nodelevel.py#L97) 与 [nodelevel.py:728](nodelevel.py#L728)）：
- `config.get_has_named_domain()` 为真 → 正常检测通信域组间指标（情况 B / 正常数据）。
- 为假（情况 A，无命名通信域）→ 直接跳过（无域名无法解释该域对应 tp/ep；检出也无从向用户说明）。

`_detect_comm_group_metric` 对每个并行域做组间对比，异常组写入对应类别（comm 为组键，`display_key` 带域名）。

### 8.6 慢 CPU 卡 cpu / host_duration（get_slow_host_ranks_by_homogenize / _get_slow_host_metric_ranks）

- 收集 `ZP_Host` 有效值（排除 -99999）。
- `process_cpu_data_by_node`：按**物理节点**（`config.HostRankMap`）分组，组内去首尾后求均值覆盖组内卡值；无节点映射时回退 `process_cpu_data`（按 4 卡分组 + 去首尾均值）。
- 通用算法 max，写入 `cpu`。
- `HostDuration` 列 → `host_duration`，集群整体拉齐后进入通用算法。

---

## 9. 结果输出

### 9.1 JSON（utils.write_result）

`straggler_detection_result.json`：
```json
{
  "KERNEL_AICORE": [{"display_key":"0","metric_value":1.5,"is_abnormal":true}],
  "comm": [{"display_key":"tp[0, 1]","metric_value":1.8,"is_abnormal":true}],
  "cpu": [...], "npu_bubble": [...],
  "kernel_aivec": [...], "memcpy_async": [...],
  "host_duration": [...], "step_duration": [...]
}
```
始终包含全部 8 类（空则 `[]`）；非 bubble 降序，bubble 升序；comm/step_duration 的 `display_key` 带域名。

### 9.2 清理（utils）

`clean_detection_outputs` 清理：`op_metric`、`straggler_analysis_output`、`analysis_result`、`straggler_detection_result.json`、`straggler_detection_result`。
`confirm_clean` 交互式询问（skill 调用时**须先问用户**）。

---

## 10. 故障联合分析（joint_analysis.py）

### 10.1 硬件流水线因果模型

```
计算(compute) → 通信(communication) → 等待/空转(wait)
```
- **计算阶段**：`KERNEL_AICORE`, `kernel_aivec`, `memcpy_async`（根因候选起点）
- **通信阶段**：`comm`, `step_duration`（受慢卡拖累）
- **等待/空转**：`cpu`, `host_duration`, `npu_bubble`（下游影响）

### 10.2 类别与指标映射

| 类别 | 单卡指标列 | 检测方式 / 异常方向 |
|------|-----------|----------|
| `KERNEL_AICORE` | `KERNEL_AICORE` | 单卡 / 大值 |
| `kernel_aivec` | `KERNEL_AIVEC` | 单卡 / 大值 |
| `memcpy_async` | `MEMCPY_ASYNC` | 单卡 / 大值 |
| `cpu` | `ZP_Host` | 单卡 / 大值 |
| `host_duration` | `HostDuration` | 单卡 / 大值 |
| `npu_bubble` | `ZP_Bubble` | 单卡 / 小值（固定 <5000ns） |
| `comm` / `step_duration` | `{xp}_Duration` / `StepDuration` | 通信域组级别 |

### 10.3 报告结构（joint_failure_analysis.log，log 格式）

每行带 `[时间戳][JOINT]` 前缀与 `[INFO]/[WARN]` 级别。四大部分：

1. **一、各列检测结果汇总**：每个类别输出标题行；异常类别先列异常项概览，再对单卡类展示该类别下全部卡并画文本条形图（`█`/`▒`，宽度 40，异常卡标 `<-- 异常`）；通信域类展示异常域及各卡域时长条形图。
2. **二、故障传播链分析**：按流水线阶段列出命中卡，标注“根因候选 / 受慢卡拖累 / 下游影响”。
3. **三、根因卡判定**：计算阶段（流水线最上游）命中卡判为根因。
4. **四、结论与建议**：优先排查根因卡。

### 10.4 根因判定规则

`_determine_root_causes`：把各类别异常卡归并到各阶段；**计算阶段（KERNEL_AICORE/kernel_aivec/memcpy_async）命中的卡**即根因候选（其在最上游，变慢会通过集合通信传导到同域其他卡）。

### 10.5 调用签名

`generate_joint_report(result, parallels, step_data=None, output_dir=None)`
- `step_data` 用于绘制全部卡（含正常卡）条形图，由 `main.py` 传入 `last_step_data`。

### 10.6 最终输出逐类别汇总表（build_summary_table）

`build_summary_table(result, parallels=None, step_data=None, degradation=None) -> str`：生成 **Unicode 框线表格**（`_render_box_table`，按列宽 + CJK 显示宽度自动对齐），一行一个"有异常的类别"，**由 `main.py` 经 `_safe_print` 打印到调用方 agent 的 stdout，不进任何 log 文件**。表头：`类别 | 异常卡 | 劣化指数 | 劣化阈值 | 数据要点`。

- **类别**：`{code}（{SHORT_CATEGORY_LABELS}）`，如 `KERNEL_AICORE（慢计算卡）`。
- **异常卡**：由 result 各 key 解析 rank 列表（组键类别归并组内所有 rank），如 `rank 0` / `rank 0, 1`。
- **劣化指数**：该类别的最大劣化值（3 位小数）。
- **劣化阈值**：`npu_bubble` → `5000ns`；通信域类（comm/step_duration）→ `config.get_comm_multiplier()`（`1+5*deg`）；其余计算/IO/Host 类 → `config.get_compute_multiplier()`（`1+deg`）。
- **数据要点**：单卡类别用 `CATEGORY_METRIC` 列 + 本地 `_fmt_ns`（ns→s/ms/us/ns），形如 `rank0=1.76ms，其他≈568~574us（约 3.1 倍）`（倍数 = 异常卡最大值/其他均值；min==max 时 `其他≈x`）；通信域类用域时长列（如 `tp_Duration`）；无数据兜底 `无详细数据`。
- 无任何异常时返回含"无异常"提示的单行表。

---

## 11. 可视化（visualizer.py + markdown_viz.py）

- `visualizer.run_visualization`：控制台实时反馈 + 调用 `markdown_viz.write_report` 生成 `analysis_result/detection_report.log`。
- `markdown_viz`：文本报告，含指标排序柱状图、异常卡高亮、统计信息、通信域分组表、总通信耗时。

---

## 12. 主入口（main.py）

- CLI：`python main.py path=<dir> [degradation=0.3] [clean=ask|yes|no]`。
- `run_detection(input_path, degradation=0.3, skip_parsing=False, clean='ask')`：skill 调用入口，返回 `{category: {key: degradation}}`。
- 流程：确认 degradation → 清理/解析 → 获取并行域与有效 ranks → 取最新 step 快照 → `delimit_detection` → `write_result` → `generate_joint_report` → `run_visualization` → `build_summary_table`（`_safe_print` 打印到 stdout，失败仅告警不中断）。

---

## 13. 关键设计要点与约定

1. **单快照**：不跨 step 做时间序列分析；CSV 只落 1 条聚合数据。
2. **倒数第二点**：多行 CSV 取 n-2 行，规避最后一行不完整。
3. **无效标记 `-99999`**：贯穿解析、读取、各检测函数，用于跳过缺失数据。
4. **统一异常算法**：`kmeans_detector.general_anomaly_detection`（KMeans + Z-score + 肘部法 + 逐轮剥离），唯一参数为倍率（由 degradation 决定）。
5. **逐轮剥离，检测/劣化分离**：每轮剔出异常簇后对剩余数据再聚类，**检测**各用当轮基线（剥掉大值后基线单调不增，逐步检出更细微离群点）；**劣化指数**统一用最后一次得到的基线簇（最严格地板）作分母，跨轮同一刻度可比。
6. **倍率分组**：计算/IO/Host = `1+degradation`，通信域 = `1+5×degradation`。
7. **8 类指标**：`KERNEL_AICORE`, `kernel_aivec`, `memcpy_async`, `npu_bubble`, `cpu`, `host_duration`, `comm`, `step_duration`。
8. **无命名域退化（情况 A）**：检测组按 hostUid 物理节点分组；通信域组间指标直接跳过；单卡指标在节点组内检测。
9. **未命中优先级（情况 B）**：检测组同样退化到物理节点分组，但通信域组间指标仍检测（HasNamedDomain=True），检出慢通信组时可带域名。
10. **CPU/节点分组**：使用内存 `config.HostRankMap`（源自 `HOST_INFO.hostUid`），不落盘文件；无映射时回退按 4 卡。
11. **输出为 log 格式**：联合分析报告用 `[JOINT]` 前缀 + 级别，含全部卡条形图。
