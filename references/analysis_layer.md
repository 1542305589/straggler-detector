# 结果分析层（Diagnosis）规格

> 本层**只读**原始 profiler 数据（`ascend_pytorch_profiler_*.db` / `analysis.db` / `cluster_analysis.db`）与检测产物
> （`straggler_detection_result.json`）；**不写库、不改检测数据、不重跑检测**。
> 正式结论**只由** `diagnosis/rules.py` 的规则产出；LLM 只能在报告第 9 节写推测，且不得进入正式结论。
>
> 实现：`diagnose.py`（入口）→ `diagnosis/evidence.py`（证据）→ `diagnosis/rules.py`（规则）→ `diagnosis/render.py`（渲染）
> / `diagnosis/validate.py`（校验门）。

---

## 1. 一句话目标

把「哪一类指标、哪张卡/哪个组、劣化多少」的检测结果，升级成「**最可能的根因类型 + 可回溯的证据**」，
同时保证：**结论可复现、证据可追溯、模型的自由发挥被隔离在推测区**。

---

## 2. 硬约束

1. 只读：数据库一律以 `file:...?mode=ro` 只读 URI 打开（`diagnosis/db.py::open_ro`）。
2. 不改检测：`main.py` 的检测流程、`nodelevel.py`、`kmeans_detector.py`、`profilingdataparse.py` 一律不动。
3. 正式结论只由规则产出：`verdict / primary / secondary_findings / rules_fired` 只能来自 `rules.evaluate()`。
4. 模型不得越界：`speculations`（推测）不得出现在 `verdict / primary / rules_fired / actions` 里（由 `validate.py` 卡）。
5. 字段缺失 → 该类别标 `not_evaluated` 并写入 `limitations`，**不得猜**。

---

## 3. 输入与取数

### 3.1 数据级别（决定取哪条路径）

| 级别 | 判据 | 取数路径 |
|---|---|---|
| `level1` | profiler db 同目录有 `analysis.db` 且 `CommAnalyzerBandwidth` 有数据 | `analysis.db` 优先，缺表回退裸 db |
| `level0` | `TASK.taskType` 含 `KERNEL_AICORE` / `KERNEL_AIVEC` / `MEMCPY_ASYNC` | 同上（无 `analysis.db` 时裸 db 粗算带宽） |
| `levelnone` | `TASK` 仅 `MsTx` | 裸 db；`compute` 未评估 |
| `cluster` | 输入是含 `CommunicationGroupMapping` + `ClusterCommunicationBandwidth` 的单文件 db | 只走 cluster 路径；F1/F2/F3/F6 不适用 |

### 3.2 evidence.json 结构

```jsonc
{
  "schema_version": "1.0",
  "meta": {
    "data_dir": "...", "level": "level1", "ranks": [0,1,...],
    "has_analysis_db": true, "has_cluster_db": false,
    "parallels": {"tp": [[0,1,2,3]], "ep": [[...]]},
    "group_ranks": {...}, "rank_device": {"0": {"host_name":"...","host_uid":"...","npu_id":0}},
    "host_info": {...}, "job_type": "training|rollout|unknown",
    "session_ms": 8300.0, "reference_step": "5612"
  },
  "detection": { /* straggler_detection_result.json 原样 */ },
  "macro": {
    "step_trace": {"by_rank": {"0": {"totals": {"computing_ms":.., "communication_ms":..,
                   "free_ms":.., "stage_ms":.., "bubble_ms":.., "preparing_ms":..},
                   "steps": [...], "reference_step": "5612"}}, "source": "analysis.db"},
    "comm_time":  {"by_rank": {"0": {"ops": [{"hccl_op_name":.., "elapse_ms":..,
                   "transit_ms":.., "idle_ms":..}]}}, "source": "analysis.db"}
  },
  "host": {
    "gloo":         {"0": {"max_ms":.., "total_ms":..}},   // PYTORCH_API 集合通信（Host 侧）
    "sync_device":  {"0": {"max_ms":.., "total_ms":..}},   // aclrtSynchronizeDeviceWithTimeout
    "sync_event":   {"0": {...}}, "sync_stream": {"0": {...}},
    "free_windows": {"0": {"gaps": [{"start_ms":.., "end_ms":.., "duration_ms":..}], "total_ms":..}},
    "session_start_ms": {"0": ..}, "source": "..."
  },
  "compute": {"by_rank": {"0": {"ops": {"MatMulV2": {"sum_ms":.., "count":.., "avg_ms":..}}}},
              "source": "profiler.db"},
  "comm": {
    "by_rank_group": {"0": {"0,1,2,3": {"allgather": {"bandwidth_gbps":.., "count":..,
                       "size_mb":.., "transit_ms":.., "transport_type":..}}}},
    "matrix": {"mc2": [{"hccl_op_name":.., "transport_type":.., "max_transit_ms":..,
               "min_bandwidth_gbps":.., "max_bandwidth_gbps":.., "slowest_pair":[0,1]}]},
    "source": "analysis.db"
  },
  "freq":  {"by_rank": {"0": {"0": {"min_mhz":.., "max_mhz":.., "avg_mhz":..}}}, "source": "profiler.db"},
  "coverage": {"missing": ["..."], "notes": ["..."]}
}
```

> 缺表 / 缺列的类别会在 `coverage.missing` 里留痕，并让对应规则落到 `not_evaluated`。

---

## 4. 输出契约

### 4.1 `formal_result.json`

| 字段 | 说明 |
|---|---|
| `schema_version` | 固定 `"1.0"` |
| `verdict` | `slow_rank` / `slow_group` / `multiple` / `no_anomaly` / `insufficient_evidence` |
| `confidence` | `high` / `medium` / `low` |
| `conclusion_source` | `detection`（检测结果优先） / `analysis_fallback`（检测层无异常，分析层兜底） |
| `primary` | `{category, rule_id, slow_side, baseline_side, degradation, summary}` |
| `secondary_findings` | `primary` 之外的命中项（同 `findings` 结构） |
| `findings` | 全部结论项，**每条带 `rule_id` + `evidence_refs`** |
| `rules_fired` | 命中的规则 ID 列表 |
| `rules_checked` | 已检查的规则 ID 列表（含 `DET`） |
| `limitations` | 未评估 / 数据缺失说明 |
| `actions` | 按 `primary.category` 给出的处置建议 |
| `validation` | 由 `diagnose.py` 写入：`{passed, errors, warnings}` |
| `speculations` | **仅由调用方（agent）填写**；不进正式结论 |

`findings[]` 结构：

| 字段 | 说明 |
|---|---|
| `rule_id` | 规则 ID（必填） |
| `category` | `comm` / `compute` / `io` / `host` / `npu_bubble` / `pp_comm` / `multiple` / `insufficient_evidence` / `none` / `unknown` |
| `status` | `authoritative`（检测层原始结论）/ `fired`（规则命中）/ `conflict`（与检测层冲突）/ `auxiliary`（检测层已下结论时的辅助发现）/ `not_evaluated` |
| `confidence` | `high` / `medium` / `low` |
| `slow_side` / `baseline_side` / `waiters` | rank 列表 |
| `metric` | 指标名（如 `compute.total_ms`） |
| `value` / `baseline_value` / `ratio` | 数值（可为 `null`） |
| `summary` | 一句话结论 |
| `evidence_refs` | 指向 `evidence.json` 的点分路径（校验门逐个解析） |
| `derived` | `true` = 数值为派生比值，跳过数值直接命中检查 |

> 非有限值（`Infinity` / `NaN`）在写盘前会被统一替换为 `null`，并在 `validation.warnings` 里记一笔——
> 保证 `formal_result.json` / `evidence.json` 一定是严格合法 JSON（否则 JS 的 `JSON.parse` 等严格解析器会直接报错）。

### 4.2 `diagnosis_report.md`（固定 9 节骨架）

```
1. 数据与结果概览      2. 检测结论（权威）     3. 慢侧 / 基准侧
4. 类别                5. 规则命中与证据       5.1 宏观迭代拆解
                                              5.2 Host 侧阻塞（PYTORCH_API / CANN_API）
                                              5.2b 空闲窗口
                                              5.3 计算耗时（含 count）
                                              5.4 通信带宽 Level-1 / Level-2
                                              5.5 频率
6. 检测层与辅助分析一致性   7. 结论与建议       8. 置信度与局限
9. 推测与待验证假设（不进正式结论）
```

未通过校验时，报告**顶部**会出现 `> ⚠️ 本报告未通过校验，结论不可直接采信`。

### 4.3 控制台证据摘要

`render.console_evidence()` 把关键证据直接打到 stdout（不打印完整 `evidence.json`）：证据摘要、各卡耗时拆解、
主机侧等待、空闲窗口、计算耗时、慢卡 Top 算子、通信带宽 L1/L2、AI Core 频率、覆盖/局限。

---

## 5. 规则库

### 5.1 检测结论（权威，rule_id = `DET_*`）

由 `straggler_detection_result.json` 直接转写，`status=authoritative`。**检测结果永远优先**。

### 5.2 A–E（辅助层，由旧分析层迁移）

| ID | 触发 | 结论 |
|---|---|---|
| A1 | 报告某域只有 1 个 Group | 该域不能组间对比，标未评估，**不能判正常** |
| A2 | 某域无带宽数据 | 带宽未评估，**不能判正常** |
| A3 | 检测结果缺某类别 | 对应类别未评估，列入 `limitations` |
| A4 | 有效 rank 数少于排序表出现的 rank | 数据可能不完整，降置信度 |
| B1 | `comm` 非空 | 转写检测层通信结论（高置信） |
| B2 | `comm` 为空但组间带宽差 ≥ 阈值 | 检测层未报 → `conflict`，建议复核 |
| B3 / B4 | 多 opType 同向 / 单组或无带宽 | 提升或降级置信度 |
| C1 | 计算类检测非空 | 计算慢卡 |
| C2 | 检测未报但计算组间平移 ≥ 阈值 | `conflict`，建议复核 |
| C3 | 只有组内少数卡异常 | 倾向单卡故障而非整组 |
| C4 | 搬运与计算同向异常 | 疑似整芯片内存带宽 / NUMA |
| D1 / D2 / D3 | `cpu` / `npu_bubble` / `pp_comm` 非空 | Host 下发 / 调度间隔 / PP 等待 |
| D4 | ZP_Host 只有 1 个节点 | 无法节点间对比 |
| E1–E4 | 多 device / 多类别同向 / 与检测矛盾 / 关键字段缺失 | 元规则；E4 → `insufficient_evidence` |

### 5.3 F1–F6（由裸 db 证据驱动，新增）

| ID | 触发条件 | 结论 |
|---|---|---|
| **F1** | 某卡 Host 侧集合通信（`host.gloo.max_ms`）≥ max(0.5×最大阻塞, 会话 5%, 1000ms)，且空闲占比 ≥ 10%、计算占比 ≤ 10% | **Host 阻塞 / 伪快卡**：这些卡在等 Host，不是它们本身慢 |
| **F2** | 计算慢卡的同类算子 `count` 与他卡相同（±5%）但 `avg_ms` ≥ 阈值倍 | 算子 / 硬件**劣化**（非负载不均） |
| **F3** | 计算慢卡的同类算子 `count` 与他卡差 ≥ 5% | **负载切分不均** |
| **F4** | (a) 某算子 `transit/elapse ≥ 0.5`；或 (b) 组间带宽差 ≥ 通信阈值；或 (c) matrix 同 transport 内慢链路 ≥ 通信阈值 | **通信真实传输慢 / 慢链路**（level0 与已判计算主因时跳过） |
| **F5** | 某卡通信占比 ≥ 10%、空闲 < 10%，且最大算子 `transit ≤ 5%×elapse`、`idle ≥ 90%×elapse` | 该卡**纯等待**（是等待者，不是慢卡） |
| **F6** | 某卡平均频率 ≤ 0.9×中位 | **频率异常** |

---

## 6. 权威顺序与汇总

```
检测结果（authoritative，永远优先）
  └─ 命中 → conclusion_source = "detection"，confidence = high
            分析层其余命中项降级为 auxiliary，不改变 verdict/primary/actions
  └─ 未命中 → conclusion_source = "analysis_fallback"
            按优先级取 primary：F1 > F4 > F2 > F3 > C1 > C2 > D1 > D2 > D3 > B1 > B2 > F6 > F5
            B1 与 F1 同时命中时，比较「等待者通信耗时」与「阻塞者空闲」取大者为主因
            confidence 封顶 medium
```

---

## 7. 推测区隔离与校验门

- 报告第 9 节 `speculations[]`：每条写 `hypothesis` / `evidence_refs` / `verification`，标注 `confidence=speculative`。
- `diagnosis/validate.py` 检查：
  1. 每条 finding 必带 `rule_id`，`status` 合法；
  2. `evidence_refs` 每个点分路径都能在 `evidence.json` 里解析到（解析不到 → error）；
  3. 推测内容不得出现在 `verdict / primary / actions`（出现 → error）。
- `diagnose.py` 把 `{passed, errors, warnings}` 写进 `formal_result.json.validation`；有 error 时报告顶部打警告，进程返回码 `2`。

---

## 8. 置信度

| 置信度 | 条件 |
|---|---|
| 高 | 检测层命中（`conclusion_source=detection`）且证据链完整 |
| 中 | 分析层兜底命中；或仅单侧证据；或计算组间差未超会话 5% |
| 低 | 单组域 / 无带宽 / 报告缺失 / 规则冲突 |
| — | `insufficient_evidence` |

---

## 9. 已知限制（尚未验证项）

- 本层**尚未在真机运行验证**（开发机无 Python）：断言脚本见 `verify_comm_domain_key.py`。
- 裸 db 取数依赖的表/列（`PYTORCH_API` / `CANN_API` / `COMPUTE_TASK_INFO` / `AICORE_FREQ` 等）为**推断**，
  需在真实数据上确认；缺失时会落到 `coverage.missing` 与 `not_evaluated`。
- `cluster` 单文件无逐算子计算 count / Host API / 频率，F1/F2/F3/F6 不适用。
- `no_align` 只影响**检测层**的带宽回填，与本层无关。

---

## 10. 回归样例

以下两组场景**已在真机验证通过**，可作为回归基线。场景特征：数据由 `tp4dp2` 的 30B MoE
训练产生，人为在 rank 0–3 上注入通信压力。**此处只描述场景特征，不记录任何内部数据路径。**

### 10.1 level0（8 卡）

**场景**：`tp=[0,1,2,3]` 与 `tp=[4,5,6,7]` 两组之间做通信压测（旧带宽口径）。

**验证过的输出基线**：

| 字段 | 期望值 |
|---|---|
| 汇总表 | `comm` 一行：异常卡 `tp[0, 1, 2, 3]`，劣化指数 `tp[0, 1, 2, 3]:2.92339`，阈值 `1.3x` |
| `conclusion_source` | `detection` |
| `primary.category` / `primary.rule_id` | `comm` / `DET_COMM` |
| `primary.slow_side` | `[0, 1, 2, 3]` |
| `primary.degradation` | `2.92339` |

**人工核对用的事实**：allgather 6.364 vs 11.94 GB/s、allgatherv 0.34 vs 0.994 GB/s；
KERNEL_AICORE 组间平移 1.59x、MEMCPY_ASYNC 4.49x（`mean/median ≈ 1`，双峰）；
计算绝对差仅 10–13 ms，**远小于**会话量级 → 不作为主因。

### 10.2 level1（8 卡，带 `analysis.db`）

**场景**：`tp=[0,1,2,3]`，`ep` / `mc2` 均为全组 `[0..7]`，rank 0–3 被注入通信压力。
走**新两级带宽口径**，`pp_comm` 按规则跳过。

**验证过的输出基线**：

| 层级 | 项 | 劣化指数 |
|---|---|---|
| Level-1（单卡，带域名） | `mc2[2]` / `ep[2]` | `8.09813` |
| Level-1 | `mc2[0]` / `ep[0]` | `8.07316` |
| Level-1 | `mc2[1]` / `ep[1]` | `7.02319` |
| Level-1 | `mc2[3]` / `ep[3]` | `6.68333` |
| Level-2（组键） | `tp[0, 1, 2, 3]` | `7.33385` |

三个要点：

1. Level-1 的键**必须带域名**（`ep[0]`，不是 `[0]`）——同一 rank 在 `ep` / `mc2` 两个域各报一次，
   键不同所以不会互相覆盖；
2. 域名**含数字时也要正确解析出 rank**（`mc2[0]` → rank `0`，不是 `[2, 0]`）；
3. 汇总表里**不应出现 `pp_comm`**。

### 10.3 cluster 单文件（`cluster_analysis.db`）

**场景**：同一批数据生成的 `cluster_analysis.db` 单文件；识别到 `2` 个域、7036 条 per-op 记录。

**验证过的输出基线**：

| 层级 | 项 | 劣化指数 |
|---|---|---|
| Level-2（组键） | `tp[0, 1, 2, 3]` | `5.33032` |
| Level-1（单卡，带域名） | `ep[3]` | `2.86893` |
| Level-1 | `ep[2]` | `2.76` |
| Level-1 | `ep[1]` | `2.63832` |
| Level-1 | `ep[0]` | `2.49197` |

要点：

1. **只检测 `comm`**（不会出现 `KERNEL_AICORE` / `cpu` / `npu_bubble` / `pp_comm`），日志会提示「无单卡快照数据」；
2. 产物落在 **`<db 所在目录>/detection_output/`**，而不是 db 所在目录本身（避免把结果写到 db 文件旁边）；
3. Level-1 的键同样**带域名**（`ep[0]` 而非 `[0]`）。
