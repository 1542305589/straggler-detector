# 结果分析层（Diagnosis）

> **本层只读 `straggler_detection_result.json` 和 `analysis_result/detection_report.log`；不读 DB、不跑 SQL、不写脚本。需要额外数据时输出 `insufficient_evidence`，并列出缺失项。**

本层是**提示词/文档**，不是 Python 分析器。检测脚本、算法、输出格式一律不动。

---

## 0. 一句话目标

在不改动现有检测脚本、算法和输出格式的前提下，给 skill 增加一层「结果分析」提示词：让 Agent **只读取检测脚本已经产出的两个文件**，按自然语言规则推导慢节点结论，并输出**固定 JSON** 和**固定 Markdown**。

---

## 1. 硬约束（必须先读，违反即返工）

1. **只允许读两个文件**：
   - `<数据目录>/straggler_detection_result.json`
   - `<数据目录>/analysis_result/detection_report.log`
2. 禁止读取 `master_*_ascend_pt/ASCEND_PROFILER_OUTPUT/*.db`，禁止 SQL，禁止调用 `msprof-analyze`，禁止写 Python 分析脚本，禁止让 Agent 自己找数据。
3. 禁止修改检测代码：`main.py`、`nodelevel.py`、`kmeans_detector.py`、`profilingdataparse.py`、`nodelevel_data_handler.py` 等一律不动。
4. 规则只能引用上述两个文件中**已经存在**的字段。字段缺失时，输出 `insufficient_evidence`，并列出缺失项，不得猜测。
5. 检测结果与报告排序冲突时，不直接下结论，标为「一致性异常，建议复核」。
6. 分析层是**提示词/文档**，不是条件匹配脚本。不要新增 `analyze_result.py` 之类的 Python 分析器。

---

## 2. 输入文件与字段

### 2.1 straggler_detection_result.json

固定 7 个类别，每个类别是列表：

```json
{
  "KERNEL_AICORE": [],
  "comm": [{"display_key": "tp[0, 1, 2, 3]", "metric_value": 2.9234, "is_abnormal": true}],
  "pp_comm": [],
  "cpu": [],
  "npu_bubble": [],
  "memcpy_async": [],
  "kernel_aivec": []
}
```

可用字段只有：**类别、`display_key`（rank 或组）、`metric_value`（劣化指数）、`is_abnormal`**。

> `comm` 的 `display_key` 有两种形态：**组键** `tp[0, 1, 2, 3]`（Level-2 组间慢组）、
> **单卡键** `ep[5]`（Level-1 组内慢 rank，带域名说明这个 rank 的异常是哪个域报出来的）。

### 2.2 analysis_result/detection_report.log

必须能识别以下区块：

| 区块 | 字段 | 用途 |
|------|------|------|
| 头部 | 数据目录、Job 类型、生成时间、有效 Rank 数 | 上下文 |
| 并行域拓扑 | ep / mc2 / tp 各有几个 Group | 单组域判断 |
| 检测结果摘要 | 类别、状态、劣化阈值、异常卡、劣化指数、物理设备 | 检测结论 |
| KERNEL_AICORE 耗时排序 | 全 rank 的耗时、相对倍数；max/min、mean/median | 单卡 vs 组间、组间平移 |
| KERNEL_AIVEC 耗时排序 | 同上 | 同上 |
| MEMCPY_ASYNC 耗时排序 | 同上 | 同上 |
| ZP_Host 耗时排序 | 按节点的耗时、相对倍数 | 只能做节点级 |
| ZP_Bubble 耗时排序 | 全 rank 的耗时、相对倍数 | 单卡 vs 组间 |
| tp 带宽表 | 每个 opType、每个 Group 的带宽、`***` | 通信组间带宽 |
| 各域带宽概览 | 每个域/opType 的 min~max、max/min、Group 数 | 通信组间差异 |
| `[ep] 无带宽数据` / `[mc2] 无带宽数据` | 缺失标记 | **不能判「正常」** |
| 慢通信域（两级带宽口径）（level1/cluster） | Level-1 逐 rank、Level-2 逐组，指标 + `***` | 新口径下的 comm 明细 |

> 说明：`level0`/`levelnone` 报告里 comm 是「tp 带宽表 + 各域带宽概览」；`level1`/`cluster` 报告里 comm 是「慢通信域（两级带宽口径）」段（Level-1 逐 rank / Level-2 逐组）。两者都只能引用报告里**已出现**的内容。
>
> **单组域**（如 `tp` 只有 1 个 Group）在 `level1`/`cluster` 报告里：**Level-1（组内逐 rank）仍然有效**（组内比较只依赖本组 rank）；只是**没有 Level-2**（组间无法比较）。**不要把「无 Level-2」当成「正常」** —— 应看 Level-1 是否有 `***` 慢 rank。

### 2.3 口径对照表（必须遵守）

| 指标 | JSON 字段 | 报告字段 | 口径差 / 注意事项 |
|------|-----------|----------|-------------------|
| comm | `comm[].metric_value` | 带宽表 + 概览 | 检测劣化指数 vs 组间带宽比 —— **不要直接和「相对倍数」比较** |
| KERNEL_AICORE | `KERNEL_AICORE[].metric_value` | 耗时排序 + 相对倍数 | 检测劣化指数 vs 耗时/最快卡 —— 报告明确写「与检测劣化指数口径不同」 |
| KERNEL_AIVEC | `kernel_aivec[].metric_value` | 耗时排序 + 相对倍数 | 同上 |
| MEMCPY_ASYNC | `memcpy_async[].metric_value` | 耗时排序 + 相对倍数 | 同上 |
| cpu | `cpu[].metric_value` | ZP_Host 排序 | 检测 per-rank vs 报告 per-node —— 报告不能定位 rank |
| npu_bubble | `npu_bubble[].metric_value` | ZP_Bubble 排序 | 检测单次阈值 ns vs 报告总时间 ms —— **不要直接比较** |
| pp_comm | `pp_comm[].metric_value` | 无明细 | 报告没有 PP 明细，只能引用 JSON |

---

## 3. 规则库

### A. 数据覆盖库（先跑）

| ID | 触发条件（现有字段） | 结论 | 置信度 |
|----|----------------------|------|--------|
| A1 | 报告 `ep: 1 个 Group` 或 `mc2: 1 个 Group` | 该域不能做组间对比，标「未评估」，**不能判「正常」** | 高 |
| A2 | 报告出现 `[ep] 无带宽数据` 或 `[mc2] 无带宽数据` | 该域带宽未评估，**不能判「正常」** | 高 |
| A3 | 报告缺少某个排序表/摘要表，或 JSON 缺少某个类别 | 对应类别「未评估」，列入 `limitations` | 高 |
| A4 | 报告「有效 Rank 数」小于带宽表/排序表中出现的 rank 数 | 数据可能不完整，降低置信度 | 中 |

### B. 通信库

| ID | 触发条件 | 结论 | 置信度 |
|----|----------|------|--------|
| B1 | `comm` 非空；报告 `tp/<opType>` 表某组带 `***`；概览 `max/min ≥ 1.3` | 通信型慢组；慢侧 = `***` 组；基准侧 = 同域另一组 | 高（≥2 个 opType 同向）/ 中（1 个） |
| B2 | `comm` 为空；但报告带宽表某组带 `***`，或概览 `max/min ≥ 1.3` | 检测层未报该差异；标「一致性异常，建议复核」，不直接判故障 | 中 |
| B3 | B1 命中且 ≥2 个 opType 的慢侧相同 | 提高 B1 置信度到**高** | 高 |
| B4 | 带宽表只有 1 个 Group，或该域无带宽数据 | 该域不能做组间对比，标「未评估」 | 高 |

### C. 计算 / 搬运库

| ID | 触发条件 | 结论 | 置信度 |
|----|----------|------|--------|
| C1 | `KERNEL_AICORE` / `kernel_aivec` / `memcpy_async` 非空；报告对应排序表有同 rank | 计算/搬运慢卡；用排序表判断单卡 vs 组间 | 高（JSON+报告同向） |
| C2 | 对应 JSON 类别为空；报告排序表 `max/min ≥ 阈值`（计算 1.3×、IO 2.5×），且 `mean/median ≈ 1`（双峰） | 检测层组内盲区；报告显示组间平移；标「建议复核」，不直接判故障 | 中 |
| C3 | 报告排序表中只有 1 个 rank 远离组内其他 rank；或整组远离另一组 | 单卡故障 vs 组间故障，两者建议不同 | 高（差距大且组结构清晰）/ 中（边界值） |
| C4 | `memcpy_async` 与 `KERNEL_AICORE`/`KERNEL_AIVEC` 同向，或报告多个排序表同向 | 可能是整芯片/内存带宽/NUMA 问题，不是单一算子 | 中 |

### D. Host / 气泡 / PP 库

| ID | 触发条件 | 结论 | 置信度 |
|----|----------|------|--------|
| D1 | `cpu` 非空 | Host/下发慢；以 JSON 的 rank 为准；报告 ZP_Host 是节点级，不能定位 rank | 高（JSON 有 rank）/ 低（只有节点级） |
| D2 | `npu_bubble` 非空；报告 ZP_Bubble 排序有同 rank | 调度/启动间隔问题；注意报告单位 ms、检测阈值 ns，**不能直接比较** | 高 |
| D3 | `pp_comm` 非空 | PP 流水线等待；报告无 PP 明细，只能引用 JSON | 中 |
| D4 | 报告 ZP_Host 排序只有 1 个节点 | 不能做节点间对比；只能做节点级描述 | 高 |

### E. 元规则库

| ID | 触发条件 | 结论 | 置信度 |
|----|----------|------|--------|
| E1 | 报告摘要「物理设备」列有多个 device | 异常组跨多个 device，不是单 device 硬件故障；单 device 才考虑 device 级 | 中（设备列可能截断） |
| E2 | 多个 JSON 类别非空，或多个报告排序表同向 | 多故障叠加/共同根因；列 primary/secondary | 中 |
| E3 | JSON 状态与报告排序/带宽表矛盾 | 标「一致性异常，建议复核」，不直接下结论 | 低/中 |
| E4 | 关键字段缺失、单组域、无带宽数据、报告截断 | `insufficient_evidence`；列出缺失项 | — |

---

## 4. Agent 输出契约

### 4.1 固定 JSON

```json
{
  "schema_version": "1.0",
  "verdict": "slow_group | slow_rank | multiple | no_anomaly | insufficient_evidence",
  "confidence": "high | medium | low",
  "primary": {
    "category": "comm | compute | io | host | pp_comm | npu_bubble | multiple",
    "rule_id": "B1",
    "slow_side": [0, 1, 2, 3],
    "baseline_side": [4, 5, 6, 7],
    "degradation": 2.9234,
    "summary": "tp[0,1,2,3] 的 allgather/allgatherv 带宽显著低于 tp[4,5,6,7]"
  },
  "secondary_findings": [
    {
      "category": "compute",
      "rule_id": "C2",
      "slow_side": [0, 1, 2, 3],
      "baseline_side": [4, 5, 6, 7],
      "ratio": 1.59,
      "confidence": "medium",
      "summary": "报告排序显示 KERNEL_AICORE 组间平移 1.59x，检测层未报"
    }
  ],
  "rules_fired": ["B1", "C2", "C3", "E1", "E2"],
  "rules_checked": ["A1", "A2", "B2", "B3", "C1", "C4", "D1", "D2", "D3", "D4", "E3", "E4"],
  "evidence": [
    {
      "source": "straggler_detection_result.json",
      "field": "comm[0].metric_value",
      "slow_value": 2.9234,
      "direction": "min",
      "note": "display_key=tp[0, 1, 2, 3]"
    },
    {
      "source": "detection_report.log",
      "field": "tp/allgatherv 带宽表",
      "slow_value": 0.34,
      "baseline_value": 0.994,
      "ratio": 2.92,
      "direction": "min"
    },
    {
      "source": "detection_report.log",
      "field": "KERNEL_AICORE 排序",
      "slow_value": 15.00,
      "baseline_value": 9.41,
      "ratio": 1.59,
      "direction": "max"
    }
  ],
  "limitations": [
    "ep/mc2 各 1 个 Group 且无带宽数据，未评估",
    "ZP_Host 为节点级，无法定位 rank",
    "报告未提供 count，无法区分负载不均与算子劣化"
  ],
  "actions": [
    "检查 tp[0,1,2,3] 的通信链路/带宽/包大小",
    "复核 KERNEL_AICORE/KERNEL_AIVEC/MEMCPY_ASYNC 组间平移（检测层组内盲区）",
    "补采 count/频率数据后再判定负载不均/硬件劣化"
  ]
}
```

### 4.2 固定 Markdown

```
# 慢节点诊断报告

## 1. 数据与结果概览
- 数据目录、Job 类型、有效 Rank 数、并行域 Group 数
- 检测结果原文摘要

## 2. 检测结论
- verdict + confidence + 一句话

## 3. 慢侧/基准侧
- 慢侧 rank 列表、基准侧 rank 列表

## 4. 类别
- comm / compute / io / host / pp_comm / npu_bubble / multiple / insufficient_evidence

## 5. 规则命中与证据
| 规则 | 状态 | 证据（source::field = slow vs baseline, ratio） | 结论 |
|---|---|---|---|

## 6. 检测层与报告层一致性
- 已报且报告同向
- 未报但报告显示差异（标「一致性异常，建议复核」）
- 单组域/无带宽数据（未评估）

## 7. 结论与建议
- 主因、次要、建议动作

## 8. 置信度与局限
- high/medium/low + 理由 + 缺失数据
```

### 4.3 证据引用格式

每条结论必须写成：

```
<source>::<field> = <slow_value> vs <baseline_value>, ratio=<x>, direction=<max|min>
```

- `source` 只能是 `straggler_detection_result.json` 或 `analysis_result/detection_report.log`。
- 不允许出现 DB 表名、SQL、脚本输出。
- 字段缺失写 `insufficient_evidence`，并列出缺失项。

### 4.4 置信度

| 置信度 | 条件 |
|--------|------|
| 高 | JSON 异常 + 报告同向 + 组间模式清晰 + 无缺失数据 |
| 中 | 仅 JSON 或仅报告；或只有 1 个 opType；或报告未报但排序超阈值 |
| 低 | 单组域、无带宽数据、报告缺失、规则冲突 |
| — | `insufficient_evidence` |

---

## 5. 执行顺序

1. 先读 JSON + 报告，做数据覆盖检查（A1–A4）。
2. 再跑通信库（B1–B4）。
3. 再跑计算/搬运库（C1–C4）。
4. 再跑 Host/气泡/PP 库（D1–D4）。
5. 最后跑元规则库（E1–E4）。
6. 汇总 `primary` + `secondary_findings`，生成 JSON + Markdown。
7. 每条结论必须带规则 ID 和证据引用；**没有证据的结论不得写入**「根因假设」。

**停止条件：**

- JSON 为空且报告无异常排序/带宽差异 → `no_anomaly`，置信度最多 `medium`（因为覆盖限制）。
- JSON 为空但报告显示组间差异 → `insufficient_evidence` 或一致性异常，**不能写 `no_anomaly`**。
- JSON 非空但报告矛盾 → `insufficient_evidence`，并标「一致性异常」。

---

## 6. 明确禁止的规则（不要写进默认规则库）

以下规则需要现有输出之外的数据，**不要写进 Agent 默认规则库**；最多输出「建议人工复核」：

| 规则 | 需要的数据 | 处理 |
|------|-----------|------|
| 慢侧 vs 等待侧 | `COMMUNICATION_OP` 逐 rank duration | 移出；只输出「建议人工查时间线」 |
| 伪快卡（Free Time） | `StepDuration - ZP_Device` 逐 rank | 移出；标「证据不足」 |
| 逐算子带宽 | `COMMUNICATION_OP` + `STRING_IDS` | 移出；只引用报告带宽表 |
| gloo/CPU 停顿 | `PYTORCH_API` | 移出；只输出「建议人工查 Host」 |
| CANN API 慢 | `CANN_API` | 移出；只引用报告 ZP_Host 节点级 |
| 采集截断 | `SESSION_TIME_INFO`/算子数 | 移出；标「未评估」 |
| 负载不均（count） | 逐算子 count | 移出；标「证据不足」 |
| 频率异常 | `AICORE_FREQ` | 移出；标「证据不足」 |

> 如果以后要放开，必须单独写清楚**数据来源、字段、公式**，并作为 Tier 2 可选层，不能混进默认规则库。

---

## 7. A3 回归样例

数据目录：

```
D:\Sessions\a3_data\a3_data\2026_0921_qwen30b_a3b_tp4dp2_level0_comm_stress_rank0_1_2_3
```

**期望结果：**

- `verdict = slow_group`
- `confidence = high`
- `primary.category = comm`
- `primary.rule_id = B1`
- `primary.slow_side = [0,1,2,3]`
- `primary.baseline_side = [4,5,6,7]`
- `primary.degradation = 2.9234`
- `secondary_findings` 至少包含：
  - compute，C2，ratio=1.59（KERNEL_AICORE）
  - io，C2，ratio=4.49（MEMCPY_ASYNC）
- `limitations` 至少包含：
  - ep/mc2 各 1 个 Group 且无带宽数据，未评估
  - ZP_Host 为节点级，无法定位 rank
- `actions` 至少包含：
  - 检查 tp[0,1,2,3] 通信链路/带宽/包大小
  - 复核计算/搬运组间平移（检测层组内盲区）

**关键证据：**

- JSON：`comm[0].display_key = "tp[0, 1, 2, 3]"`，`metric_value = 2.9234`
- 报告 tp/allgather：`0,1,2,3 = 6.364`，`4,5,6,7 = 11.94`，`***` 在 0–3
- 报告 tp/allgatherv：`0,1,2,3 = 0.34`，`4,5,6,7 = 0.994`，`***` 在 0–3
- 报告 KERNEL_AICORE：0–3 为 14.60–15.00us，4–7 为 9.41–9.59us，max/min 1.59x，mean/median 1.01x
- 报告 MEMCPY_ASYNC：0–3 为 5.34–5.56us，4–7 为 1.24–1.27us，max/min 4.49x，mean/median 1.0x

**反例（不要出现）：**

- 把 `gloo:all_reduce 7.84s` 当作主因（报告和 JSON 都没有这个字段）。
- 把 `ZP_Bubble 19.27ms` 直接和 5000ns 阈值比较（口径不同）。
- 把报告「相对倍数 1.59x」直接当作检测劣化指数（口径不同）。
- 声称 ep/mc2 正常（报告明确写「无带宽数据」）。
- 让 Agent 去读 DB、写 SQL、写脚本。

---

## 8. 验收标准

1. 给定 A3 数据目录，Agent **只读两个文件**，输出符合 4.1 的 JSON 和 4.2 的 Markdown。
2. 输出中每条结论都能在 `evidence` 中找到对应 `source::field`。
3. 不出现 DB 表名、SQL、脚本调用。
4. 不把 ep/mc2 判为「正常」；不把 ZP_Bubble 和 5000ns 直接比较；不把报告「相对倍数」当作检测劣化指数。
5. 对 A3：`verdict=slow_group`、`primary.category=comm`、`primary.rule_id=B1`、`slow_side=[0,1,2,3]`、`baseline_side=[4,5,6,7]`、`degradation=2.9234`。
6. 对 A3：至少命中 C2（计算/搬运组间平移）和 E1（多 device）、E2（多类别同向）。
7. 对 A3：`limitations` 至少包含 ep/mc2 未评估、ZP_Host 节点级、count 缺失。

---

## 9. 文件修改建议

- 若 skill 已有 `SKILL.md`：新增一节「结果分析层」，把第 1–8 节写进去；**不要改检测流程**。
- 若 skill 没有 `SKILL.md`：新增 `SKILL.md` 或 `references/analysis_layer.md`，内容同上。
- 可新增 `references/output_schema.json` 放 4.1 的 JSON schema，方便校验。
- **不要新增 `analyze_result.py` 之类的 Python 分析器**；分析层是提示词/文档。
- 若 skill 有触发词，增加：
  - 「分析检测结果 / 为什么慢 / 根因是什么 / 谁慢」 → 走分析层。
  - 「重新检测 / 跑检测」 → 走原检测流程。
- 分析层开头必须写一句：
  > 本层只读 `straggler_detection_result.json` 和 `analysis_result/detection_report.log`；不读 DB、不跑 SQL、不写脚本。需要额外数据时输出 `insufficient_evidence`，并列出缺失项。

---

## 10. 一句话总结（给外部 Agent 的验收口径）

> 检测脚本和算法完全不动；新增的只是「读两个已有输出文件 → 按自然语言规则推理 → 输出固定 JSON + 固定 Markdown」的分析层。凡是需要新数据、新脚本、DB 查询的规则，一律降级为「建议人工复核/证据不足」，不允许 Agent 自行发挥。
