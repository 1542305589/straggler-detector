# 原始 profiler `.db` 侦察清单（分析层补数据用）

> **目的**：确认「只读查原始 profiler `.db`」能不能把分析层需要的证据算出来。
>
> **现状**：本机**没有任何** `ascend_pytorch_profiler_*.db`（已全盘确认）。下面所有表/列都是从
> `profilingdataparse.py`（我们）和 msagent 的 `compare_*.py`（他们）**推断**出来的，**未经真数据验证**。
>
> **用法**：到有数据的机器上逐项探明。探不到的项 → 对应证据一律标 `not_evaluated`，不许猜。

---

## 0. 总目标：这 5 条证据链能不能自己算出来

| # | 证据（来自那份诊断报告） | 他们用的来源 | 我们能否从裸 `.db` 算出 |
|---|---|---|---|
| 1 | Host 侧 `gloo:all_reduce` 7.84s | `PYTORCH_API` | ? |
| 2 | 空闲窗口 7844ms（NPU 被饿死） | `free_analysis` | ? |
| 3 | `transit_time=0` / `idle_time=elapsed`（纯等待，非传输） | `ClusterCommunicationTime` | ? |
| 4 | `aclrtSynchronizeDeviceWithTimeout` 7.86s | `cann_api_sum` | ? |
| 5 | 每卡算子次数 / 总耗时 / 平均 | `compute_op_sum` | ? |

> 注意：他们的规则**吃的是 `msprof-analyze` 的产物**，我们是裸 db。**能搬的是判定规则，取数必须自己重写。**

---

## 1. Schema 基础（先做，其余全部依赖它）

- [ ] `SELECT name FROM sqlite_master WHERE type='table'` —— 全表清单
- [ ] `PRAGMA table_info(<t>)` —— 关键表逐列确认：
      `TASK` / `COMPUTE_TASK_INFO` / `COMMUNICATION_OP` / `CANN_API` / `PYTORCH_API` /
      `MSTX_EVENTS` / `STEP_TIME` / `STRING_IDS` / `HOST_INFO` / `NPU_INFO` /
      `META_DATA` / `SESSION_TIME_INFO`
- [ ] 每张表的行数量级（决定要不要索引、要不要强制 `LIMIT`）
- [ ] 时间列单位确认（ns？）
- [ ] 哪些列是 `STRING_IDS` 的 id 外键：
      `TASK.taskType`、`PYTORCH_API.name`、`CANN_API.name`、
      `COMMUNICATION_OP.opName`、`COMPUTE_TASK_INFO.name`

---

## 2. 计算 / 搬运类（复现 `compute_op_sum`）

- [ ] `TASK.taskType` 的**取值全集**（`KERNEL_AICORE` / `KERNEL_AIVEC` / `MEMCPY_ASYNC` / …）
- [ ] `COMPUTE_TASK_INFO` 能否**按算子名**聚合；`COUNT(*)` 与 `SUM(endNs-startNs)` 是否合理
- [ ] 单 rank 的 `TASK` 行数（`COMMUNICATION_OP` 可能上千万行）

---

## 3. Host 阻塞 / 伪快卡（最关键的一块）

- [ ] `PYTORCH_API` 有 `startNs` / `endNs` 吗？能不能按名字聚合出 `gloo:all_reduce` 的**时长**
- [ ] `CANN_API.name` 里有没有：`aclrtSynchronizeDeviceWithTimeout` /
      `aclrtSynchronizeEvent` / `aclrtSynchronizeStreamWithTimeout`
- [ ] **空闲窗口能否从 `TASK` 时间线自己推**：按 start 排序 → 相邻任务间隙求和
- [ ] `MSTX_EVENTS.message` 的取值（有没有 step / iteration 锚点）
- [ ] `CANN_API` ↔ `TASK` 的关联键（`connectionId`？）

---

## 4. 通信 / 慢链路

- [ ] `COMMUNICATION_OP` 是只有 `start/end/count`，**还是**有 `wait_time` / `transit_time` / `idle_time` 列？
      （他们报告里的 `transit/idle` 来自 msprof-analyze，我们可能得自己从 start/end 推）
- [ ] `groupName` / `opName` 与并行域的映射；`META_DATA.parallel_group_info` 的**实际结构**
- [ ] `count` 的单位（元素数？字节？）→ 能不能算带宽
- [ ] `HOST_INFO.hostUid` / `hostName`、`NPU_INFO` 与 rank 的映射关系

---

## 5. 采集完整性 / 时间对齐

- [ ] `SESSION_TIME_INFO` 的窗口
- [ ] `STEP_TIME` 行数 vs `TASK` 覆盖范围（判断是否**采集截断**）
- [ ] 多个 rank 的 db 时间戳是否同一**墙钟基准**
      （他们的报告强调：跨 Rank 比较必须以墙钟时间戳为准，不能只看 step 编号）

---

## 6. 二级数据（不用探，已知）

- [ ] `op_metric/global_rank_N.csv`：固定列 + `{域}_Duration` / `{域}_Count` +
      `{域}_{opType}_{count}` 带宽列
- [ ] `op_metric/group_info_N.json`：并行域信息

---

## 7. 灵敏度 / 兼容性

- [ ] 换一个 CANN 版本的 db，哪些表/列会缺？（决定「能力矩阵」怎么降级）
- [ ] 哪些查询必须建索引才不超时
