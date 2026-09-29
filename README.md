# Slow Node Detection - Python 版本

检测 AI 训练/推理集群中的慢节点（straggler / 亚健康检测）。解析 Ascend PyTorch Profiler 生成的 `.db` 文件，识别慢计算卡、慢通信域、慢 CPU 卡、NPU 空泡，并生成文本检测报告。

## 目录结构

```
straggler-detector/
├── __init__.py                 # 包初始化
├── config.py                   # 配置、阈值、劣化数据容器、节点/域名标志
├── utils.py                    # 结果写入、清理、通用工具
├── kmeans_detector.py          # 通用检测算法（KMeans + Z-score + 肘部法 + 异常簇递归细分）
├── profilingdataparse.py       # Profiling 数据解析（SQLite → CSV/JSON）
├── nodelevel.py                # 慢节点检测核心逻辑
├── nodelevel_data_handler.py   # 数据读取、检测组选择、节点分组
├── summary_table.py            # 检测结果汇总表工具（ASCII 表格渲染 + 单元格格式化）
├── markdown_viz.py / visualizer.py  # 可视化报告
├── main.py                     # 主入口
├── skill.md                    # skill 说明（与 SKILL.md 同步）
└── README.md / SPEC.md         # 本文档 / 方案设计
```

## 使用方法

### 作为 skill 被 Claude Code 调用

```python
import main
result = main.run_detection("/path/to/data", compute=1.3, io=2.5, comm=1.3, clean="ask")
```

### 命令行执行

```bash
cd "C:\Users\n30082019\.claude\skills\straggler-detector"
python main.py path=/your/data/path compute=1.3 io=2.5 comm=1.3 clean=ask
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `path` | 数据目录路径（必需） | - |
| `compute` | 计算类阈值（KERNEL_AICORE / kernel_aivec） | 1.3 |
| `io` | IO/CPU 类阈值（cpu / memcpy_async） | 2.5 |
| `comm` | 通信类阈值（comm / pp_comm） | 1.3 |
| `clean` | `yes`/`no`/`ask`：是否清理中间数据并重新解析 | `ask` |

> `npu_bubble` 为固定硬阈值 `< 5000ns`，不询问。
> **执行前必须询问用户**三组阈值（计算/IO/通信）与是否删除已有中间数据（`op_metric` 等），得到明确答复后再执行：删除 → `clean=yes`，保留 → `clean=no`。

## 输入数据格式

`path` 可为一个含多层子目录的父目录（`os.walk` 递归查找 `ascend_pytorch_profiler_*.db`）：

```
<path>/
├── ascend_pytorch_profiler_0.db
├── ascend_pytorch_profiler_1.db
└── ...
```

或常见 dump 结构：`master_xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_N.db`（空 master db 会被跳过，仅用含核心表的 rank db）。

## 输出格式

### 1. CSV 文件（op_metric/）

每张卡一条聚合数据行，列：`StepIndex, StepDuration, ZP_Device, ZP_Duration, ZP_Host, ZP_Bubble, ZP_Count, KERNEL_AICORE, MEMCPY_ASYNC, KERNEL_AIVEC, HostDuration, DataLoader` + 各并行域 `{xp}_Duration, {xp}_Count`。

### 2. 检测结果（straggler_detection_result.json）

```json
{
  "KERNEL_AICORE": [{"display_key": "0", "metric_value": 1.5, "is_abnormal": true}],
  "comm": [{"display_key": "tp[0, 1]", "metric_value": 1.8, "is_abnormal": true}],
  "cpu": [...],
  "npu_bubble": [...],
  "kernel_aivec": [...],
  "memcpy_async": [...]
}
```

### 3. 报告文件

- `analysis_result/detection_report.log` — 可视化详情报告

### 4. 最终输出逐类别汇总表（stdout）

检测结束时在 stdout 打印一张 **ASCII 框线表格**，一行一个"有异常的类别"，列：`类别 | 异常卡 | 劣化指数 | 劣化阈值`。它渲染到**调用方 agent 的最终输出**，不进任何 log 文件；无异常时打印含"无异常"提示的单行表：

```
+---------------+-----------+------------------+----------+
| 类别          | 异常卡    | 劣化指数         | 劣化阈值 |
+---------------+-----------+------------------+----------+
| KERNEL_AICORE | rank 0, 3 | 0:1.397，3:1.398 | 1.3x     |
+---------------+-----------+------------------+----------+
| comm          | tp[0, 1]  | tp[0, 1]:2.5     | 1.3x     |
+---------------+-----------+------------------+----------+
| pp_comm       | 0->4      | 0->4:2.5         | 1.3x     |
+---------------+-----------+------------------+----------+
```

> 类别列用 op_metric 指标列名（如 `KERNEL_AIVEC`、`MEMCPY_ASYNC` 大写）；劣化指数列为 `项:值`（单卡 `rank:值`、通信组 `tp[0,1]:值`、PP 链路 `0->4:值`）。

## 检测的 7 类指标

| 类别 | 指标列 | 检测方式 |
|------|--------|----------|
| `comm` | `{domain}_{opType}_{count}`（带宽） | 带宽聚类（min 方向，多 opType 交叉验证） |
| `pp_comm` | `PP_Overlap` + `PP_Count`（PP 链路重叠 / 传输字节数） | 按 stage 位置聚类（max 方向，按 count 容差细分；batch `<->` / send-recv `->`） |
| `KERNEL_AICORE` | `KERNEL_AICORE` | 检测组内 + 通用算法 |
| `kernel_aivec` | `KERNEL_AIVEC` | 检测组内 + 通用算法 |
| `memcpy_async` | `MEMCPY_ASYNC` | 检测组内 + 通用算法 |
| `npu_bubble` | `ZP_Bubble` | 单阈值 < 5000ns |
| `cpu` | `ZP_Host` | 按物理节点拉齐 + 通用算法（异常项 key=hostName） |

## 算法流程

```
输入：SQLite profiling 数据库
        │
        ▼
[profilingdataparse.data_parsing]
  递归查找 *.db → 解析各表 → op_metric/global_rank_N.csv + group_info_N.json
  读取 HOST_INFO.hostUid → config.HostRankMap（内存）
        │
        ▼
[nodelevel_data_handler.get_cur_detection_info]
  聚合 group_info → parallels {域名: [[rank组]...]}
  设置 HasNamedDomain / IsClusterData 标志
        │
        ▼
[nodelevel_data_handler.get_cur_job_last_step_data]
  读 CSV → 单快照 {metric: {rank: value}}（多行取倒数第二行）
        │
        ▼
[nodelevel.delimit_detection]
  ├── detection_zp_bubble_data()           → npu_bubble
  ├── get_slow_calculate_ranks()           → KERNEL_AICORE
  ├── get_slow_metric_ranks() ×2           → kernel_aivec / memcpy_async
  ├── detect_slow_domain_by_bandwidth()    → comm（带宽聚类，有命名域时）
  ├── detect_pp_slow_domain()              → pp_comm（PP 等待占比）
  ├── get_slow_host_ranks_by_homogenize()  → cpu
        │
        ▼
输出：straggler_detection_result.json / analysis_result/detection_report.log
```

## 核心算法（kmeans_detector.py）

`general_anomaly_detection`：过滤 ≤0/-99999 → Z-score → 肘部法选 K → KMeans++ → 偏大方向异常簇（簇均值 > 基线×阈值）→ **异常簇递归细分**（对异常簇数据再次聚类，更深层异常替换父层、更深层无异常保持父层，向外排除边缘成员减少误检；劣化指数统一用**第一次 KMeans（全数据）的基线簇均值**为分母，degradation = 异常值/第一次基线，同一刻度可比）。阈值分组（skill 调用时询问用户）：计算类（KERNEL_AICORE / kernel_aivec）= `compute`（默认 1.3），IO/CPU 类（cpu / memcpy_async）= `io`（默认 2.5），通信类（comm / pp_comm）= `comm`（默认 1.3）；`npu_bubble` 用固定阈值 `< 5000ns`。

检测组由 `nodelevel.get_cal_detection_group` 按优先级（tp→exp→ep→…→dp）选定，集群数据用完整分组、非集群按节点过滤；无命名通信域时退化按 hostUid 物理节点分组（通信域组间指标直接跳过）。

## 依赖

- Python 3.8+
- 标准库：`sqlite3`, `csv`, `json`, `os`, `logging`, `math`, `random`

## 版本

2.2.0 - 移除 host_duration 指标（6 类），Host 侧仅保留 cpu
