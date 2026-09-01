# Slow Node Detection - Python 版本

检测 AI 训练/推理集群中的慢节点（straggler / 亚健康检测）。解析 Ascend PyTorch Profiler 生成的 `.db` 文件，识别慢计算卡、慢通信域、慢 CPU 卡、NPU 空泡，并生成故障联合分析报告。

## 目录结构

```
straggler-detector/
├── __init__.py                 # 包初始化
├── config.py                   # 配置、阈值、劣化数据容器、节点/域名标志
├── utils.py                    # 结果写入、清理、通用工具
├── kmeans_detector.py          # 通用检测算法（KMeans + Z-score + 肘部法 + 逐轮剥离）
├── profilingdataparse.py       # Profiling 数据解析（SQLite → CSV/JSON）
├── nodelevel.py                # 慢节点检测核心逻辑
├── nodelevel_data_handler.py   # 数据读取、检测组选择、节点分组
├── joint_analysis.py           # 故障联合分析（传播链 + 根因 + 报告）
├── markdown_viz.py / visualizer.py  # 可视化报告
├── main.py                     # 主入口
├── skill.md                    # skill 说明（与 SKILL.md 同步）
└── README.md / SPEC.md         # 本文档 / 方案设计
```

## 使用方法

### 作为 skill 被 Claude Code 调用

```python
import main
result = main.run_detection("/path/to/data", degradation=0.3, clean="ask")
```

### 命令行执行

```bash
cd "C:\Users\n30082019\.claude\skills\straggler-detector"
python main.py path=/your/data/path degradation=0.3 clean=ask
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `path` | 数据目录路径（必需） | - |
| `degradation` | 劣化阈值 | 0.3 |
| `clean` | `yes`/`no`/`ask`：是否清理中间数据并重新解析 | `ask` |

> **执行前必须询问用户**是否删除已有中间数据（`op_metric` 等），得到明确答复后再执行：删除 → `clean=yes`，保留 → `clean=no`。

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
  "memcpy_async": [...],
  "host_duration": [...],
  "step_duration": [...],
  "xp_count": [...]
}
```

### 3. 报告文件

- `joint_failure_analysis.log` — 故障联合分析报告（传播链 + 根因 + 条形图）
- `analysis_result/detection_report.log` — 可视化详情报告

## 检测的 9 类指标

| 类别 | 指标列 | 检测方式 |
|------|--------|----------|
| `step_duration` | `StepDuration` | 通信域组间对比 |
| `comm` | `{xp}_Duration` | 通信域组间对比 |
| `xp_count` | `{xp}_Count` | 通信域组间对比 |
| `KERNEL_AICORE` | `KERNEL_AICORE` | 检测组内 + 通用算法 |
| `kernel_aivec` | `KERNEL_AIVEC` | 检测组内 + 通用算法 |
| `memcpy_async` | `MEMCPY_ASYNC` | 检测组内 + 通用算法 |
| `npu_bubble` | `ZP_Bubble` | 单阈值 < 5000ns |
| `host_duration` | `HostDuration` | 节点对齐 + 通用算法 |
| `cpu` | `ZP_Host` | 节点对齐 + 通用算法 |

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
  ├── detection_all_communication_parallel()→ comm / step_duration / xp_count（有命名域时）
  ├── get_slow_host_ranks_by_homogenize()  → cpu
  └── _get_slow_host_metric_ranks()        → host_duration
        │
        ▼
输出：straggler_detection_result.json / joint_failure_analysis.log / detection_report.log
```

## 核心算法（kmeans_detector.py）

`general_anomaly_detection`：过滤 ≤0/-99999 → Z-score → 肘部法选 K → KMeans++ → 偏大方向异常簇（簇均值 > 基线×倍率）→ **逐轮剥离**（剔除异常簇数据后对剩余数据再聚类，轮数 ≤10，各轮异常与当轮基线累积，degradation = 值/当轮基线）。异常倍率由 `degradation` 决定：计算/IO/Host 类 = `1+degradation`，通信域类 = `1+5×degradation`；`npu_bubble` 用固定阈值 `< 5000ns`。

检测组由 `nodelevel.get_cal_detection_group` 按优先级（tp→exp→ep→…→dp）选定，集群数据用完整分组、非集群按节点过滤；无命名通信域时退化按 hostUid 物理节点分组（通信域组间指标直接跳过）。

## 依赖

- Python 3.8+
- 标准库：`sqlite3`, `csv`, `json`, `os`, `logging`, `math`, `random`

## 版本

2.0.0 - 新核心通用检测算法（KMeans + Z-score + 肘部法 + 逐轮剥离），9 类指标
