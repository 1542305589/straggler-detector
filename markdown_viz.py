"""
报告生成模块 - 为慢节点检测结果生成文本格式报告
替代原有 matplotlib PNG 图表和 Markdown 格式，直接输出到 log 文件

功能:
1. 水平柱状图（使用 Unicode 字符）
2. 排序表格 + 统计摘要
3. 异常高亮
4. 并行域通信耗时排序
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime

# 添加父目录到路径以便导入 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger("[REPORT]")

# 柱状图配置
BAR_CHAR = "█"
BAR_MAX_WIDTH = 40
TOP_N = 30  # 显示 Top N 最慢
BOTTOM_N = 5  # 显示 Bottom N 最快
SEP = "="  # 分隔符字符

# 类别 -> 该类别在 step_data 中对应的单卡指标列（用于生成排序柱状图并高亮异常卡）
CATEGORY_METRIC = {
    "KERNEL_AICORE": "KERNEL_AICORE",
    "kernel_aivec": "KERNEL_AIVEC",
    "memcpy_async": "MEMCPY_ASYNC",
    "cpu": "ZP_Host",
    "host_duration": "HostDuration",
    "npu_bubble": "ZP_Bubble",
}


def _fmt_ns(value: float) -> str:
    """将纳秒格式化为可读单位"""
    if value >= 1e9:
        return f"{value/1e9:.2f}s"
    elif value >= 1e6:
        return f"{value/1e6:.2f}ms"
    elif value >= 1e3:
        return f"{value/1e3:.2f}us"
    else:
        return f"{value:.0f}ns"


def _filter_valid(data: Dict[int, float]) -> Dict[int, float]:
    """过滤掉 -99999 和 <=0 的无效数据"""
    return {k: v for k, v in data.items() if v != -99999 and v > 0}


def _bar(value: float, max_value: float) -> str:
    """生成水平柱状图字符串"""
    if max_value <= 0:
        return ""
    width = max(1, int(value / max_value * BAR_MAX_WIDTH))
    return BAR_CHAR * width


def _sep_line(title: str = "", width: int = 70) -> str:
    """生成装饰分隔线"""
    if title:
        pad = (width - len(title) - 2) // 2
        return f"{SEP * pad} {title} {SEP * pad}" if pad > 0 else title
    return SEP * width


def _metric_section(
    metric_name: str,
    data: Dict[int, float],
    abnormal_ranks: Optional[List[int]] = None,
) -> str:
    """生成单个指标的排序柱状图（纯文本）"""
    filtered = _filter_valid(data)
    if not filtered:
        return f"\n[{metric_name}] 无有效数据\n"

    abnormal_set = set(abnormal_ranks or [])
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_items]
    max_value = values[0] if values else 1
    mean_val = sum(values) / len(values)

    lines = []
    lines.append("")
    lines.append(_sep_line(f"{metric_name} 耗时排序", 70))
    lines.append(f"  展示 Top {TOP_N} 最慢 + Bottom {BOTTOM_N} 最快")
    lines.append("")

    total = len(sorted_items)
    display_items = []
    display_items.extend(sorted_items[:TOP_N])
    if total > TOP_N + BOTTOM_N:
        display_items.append(None)
    if BOTTOM_N > 0:
        display_items.extend(sorted_items[-BOTTOM_N:] if total > TOP_N else [])

    top_max = sorted_items[0][1] if sorted_items else 1

    # 列头
    lines.append(f"  {'#':>3}  {'Rank':>6}  {'耗时':>10}  {'劣化指数':>8}  柱状图")
    lines.append(f"  {'---':>3}  {'------':>6}  {'----------':>10}  {'--------':>8}  -------")

    idx = 0
    for item in display_items:
        if item is None:
            mid = total - TOP_N - BOTTOM_N
            lines.append(f"  ...  ......  ..........  ........  (中间 {mid} 卡略)")
            continue

        rank, val = item
        idx += 1
        bar = _bar(val, top_max)
        ratio = val / mean_val if mean_val > 0 else 1
        is_abnormal = rank in abnormal_set
        marker = " ***" if is_abnormal else ""
        lines.append(f"  {idx:>3}  {rank:>6}  {_fmt_ns(val):>10}  {ratio:>7.2f}x  {bar}{marker}")

    # 标记说明
    if abnormal_set:
        lines.append("")
        lines.append("  *** = 异常卡")

    # 统计信息
    lines.append("")
    lines.append(f"  {'-- 统计信息':-<40}")
    sorted_vals = sorted(values)
    n = len(values)
    median_val = sorted_vals[n // 2] if n % 2 == 1 else \
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    max_val = sorted_vals[-1]
    min_val = sorted_vals[0]
    lines.append(f"    总卡数:      {n}")
    lines.append(f"    最大值:      {_fmt_ns(max_val)}")
    lines.append(f"    最小值:      {_fmt_ns(min_val)}")
    lines.append(f"    均值:        {_fmt_ns(mean_val)}")
    lines.append(f"    中位数:      {_fmt_ns(median_val)}")
    if n >= 2:
        max_min_ratio = max_val / min_val if min_val > 0 else float('inf')
        mean_median_ratio = mean_val / median_val if median_val > 0 else float('inf')
        lines.append(f"    最大/最小比:  {max_min_ratio:.2f}x")
        lines.append(f"    均值/中位数比: {mean_median_ratio:.2f}x")
    lines.append("")

    return "\n".join(lines)


def _comm_section(
    domain_name: str,
    domain_groups: List[List[int]],
    comm_data: Dict[int, float],
    abnormal_groups: Optional[List[List[int]]] = None,
) -> str:
    """生成并行域通信耗时（纯文本）"""
    group_stats = []
    for group in domain_groups:
        valid_vals = [comm_data.get(r, -99999) for r in group
                      if comm_data.get(r, -99999) != -99999 and comm_data.get(r, -99999) > 0]
        if not valid_vals:
            continue
        min_val = min(valid_vals)
        max_val = max(valid_vals)
        mean_val = sum(valid_vals) / len(valid_vals)
        group_label = ",".join(str(r) for r in group)
        group_stats.append((group_label, min_val, max_val, mean_val, valid_vals, group))

    if not group_stats:
        return f"\n[{domain_name}] 无有效通信数据\n"

    group_stats.sort(key=lambda x: x[1], reverse=True)

    abnormal_set = set()
    if abnormal_groups:
        for ag in abnormal_groups:
            abnormal_set.add(",".join(str(r) for r in ag))

    lines = []
    lines.append("")
    lines.append(_sep_line(f"{domain_name} 并行域 - 实际集合通信耗时", 70))
    lines.append("  实际集合通信耗时 = 组内通信耗时最短的卡的值")
    lines.append("")

    min_vals = [s[1] for s in group_stats]
    max_min = max(min_vals) if min_vals else 1

    lines.append(f"  {'#':>3}  {'Group':>20}  {'实际耗时(min)':>14}  {'组内均值':>10}  {'组内最大':>10}  柱状图")
    lines.append(f"  {'---':>3}  {'--------------------':>20}  {'--------------':>14}  {'----------':>10}  {'----------':>10}  -------")

    for i, (label, min_v, max_v, mean_v, vals, group) in enumerate(group_stats, 1):
        bar = _bar(min_v, max_min)
        is_abnormal = label in abnormal_set
        marker = " ***" if is_abnormal else ""
        lines.append(
            f"  {i:>3}  {label:>20}  {_fmt_ns(min_v):>14}  "
            f"{_fmt_ns(mean_v):>10}  {_fmt_ns(max_v):>10}  {bar}{marker}"
        )

    lines.append("")
    overall_mean = sum(min_vals) / len(min_vals)
    overall_min = min(min_vals)
    overall_max = max(min_vals)
    lines.append(f"  总组数: {len(group_stats)}  |  总均值: {_fmt_ns(overall_mean)}  |  范围: {_fmt_ns(overall_min)} ~ {_fmt_ns(overall_max)}")
    if overall_min > 0:
        lines.append(f"  最大/最小比: {overall_max / overall_min:.2f}x")

    if abnormal_set:
        lines.append("  *** = 异常 Group")
    lines.append("")

    return "\n".join(lines)


def _detection_summary(
    detection_result: Dict[str, Dict[str, float]],
    valid_ranks: List[int],
    degradation: float,
) -> str:
    """生成检测结果摘要"""
    type_names = {
        "KERNEL_AICORE": "慢计算 (KERNEL_AICORE)",
        "kernel_aivec": "矢量计算 (kernel_aivec)",
        "memcpy_async": "内存搬运 (memcpy_async)",
        "comm": "慢通信 (comm)",
        "step_duration": "Step时长 (step_duration)",
        "xp_count": "通信计数 (xp_count)",
        "cpu": "慢CPU (cpu)",
        "host_duration": "Host耗时 (host_duration)",
        "npu_bubble": "Bubble (npu_bubble)",
    }
    # 覆盖动态类别
    for key in detection_result:
        if key not in type_names:
            type_names[key] = key

    lines = []
    lines.append(f"  {'检测类型':<22} {'状态':<10} {'异常卡数':<10} 异常详情")
    lines.append(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*30}")

    for key, name in type_names.items():
        if key in detection_result and detection_result[key]:
            items = detection_result[key]
            details = "; ".join(f"{rk}: {ratio:.2f}x" for rk, ratio in items.items())
            if len(details) > 80:
                details = details[:77] + "..."
            lines.append(f"  {name:<22} {'异常':<10} {len(items):<10} {details}")
        else:
            lines.append(f"  {name:<22} {'正常':<10} {0:<10} -")

    lines.append("")
    lines.append(f"  总 Rank 数: {len(valid_ranks)}  |  劣化阈值: {degradation}")
    return "\n".join(lines)


def _comm_total_section(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
) -> str:
    """
    生成总通信耗时排序（对所有并行域的通信耗时求和）

    遍历所有并行域，将每张卡在各域中的通信耗时相加得到总通信时间。
    如果没有域级别的 Duration 数据，则使用 ZP_Duration 作为备选。
    """
    # 找出所有并行域对应的 _Duration key（包含所有域）
    parallel_names = [n for n in (list(parallels.keys()) if parallels else [])
                      if n]
    domain_keys = [f"{name}_Duration" for name in parallel_names]

    # 检查是否有有效的域 Duration 数据
    has_domain_data = False
    for dk in domain_keys:
        if dk in step_data:
            vals = [v for v in step_data[dk].values() if v != -99999 and v > 0]
            if vals:
                has_domain_data = True
                break

    # 检查是否还有 _Duration（空域名）也应该加入
    if "" in (list(parallels.keys()) if parallels else []):
        if "_Duration" in step_data:
            domain_keys.append("_Duration")

    if has_domain_data:
        # 方法1: 对各域 Duration 求和
        comm_totals: Dict[int, float] = {}
        for dk in domain_keys:
            if dk not in step_data:
                continue
            for rank, val in step_data[dk].items():
                if val != -99999 and val > 0:
                    comm_totals[rank] = comm_totals.get(rank, 0) + val

        subtitle = "各域通信耗时求和: " + ", ".join(
            dk.replace("_Duration", "") for dk in domain_keys
            if dk in step_data and any(v > 0 and v != -99999 for v in step_data[dk].values())
        )
    else:
        # 方法2: 备选，使用 ZP_Duration
        zp_dur = step_data.get("ZP_Duration", {})
        comm_totals = {k: v for k, v in zp_dur.items() if v != -99999 and v > 0}
        subtitle = "无域通信数据，使用 ZP_Duration（总通信耗时）"

    if not comm_totals:
        return ""

    # 与 _metric_section 相同的格式输出
    sorted_items = sorted(comm_totals.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_items]
    max_value = values[0] if values else 1
    mean_val = sum(values) / len(values)
    n = len(values)

    lines = []
    lines.append("")
    lines.append(_sep_line("总通信耗时排序", 70))
    lines.append(f"  {subtitle}")
    lines.append(f"  展示 Top {TOP_N} 最慢 + Bottom {BOTTOM_N} 最快")
    lines.append("")

    total = len(sorted_items)
    display_items = []
    display_items.extend(sorted_items[:TOP_N])
    if total > TOP_N + BOTTOM_N:
        display_items.append(None)
    if BOTTOM_N > 0:
        display_items.extend(sorted_items[-BOTTOM_N:] if total > TOP_N else [])

    top_max = sorted_items[0][1] if sorted_items else 1

    lines.append(f"  {'#':>3}  {'Rank':>6}  {'耗时':>10}  {'占比':>8}  柱状图")
    lines.append(f"  {'---':>3}  {'------':>6}  {'----------':>10}  {'--------':>8}  -------")

    idx = 0
    for item in display_items:
        if item is None:
            mid = total - TOP_N - BOTTOM_N
            lines.append(f"  ...  ......  ..........  ........  (中间 {mid} 卡略)")
            continue
        rank, val = item
        idx += 1
        bar = _bar(val, top_max)
        ratio = val / mean_val if mean_val > 0 else 1
        lines.append(f"  {idx:>3}  {rank:>6}  {_fmt_ns(val):>10}  {ratio:>7.2f}x  {bar}")

    # 统计信息
    lines.append("")
    lines.append(f"  {'-- 统计信息':-<40}")
    sorted_vals = sorted(values)
    median_val = sorted_vals[n // 2] if n % 2 == 1 else \
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    max_val = sorted_vals[-1]
    min_val = sorted_vals[0]
    lines.append(f"    总卡数:      {n}")
    lines.append(f"    总通信最大:  {_fmt_ns(max_val)}")
    lines.append(f"    总通信最小:  {_fmt_ns(min_val)}")
    lines.append(f"    均值:        {_fmt_ns(mean_val)}")
    lines.append(f"    中位数:      {_fmt_ns(median_val)}")
    if n >= 2:
        max_min_ratio = max_val / min_val if min_val > 0 else float('inf')
        lines.append(f"    最大/最小比:  {max_min_ratio:.2f}x")
    lines.append("")

    return "\n".join(lines)


def generate_report(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    input_path: str = "",
    degradation: float = 0.3,
) -> str:
    """
    生成完整文本检测报告

    返回:
        纯文本报告字符串
    """
    sections = []
    sections.append("")
    sections.append(_sep_line("慢节点检测报告", 70))
    sections.append("")
    sections.append(f"  数据目录: {input_path}")
    sections.append(f"  Job 类型: {config.get_job_type()}")
    sections.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"  有效 Rank 数: {len(valid_ranks)}")
    sections.append("")

    # 并行域拓扑
    if parallels:
        sections.append(_sep_line("并行域拓扑", 50))
        sections.append("")
        for domain_name, domain_groups in parallels.items():
            name = domain_name if domain_name else "(unnamed)"
            sections.append(f"  {name}: {len(domain_groups)} 个 Group")
        sections.append("")

    # 检测结果摘要
    if detection_result and any(detection_result.values()):
        sections.append(_sep_line("检测结果摘要", 50))
        sections.append("")
        sections.append(_detection_summary(detection_result, valid_ranks, degradation))
        sections.append("")

    # Part 1: 单卡指标排序柱状图（KERNEL_AICORE/kernel_aivec/memcpy_async/cpu/host_duration/npu_bubble）
    cat_to_metric = dict(CATEGORY_METRIC)
    rendered_metric_cols = set()
    for cat, metric_name in cat_to_metric.items():
        if metric_name not in step_data or metric_name in rendered_metric_cols:
            continue
        rendered_metric_cols.add(metric_name)
        if not _filter_valid(step_data[metric_name]):
            logger.warning(f"{metric_name} 所有数据均无效，跳过")
            continue

        abnormal_ranks = []
        if detection_result and cat in detection_result:
            try:
                abnormal_ranks = [int(rk) for rk in detection_result[cat].keys()]
            except ValueError:
                pass

        sections.append(_metric_section(metric_name, step_data[metric_name], abnormal_ranks))

    # Part 1.5: 总通信耗时排序（各域通信耗时求和）
    if parallels:
        sections.append(_comm_total_section(step_data, parallels))

    # Part 2: 每个并行域的通信耗时
    if parallels:
        comm_abnormal = detection_result.get("comm", {}) if detection_result else {}

        for domain_name, domain_groups in parallels.items():
            if not domain_name or not domain_groups:
                continue

            duration_key = f"{domain_name}_Duration"
            if duration_key in step_data and step_data[duration_key]:
                comm_data = step_data[duration_key]
            else:
                raw_data = step_data.get(domain_name, {})
                if not raw_data:
                    continue
                comm_data = raw_data

            valid_comm = _filter_valid(comm_data)
            if not valid_comm:
                continue

            abnormal_groups = []
            for group_key in comm_abnormal:
                try:
                    ranks = [int(r) for r in group_key.split(",")]
                    abnormal_groups.append(ranks)
                except (ValueError, AttributeError):
                    pass

            sections.append(_comm_section(domain_name, domain_groups, comm_data, abnormal_groups))

    # Part 3: 各并行域 per-rank 通信耗时排序（不加和）
    if parallels:
        comm_abnormal = detection_result.get("comm", {}) if detection_result else {}
        abnormal_comm_ranks = set()
        for group_key in comm_abnormal:
            try:
                for r in group_key.split(","):
                    abnormal_comm_ranks.add(int(r))
            except (ValueError, AttributeError):
                pass

        for domain_name in parallels:
            if not domain_name:
                continue
            duration_key = f"{domain_name}_Duration"
            if duration_key not in step_data:
                continue
            filtered = _filter_valid(step_data[duration_key])
            if not filtered:
                continue
            sections.append(_metric_section(
                duration_key,
                step_data[duration_key],
                list(abnormal_comm_ranks) if abnormal_comm_ranks else None,
            ))

    sections.append(_sep_line("", 70))
    sections.append("")

    return "\n".join(sections)


def write_report(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    input_path: str = "",
    degradation: float = 0.3,
) -> str:
    """
    生成并写入文本报告

    返回:
        报告文件路径
    """
    report = generate_report(
        step_data, parallels, valid_ranks, output_dir,
        detection_result=detection_result,
        input_path=input_path,
        degradation=degradation,
    )

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "detection_report.log")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"检测报告已保存: {report_path}")
    print(f"\n检测报告已保存: {report_path}")
    return report_path
