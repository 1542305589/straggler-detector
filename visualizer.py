"""
可视化模块 - 为慢节点检测结果生成 Markdown 格式报告

功能:
1. KERNEL_AICORE / ZP_Host 排序柱状图（Markdown 版本）
2. 每个并行域的集合通信耗时图（Markdown 版本）
3. 检测结果摘要
"""

import os
import logging
from typing import Dict, List, Optional
from collections import defaultdict

import markdown_viz

logger = logging.getLogger("[VISUALIZER]")


def _filter_valid(data: Dict[int, float]) -> Dict[int, float]:
    """过滤掉 -99999 和 <=0 的无效数据"""
    return {k: v for k, v in data.items() if v != -99999 and v > 0}


def _format_ns(value: float) -> str:
    """将纳秒格式化为可读单位"""
    if value >= 1e9:
        return f"{value/1e9:.2f}s"
    elif value >= 1e6:
        return f"{value/1e6:.2f}ms"
    elif value >= 1e3:
        return f"{value/1e3:.2f}us"
    else:
        return f"{value:.0f}ns"


def plot_metric_bar(
    data: Dict[int, float],
    metric_name: str,
    output_path: str,
    threshold_value: float = None,
    threshold_label: str = None,
    abnormal_ranks: Optional[List[int]] = None,
):
    """
    生成单个指标的 Markdown 柱状图（替代原有 matplotlib 版本）

    注意: output_path 参数保留是为了兼容，实际输出为 Markdown 报告的一部分
    """
    # 直接输出到控制台作为实时反馈
    filtered = _filter_valid(data)
    if not filtered:
        logger.warning(f"{metric_name} 无有效数据")
        return

    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    print(f"\n【{metric_name} 排序】")
    print(f"{'Rank':>8}  {'耗时':>12}")
    print("-" * 24)
    for rank_str, val in sorted_items:
        print(f"{rank_str:>8}  {_format_ns(val):>12}")


def plot_communication_per_domain(
    domain_name: str,
    domain_groups: List[List[int]],
    comm_data: Dict[int, float],
    output_path: str,
    abnormal_groups: Optional[List[List[int]]] = None,
):
    """
    生成并行域通信耗时 Markdown 图表（替代原有 matplotlib 版本）

    注意: output_path 参数保留是为了兼容，实际输出为 Markdown 报告的一部分
    """
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
        group_stats.append((group_label, min_val, max_val, mean_val, valid_vals))

    if not group_stats:
        logger.warning(f"并行域 {domain_name} 无有效通信数据")
        return

    group_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n【{domain_name} 并行域 - 实际集合通信耗时排序】")
    print(f"{'Group':>20}  {'实际耗时(min)':>14}  {'组内均值':>14}  {'组内最大':>14}")
    print("-" * 68)
    for label, min_v, max_v, mean_v, vals in group_stats:
        print(f"{label:>20}  {_format_ns(min_v):>14}  {_format_ns(mean_v):>14}  {_format_ns(max_v):>14}")


def run_visualization(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    data_path: str = "",
    degradation: float = 0.3,
):
    """
    运行所有可视化，生成 Markdown 报告

    参数:
        step_data: step 快照数据
        parallels: 并行域信息
        valid_ranks: 有效 rank 列表
        output_dir: 输出目录
        detection_result: 检测结果（用于异常高亮）
        input_path: 原始数据目录（用于报告中显示）
        degradation: 劣化阈值
    """
    result_dir = os.path.join(output_dir, "analysis_result")
    os.makedirs(result_dir, exist_ok=True)

    # 先打印控制台实时反馈（含新增单卡指标列）
    for metric_name in ["KERNEL_AICORE", "ZP_Host", "KERNEL_AIVEC", "MEMCPY_ASYNC", "HostDuration"]:
        if metric_name in step_data:
            plot_metric_bar(step_data[metric_name], metric_name, "")

    # 控制台：总通信耗时排序（包含所有域）
    if parallels:
        domain_keys = [f"{name}_Duration" for name in parallels
                       if name]
        has_domain_data = any(
            dk in step_data and any(v > 0 and v != -99999 for v in step_data[dk].values())
            for dk in domain_keys
        )
        if has_domain_data:
            comm_totals: Dict[int, float] = {}
            for dk in domain_keys:
                if dk not in step_data:
                    continue
                for rank, val in step_data[dk].items():
                    if val != -99999 and val > 0:
                        comm_totals[rank] = comm_totals.get(rank, 0) + val
            label = "各域求和: " + ", ".join(
                d.replace("_Duration", "") for d in domain_keys if d in step_data
            )
        else:
            zp_dur = step_data.get("ZP_Duration", {})
            comm_totals = {k: v for k, v in zp_dur.items() if v != -99999 and v > 0}
            label = "无域通信数据，使用 ZP_Duration（总通信耗时）"

        if comm_totals:
            print(f"\n【总通信耗时排序】{label}")
            print(f"{'Rank':>8}  {'耗时':>12}")
            print("-" * 24)
            sorted_items = sorted(comm_totals.items(), key=lambda x: x[1], reverse=True)
            for rank_str, val in sorted_items:
                print(f"{rank_str:>8}  {_format_ns(val):>12}")

    if parallels:
        for domain_name, domain_groups in parallels.items():
            if not domain_name or not domain_groups:
                continue
            duration_key = f"{domain_name}_Duration"
            if duration_key in step_data and step_data[duration_key]:
                plot_communication_per_domain(
                    domain_name, domain_groups, step_data[duration_key], ""
                )

    # 控制台：各并行域 per-rank 通信耗时排序（不加和）
    if parallels:
        for domain_name in parallels:
            if not domain_name:
                continue
            duration_key = f"{domain_name}_Duration"
            if duration_key not in step_data:
                continue
            filtered = _filter_valid(step_data[duration_key])
            if not filtered:
                continue
            plot_metric_bar(step_data[duration_key], duration_key, "")

    # 生成完整 Markdown 报告
    report_path = markdown_viz.write_report(
        step_data, parallels, valid_ranks, result_dir,
        detection_result=detection_result,
        input_path=data_path,
        degradation=degradation,
    )

    logger.info(f"可视化完成，报告已保存至: {report_path}")
    print(f"可视化完成，报告已保存至: {report_path}")
    return report_path
