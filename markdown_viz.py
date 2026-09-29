"""
报告生成模块 - 为慢节点检测结果生成文本格式报告
替代原有 matplotlib PNG 图表和 Markdown 格式，直接输出到 log 文件

功能:
1. 水平柱状图（使用 Unicode 字符）
2. 排序表格 + 统计摘要
3. 异常高亮
4. 并行域集合通信带宽排序
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime

# 添加父目录到路径以便导入 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import utils
import summary_table

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
    abnormal_map: Optional[Dict[str, float]] = None,
    note: str = "",
) -> str:
    """生成单个指标的排序柱状图（纯文本）。

    柱状图的"相对倍数"列以**组内最小耗时**为基准（最快的卡=1.00x，其余为相对其倍数，
    与排序天然单调）。该列是直观的组内对比量，**不**混同于检测的劣化指数
    （后者以第一次 KMeans 基线簇均值为分母，见摘要/汇总表）。
    """
    filtered = _filter_valid(data)
    if not filtered:
        return f"\n[{metric_name}] 无有效数据\n"

    abnormal_set = set()
    if abnormal_map:
        try:
            abnormal_set = {int(rk) for rk in abnormal_map.keys()}
        except ValueError:
            abnormal_set = set()

    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_items]
    max_value = values[0] if values else 1

    # 基准 = 组内最小耗时（最快卡），该卡显示 1.00x
    baseline = min(values) if values else 0

    sorted_vals = sorted(values)
    n = len(values)
    median_val = sorted_vals[n // 2] if n % 2 == 1 else \
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    lines = []
    lines.append("")
    lines.append(_sep_line(f"{metric_name} 耗时排序", 70))
    if note:
        lines.append(f"  {note}")
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

    # 列头（相对倍数 = 该卡耗时 / 组内最小耗时，最快卡为 1.00x）
    lines.append(f"  {'#':>3}  {'Rank':>6}  {'耗时':>10}  {'相对倍数':>8}  柱状图")
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
        ratio = val / baseline if baseline > 0 else 1
        is_abnormal = rank in abnormal_set
        marker = " ***" if is_abnormal else ""
        lines.append(f"  {idx:>3}  {rank:>6}  {_fmt_ns(val):>10}  {ratio:>7.2f}x  {bar}{marker}")

    # 标记说明
    if abnormal_set:
        lines.append("")
        lines.append("  *** = 异常卡")
    lines.append("")
    lines.append(f"  相对倍数 = 耗时 / 组内最小耗时（最快卡=1.00x；与检测劣化指数口径不同）")

    # 统计信息
    lines.append("")
    lines.append(f"  {'-- 统计信息':-<40}")
    mean_val = sum(values) / len(values)
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


def _comm_bandwidth_section(
    domain_name: str,
    domain_groups: List[List[int]],
    step_data: Dict[str, Dict[int, float]],
    abnormal_groups: Optional[List[List[int]]] = None,
) -> str:
    """生成并行域集合通信带宽（越小越慢；数据来自带宽列 <domain>_<opType>_<count>）。"""
    cols = utils.domain_bandwidth_cols(step_data, domain_name)
    if not cols:
        return f"\n[{domain_name}] 无带宽数据\n"

    op_types = sorted({op for op, _c, _col in cols})
    abnormal_set = set()
    if abnormal_groups:
        for ag in abnormal_groups:
            abnormal_set.add(",".join(str(r) for r in sorted(ag)))

    lines = []
    lines.append("")
    lines.append(_sep_line(f"{domain_name} 并行域 - 集合通信带宽（越小越慢）", 70))
    lines.append("  带宽 = count / 组内最快 10% 最短耗时均值；每组每个 opType 取 count 最大的条目为代表")
    lines.append("")

    for op in op_types:
        rows = []
        for g in domain_groups:
            reps = utils.group_representative_bandwidth(step_data, domain_name, g)
            bw = next((b for o, _c, b in reps if o == op), None)
            if bw is None:
                continue
            rows.append((",".join(str(r) for r in sorted(g)), bw))
        if not rows:
            continue
        rows.sort(key=lambda x: x[1])  # 升序：带宽最小（最慢）在前
        max_bw = max(b for _, b in rows)
        lines.append(f"  [opType={op}]")
        lines.append(f"  {'#':>3}  {'Group':>20}  {'带宽':>14}  柱状图")
        lines.append(f"  {'---':>3}  {'--------------------':>20}  {'--------------':>14}  -------")
        for i, (label, bw) in enumerate(rows, 1):
            bar = _bar(bw, max_bw)
            marker = " ***" if label in abnormal_set else ""
            lines.append(f"  {i:>3}  {label:>20}  {bw:>14.4g}  {bar}{marker}")
        lines.append("")

    if abnormal_set:
        lines.append("  *** = 异常 Group（带宽显著偏小）")
    lines.append("")
    return "\n".join(lines)


def _comm_bandwidth_overview(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
) -> str:
    """各域集合通信带宽概览（每域每 opType 的组间带宽范围）。"""
    lines = []
    lines.append("")
    lines.append(_sep_line("各域集合通信带宽概览", 70))
    lines.append("  仅展示各域带宽分布，不参与检测")
    lines.append("")
    any_row = False
    for domain_name, domain_groups in (parallels or {}).items():
        if not domain_name or not domain_groups:
            continue
        cols = utils.domain_bandwidth_cols(step_data, domain_name)
        if not cols:
            continue
        for op in sorted({op for op, _c, _col in cols}):
            bws = []
            for g in domain_groups:
                reps = utils.group_representative_bandwidth(step_data, domain_name, g)
                bw = next((b for o, _c, b in reps if o == op), None)
                if bw is not None:
                    bws.append(bw)
            if not bws:
                continue
            any_row = True
            lo, hi = min(bws), max(bws)
            ratio = (hi / lo) if lo > 0 else float('inf')
            lines.append(f"  {domain_name}/{op}: 组间带宽 {lo:.4g} ~ {hi:.4g}  "
                         f"（最大/最小 {ratio:.2f}x，{len(bws)} 组）")
    if not any_row:
        return ""
    lines.append("")
    return "\n".join(lines)


def _category_threshold(category: str) -> str:
    """返回某检测类别对应的劣化阈值显示文本（按类别组取阈值）。

    - 计算类（KERNEL_AICORE / kernel_aivec）→ COMPUTE_THRESHOLD（默认 1.3）
    - IO/CPU 类（cpu / memcpy_async）→ IO_THRESHOLD（默认 2.5）
    - 通信类（comm / pp_comm）→ COMM_THRESHOLD（默认 1.3）
    - npu_bubble → 固定硬阈值 < BUBBLE_THRESHOLD_NS
    """
    if category == "npu_bubble":
        return f"<{config.BUBBLE_THRESHOLD_NS}ns"
    return f"{config.get_threshold_for_category(category):g}x"


# 各检测类别的口径说明（置于汇总表下方脚注）
CATEGORY_DESC = {
    "KERNEL_AICORE": "所有 KERNEL_AICORE 算子的平均时间",
    "kernel_aivec": "所有 KERNEL_AIVEC 算子的平均时间",
    "memcpy_async": "所有 MEMCPY_ASYNC 算子的平均时间",
    "comm": "各通信域 {domain}_{opType}_{count} 带宽聚类",
    "pp_comm": "PP 链路 Send/Recv 时间窗重叠（长则慢），按阶段位置聚类",
    "cpu": "ZP_Host：通信算子与 KERNEL_AICORE 的 Host 耗时均值",
    "npu_bubble": "ZP_Bubble：通信算子启动间隔，小于 5000ns 记异常",
}


def _detection_summary(
    detection_result: Dict[str, Dict[str, float]],
    valid_ranks: List[int],
    parallels: Dict[str, List[List[int]]] = None,
) -> str:
    """生成检测结果摘要（ASCII 框线表 + 口径脚注）。"""
    headers = ["类别", "状态", "劣化阈值", "异常卡", "劣化指数", "物理设备"]
    ordered = ["KERNEL_AICORE", "kernel_aivec", "memcpy_async",
               "comm", "pp_comm", "cpu", "npu_bubble"]
    known = set(ordered)
    dynamic = [c for c in detection_result.keys() if c not in known]
    all_categories = ordered + sorted(dynamic)

    rows = []
    for category in all_categories:
        items = detection_result.get(category) or {}
        name = summary_table.CATEGORY_DISPLAY.get(category, category)
        th = _category_threshold(category)
        if items:
            cards_str, deg_str, dev_str = summary_table._summary_cells(
                category, items, parallels)
            rows.append([name, "异常", th, cards_str, deg_str, dev_str or "-"])
        else:
            rows.append([name, "正常", th, "-", "-", "-"])

    lines = [summary_table._render_box_table(headers, rows)]

    desc_categories = [c for c in all_categories if c in CATEGORY_DESC]
    if desc_categories:
        name_width = max(
            summary_table._disp_len(summary_table.CATEGORY_DISPLAY.get(c, c))
            for c in desc_categories
        )
        lines.append("  口径说明:")
        for category in desc_categories:
            name = summary_table.CATEGORY_DISPLAY.get(category, category)
            lines.append("    - " + summary_table._disp_ljust(name, name_width)
                         + ": " + CATEGORY_DESC[category])

    lines.append("")
    lines.append(
        f"  总 Rank 数: {len(valid_ranks)}  |  阈值(计算/IO/通信): "
        f"{config.get_compute_threshold():g}/{config.get_io_threshold():g}/{config.get_comm_threshold():g}"
    )
    return "\n".join(lines)


def generate_report(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    input_path: str = "",
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
        sections.append(_detection_summary(detection_result, valid_ranks, parallels))
        sections.append("")

    # Part 1: 单卡指标排序柱状图（KERNEL_AICORE/kernel_aivec/memcpy_async/cpu/npu_bubble）
    cat_to_metric = dict(CATEGORY_METRIC)
    rendered_metric_cols = set()
    for cat, metric_name in cat_to_metric.items():
        if metric_name not in step_data or metric_name in rendered_metric_cols:
            continue
        rendered_metric_cols.add(metric_name)
        if not _filter_valid(step_data[metric_name]):
            logger.warning(f"{metric_name} 所有数据均无效，跳过")
            continue

        abnormal_map = {}
        if detection_result and cat in detection_result:
            abnormal_map = detection_result[cat]
            if cat == "cpu" and abnormal_map:
                # cpu 的 key 是节点显示名（hostName），转成 rank 集合供 ZP_Host 段高亮
                node_ranks = config.get_node_ranks_map()
                abnormal_map = {
                    str(r): deg
                    for node, deg in abnormal_map.items()
                    for r in node_ranks.get(node, [])
                }

        sections.append(_metric_section(metric_name, step_data[metric_name], abnormal_map))

    # Part 2: 每个并行域的集合通信带宽（跳过 pp / embd：不在带宽白名单）
    if parallels:
        comm_abnormal = detection_result.get("comm", {}) if detection_result else {}

        abnormal_groups = []
        for group_key in comm_abnormal:
            try:
                ranks = [int(r) for r in group_key.split(",")]
                abnormal_groups.append(ranks)
            except (ValueError, AttributeError):
                pass

        for domain_name, domain_groups in parallels.items():
            if not domain_name or not domain_groups or domain_name in ("pp", "embd"):
                continue
            sections.append(_comm_bandwidth_section(
                domain_name, domain_groups, step_data, abnormal_groups))

    # 各域集合通信带宽概览（置于最后）
    if parallels:
        sections.append(_comm_bandwidth_overview(step_data, parallels))

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
    )

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "detection_report.log")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"检测报告已保存: {report_path}")
    print(f"\n检测报告已保存: {report_path}")
    return report_path
