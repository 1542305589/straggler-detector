"""
故障联合分析模块
汇总各列检测结果，按硬件流水线因果推断故障传播链，判定根因卡，
生成整体故障联合分析报告（log 格式）。
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# 硬件流水线各阶段及其对应的检测类别（按因果先后顺序）
# 计算 → 搬运 → 通信 → 等待/空转
PIPELINE_STAGES = [
    {
        "stage": "compute",           # 计算
        "label": "计算",
        "categories": ["KERNEL_AICORE", "kernel_aivec", "memcpy_async"],
        "description": "卡内算子执行（KERNEL_AICORE / KERNEL_AIVEC / MEMCPY_ASYNC）变慢，通常是根因的起点",
    },
    {
        "stage": "communication",     # 通信
        "label": "通信",
        "categories": ["comm"],
        "description": "集合通信变慢，慢卡会导致同域其他卡等待通信",
    },
    {
        "stage": "wait",              # 等待/空转
        "label": "等待/空转",
        "categories": ["cpu", "npu_bubble"],
        "description": "其他卡因等待通信/数据而 CPU 空转或产生 bubble，通常是下游影响",
    },
]

# 类别中文名映射（新增类别无映射时默认用类别名）
CATEGORY_LABELS = {
    "KERNEL_AICORE": "慢计算卡(KERNEL_AICORE)",
    "kernel_aivec": "矢量计算(KERNEL_AIVEC)",
    "memcpy_async": "内存搬运(MEMCPY_ASYNC)",
    "comm": "慢通信域(comm)",
    "step_duration": "Step时长(step_duration)",
    "cpu": "慢CPU卡(cpu)",
    "host_duration": "Host耗时(host_duration)",
    "npu_bubble": "NPU空泡(npu_bubble)",
}

# 类别短标签（用于最终输出汇总表的第一列）
SHORT_CATEGORY_LABELS = {
    "KERNEL_AICORE": "慢计算卡",
    "kernel_aivec": "矢量计算",
    "memcpy_async": "内存搬运",
    "comm": "慢通信域",
    "step_duration": "Step时长",
    "cpu": "慢CPU卡",
    "host_duration": "Host耗时",
    "npu_bubble": "NPU空泡",
}

# 类别 -> step_data 中对应的单卡指标列（用于展示全部卡值）
# comm / step_duration 为组键类别（域），单独处理
CATEGORY_METRIC = {
    "KERNEL_AICORE": "KERNEL_AICORE",
    "kernel_aivec": "KERNEL_AIVEC",
    "memcpy_async": "MEMCPY_ASYNC",
    "cpu": "ZP_Host",
    "host_duration": "HostDuration",
    "step_duration": "StepDuration",
    "npu_bubble": "ZP_Bubble",
}

# 条形图宽度（字符数）
BAR_WIDTH = 40


def _bar(label: str, value: float, max_val: float, abnormal: bool = False) -> str:
    """生成文本条形图行：条形 + 标签 + 数值 + 异常标记"""
    ratio = value / max_val if max_val > 0 else 0
    n = int(round(ratio * BAR_WIDTH))
    bar = "█" * n + "▒" * (BAR_WIDTH - n)
    tag = " <-- 异常" if abnormal else ""
    return f"{bar} | {label:<8} {value:>12.2f}{tag}"


def _print_per_rank_bars(
    prefix: str,
    metric_col: str,
    step_data: dict,
    abnormal_keys: set,
    bubble: bool = False,
) -> List[str]:
    """
    针对单卡类别，展示该类别下所有卡的值并画条形图。
    metric_col: step_data 中的指标列
    abnormal_keys: 该类别中异常的卡 key 集合（字符串形式的 rank）
    bubble: npu_bubble 异常是小值，升序显示
    """
    ranks_map = step_data.get(metric_col, {}) if step_data else {}
    if not ranks_map:
        return [f"{prefix} [INFO] 无 {metric_col} 数据，无法绘制全部卡条形图"]

    # 过滤无效数据（-99999 / <=0）
    valid = {r: v for r, v in ranks_map.items() if v != -99999}
    if not valid:
        return [f"{prefix} [INFO] {metric_col} 数据均为无效值(-99999)"]

    order = sorted(valid.items(), key=lambda kv: kv[1], reverse=not bubble)
    max_val = max(v for v in valid.values()) if not bubble else max(v for v in valid.values())

    lines = []
    lines.append(f"{prefix} [INFO] 全部卡 {metric_col} 值（◄ 小值 / 大值 ►）：")
    for rank, val in order:
        abnormal = str(rank) in abnormal_keys
        lines.append(f"{prefix}  " + _bar(f"rank{rank}", val, max_val, abnormal))
    return lines


def _print_comm_bars(prefix: str, category: str, abnormal_items: dict, step_data: dict, parallels: dict) -> List[str]:
    """
    通信域类别：展示异常通信域，并尽量展示该域下各卡的通信时长（若有数据）。
    返回 log 行列表。
    """
    lines = []
    abnormal_keys = set(abnormal_items.keys())

    # 尝试从 step_data 中找出各并行域时长指标（如 tp_Duration）
    domain_metric = None
    if parallels:
        for domain_name in parallels.keys():
            col = f"{domain_name}_Duration"
            if step_data.get(col):
                domain_metric = col
                break

    if domain_metric:
        ranks_map = {str(r): v for r, v in step_data.get(domain_metric, {}).items()
                     if v != -99999}
        if ranks_map:
            max_val = max(ranks_map.values()) or 1
            lines.append(f"{prefix} [INFO] 全部卡 {domain_metric} 值：")
            for rank, val in sorted(ranks_map.items(), key=lambda kv: kv[1], reverse=True):
                # 该卡是否落在任一异常通信域内
                abnormal = False
                for gk in abnormal_keys:
                    if rank in gk.split(","):
                        abnormal = True
                        break
                lines.append(f"{prefix}  " + _bar(f"rank{rank}", val, max_val, abnormal))
    else:
        # 无域时长数据，仅展示异常通信域
        for key, val in sorted(abnormal_items.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"{prefix}   [WARN] 慢通信域组 [{key}] -> 劣化指数 = {val:.2f}")
    return lines


def _parse_ranks_from_key(key: str):
    """把类别结果的 key 解析成 rank 列表（单卡 key 是 "0"，组 key 是 "0,1,2"）"""
    return [int(r) for r in key.split(",")]


def _collect_abnormal_cards(result: dict) -> Dict[str, set]:
    """收集每一类别里异常涉及的卡集合（组类型归并到每张卡）"""
    per_category_cards: Dict[str, set] = {}
    for category, items in result.items():
        if not items:
            continue
        cards = set()
        for key in items:
            cards.update(_parse_ranks_from_key(key))
        per_category_cards[category] = cards
    return per_category_cards


def _determine_root_causes(per_category_cards: dict) -> List[Dict]:
    """
    判定根因卡：
    按流水线阶段从早到晚扫描，最早阶段命中的卡优先判定为根因。
    传播链：计算慢 → 通信慢 → 其他卡等待
    """
    # 命中各阶段的卡
    stage_hits = {}
    for stage_info in PIPELINE_STAGES:
        cats = stage_info["categories"]
        hit_cards = set()
        for c in cats:
            hit_cards |= per_category_cards.get(c, set())
        stage_hits[stage_info["stage"]] = hit_cards

    compute_cards = stage_hits.get("compute", set())

    root_causes = []

    # 根因：在计算/搬运阶段（流水线最上游）出现异常的卡
    for card in compute_cards:
        root_causes.append({
            "rank": card,
            "stage": "compute",
            "categories": [c for c in ["KERNEL_AICORE", "kernel_aivec", "memcpy_async"] if card in per_category_cards.get(c, set())],
            "reason": f"rank{card} 在计算/搬运阶段出现异常，构成故障传播的源头",
        })

    return root_causes


def _log_prefix() -> str:
    """生成 log 行前缀：时间戳 + 标记"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts}][JOINT]"


def generate_joint_report(result: dict, parallels: dict = None, step_data: dict = None, output_dir: str = None) -> str:
    """
    生成故障联合分析 log 报告

    参数:
        result: 检测结果（含所有类别，含 kernel_aivec/memcpy_async 等）
        parallels: 并行域信息（可选）
        step_data: 单 step 快照数据（可选），用于展示每个类别下全部卡（含正常卡）并绘制条形图
        output_dir: 输出目录（默认 config.get_file_path()）

    返回:
        报告文件路径
    """
    if output_dir is None:
        output_dir = config.get_file_path()

    lines = []
    prefix = _log_prefix()

    # 一、各列检测结果汇总
    lines.append(f"{prefix} ============ 故障联合分析 ============")
    lines.append(f"{prefix} 输出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{prefix} ---------- 一、各列检测结果汇总 ----------")

    group_categories = ("comm", "step_duration")
    # 单卡类别集合：CATEGORY_METRIC 中映射到指标列且非组键类别
    single_card_categories = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async", "cpu", "host_duration", "npu_bubble")

    # 动态类别集合：优先展示已知/存在的类别，同时覆盖动态类别
    ordered_categories = [
        "KERNEL_AICORE", "kernel_aivec", "memcpy_async",
        "comm", "step_duration", "cpu", "host_duration", "npu_bubble",
    ]
    known = set(ordered_categories)
    dynamic = [c for c in result.keys() if c not in known]
    all_categories = ordered_categories + sorted(dynamic)

    total_anomalies = 0
    for category in all_categories:
        # result 是原始 DegradationData 格式：{category: {key: value}}
        items = result.get(category) or {}
        label = CATEGORY_LABELS.get(category, category)
        lines.append(f"{prefix} [INFO] {label}:")

        if not items:
            lines.append(f"{prefix}   [INFO] 无异常（该维度全部卡表现正常）")
            continue

        total_anomalies += len(items)
        # 异常项概览
        sorted_items = sorted(items.items(), key=lambda kv: kv[1], reverse=True)
        display = "; ".join(f"{key}({val:.2f})" for key, val in sorted_items)
        lines.append(f"{prefix}   [WARN] 异常项（共 {len(items)} 项）: {display}")

        metric_col = CATEGORY_METRIC.get(category)
        if metric_col and category in single_card_categories:
            abnormal_keys = set(items.keys())
            bubble = (category == "npu_bubble")
            lines.extend(_print_per_rank_bars(prefix, metric_col, step_data, abnormal_keys, bubble))
        elif category in group_categories and step_data:
            # 组键类别（通信域）：展示各域时长（以异常域为核心）
            lines.extend(_print_comm_bars(prefix, category, items, step_data, parallels))
        elif metric_col and step_data:
            # 通信算子类型类别：按单卡展示该算子类型的各卡平均耗时
            abnormal_keys = set(items.keys())
            lines.extend(_print_per_rank_bars(prefix, metric_col, step_data, abnormal_keys, False))

    lines.append(f"{prefix} [INFO] 共检测到 {total_anomalies} 项异常。")
    lines.append(f"{prefix} ---------- 二、故障传播链分析 ----------")
    lines.append(f"{prefix} [INFO] 故障按硬件流水线因果传播: 计算慢 -> 搬运慢 -> 通信慢 -> 其他卡等待/空转")

    per_category_cards = _collect_abnormal_cards(result)

    # 各阶段命中的卡
    stage_ranks = {}
    for stage_info in PIPELINE_STAGES:
        hit = set()
        for c in stage_info["categories"]:
            hit |= per_category_cards.get(c, set())
        stage_ranks[stage_info["stage"]] = hit

    for stage_info in PIPELINE_STAGES:
        hit = stage_ranks[stage_info["stage"]]
        suffix = ""
        if stage_info["stage"] == "compute" and hit:
            suffix = " <- 根因候选"
        elif stage_info["stage"] == "communication" and hit:
            suffix = " <- 受慢卡拖累"
        elif stage_info["stage"] == "wait" and hit:
            suffix = " <- 下游影响"
        hit_str = sorted(hit) if hit else "无"
        lines.append(f"{prefix} [INFO] {stage_info['label']}阶段{suffix}: 命中卡 {hit_str} - {stage_info['description']}")

    lines.append(f"{prefix} ---------- 三、根因卡判定 ----------")
    root_causes = _determine_root_causes(per_category_cards)
    if root_causes:
        lines.append(f"{prefix} [WARN] 根因卡: {', '.join(str(r['rank']) for r in root_causes)}")
        lines.append(f"{prefix} [INFO] 判定依据: 这些卡在计算/搬运阶段(流水线最上游)出现异常,")
        lines.append(f"{prefix} [INFO] 其变慢会通过集合通信传递到同域其他卡, 导致下游卡等待/空转。")
    else:
        lines.append(f"{prefix} [INFO] 未发现明确根因卡(可能存在单一维度的轻微退化, 或异常维度不在计算/搬运阶段)。")

    lines.append(f"{prefix} ---------- 四、结论与建议 ----------")
    if root_causes:
        lines.append(f"{prefix} [WARN] 建议优先排查根因卡 {', '.join(str(r['rank']) for r in root_causes)}:")
        lines.append(f"{prefix} [INFO] 降低其计算/搬运耗时可缓解同域其他卡的等待, 提升整体训练吞吐。")
    else:
        lines.append(f"{prefix} [INFO] 本次未检测到明确的慢节点根因链, 建议结合更长时序数据进一步观察。")
    lines.append(f"{prefix} ============ 故障联合分析结束 ============")
    lines.append("")

    report_content = "\n".join(lines)

    # 写 log 文件
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "joint_failure_analysis.log")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    # 打印到控制台（兼容 Windows GBK 等非 UTF-8 控制台：
    # ◄► 等字符不在其字符集时降级为 '?'，避免 UnicodeEncodeError 崩溃）
    try:
        print(report_content)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            sys.stdout.write(report_content.encode(enc, errors="replace").decode(enc))
        except Exception:
            sys.stdout.write(report_content.encode("ascii", errors="replace").decode("ascii"))

    return report_path


# ---- 最终输出汇总表（渲染到调用 skill 的 agent 的 stdout，不进任何 log 文件） ----

# 计算/IO/Host 类（倍率 = 1 + degradation）与通信域类（倍率 = 1 + 5*degradation）的类别集合
COMPUTE_METRIC_CATEGORIES = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async", "cpu", "host_duration")
COMM_GROUP_CATEGORIES = ("comm", "step_duration")


def _fmt_ns(value: float) -> str:
    """将纳秒格式化为可读单位（与 markdown_viz._fmt_ns 逻辑一致）"""
    if value >= 1e9:
        return f"{value/1e9:.2f}s"
    elif value >= 1e6:
        return f"{value/1e6:.2f}ms"
    elif value >= 1e3:
        return f"{value/1e3:.2f}us"
    else:
        return f"{value:.0f}ns"


def _cell_metric_summary(metric_col: str, abnormal_ranks, step_data) -> str:
    """
    数据要点：对某指标列，展示异常卡的值、其他卡范围与倍数。
    例：rank0=1.76ms，其他≈568~574us（约 2.9 倍）
    """
    ranks_map = {}
    if step_data:
        ranks_map = step_data.get(metric_col) or {}
    # 过滤无效值（-99999 / <=0），key 转 int
    valid = {}
    for r, v in ranks_map.items():
        try:
            ri = int(r)
        except (TypeError, ValueError):
            continue
        if v != -99999 and v > 0:
            valid[ri] = v
    if not valid:
        return "无详细数据"

    anom_set = set(abnormal_ranks)
    anom_vals = [(r, valid[r]) for r in sorted(anom_set) if r in valid]
    normal_vals = [v for r, v in valid.items() if r not in anom_set]

    parts = [f"rank{r}={_fmt_ns(v)}" for r, v in anom_vals]
    if normal_vals:
        nmin, nmax = min(normal_vals), max(normal_vals)
        parts.append(f"其他≈{_fmt_ns(nmin)}~{_fmt_ns(nmax)}"
                     if nmin != nmax else f"其他≈{_fmt_ns(nmin)}")

    text = "，".join(parts)
    if anom_vals and normal_vals:
        max_anom = max(v for _, v in anom_vals)
        normal_avg = sum(normal_vals) / len(normal_vals)
        mult = max_anom / normal_avg if normal_avg > 0 else 0.0
        text += f"（约 {mult:.1f} 倍）"
    return text or "无详细数据"


def _cell_comm_summary(items: dict, parallels: dict, step_data: dict) -> str:
    """通信域类别的数据要点：优先用域时长列（如 tp_Duration）展示，否则兜底。"""
    domain_metric = None
    if parallels:
        for domain_name in parallels.keys():
            col = f"{domain_name}_Duration"
            if step_data and step_data.get(col):
                domain_metric = col
                break
    if not domain_metric:
        return "无详细数据"
    abnormal_ranks = []
    for key in items:
        abnormal_ranks.extend(_parse_ranks_from_key(key))
    return _cell_metric_summary(domain_metric, abnormal_ranks, step_data)


def _disp_len(text: str) -> int:
    """估算字符串在终端中的显示宽度（CJK/全角字符按 2 列计）。"""
    return sum(2 if 0x2E80 <= ord(ch) <= 0x9FFF or 0xFF00 <= ord(ch) <= 0xFFEF else 1
               for ch in text)


def _disp_ljust(text: str, width: int) -> str:
    """按显示宽度左填充空格，保证 CJK 与 ASCII 在终端中对齐。"""
    pad = width - _disp_len(text)
    return text + " " * max(pad, 0)


def _render_box_table(headers: List[str], rows: List[List[str]]) -> str:
    """渲染 Unicode 框线表格（┌─┬─┐），按列自动对齐（考虑 CJK 显示宽度）。"""
    all_rows = [headers] + rows
    ncols = len(headers)
    widths = [0] * ncols
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _disp_len(cell))

    def _line(left, mid, right, fill="─"):
        return left + mid.join(fill * w for w in widths) + right

    top = _line("┌", "┬", "┐")
    mid = _line("├", "┼", "┤")
    bot = _line("└", "┴", "┘")
    lines = [top]
    for idx, row in enumerate(all_rows):
        cells = "│" + "│".join(" " + _disp_ljust(cell, widths[i]) + " "
                               for i, cell in enumerate(row)) + "│"
        lines.append(cells)
        if idx == 0:
            lines.append(mid)
    lines.append(bot)
    return "\n".join(lines)


def build_summary_table(result: dict, parallels: dict = None, step_data: dict = None,
                        degradation: float = None) -> str:
    """
    生成逐类别汇总的 Unicode 框线表格字符串（渲染到调用方 agent 的最终输出，不进任何 log 文件）。

    参数:
        result: 检测结果 {category: {key: degradation}}
        parallels: 并行域信息（可选，用于通信域类别的域时长列）
        step_data: 单 step 快照数据（可选，用于数据要点的各卡值）
        degradation: 劣化阈值（默认取 config.Degradation，用于计算劣化阈值列）

    返回:
        Unicode 框线表格字符串；无任何异常时返回标题行 + 表头 + “无异常”提示。
    """
    if degradation is None:
        degradation = config.Degradation

    headers = ["类别", "异常卡", "劣化指数", "劣化阈值", "数据要点"]

    ordered_categories = [
        "KERNEL_AICORE", "kernel_aivec", "memcpy_async",
        "comm", "step_duration", "cpu", "host_duration", "npu_bubble",
    ]
    known = set(ordered_categories)
    dynamic = [c for c in result.keys() if c not in known]
    all_categories = ordered_categories + sorted(dynamic)

    rows = []
    threshold = {
        "compute": config.get_compute_multiplier(),
        "comm": config.get_comm_multiplier(),
        "bubble": 5000,
    }

    for category in all_categories:
        items = result.get(category) or {}
        if not items:
            continue

        # 异常卡列：解析出涉及的所有 rank
        abnormal_ranks = []
        for key in items:
            abnormal_ranks.extend(_parse_ranks_from_key(key))
        abnormal_ranks = sorted(set(abnormal_ranks))
        cards_str = "rank " + ", ".join(str(r) for r in abnormal_ranks)

        # 劣化指数列：该类别的最大劣化值
        max_deg = max(items.values())
        deg_str = f"{max_deg:.3f}"

        # 劣化阈值列
        if category == "npu_bubble":
            th_str = f"{threshold['bubble']}ns"
        elif category in COMM_GROUP_CATEGORIES:
            th_str = f"{threshold['comm']:.3f}"
        else:
            th_str = f"{threshold['compute']:.3f}"

        # 类别列
        label = CATEGORY_LABELS.get(category, category)
        short = SHORT_CATEGORY_LABELS.get(category)
        category_str = f"{category}（{short}）" if short else label

        # 数据要点列
        if category in COMM_GROUP_CATEGORIES:
            summary = _cell_comm_summary(items, parallels, step_data)
        else:
            metric_col = CATEGORY_METRIC.get(category)
            summary = _cell_metric_summary(metric_col, abnormal_ranks, step_data) if metric_col else "无详细数据"

        rows.append([category_str, cards_str, deg_str, th_str, summary])

    if not rows:
        return _render_box_table(headers, [["无异常", "-", "-", "-", "该维度全部卡表现正常"]])

    return _render_box_table(headers, rows)
