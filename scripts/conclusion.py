from __future__ import annotations
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

_CATEGORY_CN = {
    "KERNEL_AICORE": "KERNEL_AICORE类算子",
    "kernel_aivec": "KERNEL_AIVEC类算子",
    "memcpy_async": "MEMCPY_ASYNC类算子",
    "comm": "comm（通信域带宽）",
    "cpu": "cpu（主机侧 ZP_Host）",
    "npu_bubble": "npu_bubble（NPU 气泡 ZP_Bubble）",
    "pp_comm": "pp_comm（PP 流水线等待）",
}
_ORDER = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async", "comm",
          "cpu", "npu_bubble", "pp_comm")
_COLUMN = {
    "KERNEL_AICORE": "KERNEL_AICORE",
    "kernel_aivec": "KERNEL_AIVEC",
    "memcpy_async": "MEMCPY_ASYNC",
    "cpu": "ZP_Host",
    "npu_bubble": "ZP_Bubble",
}
_DETAIL_CATS = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async")
_LEVEL_NOTE = {
    "levelnone": "本次数据只含 MsTx，没有计算 / 搬运算子，计算类与搬运类未参与检测。",
    "cluster": "cluster 单文件只评估通信，其他类别未参与。",
}
_MEMCPY_NOTE = """关于 memcpy_async 的说明：该指标统计的是「单次拷贝算子的平均耗时」，而这类任务里混了
主机-设备拷贝（H2D/D2H）、设备内拷贝、设备间拷贝（D2D）。下面这些已知情况会让少数卡的均值
结构性偏高，进而导致误报：

1. **PP>1 且 TP>1（Megatron 框架特有）**：只有首/末 stage 的 tp0 需要把 data/label 从主机搬到
   卡上、再广播给同 TP 组，其余卡为 0。例：PP4TP4 时就是 rank0 和 rank12。
2. **PP=1 且 TP>1（Megatron 框架特有，如 TP4DP4）**：各 DP 组的 tp0 承担同样的 H2D，
   例：rank0/4/8/12。
3. **任务数不均**：任务少的卡缺少短任务，平均值容易贴线偏高。
4. **采集窗口残缺**：某些卡 STEP_TIME 只覆盖少量 step 时，均值会假性偏移，还会把其它卡带成
   镜像误报。

当前数据既没有拷贝方向（H2D/D2H/D2D），也没有逐次数据量，所以只看这个指标无法定论；
如需确认是否真的存在搬运劣化，需要补充这两项数据。"""


def _first_rank(item: Dict) -> str:
    ranks = _ranks_from_key(item.get("display_key"))
    return str(ranks[0]) if ranks else str(item.get("display_key"))

def _pp_percent() -> str:
    try:
        import config
        return f"{config.PP_WAIT_RATIO_THRESHOLD * 100:g}"
    except Exception:
        return "5"

def _fmt_ranks(ranks) -> str:
    rs = sorted({int(r) for r in (ranks or [])})
    if not rs:
        return ""
    if len(rs) <= 6:
        return "rank " + "、".join(str(r) for r in rs)
    return f"rank {rs[0]}–{rs[-1]} 等 {len(rs)} 张"


def _ranks_from_key(key) -> List[int]:
    s = str(key or "")
    if "[" in s and s.endswith("]"):
        s = s[s.index("[") + 1:-1]
    return sorted(int(x) for x in re.findall(r"\d+", s))


def _ranks_text(key) -> str:
    s = str(key or "").strip()
    m = re.match(r"^([A-Za-z0-9_]+)\s*\[(.*)\]$", s)
    if m:
        ranks = [x.strip() for x in m.group(2).split(",") if x.strip()]
        return f"{m.group(1)} 组的 rank {'、'.join(ranks)}"
    m = re.match(r"^\[(.*)\]$", s)
    if m:
        ranks = [x.strip() for x in m.group(1).split(",") if x.strip()]
        ranks = sorted(ranks, key=lambda x: int(x) if x.isdigit() else 0)
        return "rank " + "、".join(ranks)
    m = re.match(r"^rank(\d+)-(\d+)$", s)
    if m:
        return f"rank {m.group(1)}–{m.group(2)}"
    m = re.match(r"^rank(\d+)$", s)
    if m:
        return f"rank {m.group(1)}"
    if s.isdigit():
        return f"rank {s}"
    return s


def _pick_step(values: List[float]) -> Optional[float]:
    """与检测一致：n>1 取倒数第二个 step，n==1 取第 0 个。"""
    if not values:
        return None
    return values[-2] if len(values) > 1 else values[0]


def _read_metric(metric_dir: str, rank: int, column: str) -> Optional[float]:
    path = os.path.join(metric_dir, f"global_rank_{rank}.csv")
    if not os.path.isfile(path):
        return None
    vals: List[float] = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get(column)
                if raw in (None, ""):
                    continue
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    pass
    except OSError:
        return None
    v = _pick_step(vals)
    return None if v is None else v / 1_000_000.0   # ns → ms


def _detection_groups(data_dir: str):
    """rank -> 所属检测组。直接读 op_metric/group_info_*.json，按检测层同一套优先级取域。"""
    priority = ("tp", "exp", "ep", "tp_exp", "cp", "cp2", "cp_ulysses",
                "cp_ring", "dp", "dp_cp", "dp_modulo_exp_cp", "default_group")
    metric_dir = os.path.join(data_dir, "op_metric")
    domains: Dict[str, Dict[tuple, List[int]]] = {}
    ranks_all = set()
    try:
        names = sorted(os.listdir(metric_dir))
    except OSError:
        return {}, []
    for name in names:
        if not (name.startswith("group_info_") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(metric_dir, name), encoding="utf-8") as f:
                info = json.load(f) or {}
        except (OSError, ValueError):
            continue
        for entry in info.values():
            if not isinstance(entry, dict):
                continue
            pg = str(entry.get("group_name") or "")
            try:
                ranks = sorted(int(r) for r in (entry.get("global_ranks") or []))
            except (TypeError, ValueError):
                continue
            if not ranks:
                continue
            ranks_all.update(ranks)
            domains.setdefault(pg, {})[tuple(ranks)] = ranks
    for pg in priority:
        groups = list(domains.get(pg, {}).values())
        if groups:
            mapping = {}
            for g in groups:
                for r in g:
                    mapping[r] = list(g)
            return mapping, groups
    if ranks_all:
        allr = sorted(ranks_all)
        return {r: allr for r in allr}, [allr]
    return {}, []

def _baseline(column: str, metric_dir: str, group: List[int],
              flagged: List[int]) -> Tuple[Optional[float], List[int]]:
    """组内非异常卡的平均值（ms）。"""
    peers = [r for r in group if r not in flagged]
    if not peers:
        peers = list(group)
    vals = [v for v in (_read_metric(metric_dir, r, column) for r in peers) if v is not None]
    if not vals:
        return None, []
    return sum(vals) / len(vals), sorted(peers)


def _detail_line(cat: str, items: List[Dict], metric_dir: str,
                 mapping: Dict[int, List[int]]) -> str:
    column = _COLUMN.get(cat)
    label = _CATEGORY_CN.get(cat, cat)
    if not column or not os.path.isdir(metric_dir):
        detail = "、".join(
            f"{_ranks_text(i.get('display_key'))} 约 {float(i.get('metric_value') or 0):.2f} 倍"
            for i in items)
        return f"{label}：{detail}"

    parts = []
    buckets: Dict[Tuple[int, ...], List[Dict]] = {}
    for i in items:
        ranks = _ranks_from_key(i.get("display_key"))
        group = mapping.get(ranks[0]) if ranks else None
        key = tuple(sorted(group)) if group else ()
        buckets.setdefault(key, []).append(i)

    for key, group_items in sorted(buckets.items()):
        group = list(key) if key else sorted(mapping) or []
        flagged = sorted({r for i in group_items for r in _ranks_from_key(i.get("display_key"))})
        ratios = "、".join(
            f"rank{_first_rank(i)} 相对检测组内基线约 "
            f"{float(i.get('metric_value') or 0):.2f} 倍" for i in group_items)
        values = []
        for r in flagged:
            v = _read_metric(metric_dir, r, column)
            if v is not None:
                values.append(f"rank{r} 为 {v:.3f} ms")
        base, base_ranks = _baseline(column, metric_dir, group, flagged)
        who = ("rank " + "、".join(str(r) for r in base_ranks)) if base_ranks else "—"
        sentence = ratios
        if values:
            sentence += "，" + "，".join(values)
        if base is not None:
            sentence += f"，检测组内基线平均（{who}）为 {base:.3f} ms"
        parts.append(sentence)
    return f"{label}：" + "。".join(parts)


def _node_ranks_map() -> Dict[str, List[int]]:
    """节点名 -> 该节点的 rank 列表（来自 config 的 host 映射）。"""
    try:
        import config
        host_map = config.get_host_rank_map() or {}
    except Exception:
        return {}
    out: Dict[str, List[int]] = {}
    for rank_str in host_map:
        try:
            r = int(rank_str)
        except (TypeError, ValueError):
            continue
        try:
            label = config.get_node_name(r)
        except Exception:
            label = ""
        if label:
            out.setdefault(str(label), []).append(r)
    return {k: sorted(v) for k, v in out.items()}

def _node_value(metric_dir: str, nodes: Dict[str, List[int]], name: str):
    ranks = nodes.get(name) or []
    vals = [v for v in (_read_metric(metric_dir, r, "ZP_Host") for r in ranks) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def _node_line(items: List[Dict], metric_dir: str) -> str:
    """cpu 类：节点维度描述（基线 = 其它节点）。"""
    label = _CATEGORY_CN.get("cpu", "cpu")
    nodes = _node_ranks_map()
    names = [str(i.get("display_key")) for i in items]
    ratios = "、".join(
        f"{n} 相对其它节点基线约 {float(i.get('metric_value') or 0):.2f} 倍"
        for n, i in zip(names, items))
    sentence = ratios
    values = []
    for n in names:
        v = _node_value(metric_dir, nodes, n)
        if v is not None:
            values.append(f"{n} 为 {v:.3f} ms")
    if values:
        sentence += "，" + "，".join(values)
    base_nodes = [n for n in nodes if n not in names]
    base_vals = [v for v in (_node_value(metric_dir, nodes, n) for n in base_nodes) if v is not None]
    if base_vals:
        who = "、".join(base_nodes)
        sentence += (f"，其它节点基线平均（{who}）为 {sum(base_vals) / len(base_vals):.3f} ms")
    else:
        sentence += "，其它节点基线无可用数据"
    return f"{label}：{sentence}"

def _comm_line(items: List[Dict]) -> str:
    ratios = "、".join(
        f"{_ranks_text(i.get('display_key'))} 相对组内基线约 "
        f"{float(i.get('metric_value') or 0):.2f} 倍" for i in items)
    return (f"{_CATEGORY_CN.get('comm')}：{ratios}"
            "（取该域内劣化最严重的［并行域, 算子, Count］组合）")

def _pp_line(items: List[Dict]) -> str:
    parts = []
    for i in items:
        pct = float(i.get("metric_value") or 0) * 100.0
        parts.append(f"{_ranks_text(i.get('display_key'))} 的点对点等待时间占会话时长约 {pct:.2f}%")
    return f"{_CATEGORY_CN.get('pp_comm')}：" + "、".join(parts) + f"，超过阈值 {_pp_percent()}%"

def _bubble_line(items: List[Dict]) -> str:
    keys = "、".join(_ranks_text(i.get("display_key")) for i in items)
    return f"{_CATEGORY_CN.get('npu_bubble')}：{keys} 的算子启动间隔低于固定阈值 5000 ns"

def render(detection: Dict, level: str, thresholds, metric_dir: str = "",
           data_dir: str = "") -> str:
    """把检测结果表翻成自然语言。只描述检测层给的结果，不做任何额外判断。"""
    lines: List[str] = ["# 检测结论", ""]
    found = []
    for cat in _ORDER:
        items = [i for i in (detection.get(cat) or [])
                 if i.get("is_abnormal") is not False]
        if items:
            items.sort(key=lambda i: float(i.get("metric_value") or 0), reverse=True)
            found.append((cat, items))

    level_text = f"`{level}`" if level else "未知"
    if not found:
        lines.append(f"数据级别 {level_text}。**本次未检出异常。**")
        return "\n".join(lines) + "\n"

    lines.append(f"数据级别 {level_text}。共检出 **{len(found)} 类异常**：")
    lines.append("")
    mapping, _groups = _detection_groups(data_dir) if data_dir else ({}, [])
    for cat, items in found:
        if cat == "cpu":
            lines.append(f"- {_node_line(items, metric_dir)}")
        elif cat in _DETAIL_CATS:
            lines.append(f"- {_detail_line(cat, items, metric_dir, mapping)}")
        elif cat == "comm":
            lines.append(f"- {_comm_line(items)}")
        elif cat == "pp_comm":
            lines.append(f"- {_pp_line(items)}")
        elif cat == "npu_bubble":
            lines.append(f"- {_bubble_line(items)}")
        else:
            detail = "、".join(
                f"{_ranks_text(i.get('display_key'))} 约 {float(i.get('metric_value') or 0):.2f} 倍"
                for i in items)
            lines.append(f"- {_CATEGORY_CN.get(cat, cat)}：{detail}")
    lines.append("")
    lines.append(
        "> 倍数 = 该卡相对检测组内基线的劣化倍数，越大越异常。本次阈值："
        f"计算（针对KERNEL_AICORE和KERNEL_AIVEC）{thresholds[0]} 倍、"
        f"IO/CPU（针对MEMCPY_ASYNC和Host）{thresholds[1]} 倍、"
        f"集合通信（comm）{thresholds[2]} 倍、"
        f"PP 点对点通信等待占比（pp_comm）{_pp_percent()}%。")

    if any(cat == "memcpy_async" for cat, _ in found):
        lines.append("")
        lines.extend(_MEMCPY_NOTE.splitlines())

    note = _LEVEL_NOTE.get(level or "")
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    return "\n".join(lines)


def render_from_dir(out_dir: str, level: str, thresholds) -> str:
    """读检测结果 JSON，返回结论文本（只在控制台输出，不落盘）。"""
    det = {}
    src = os.path.join(out_dir, "straggler_detection_result.json")
    try:
        with open(src, "r", encoding="utf-8") as f:
            det = json.load(f) or {}
    except (OSError, ValueError):
        det = {}
    return render(det, level, thresholds,
                  metric_dir=os.path.join(out_dir, "op_metric"), data_dir=out_dir)
