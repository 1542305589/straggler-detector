from __future__ import annotations
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

_CATEGORY_CN = {
    "KERNEL_AICORE": "计算类（AI Core 算子）",
    "kernel_aivec": "计算类（AI Vector 算子）",
    "memcpy_async": "内存搬运类",
    "comm": "通信类",
    "cpu": "主机侧（Host 下发）",
    "npu_bubble": "NPU 气泡（算子间空转）",
    "pp_comm": "PP 流水线等待",
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
外部拷贝（aclrtMemcpyAsync）、设备内拷贝和集合通信拷贝。下面这些已知情况会让少数卡的均值
结构性偏高，看着像慢卡、其实不是：

1. **PP>1 且 TP>1**：只有首/末 stage 的 tp0 需要把 data/label 从主机搬到卡上、再广播给同 TP 组，
   其余卡为 0。例：PP4TP4 时就是 rank0 和 rank12。
2. **PP=1 且 TP>1**（如 TP4DP4）：各 DP 组的 tp0 承担同样的 H2D，例：rank0/4/8/12。
3. **任务数不均**：任务少的卡缺少短任务，平均值容易贴线偏高。
4. **采集窗口残缺**：某些卡 STEP_TIME 只覆盖少量 step 时，均值会假性偏移，还会把其它卡带成
   镜像误报。

当前数据既没有拷贝方向（H2D/D2H/D2D），也没有逐次数据量，所以只看这个指标无法定论；
如需确认是否真的存在搬运劣化，需要补充这两项数据。"""


def _first_rank(item: Dict) -> str:
    ranks = _ranks_from_key(item.get("display_key"))
    return str(ranks[0]) if ranks else str(item.get("display_key"))

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
        ratios = "，".join(
            f"rank{_first_rank(i)} 相对检测组内基线约 "
            f"{float(i.get('metric_value') or 0):.2f} 倍" for i in group_items)
        values = []
        for r in flagged:
            v = _read_metric(metric_dir, r, column)
            if v is not None:
                values.append(f"rank{r} 为 {v:.3f} ms")
        base, base_ranks = _baseline(column, metric_dir, group, flagged)
        sentence = ratios
        if values:
            sentence += "；" + "、".join(values)
        if base is not None:
            who = _fmt_ranks(base_ranks) if base_ranks else "—"
            sentence += f"；检测组内基线平均（{who}）为 {base:.3f} ms"
        parts.append(sentence)
    return f"{label}：" + "。".join(parts)


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
        if cat in _DETAIL_CATS:
            lines.append(f"- {_detail_line(cat, items, metric_dir, mapping)}")
        else:
            detail = "、".join(
                f"{_ranks_text(i.get('display_key'))} 约 {float(i.get('metric_value') or 0):.2f} 倍"
                for i in items)
            lines.append(f"- {_CATEGORY_CN.get(cat, cat)}：{detail}")
    lines.append("")
    lines.append(
        "> 倍数 = 该卡相对检测组内基线的劣化倍数，越大越异常。"
        f"本次阈值：计算 {thresholds[0]} 倍、IO/CPU {thresholds[1]} 倍、通信 {thresholds[2]} 倍。")

    if any(cat == "memcpy_async" for cat, _ in found):
        lines.append("")
        lines.extend(_MEMCPY_NOTE.splitlines())

    note = _LEVEL_NOTE.get(level or "")
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    return "\n".join(lines)


def write(out_dir: str, level: str, thresholds):
    """读检测结果 JSON → 生成 conclusion.md，返回 (文本, 路径)。"""
    det: Dict = {}
    src = os.path.join(out_dir, "straggler_detection_result.json")
    try:
        with open(src, "r", encoding="utf-8") as f:
            det = json.load(f) or {}
    except (OSError, ValueError):
        det = {}
    text = render(det, level, thresholds,
                  metric_dir=os.path.join(out_dir, "op_metric"), data_dir=out_dir)
    path = os.path.join(out_dir, "conclusion.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, path
