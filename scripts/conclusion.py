from __future__ import annotations
import csv
import logging
import json
import os
import re
from typing import Dict, List, Optional, Tuple

_CATEGORY_CN = {
    "KERNEL_AICORE": "KERNEL_AICORE类算子",
    "kernel_aivec": "KERNEL_AIVEC类算子",
    "memcpy_async": "MEMCPY_ASYNC类算子",
    "computing_load": "computing_load（混合负载粗筛）",
    "comm": "comm（通信组带宽）",
    "cpu": "Host（CPU）粗筛",
    "pp_comm": "pp_comm（PP 流水线等待）",
}
_ORDER = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async", "computing_load",
          "comm", "cpu", "pp_comm")
_COLUMN = {
    "KERNEL_AICORE": "KERNEL_AICORE",
    "kernel_aivec": "KERNEL_AIVEC",
    "memcpy_async": "MEMCPY_ASYNC",
    "cpu": "ZP_Host",
}
_DETAIL_CATS = ("KERNEL_AICORE", "kernel_aivec", "memcpy_async")
_LEVEL_NOTE = {
    "levelnone": "本次数据只含 MsTx，没有计算 / 搬运算子，计算类与搬运类未参与检测。",
    "cluster": "cluster 单文件评估通信、Host 级 stall 粗筛（ClusterStepTraceTime.free）和混合负载粗筛（ClusterStepTraceTime.computing）；慢计算需原始 KERNEL_AICORE/AIVEC 复核。",
}
_MEMCPY_NOTE_HEAD = """关于 memcpy_async 的说明：该指标统计的是「拷贝算子的平均耗时」，包括：设备拷贝（H2D/D2H）、
设备内拷贝、设备间拷贝（D2D）。部分情况下会让少数卡的均值结构性偏高，进而导致误报：

1. **PP>1 且 TP>1（Megatron 框架特有）**：只有首/末 stage 的 tp0 需要把 data/label 从主机搬到
   卡上、再广播给同 TP 组，其余卡为 0。例：PP4TP4 时就是 rank0 和 rank12。
2. **PP=1 且 TP>1（Megatron 框架特有，如 TP4DP4）**：各 DP 组的 tp0 承担同样的 H2D，
   例：rank0/4/8/12。
3. **任务数不均**：任务少的卡缺少短任务，平均值容易贴线偏高。"""

def _memcpy_note(ranks) -> List[str]:
    rs = sorted({int(r) for r in ranks})
    if not rs:
        return _MEMCPY_NOTE_HEAD.splitlines()
    if len(rs) == 1:
        who = f"rank{rs[0]}"
    elif len(rs) == 2:
        who = f"rank{rs[0]} 和 rank{rs[1]}"
    else:
        who = "、".join(f"rank{r}" for r in rs)
    tail = (f"单从「拷贝算子的平均耗时」来看，{who} 确实存在内存拷贝耗时长的问题，"
            "如需确定是否真实存在 IO 劣化，需进一步排查。")
    return _MEMCPY_NOTE_HEAD.splitlines() + ["", tail]


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
        base_ranks = [r for r in group if r not in flagged] or list(group)
        base = None
        for item in group_items:
            try:
                v = _read_metric(metric_dir, int(_first_rank(item)), column)
            except (TypeError, ValueError):
                v = None
            ratio = float(item.get("metric_value") or 0)
            if v is not None and ratio > 0:
                base = v / ratio      # 与检测层 kmeans 基线一致
                break
        if base is None:
            base, base_ranks = _baseline(column, metric_dir, group, flagged)
        who = ("rank " + "、".join(str(r) for r in base_ranks)) if base_ranks else "—"
        sentence = ratios
        if values:
            sentence += "，" + "，".join(values)
        if base is not None:
            sentence += f"，检测组内基线（{who}）为 {base:.3f} ms"
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
    if not os.path.isdir(metric_dir):
        return f"{label}：{ratios}"
    sentence = ratios
    values = []
    for n in names:
        v = _node_value(metric_dir, nodes, n)
        if v is not None:
            values.append(f"{n} 为 {v:.3f} ms")
    if values:
        sentence += "，" + "，".join(values)
    base_nodes = [n for n in nodes if n not in names]
    base = None
    for n, item in zip(names, items):
        v = _node_value(metric_dir, nodes, n)
        ratio = float(item.get("metric_value") or 0)
        if v is not None and ratio > 0:
            base = v / ratio          # 与检测层 kmeans 基线一致
            break
    if base is None:
        base_vals = [v for v in (_node_value(metric_dir, nodes, n) for n in base_nodes)
                     if v is not None]
        if base_vals:
            base = sum(base_vals) / len(base_vals)
    if base is not None:
        who = "、".join(base_nodes) if base_nodes else "—"
        sentence += f"，其它节点基线（{who}）为 {base:.3f} ms"
    else:
        sentence += "，其它节点基线无可用数据"
    return f"{label}：{sentence}"

def _cluster_domain_groups(db_path: str):
    """cluster_analysis.db 的并行域 -> 组列表。直接复用检测层的实现。"""
    try:
        from nodelevel_data_handler import load_cluster_parallels
        parallels = load_cluster_parallels(db_path)
        if parallels:
            return {k: [list(g) for g in v] for k, v in parallels.items()}
        logging.getLogger(__name__).warning(
            "[CONCLUSION] CommunicationGroupMapping 未解析出组信息（检测层同款查询）")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[CONCLUSION] 读取 cluster 组信息失败：{e}")
    return {}

def _domain_groups_map(data_dir: str) -> Dict[str, List[List[int]]]:
    """通信域 -> 该域内的所有 Group。支持 op_metric 目录与 cluster 单文件 db。"""
    if data_dir and os.path.isfile(data_dir):
        return _cluster_domain_groups(data_dir)
    metric_dir = os.path.join(data_dir, "op_metric") if data_dir else ""
    out: Dict[str, List[List[int]]] = {}
    if not metric_dir:
        return out
    try:
        names = sorted(os.listdir(metric_dir))
    except OSError:
        return out
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
            if not pg or not ranks:
                continue
            bucket = out.setdefault(pg, [])
            if ranks not in bucket:
                bucket.append(ranks)
    return {k: sorted(v) for k, v in out.items()}

def _brk(domain: str, ranks) -> str:
    """组标签统一用全角方括号（半角 [x] 会被 Markdown 当引用链接吞掉）。"""
    inner = ",".join(str(r) for r in ranks)
    return f"{domain}［{inner}］" if domain else f"［{inner}］"

def _groups_text(domain: str, groups: List[List[int]], limit: int = 6) -> str:
    shown = groups[:limit]
    text = "、".join(_brk(domain, g) for g in shown)
    if len(groups) > limit:
        text += f" 等 {len(groups)} 个组"
    return text

def _parse_comm_key(key) -> tuple:
    s = str(key or "")
    m = re.match(r"^([A-Za-z0-9_]+)\s*\[(.*)\]$", s)
    if not m:
        return "", []
    try:
        ranks = sorted(int(x) for x in m.group(2).split(",") if x.strip())
    except (TypeError, ValueError):
        ranks = []
    return m.group(1), ranks

_CATEGORY_GROUPS = (
    ("计算类异常", ("KERNEL_AICORE", "kernel_aivec")),
    ("混合负载粗筛", ("computing_load",)),
    ("IO类异常", ("memcpy_async",)),
    ("通信类异常", ("comm", "pp_comm")),
    ("Host粗筛", ("cpu",)),
)

def _body(text: str) -> str:
    """去掉「标签：」前缀，只留描述。"""
    return text.split("：", 1)[1] if "：" in text else text

_COMM_NOTE = "（取该组内劣化最严重的［并行域, 算子, Count］组合）"
_COMM_NOTE_LEVEL = "（各［算子, Count］组合的带宽与中位数之比取平均）"

def _comm_rows(items: List[Dict], data_dir: str = "", level: str = ""):
    """通信类描述。基线一律排除本类别已判为异常的卡 / 组。

    level1 / cluster 为两级口径，分两段输出：组内逐卡检测、组间逐组检测。
    """
    domains = _domain_groups_map(data_dir) if data_dir else {}
    two_level = level in ("level1", "cluster")

    parsed = [(i,) + _parse_comm_key(i.get("display_key")) for i in items]
    flagged_ranks: Dict[str, set] = {}
    flagged_groups: Dict[str, List[List[int]]] = {}
    for _i, domain, target in parsed:
        if not domain or not target:
            continue
        flagged_ranks.setdefault(domain, set()).update(target)
        if len(target) > 1:
            flagged_groups.setdefault(domain, []).append(target)

    def _clean_groups(domain: str, groups: List[List[int]], exclude) -> List[List[int]]:
        bad = flagged_ranks.get(domain) or set()
        out = []
        for g in groups:
            if exclude and g == exclude:
                continue
            if set(g) & bad:
                continue
            out.append(g)
        return out

    plain: List[str] = []
    within: List[str] = []
    between: List[str] = []
    for item, domain, target in parsed:
        ratio = float(item.get("metric_value") or 0)
        groups = domains.get(domain) or []

        if not two_level or len(target) > 1:
            others = _clean_groups(domain, groups, target)
            if others:
                who = _groups_text(domain, others)
            elif groups:
                who = "域内其它组均含异常卡，无可用基线"
            else:
                who = "域内组信息不可用"
            label = (_brk(domain, target)
                     if domain and target else _ranks_text(item.get("display_key")))
            head = f"{domain} 域的 " if not two_level and domain else ""
            text = f"{head}{label} 相对域内基线（{who}）约 {ratio:.2f} 倍"
            (between if two_level else plain).append(text)
            continue

        rank = target[0]
        group = next((g for g in groups if rank in g), [])
        bad = flagged_ranks.get(domain) or set()
        peers = [r for r in group if r != rank and r not in bad]
        if peers:
            who = _brk(domain, peers)
        elif group:
            who = "同组其它卡均为异常卡，无可用基线"
        else:
            who = "组信息不可用"
        within.append(f"{_brk(domain, [rank])} 相对同组基线（{who}）约 {ratio:.2f} 倍")

    out = []
    if plain:
        # 旧口径（levelnone / level0）：劣化指数取各 opType 的最大值
        out.append(("集合通信带宽", "；".join(plain) + _COMM_NOTE))
    if within:
        # 两级口径：指标 = 各（算子, Count）组合的带宽/中位数 的平均值
        out.append(("集合通信带宽（组内逐卡检测）",
                    "；".join(within) + _COMM_NOTE_LEVEL))
    if between:
        out.append(("集合通信带宽（组间逐组检测）",
                    "；".join(between) + _COMM_NOTE_LEVEL))
    return out

def _pp_line(items: List[Dict]) -> str:
    parts = []
    for i in items:
        pct = float(i.get("metric_value") or 0) * 100.0
        ranks = sorted({int(r) for r in _ranks_from_key(i.get("display_key"))})
        label = _brk("pp", ranks) if ranks else _ranks_text(i.get("display_key"))
        parts.append(f"{label} 的点对点等待时间占会话时长约 {pct:.2f}%")
    return f"{_CATEGORY_CN.get('pp_comm')}：" + "、".join(parts) + f"，超过阈值 {_pp_percent()}%"

def _threshold_text(level: str, thresholds) -> str:
    """只列出该数据级别实际用到的阈值。"""
    compute, io, comm = thresholds[0], thresholds[1], thresholds[2]
    parts = []
    if level == "cluster":
        return (f"计算类（KERNEL_AICORE / KERNEL_AIVEC / computing）{compute} 倍、"
                f"IO/CPU（MEMCPY_ASYNC / Host）{io} 倍、集合通信（comm）{comm} 倍")
    if level in ("level0", "level1"):
        parts.append(f"计算（针对KERNEL_AICORE和KERNEL_AIVEC）{compute} 倍")
    parts.append(f"IO/CPU（针对MEMCPY_ASYNC和Host）{io} 倍")
    parts.append(f"集合通信（comm）{comm} 倍")
    if level in ("level0", "levelnone"):
        parts.append(f"PP 点对点通信等待占比（pp_comm）{_pp_percent()}%")
    return "、".join(parts)

_METRIC_NOTES = {
    "computing_load": "cluster 单文件聚合值（cluster数据独有），含 MIX/AICPU 等，仅作混合负载粗筛。",
    "ZP_Host": "通信算子 + KERNEL_AICORE 的 Host 侧耗时均值",
    "free": "cluster 单文件 NPU 空闲时间（cluster数据独有），仅作 Host 粗筛",
    "pp_comm": "pp点对点通信，劣化数值为pp通信的等待占整个会话的占比",
}


def _abnormal_items(detection: Dict, cat: str) -> List[Dict]:
    items = [i for i in (detection.get(cat) or [])
             if i.get("is_abnormal") is not False]
    items.sort(key=lambda i: float(i.get("metric_value") or 0), reverse=True)
    return items


def _pct(degradation) -> str:
    """劣化倍数 → 百分数：倍数 1.68 表示比基线慢 68%。"""
    try:
        value = (float(degradation) - 1.0) * 100.0
    except (TypeError, ValueError):
        return "?"
    return f"{max(value, 0.0):.0f}%"


def _ratio_pct(ratio) -> str:
    """占比（0~1）→ 百分数，用于 pp_comm 的等待占比。"""
    try:
        value = float(ratio) * 100.0
    except (TypeError, ValueError):
        return "?"
    return f"{max(value, 0.0):.0f}%"


def _rank_label(item: Dict) -> str:
    key = str(item.get("display_key") or "")
    if key.startswith("Rank"):
        return key
    if key.isdigit():
        return f"Rank{key}"
    return key


def _comm_label(item: Dict, is_pp: bool = False) -> Tuple[str, str]:
    key = str(item.get("display_key") or "")
    if is_pp:
        inner = key.strip("[]")
        return f"pp[{inner}]", "group"
    m = re.match(r"^([A-Za-z0-9_]+)\[(.*)\]$", key)
    if m:
        domain, inner = m.group(1), m.group(2)
        parts = [x.strip() for x in inner.split(",") if x.strip()]
        if len(parts) == 1:
            return f"{domain} Rank{parts[0]}", "card"
        return f"{domain}[{inner}]", "group"
    if key.isdigit():
        return f"Rank{key}", "card"
    return key, "group"


def _fmt_items(items: List[Dict], label_fn) -> str:
    return "、".join(f"{label_fn(i)}({_pct(i.get('metric_value'))})" for i in items)


def _top3_line(labels: List[str], prefix: str = "") -> str:
    top = labels[:3]
    if not top:
        return ""
    return f"{prefix}其中 {'、'.join(top)} 为TOP{len(top)}点故障点，建议优先排查"


def _metric_line(name: str, items: List[Dict], label_fn, note: str = "",
                  top: bool = False) -> List[str]:
    if not items:
        return []
    out = [f"{name}：{_fmt_items(items, label_fn)}"]
    if top and len(items) > 3:
        top_line = _top3_line([label_fn(i) for i in items])
        if top_line:
            out.append(top_line)
    if note:
        out.append(f"说明：{note}")
    return out


def render(detection: Dict, level: str, thresholds, metric_dir: str = "",
           data_dir: str = "") -> str:
    """把检测结果表翻成自然语言。只描述检测层给的结果，不做任何额外判断。"""
    lines: List[str] = [f"检测结果（数据级别：{level or '未知'}）："]

    # 1、计算类
    compute_lines: List[str] = []
    aicore = _abnormal_items(detection, "KERNEL_AICORE")
    aivec = _abnormal_items(detection, "kernel_aivec")
    computing = _abnormal_items(detection, "computing_load")
    compute_lines += _metric_line("KERNEL_AICORE", aicore, _rank_label, top=True)
    compute_lines += _metric_line("KERNEL_AIVEC", aivec, _rank_label, top=True)
    compute_lines += _metric_line(
        "computing（cluster数据独有）", computing, _rank_label,
        _METRIC_NOTES["computing_load"], top=True)
    lines.append("1、计算类故障检测结果：")
    lines.extend(compute_lines or ["无"])

    # 2、通信类
    cards: List[Tuple[str, Dict]] = []
    groups: List[Tuple[str, Dict]] = []
    for item in _abnormal_items(detection, "comm"):
        label, kind = _comm_label(item)
        (cards if kind == "card" else groups).append((label, item))
    pp_items: List[Tuple[str, Dict]] = []
    for item in _abnormal_items(detection, "pp_comm"):
        label, _kind = _comm_label(item, is_pp=True)
        pp_items.append((label, item))

    lines.append("2、通信类故障检测结果：")
    if not cards and not groups and not pp_items:
        lines.append("无")
    else:
        card_text = "、".join(f"{label}({_pct(item.get('metric_value'))})"
                             for label, item in cards)
        group_text = "、".join(f"{label}({_pct(item.get('metric_value'))})"
                              for label, item in groups)
        lines.append(f"慢通信卡：{card_text or '无'}")
        top = _top3_line([label for label, _ in cards])
        if top:
            lines.append(top)
        lines.append(f"慢通信域：{group_text or '无'}")
        top = _top3_line([label for label, _ in groups])
        if top:
            lines.append(top)
        if pp_items:
            pp_text = "、".join(f"{label}({_ratio_pct(item.get('metric_value'))})"
                               for label, item in pp_items)
            lines.append(f"PP点对点通信等待：{pp_text}")
            top = _top3_line([label for label, _ in pp_items])
            if top:
                lines.append(top)
            lines.append(f"说明：{_METRIC_NOTES['pp_comm']}")

    # 3、Host 侧
    host_items = _abnormal_items(detection, "cpu")
    host_name = "free（空闲时间，cluster数据独有）" if level == "cluster" else "ZP_Host（算子Host耗时）"
    host_note = _METRIC_NOTES["free"] if level == "cluster" else _METRIC_NOTES["ZP_Host"]
    host_lines = _metric_line(host_name, host_items, lambda i: str(i.get("display_key")), host_note)
    lines.append("3、Host侧故障检测结果：")
    lines.extend(host_lines or ["无"])

    # 4、内存搬运类
    memcpy_items = _abnormal_items(detection, "memcpy_async")
    memcpy_lines = _metric_line("MEMCPY_ASYNC", memcpy_items, _rank_label, top=True)
    lines.append("4、内存搬运类故障检测结果：")
    lines.extend(memcpy_lines or ["无"])
    if memcpy_items:
        ranks = sorted({r for i in memcpy_items for r in _ranks_from_key(i.get("display_key"))})
        lines.append("")
        lines.extend(_memcpy_note(ranks))

    lines.append("")
    lines.append("（阈值：" + _threshold_text(level, thresholds) + "）")

    unevaluated = []
    if level == "cluster":
        unevaluated.append("KERNEL_AICORE/KERNEL_AIVEC、内存搬运类")
    elif level == "level1":
        unevaluated.append("PP点对点通信")
    elif level == "levelnone":
        unevaluated.append("计算类、内存搬运类、Host侧")
    if unevaluated:
        lines.append("未评估：" + "；".join(unevaluated) + "。")

    lines.append("")
    return "\n".join(lines)


def render_from_dir(out_dir: str, level: str, thresholds, source_path: str = "") -> str:
    """读检测结果 JSON，返回结论文本（只在控制台输出，不落盘）。"""
    det = {}
    src = os.path.join(out_dir, "straggler_detection_result_flat.json")
    if not os.path.exists(src):
        src = os.path.join(out_dir, "straggler_detection_result.json")
    try:
        with open(src, "r", encoding="utf-8") as f:
            det = json.load(f) or {}
    except (OSError, ValueError):
        det = {}
    return render(det, level, thresholds,
                  metric_dir=os.path.join(out_dir, "op_metric"),
                  data_dir=source_path or out_dir)
