from __future__ import annotations
import json
import os
import re
from typing import Dict, List

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
_LEVEL_NOTE = {
    "levelnone": "本次数据只含 MsTx，没有计算 / 搬运算子，计算类与搬运类未参与检测。",
    "level0": "level0 数据没有通信带宽表，通信带宽未评估（**不能当作正常**）。",
    "level1": "通信按 analysis.db 的两级带宽口径评估。",
    "cluster": "cluster 单文件只评估通信，其他类别未参与。",
}

def _ranks_text(key) -> str:
    s = str(key or "").strip()
    m = re.match(r"^([A-Za-z0-9_]+)\s*\[(.*)\]$", s)
    if m:
        domain, inner = m.group(1), m.group(2)
        ranks = [x.strip() for x in inner.split(",") if x.strip()]
        return f"{domain} 组的 rank {'、'.join(ranks)}"
    m = re.match(r"^rank(\d+)-(\d+)$", s)
    if m:
        return f"rank {m.group(1)}–{m.group(2)}"
    m = re.match(r"^rank(\d+)$", s)
    if m:
        return f"rank {m.group(1)}"
    if s.isdigit():
        return f"rank {s}"
    return s

def _item_text(item: Dict) -> str:
    return f"{_ranks_text(item.get('display_key'))} 约 {float(item.get('metric_value') or 0):.2f} 倍"

def render(detection: Dict, level: str, thresholds) -> str:
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
    else:
        lines.append(f"数据级别 {level_text}。共检出 **{len(found)} 类异常**：")
        lines.append("")
        for cat, items in found:
            detail = "、".join(_item_text(i) for i in items)
            lines.append(f"- **{_CATEGORY_CN.get(cat, cat)}**：{detail}")
        lines.append("")
        lines.append(
            "> 倍数 = 该卡 / 该组相对同组正常卡的劣化倍数，越大越异常。"
            f"本次阈值：计算 {thresholds[0]} 倍、IO/CPU {thresholds[1]} 倍、通信 {thresholds[2]} 倍。")

    note = _LEVEL_NOTE.get(level or "")
    if note:
        lines.append("")
        lines.append(f"未参与检测的部分：{note}")
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
    text = render(det, level, thresholds)
    path = os.path.join(out_dir, "conclusion.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, path
