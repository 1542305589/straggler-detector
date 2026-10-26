# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""输出层：把 formal_result + evidence 渲染成固定 Markdown，并写 formal_result.json。"""
from __future__ import annotations

import json
import os
from typing import Dict, List


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_无数据_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def _macro_table(ev: Dict) -> str:
    by_rank = ev.get("macro", {}).get("step_trace", {}).get("by_rank", {}) or {}
    rows = []
    for r in sorted(by_rank, key=lambda x: int(x)):
        t = by_rank[r].get("totals", {}) or {}
        rows.append([
            r,
            _fmt(t.get("computing_ms")),
            _fmt(t.get("communication_ms")),
            _fmt(t.get("free_ms")),
            _fmt(t.get("stage_ms")),
            _fmt(t.get("bubble_ms")),
            _fmt(t.get("preparing_ms")),
        ])
    return _table(["Rank", "Computing(ms)", "Communication(ms)", "Free(ms)",
                   "Stage(ms)", "Bubble(ms)", "Preparing(ms)"], rows)


def _host_table(ev: Dict) -> str:
    host = ev.get("host", {}) or {}
    gloo = host.get("gloo", {}) or {}
    sd = host.get("sync_device", {}) or {}
    se = host.get("sync_event", {}) or {}
    ss = host.get("sync_stream", {}) or {}
    ranks = sorted(set(gloo) | set(sd) | set(se) | set(ss), key=lambda x: int(x))
    rows = []
    for r in ranks:
        rows.append([
            r,
            _fmt((gloo.get(r) or {}).get("max_ms")),
            _fmt((sd.get(r) or {}).get("max_ms")),
            _fmt((se.get(r) or {}).get("max_ms")),
            _fmt((ss.get(r) or {}).get("max_ms")),
        ])
    return _table(["Rank", "gloo max(ms)", "SyncDevice max(ms)",
                   "SyncEvent max(ms)", "SyncStream max(ms)"], rows)


def _compute_table(ev: Dict) -> str:
    by_rank = ev.get("compute", {}).get("by_rank", {}) or {}
    rows = []
    for r in sorted(by_rank, key=lambda x: int(x)):
        ops = by_rank[r].get("ops", {}) or {}
        total = sum(_f(v.get("sum_ms", 0.0)) for v in ops.values())
        cnt = sum(_f(v.get("count", 0.0)) for v in ops.values())
        rows.append([r, _fmt(cnt, 0), _fmt(total), _fmt(total / cnt if cnt else 0.0)])
    return _table(["Rank", "Count", "Total(ms)", "Avg(ms)"], rows)


def _group_label(ev: Dict, group: str) -> str:
    for domain, groups in (ev.get("meta", {}).get("parallels", {}) or {}).items():
        for g in groups:
            if ",".join(str(x) for x in sorted(g)) == group:
                return f"{domain}[{group}]"
    return group


def _comm_level1_table(ev: Dict) -> str:
    """Level-1：逐 rank 逐算子带宽（新两级带宽口径）。"""
    by = ev.get("comm", {}).get("by_rank_group", {}) or {}
    rows = []
    for r in sorted(by, key=lambda x: int(x)):
        for g in sorted(by[r]):
            label = _group_label(ev, g)
            for op in sorted(by[r][g]):
                d = by[r][g][op] or {}
                bw = _f(d.get("bandwidth_gbps", 0.0))
                if bw <= 0:
                    continue
                rows.append([r, label, op, _fmt(d.get("count"), 0),
                             _fmt(d.get("size_mb")), _fmt(bw),
                             d.get("transport_type") or "-"])
    if len(rows) > 40:
        rows = rows[:40] + [["...", "（仅展示前 40 条）", "", "", "", "", ""]]
    return _table(["Rank", "域/组", "Op", "Count", "Size(MB)", "BW(GB/s)", "Transport"], rows)


def _comm_level2_table(ev: Dict) -> str:
    """Level-2：按域/组聚合带宽；组间 min/max 超过通信阈值标 ***。"""
    try:
        import config
        th = config.get_comm_threshold()
    except Exception:
        th = 1.3
    by = ev.get("comm", {}).get("by_rank_group", {}) or {}
    agg: Dict[tuple, Dict] = {}
    for _r, groups in by.items():
        for g, ops in (groups or {}).items():
            label = _group_label(ev, g)
            for op, d in (ops or {}).items():
                a = agg.setdefault((label, op), {"count": 0, "size": 0.0, "time": 0.0})
                a["count"] += _f(d.get("count", 0))
                a["size"] += _f(d.get("size_mb", 0.0))
                a["time"] += _f(d.get("transit_ms", 0.0))
    by_op: Dict[str, List[float]] = {}
    for (label, op), a in agg.items():
        a["bw"] = a["size"] / a["time"] if a["time"] > 0 else 0.0
        by_op.setdefault(op, []).append(a["bw"])
    rows = []
    for (label, op), a in sorted(agg.items()):
        mark = ""
        vals = [v for v in by_op.get(op, []) if v > 0]
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            ratio = hi / lo if lo > 0 else 1.0
            if ratio >= th and abs(a["bw"] - lo) < 1e-9:
                mark = "***"
        rows.append([label, op, _fmt(a["count"], 0), _fmt(a["size"]),
                     _fmt(a["bw"]), mark])
    return _table(["域/组", "Op", "Count", "Size(MB)", "BW(GB/s)", "标记"], rows)


def _free_table(ev: Dict) -> str:
    fw = ev.get("host", {}).get("free_windows", {}) or {}
    starts = ev.get("host", {}).get("session_start_ms", {}) or {}
    by_rank = ev.get("macro", {}).get("step_trace", {}).get("by_rank", {}) or {}
    rows = []
    for r in sorted(fw, key=lambda x: int(x)):
        gaps = fw[r].get("gaps", []) or []
        step_free = _f((by_rank.get(r, {}) or {}).get("totals", {}).get("free_ms", 0.0))
        if not gaps:
            rows.append([r, "-", "-", "-", _fmt(fw[r].get("total_ms")), _fmt(step_free)])
            continue
        g = gaps[0]
        base = _f(starts.get(r, 0.0))
        rows.append([r, _fmt(_f(g.get("start_ms")) - base),
                     _fmt(_f(g.get("end_ms")) - base),
                     _fmt(g.get("duration_ms")), _fmt(fw[r].get("total_ms")),
                     _fmt(step_free)])
    return _table(["Rank", "最大 TASK 间隙起点(相对 ms)", "终点(相对 ms)",
                   "间隙(ms)", "TASK 总间隙(ms)", "StepTraceTime.free(ms)"], rows)


def _freq_table(ev: Dict) -> str:
    by = ev.get("freq", {}).get("by_rank", {}) or {}
    rows = []
    for r in sorted(by, key=lambda x: int(x)):
        for dev, d in sorted((by[r] or {}).items()):
            rows.append([r, dev, _fmt(d.get("min_mhz"), 0),
                         _fmt(d.get("max_mhz"), 0), _fmt(d.get("avg_mhz"), 0)])
    return _table(["Rank", "Device", "Min(MHz)", "Max(MHz)", "Avg(MHz)"], rows)


def _findings_table(result: Dict) -> str:
    rows = []
    for f in result.get("findings", []):
        rows.append([
            f.get("rule_id", ""),
            f.get("status", ""),
            f.get("category", ""),
            f.get("metric", ""),
            _fmt(f.get("value")),
            _fmt(f.get("baseline_value")),
            _fmt(f.get("ratio")),
            ",".join(str(x) for x in (f.get("slow_side") or [])) or "-",
            f.get("summary", ""),
        ])
    return _table(["规则", "状态", "类别", "指标", "慢值", "基准值", "比值", "慢侧", "结论"], rows)


def render_markdown(ev: Dict, result: Dict) -> str:
    meta = ev.get("meta", {}) or {}
    primary = result.get("primary", {}) or {}
    lines: List[str] = []
    lines.append("# 慢节点诊断报告\n")

    if (result.get("validation") or {}).get("errors"):
        lines.append("> ⚠️ **本报告未通过校验，结论不可直接采信**"
                     "（详见 `formal_result.json` 的 `validation`）\n")

    lines.append("## 1. 数据与结果概览\n")
    lines.append(f"- 数据目录：`{meta.get('data_dir', '')}`")
    lines.append(f"- 数据级别：`{meta.get('level', '')}`")
    lines.append(f"- 有效 Rank 数：{len(meta.get('ranks') or [])}")
    lines.append(f"- 并行域：`{', '.join(sorted((meta.get('parallels') or {}).keys())) or '无'}`")
    lines.append(f"- Job 类型：`{meta.get('job_type', 'unknown')}`")
    lines.append(f"- 会话时长：{_fmt(meta.get('session_ms'))} ms")
    det = ev.get("detection", {}) or {}
    det_summary = []
    for cat in sorted(det):
        items = det.get(cat) or []
        if items:
            det_summary.append(f"{cat}={len(items)}")
    lines.append(f"- 检测结果摘要：`{', '.join(det_summary) or '空'}`")
    lines.append("")

    lines.append("## 2. 检测结论（权威）\n")
    if result.get("conclusion_source") == "detection":
        lines.append("- 结论来源：**检测结果**（检测层基于算子耗时/通信带宽，结论层仅辅助）")
    else:
        lines.append("- 结论来源：**辅助分析**（检测层无异常，置信度已降级）")
    lines.append(f"- verdict：`{result.get('verdict', '')}`")
    lines.append(f"- confidence：`{result.get('confidence', '')}`")
    lines.append(f"- summary：{primary.get('summary', '')}")
    lines.append("")

    lines.append("## 3. 慢侧 / 基准侧\n")
    lines.append(f"- 慢侧：`{primary.get('slow_side') or []}`")
    lines.append(f"- 基准侧：`{primary.get('baseline_side') or []}`")
    lines.append("")

    lines.append("## 4. 类别\n")
    lines.append(f"- primary：`{primary.get('category', '')}`（rule `{primary.get('rule_id', '')}`）")
    auth = sorted({f.get("category") for f in result.get("findings", [])
                   if f.get("status") == "authoritative"})
    aux = sorted({f.get("category") for f in result.get("findings", [])
                  if f.get("status") == "auxiliary"})
    lines.append(f"- 权威类别：`{', '.join(auth) or '无'}`")
    lines.append(f"- 辅助类别：`{', '.join(aux) or '无'}`")
    lines.append("")

    lines.append("## 5. 规则命中与证据\n")
    lines.append(_findings_table(result))
    lines.append("### 5.1 宏观迭代拆解（StepTraceTime）\n")
    lines.append(_macro_table(ev))
    lines.append("### 5.2 Host 侧阻塞（PYTORCH_API / CANN_API）\n")
    lines.append(_host_table(ev))
    lines.append("### 5.2b 空闲窗口（TASK 间隙 vs StepTraceTime.free）\n")
    lines.append(_free_table(ev))
    lines.append("> TASK 间隙可能被 NOTIFY/EVENT 等 wait 任务填满；判断 NPU 是否空闲以 `StepTraceTime.free` 为准。\n")
    lines.append("### 5.3 计算耗时（COMPUTE_TASK_INFO）\n")
    lines.append(_compute_table(ev))
    lines.append("### 5.4 通信带宽 Level-1（逐 rank，越小越慢）\n")
    lines.append(_comm_level1_table(ev))
    lines.append("### 5.4 通信带宽 Level-2（按域/组聚合，*** 为慢组）\n")
    lines.append(_comm_level2_table(ev))
    if meta.get("level") == "level0":
        lines.append("> level0 无 CommAnalyzerBandwidth，带宽为裸 db 粗算，仅供参考。\n")
    lines.append("### 5.5 频率（AICORE_FREQ）\n")
    lines.append(_freq_table(ev))
    lines.append("")

    lines.append("## 6. 检测层与辅助分析一致性\n")
    conflicts = [f for f in result.get("findings", []) if f.get("status") == "conflict"]
    if conflicts:
        for f in conflicts:
            lines.append(f"- `{f.get('rule_id')}`：{f.get('summary')}")
    else:
        lines.append("- 未发现一致性冲突。")
    lines.append("")

    lines.append("## 7. 结论与建议\n")
    for a in result.get("actions", []) or []:
        lines.append(f"- {a}")
    lines.append("")

    lines.append("## 8. 置信度与局限\n")
    lines.append("- 结论来源：`" + ("检测结果" if result.get("conclusion_source") == "detection" else "辅助分析") + "`")
    lines.append(f"- 置信度：`{result.get('confidence', '')}`")
    for lim in result.get("limitations", []) or []:
        lines.append(f"- {lim}")
    if not result.get("limitations"):
        lines.append("- 无额外局限。")
    lines.append("")

    lines.append("## 9. 推测与待验证假设（不进正式结论）\n")
    specs = result.get("speculations", []) or []
    if not specs:
        lines.append("_暂无。Agent 可在本区自主下钻，但每条必须标注 `confidence=speculative`，且不得写入正式结论。_")
    else:
        for s in specs:
            lines.append(f"- [推测] {s.get('hypothesis', '')}")
            if s.get("evidence_refs"):
                lines.append(f"  - 证据：`{', '.join(s['evidence_refs'])}`")
            if s.get("verification"):
                lines.append(f"  - 验证：{s['verification']}")
    lines.append("")
    return "\n".join(lines)


def console_evidence(ev: Dict, result: Dict) -> str:
    """把关键证据打印到调用方 console（不打印 337KB 的完整 evidence.json）。"""
    meta = ev.get("meta", {}) or {}
    primary = result.get("primary", {}) or {}
    lines: List[str] = []
    lines.append("## 证据摘要")
    lines.append(f"- 数据级别：`{meta.get('level', '')}`，有效 Rank：{len(meta.get('ranks') or [])}")
    lines.append(f"- 并行域：`{', '.join(sorted((meta.get('parallels') or {}).keys())) or '无'}`")
    lines.append(f"- 会话时长：{_fmt(meta.get('session_ms'))} ms")
    if result.get("conclusion_source") == "detection":
        lines.append("- 结论来源：**检测结果**（结论层仅辅助，不改变检测结论）")
    else:
        lines.append("- 结论来源：**辅助分析**（检测层无异常，置信度已降级）")
    verdict_text = {
        "slow_rank": "少数卡异常",
        "slow_group": "通信组异常",
        "multiple": "多个类别同时异常",
        "no_anomaly": "未发现异常",
        "insufficient_evidence": "证据不足",
    }.get(result.get("verdict"), result.get("verdict", ""))
    conf_text = {"high": "高", "medium": "中", "low": "低"}.get(
        result.get("confidence"), result.get("confidence", ""))
    lines.append(f"- 结论：{verdict_text}（置信度 {conf_text}）")
    lines.append(f"- 主要问题：{primary.get('summary', '')}")
    lines.append(f"- 涉及卡：{primary.get('slow_side') or []}")
    aux = [f for f in result.get("findings", []) if f.get("status") == "auxiliary"]
    if aux:
        lines.append("- 辅助发现（不改变检测结论）：" + "；".join(
            f.get("summary", "") for f in aux))
    lines.append("")
    lines.append("### 各卡耗时拆解")
    lines.append(_macro_table(ev))
    lines.append("### 主机侧等待")
    lines.append(_host_table(ev))
    lines.append("### 空闲窗口")
    lines.append(_free_table(ev))
    lines.append("### 计算耗时")
    lines.append(_compute_table(ev))
    slow = primary.get("slow_side") or []
    by_rank = ev.get("compute", {}).get("by_rank", {}) or {}
    for r in slow:
        ops = (by_rank.get(str(r), {}) or {}).get("ops", {}) or {}
        top = sorted(ops.items(), key=lambda x: -_f(x[1].get("sum_ms", 0.0)))[:8]
        if top:
            lines.append(f"- rank {r} Top 算子：" + "，".join(
                f"{op}={_fmt(v.get('sum_ms'))}ms/{int(_f(v.get('count')))}次"
                for op, v in top))
    lines.append("")
    lines.append("### 通信带宽 Level-1（逐 rank，越小越慢）")
    lines.append(_comm_level1_table(ev))
    lines.append("### 通信带宽 Level-2（按域/组聚合，*** 为慢组）")
    lines.append(_comm_level2_table(ev))
    lines.append("### AI Core 频率")
    lines.append(_freq_table(ev))
    lines.append("### 覆盖/局限")
    for lim in result.get("limitations", []) or []:
        lines.append(f"- {lim}")
    if not result.get("limitations"):
        lines.append("- 无额外局限")
    return "\n".join(lines)


def format_artifacts(items: List[tuple]) -> List[str]:
    """把 [(标签, 路径), ...] 格式化成「标签: 绝对路径」的行。

    不同数据级别的产物目录不同（levelnone/level0/level1 落在数据目录或 job 输出目录，
    cluster 单文件落在 `<db 所在目录>/detection_output`，diagnose.py 由 `out=` 决定），
    统一打印**绝对路径**，避免使用者不知道产物到底写在哪。
    """
    lines = []
    for label, path in items:
        lines.append(f"{label}: {os.path.abspath(path)}" if path else f"{label}: -")
    return lines


def write_outputs(ev: Dict, result: Dict, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "formal_result.json")
    report_path = os.path.join(out_dir, "diagnosis_report.md")
    evidence_path = os.path.join(out_dir, "evidence.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(ev, result))
    return {
        "evidence": evidence_path,
        "result": result_path,
        "report": report_path,
    }
