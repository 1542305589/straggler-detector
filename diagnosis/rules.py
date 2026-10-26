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
"""规则层：只读 evidence.json，输出确定性正式结论。

A–E：覆盖/通信/计算/Host/元规则（由旧分析层迁移）。
F1–F6：由裸 db 证据驱动的根因规则（Host 阻塞、纯计算、负载不均、慢链路、纯等待、频率）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

COMPUTE = 1.3
IO = 2.5
COMM = 1.3

CATEGORY_ORDER = (
    "host", "comm", "compute", "io", "pp_comm", "npu_bubble", "multiple",
)


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _ratio(a: float, b: float) -> float:
    if b > 0:
        return a / b
    return float("inf") if a > 0 else 1.0


def _median(values: List[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _ranks_from_key(key: str) -> List[int]:
    # 先剥掉域名前缀再看数字：域名里可能含数字（如 mc2），直接对整串 findall 会把 2 当成 rank。
    inner = str(key)
    if "[" in inner and inner.endswith("]"):
        inner = inner[inner.index("[") + 1:-1]
    return sorted(int(x) for x in re.findall(r"\d+", inner))


def _detection_map(detection: Dict) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for cat, items in (detection or {}).items():
        if isinstance(items, list):
            out[cat] = [x for x in items if isinstance(x, dict)]
    return out


def _detection_slow(detection: Dict, cat: str) -> List[int]:
    out: List[int] = []
    for item in _detection_map(detection).get(cat, []):
        if item.get("is_abnormal") is False:
            continue
        out.extend(_ranks_from_key(item.get("display_key", "")))
    return sorted(set(out))


def _step_totals(ev: Dict, rank: int) -> Dict:
    by_rank = ev.get("macro", {}).get("step_trace", {}).get("by_rank", {})
    return by_rank.get(str(rank), {}).get("totals", {})


def _rank_list(ev: Dict) -> List[int]:
    ranks = ev.get("meta", {}).get("ranks") or []
    if ranks:
        return sorted(int(r) for r in ranks)
    return sorted(int(r) for r in (ev.get("macro", {}).get("step_trace", {}).get("by_rank", {}) or {}))


def _group_of(ev: Dict, domain: str) -> List[List[int]]:
    return ev.get("meta", {}).get("parallels", {}).get(domain, []) or []


def _domains(ev: Dict) -> List[str]:
    return sorted((ev.get("meta", {}).get("parallels", {}) or {}).keys())


def _op_bandwidth(ev: Dict, rank: int, group: str, op: str) -> float:
    return _f(
        (ev.get("comm", {}).get("by_rank_group", {}).get(str(rank), {})
         .get(group, {}).get(op, {}) or {}).get("bandwidth_gbps", 0.0)
    )


def _op_size_time(ev: Dict, rank: int, group: str, op: str):
    d = (ev.get("comm", {}).get("by_rank_group", {}).get(str(rank), {})
         .get(group, {}).get(op, {}) or {})
    return _f(d.get("size_mb", 0.0)), _f(d.get("transit_ms", 0.0))


def _finding(rule_id: str, category: str, status: str, summary: str,
              slow_side=None, baseline_side=None, waiters=None,
              metric=None, value=None, baseline_value=None, ratio=None,
              confidence="medium", evidence_refs=None, derived=False) -> Dict:
    return {
        "rule_id": rule_id,
        "category": category,
        "status": status,
        "confidence": confidence,
        "slow_side": sorted(slow_side or []),
        "baseline_side": sorted(baseline_side or []),
        "waiters": sorted(waiters or []),
        "metric": metric or "",
        "value": value,
        "baseline_value": baseline_value,
        "ratio": ratio,
        "summary": summary,
        "evidence_refs": evidence_refs or [],
        "derived": bool(derived),
    }


def evaluate(ev: Dict, thresholds: Dict[str, float] = None) -> Dict:
    th = {"compute": COMPUTE, "io": IO, "comm": COMM}
    th.update(thresholds or {})
    detection = ev.get("detection", {}) or {}
    ranks = _rank_list(ev)
    findings: List[Dict] = []
    checked: List[str] = []
    limitations: List[str] = []

    coverage = ev.get("coverage", {}) or {}
    limitations.extend(coverage.get("missing", []) or [])

    # ---------------- A 覆盖/一致性 ----------------
    for rule_id in ("A1", "A2", "A3", "A4"):
        checked.append(rule_id)
    for domain in ("ep", "mc2"):
        groups = _group_of(ev, domain)
        if groups and len(groups) < 2:
            limitations.append(f"{domain} 只有 1 个 Group，不能组间对比（未评估，不能判正常）")
    for domain in _domains(ev):
        groups = _group_of(ev, domain)
        has_bw = any(
            _op_bandwidth(ev, r, _group_key(g), op) > 0
            for r in ranks for g in _group_of(ev, domain)
            for op in _ops_in_group(ev, r, _group_key(g))
        )
        if groups and not has_bw:
            limitations.append(f"{domain} 无带宽数据，带宽未评估（不能判正常）")
    for cat in ("KERNEL_AICORE", "kernel_aivec", "memcpy_async", "comm", "cpu", "npu_bubble", "pp_comm"):
        if cat not in (detection or {}) and cat in ("KERNEL_AICORE", "comm", "cpu", "npu_bubble"):
            limitations.append(f"检测结果缺少 {cat} 类别，对应结论未评估")

    # ---------------- B 通信慢组 ----------------
    checked.append("B1")
    det_comm = _detection_map(detection).get("comm", [])
    if det_comm:
        for item in det_comm:
            if item.get("is_abnormal") is False:
                continue
            key = str(item.get("display_key", ""))
            slow = _ranks_from_key(key)
            m = re.match(r"([A-Za-z0-9_]+)\s*\[", key)
            domain = m.group(1) if m else ""
            groups = _group_of(ev, domain)
            baseline = [r for g in groups for r in g if r not in slow]
            findings.append(_finding(
                "B1", "comm", "fired",
                f"检测层判定 {domain} 通信组 {key} 通信异常，劣化指数 {item.get('metric_value')}（阈值 {th['comm']}）。",
                slow_side=slow, baseline_side=baseline,
                metric="detection.comm.metric_value",
                value=item.get("metric_value"), ratio=item.get("metric_value"),
                confidence="high", evidence_refs=["detection.comm"],
            ))
            break
    else:
        level = ev.get("meta", {}).get("level")
        fired_b2 = False
        for domain in _domains(ev):
            groups = _group_of(ev, domain)
            if len(groups) < 2:
                continue
            for op in _all_ops(ev, groups):
                bws = []
                for g in groups:
                    size = sum(_op_size_time(ev, r, _group_key(g), op)[0] for r in g)
                    time = sum(_op_size_time(ev, r, _group_key(g), op)[1] for r in g)
                    if size > 0 and time > 0:
                        bws.append((_ratio(size, time), g))
                if len(bws) < 2:
                    continue
                lo = min(bws, key=lambda x: x[0])
                hi = max(bws, key=lambda x: x[0])
                ratio = _ratio(hi[0], lo[0])
                if ratio >= th["comm"] and level in ("level1", "cluster"):
                    findings.append(_finding(
                        "B2", "comm", "conflict",
                        f"辅助分析发现 {domain} 的 {op} 带宽组间差约 {ratio:.2f} 倍，但检测层未报，建议复核。",
                        slow_side=sorted(lo[1]), baseline_side=sorted(hi[1]),
                        metric=f"{domain}.{op}.bandwidth_gbps",
                        value=lo[0], baseline_value=hi[0], ratio=ratio,
                        confidence="medium", derived=True,
                        evidence_refs=[f"comm.by_rank_group.{r}.{_group_key(sorted(lo[1]))}.{op}.bandwidth_gbps"
                                       for r in sorted(lo[1])],
                    ))
                    fired_b2 = True
                    break
            if fired_b2:
                break
        if level == "level0":
            limitations.append("level0 无 CommAnalyzerBandwidth，组间带宽未评估（不能判正常）")
    for rule_id in ("B2", "B3", "B4"):
        checked.append(rule_id)

    # ---------------- C 计算/搬运 ----------------
    compute_slow: List[int] = []
    compute_ratio = 0.0
    compute_base = 0.0
    compute_value = 0.0
    if ranks:
        totals = {r: _compute_total(ev, r) for r in ranks}
        med = _median(list(totals.values()))
        if med > 0:
            for r in ranks:
                if totals[r] >= th["compute"] * med:
                    compute_slow.append(r)
            if compute_slow:
                compute_value = max(totals[r] for r in compute_slow)
                compute_ratio = _ratio(compute_value, med)
                compute_base = med
    detection_compute = set(_detection_slow(detection, "KERNEL_AICORE")) | \
        set(_detection_slow(detection, "kernel_aivec")) | \
        set(_detection_slow(detection, "memcpy_async"))
    compute_diff = compute_value - compute_base
    session_ms = _f(ev.get("meta", {}).get("session_ms", 0.0))
    compute_dominant_flag = compute_diff >= max(0.05 * session_ms, 50.0)
    if compute_slow and not compute_dominant_flag:
        limitations.append(
            f"计算组间差仅 {compute_diff:.1f} ms，未超过会话 5%，不作为主因"
        )
        compute_slow = []
    if compute_slow:
        rule = "C1" if detection_compute else "C2"
        status = "fired" if detection_compute else "conflict"
        findings.append(_finding(
            rule, "compute", status,
            f"rank {compute_slow} 的计算总耗时明显高于其他卡（约 {compute_ratio:.2f} 倍）"
            + ("" if detection_compute else "，但检测层未报，建议复核"),
            slow_side=compute_slow, baseline_side=[r for r in ranks if r not in compute_slow],
            metric="compute.total_ms", value=compute_value, baseline_value=compute_base,
            ratio=compute_ratio, confidence="high" if detection_compute else "medium",
            derived=True,
            evidence_refs=[f"compute.by_rank.{r}.ops" for r in compute_slow],
        ))
    # F2 / F3：区分算子劣化 vs 负载不均
    if compute_slow:
        degraded, imbalanced = _classify_compute(ev, ranks, compute_slow, th["compute"])
        if degraded:
            findings.append(_finding(
                "F2", "compute", "fired",
                f"rank {compute_slow} 的同类算子调用次数与其他卡相同，但每次平均耗时明显更高"
                f"（约 {degraded.get('ratio'):.1f} 倍），可能是算子或硬件劣化。",
                slow_side=compute_slow,
                baseline_side=[r for r in ranks if r not in compute_slow],
                metric="compute.op.avg_ms", value=degraded.get("avg_ms"),
                baseline_value=degraded.get("base_avg_ms"), ratio=degraded.get("ratio"),
                confidence="high", derived=True,
                evidence_refs=[f"compute.by_rank.{r}.ops.{degraded['op']}.avg_ms" for r in compute_slow],
            ))
        if imbalanced:
            findings.append(_finding(
                "F3", "compute", "fired",
                f"rank {compute_slow} 的同类算子调用次数与其他卡不同"
                f"（约 {imbalanced.get('ratio'):.1f} 倍），可能是负载切分不均。",
                slow_side=compute_slow,
                baseline_side=[r for r in ranks if r not in compute_slow],
                metric="compute.op.count", value=imbalanced.get("slow_count"),
                baseline_value=imbalanced.get("base_count"), ratio=imbalanced.get("ratio"),
                confidence="medium", derived=True,
                evidence_refs=[f"compute.by_rank.{r}.ops.{imbalanced['op']}.count" for r in compute_slow],
            ))
    for rule_id in ("C1", "C2", "C3", "C4"):
        checked.append(rule_id)
    # C3：单卡 vs 组间
    if compute_slow:
        groups = _group_of(ev, "tp") or _group_of(ev, "ep") or _group_of(ev, "dp")
        if groups:
            for g in groups:
                if set(compute_slow).issubset(set(g)) and len(compute_slow) < len(g):
                    findings.append(_finding(
                        "C3", "compute", "fired",
                        f"rank {compute_slow} 只是组内少数卡异常，倾向单卡故障，而不是整组问题。",
                        slow_side=compute_slow, baseline_side=[r for r in g if r not in compute_slow],
                        confidence="high",
                    ))
                    break
    # C4：搬运与计算同向
    if _detection_slow(detection, "memcpy_async") and (
        _detection_slow(detection, "KERNEL_AICORE") or _detection_slow(detection, "kernel_aivec")
    ):
        findings.append(_finding(
            "C4", "compute", "fired",
            "内存搬运与计算类同时异常，可能是整芯片内存带宽或跨 NUMA 问题，而不是单一算子。",
            slow_side=sorted(set(_detection_slow(detection, "memcpy_async")) |
                             set(_detection_slow(detection, "KERNEL_AICORE")) |
                             set(_detection_slow(detection, "kernel_aivec"))),
            confidence="medium",
        ))

    # ---------------- D Host/气泡/PP ----------------
    cpu_slow = _detection_slow(detection, "cpu")
    if cpu_slow:
        findings.append(_finding(
            "D1", "host", "fired",
            f"ZP_Host 异常卡 {cpu_slow}，Host/下发慢（报告节点级，以 JSON rank 为准）",
            slow_side=cpu_slow, confidence="high",
            evidence_refs=[f"host.free_windows.{r}.total_ms" for r in cpu_slow],
        ))
    bubble_slow = _detection_slow(detection, "npu_bubble")
    if bubble_slow:
        findings.append(_finding(
            "D2", "npu_bubble", "fired",
            f"ZP_Bubble 异常卡 {bubble_slow}，疑似调度/启动间隔问题",
            slow_side=bubble_slow, confidence="high",
        ))
    pp_slow = _detection_slow(detection, "pp_comm")
    if pp_slow:
        findings.append(_finding(
            "D3", "pp_comm", "fired",
            f"PP 流水线等待（{pp_slow}）",
            slow_side=pp_slow, confidence="medium",
        ))
    node_ranks = _node_ranks(ev)
    if len(node_ranks) < 2:
        limitations.append("只有 1 个节点，ZP_Host 不能做节点间对比（只能节点级描述）")
    for rule_id in ("D1", "D2", "D3", "D4"):
        checked.append(rule_id)

    # ---------------- E 元规则 ----------------
    fired_cats = sorted({f["category"] for f in findings if f["status"] == "fired"})
    if len(fired_cats) >= 2:
        findings.append(_finding(
            "E2", "multiple", "fired",
            f"多个类别同向异常：{fired_cats}，考虑多故障叠加/共同根因",
            confidence="medium",
        ))
    for f in findings:
        if f["status"] == "conflict":
            findings.append(_finding(
                "E3", f["category"], "conflict",
                f"{f['rule_id']} 与检测层不一致，标「一致性异常，建议复核」",
                slow_side=f["slow_side"], confidence="low",
            ))
            break
    for rule_id in ("E1", "E2", "E3", "E4"):
        checked.append(rule_id)

    # ---------------- F 新证据族 ----------------
    f1 = _f1_host_block(ev, ranks, th)
    if f1:
        findings.append(f1)
    f5 = _f5_waiter(ev, ranks, th)
    if f5:
        findings.append(f5)
    f4 = _f4_slow_link(ev, ranks, th, compute_dominant_flag)
    if f4:
        findings.append(f4)
    f6 = _f6_freq(ev, ranks, th)
    if f6:
        findings.append(f6)
    for rule_id in ("F1", "F2", "F3", "F4", "F5", "F6"):
        checked.append(rule_id)

    # ---------------- 汇总 ----------------
    # 检测结果（权威）优先；结论层只做辅助，不改变检测结论。
    det_findings = _detection_findings(detection)
    det_primary = _detection_primary(detection, det_findings, ev)
    analysis_fired = [f for f in findings if f["status"] in ("fired", "conflict")]

    if det_primary:
        conclusion_source = "detection"
        primary = det_primary
        verdict = _detection_verdict(detection, det_findings, ev)
        confidence = "high"
        skip_dup = {"B1", "E2"}
        findings = [f for f in findings if f["rule_id"] not in skip_dup]
        for f in findings:
            if f["status"] in ("fired", "conflict"):
                f["status"] = "auxiliary"
        secondary = findings
        result_findings = det_findings + findings
        rules_fired = [det_primary["rule_id"]] + [f["rule_id"] for f in analysis_fired]
    else:
        conclusion_source = "analysis_fallback"
        if not analysis_fired and limitations:
            findings.append(_finding(
                "E4", "insufficient_evidence", "not_evaluated",
                "关键字段缺失/单组域/无带宽数据，证据不足",
                confidence="low",
            ))
        if not analysis_fired:
            verdict = "no_anomaly" if not limitations else "insufficient_evidence"
            primary = {
                "category": "insufficient_evidence" if verdict == "insufficient_evidence" else "none",
                "rule_id": "E4" if verdict == "insufficient_evidence" else "",
                "slow_side": [],
                "baseline_side": [],
                "summary": "未命中任何正式规则" if verdict == "no_anomaly" else "证据不足，无法下结论",
            }
            confidence = "low" if verdict == "insufficient_evidence" else "medium"
        else:
            priority = {
                "F1": 0, "F4": 1, "F2": 2, "F3": 3, "C1": 4, "C2": 5,
                "D1": 6, "D2": 7, "D3": 8, "B1": 9, "B2": 10, "F6": 11, "F5": 12,
                "C3": 13, "C4": 14, "D4": 15, "E2": 16, "E3": 17,
            }
            analysis_fired.sort(key=lambda f: priority.get(f["rule_id"], 99))
            top = analysis_fired[0]
            root_rules = {"F1", "F4", "F2", "F3", "C1", "D1", "D2", "D3", "F6"}
            cats = {
                f["category"] for f in analysis_fired
                if f["rule_id"] in root_rules and f["category"] not in ("multiple", "insufficient_evidence")
            }
            b1 = next((f for f in analysis_fired if f["rule_id"] == "B1"), None)
            f1 = next((f for f in analysis_fired if f["rule_id"] == "F1"), None)
            if b1 and f1:
                comm_max = max(
                    (_f(_step_totals(ev, r).get("communication_ms", 0.0))
                     for r in (b1.get("slow_side") or [])), default=0.0)
                free_max = max(
                    (_f(_step_totals(ev, r).get("free_ms", 0.0))
                     for r in (f1.get("slow_side") or [])), default=0.0)
                if comm_max > free_max:
                    top = b1
                    cats = {"comm"}
                else:
                    top = f1
                    cats = {"host"}

            verdict = "multiple" if len(cats) >= 2 else _verdict_from_finding(top, ev)
            primary = {
                "category": top["category"] if len(cats) < 2 else "multiple",
                "rule_id": top["rule_id"],
                "slow_side": top["slow_side"],
                "baseline_side": top["baseline_side"],
                "degradation": top["ratio"],
                "summary": "检测层无异常；辅助分析：" + top["summary"],
            }
            confidence = top["confidence"]
            if confidence == "high":
                confidence = "medium"
        secondary = analysis_fired[1:] if analysis_fired else []
        result_findings = findings
        rules_fired = [f["rule_id"] for f in analysis_fired]

    return {
        "schema_version": "1.0",
        "verdict": verdict,
        "confidence": confidence,
        "primary": primary,
        "conclusion_source": conclusion_source,
        "secondary_findings": secondary,
        "findings": result_findings,
        "rules_fired": rules_fired,
        "rules_checked": sorted(set(checked) | {"DET"}),
        "limitations": sorted(set(limitations)),
        "actions": _actions(primary, analysis_fired),
    }


def _category_of_detection(cat: str) -> str:
    return {
        "comm": "comm", "KERNEL_AICORE": "compute", "kernel_aivec": "compute",
        "memcpy_async": "io", "cpu": "host", "npu_bubble": "npu_bubble",
        "pp_comm": "pp_comm",
    }.get(cat, "unknown")


def _detection_findings(detection: Dict) -> List[Dict]:
    out: List[Dict] = []
    for cat in ("comm", "KERNEL_AICORE", "kernel_aivec", "memcpy_async",
                 "cpu", "npu_bubble", "pp_comm"):
        for item in _detection_map(detection).get(cat, []):
            if item.get("is_abnormal") is False:
                continue
            slow = _ranks_from_key(item.get("display_key", ""))
            label = {
                "comm": f"通信组 {item.get('display_key')}",
                "KERNEL_AICORE": f"rank {item.get('display_key')} 计算",
                "kernel_aivec": f"rank {item.get('display_key')} 计算",
                "memcpy_async": f"rank {item.get('display_key')} 内存搬运",
                "cpu": f"rank {item.get('display_key')} 主机侧",
                "npu_bubble": f"rank {item.get('display_key')} NPU 气泡",
                "pp_comm": f"PP 通信组 {item.get('display_key')}",
            }.get(cat, f"{cat} {item.get('display_key')}")
            out.append(_finding(
                "DET_" + cat.upper(), _category_of_detection(cat), "authoritative",
                f"检测层判定 {label}异常，劣化指数 {_f(item.get('metric_value')):.2f}",
                slow_side=slow, metric=f"detection.{cat}",
                value=item.get("metric_value"), confidence="high",
                evidence_refs=[f"detection.{cat}"], derived=True,
            ))
    return out


def _detection_primary(detection: Dict, det_findings: List[Dict], ev: Dict) -> Optional[Dict]:
    if not det_findings:
        return None
    cats = {f["category"] for f in det_findings}
    if len(cats) > 1:
        groups: Dict[str, List[Dict]] = {}
        for f in det_findings:
            groups.setdefault(f["category"], []).append(f)
        label_map = {
            "compute": "计算", "io": "内存搬运", "comm": "通信",
            "host": "主机侧", "npu_bubble": "NPU 气泡", "pp_comm": "PP 通信",
        }
        parts = []
        for cat, fs in groups.items():
            ranks = sorted({r for f in fs for r in f["slow_side"]})
            vals = [f["value"] for f in fs if f["value"] is not None]
            label = label_map.get(cat, cat)
            if vals:
                lo, hi = min(vals), max(vals)
                span = (f"{hi:.1f} 倍" if round(hi, 1) == round(lo, 1)
                        else f"{lo:.1f}–{hi:.1f} 倍")
                parts.append(f"rank {ranks} 的{label}异常（劣化 {span}）")
            else:
                parts.append(f"rank {ranks} 的{label}异常")
        return {
            "category": "multiple",
            "rule_id": "DET_MULTIPLE",
            "slow_side": sorted({r for f in det_findings for r in f["slow_side"]}),
            "baseline_side": [],
            "degradation": None,
            "summary": "检测层判定：" + "；".join(parts) + "。",
        }
    f = det_findings[0]
    return {
        "category": f["category"],
        "rule_id": f["rule_id"],
        "slow_side": f["slow_side"],
        "baseline_side": _baseline_for(detection, ev, f["category"], f["slow_side"]),
        "degradation": f["value"],
        "summary": f["summary"],
    }


def _baseline_for(detection: Dict, ev: Dict, category: str,
                   slow: List[int]) -> List[int]:
    ranks = _rank_list(ev)
    if category == "comm":
        for domain in _domains(ev):
            groups = _group_of(ev, domain)
            for g in groups:
                if sorted(g) == sorted(slow):
                    return sorted({r for gg in groups if gg is not g for r in gg})
    if category in ("compute", "io", "host"):
        return [r for r in ranks if r not in slow]
    return []


def _detection_verdict(detection: Dict, det_findings: List[Dict], ev: Dict) -> str:
    if not det_findings:
        return "no_anomaly"
    if len({f["category"] for f in det_findings}) > 1:
        return "multiple"
    f = det_findings[0]
    if f["category"] == "comm":
        for domain in _domains(ev):
            for g in _group_of(ev, domain):
                if sorted(g) == sorted(f["slow_side"]):
                    return "slow_group"
        return "slow_rank"
    return "slow_rank"


def _verdict_from_finding(f: Dict, ev: Dict) -> str:
    slow = f.get("slow_side") or []
    if f.get("category") == "host" and slow:
        return "slow_rank"
    if f.get("category") == "comm":
        for domain in _domains(ev):
            for g in _group_of(ev, domain):
                if sorted(g) == sorted(slow):
                    return "slow_group"
        return "slow_rank"
    return "slow_rank"


def _actions(primary: Dict, fired: List[Dict]) -> List[str]:
    cat = primary.get("category")
    slow = primary.get("slow_side") or []
    if cat == "host":
        return [
            f"用 py-spy dump / gdb 抓取 {slow} 的线程栈，确认阻塞在 Gloo/TCPStore/Python 锁",
            "检查 CPU 绑核、GLOO_SOCKET_IFNAME、TCP 重传/backlog",
        ]
    if cat == "comm":
        return [
            f"检查 {slow} 的通信链路、带宽、包大小与通信域配置",
            "用通信矩阵确认慢链路（src/dst/transport）",
        ]
    if cat == "compute":
        return [
            f"对比 {slow} 的逐算子 count 与 avg，区分负载不均 vs 算子/硬件劣化",
            "检查动态 shape、融合缺失、AICore 利用率",
        ]
    if cat == "io":
        return ["检查 MEMCPY_ASYNC 包大小、SDMA 对齐与内存带宽"]
    if cat == "npu_bubble":
        return ["检查算子启动间隔、调度与 stream 同步"]
    if cat == "pp_comm":
        return ["检查 PP 相邻 stage 的点对点传输与等待"]
    return ["补充 count / 频率 / Host API 数据后再判定"]


def _ops_in_group(ev: Dict, rank: int, group: str) -> List[str]:
    d = ev.get("comm", {}).get("by_rank_group", {}).get(str(rank), {}).get(group, {}) or {}
    return list(d.keys())


def _all_ops(ev: Dict, groups: List[List[int]]) -> List[str]:
    ops = set()
    for g in groups:
        for r in g:
            for op in _ops_in_group(ev, r, _group_key(g)):
                ops.add(op)
    return sorted(ops)


def _group_key(ranks: List[int]) -> str:
    return ",".join(str(r) for r in sorted(ranks))


def _compute_total(ev: Dict, rank: int) -> float:
    ops = (ev.get("compute", {}).get("by_rank", {}).get(str(rank), {}) or {}).get("ops", {}) or {}
    return sum(_f(v.get("sum_ms", 0.0)) for v in ops.values())


def _compute_ops(ev: Dict, rank: int) -> Dict[str, Dict]:
    return (ev.get("compute", {}).get("by_rank", {}).get(str(rank), {}) or {}).get("ops", {}) or {}


def _classify_compute(ev: Dict, ranks: List[int], slow: List[int], threshold: float):
    peers = [r for r in ranks if r not in slow]
    if not peers:
        return None, None
    degraded = None
    imbalanced = None
    for op in _compute_ops(ev, slow[0]).keys():
        slow_cnt = _median([_f(_compute_ops(ev, r).get(op, {}).get("count", 0)) for r in slow])
        base_cnt = _median([_f(_compute_ops(ev, r).get(op, {}).get("count", 0)) for r in peers])
        slow_avg = _median([_f(_compute_ops(ev, r).get(op, {}).get("avg_ms", 0)) for r in slow])
        base_avg = _median([_f(_compute_ops(ev, r).get(op, {}).get("avg_ms", 0)) for r in peers])
        if base_avg <= 0:
            continue
        cnt_ratio = _ratio(slow_cnt, base_cnt) if base_cnt > 0 else 1.0
        avg_ratio = _ratio(slow_avg, base_avg)
        if 0.95 <= cnt_ratio <= 1.05 and avg_ratio >= threshold:
            degraded = {
                "op": op, "avg_ms": slow_avg, "base_avg_ms": base_avg,
                "ratio": avg_ratio,
            }
        elif (cnt_ratio >= 1.05 or cnt_ratio <= 0.95) and avg_ratio >= 1.0:
            imbalanced = {
                "op": op, "slow_count": slow_cnt, "base_count": base_cnt,
                "ratio": cnt_ratio,
            }
    return degraded, imbalanced


def _node_ranks(ev: Dict) -> Dict[str, List[int]]:
    nodes: Dict[str, List[int]] = {}
    for rank, info in (ev.get("meta", {}).get("rank_device", {}) or {}).items():
        host = str(info.get("host_uid") or info.get("host_name") or "unknown")
        nodes.setdefault(host, []).append(_i(rank))
    return {k: sorted(v) for k, v in nodes.items()}


def _f1_host_block(ev: Dict, ranks: List[int], th: Dict) -> Optional[Dict]:
    session = _f(ev.get("meta", {}).get("session_ms", 0.0))
    if session <= 0:
        session = max((_f(_step_totals(ev, r).get("stage_ms", 0.0)) for r in ranks), default=0.0)
    if session <= 0:
        return None
    gloo = ev.get("host", {}).get("gloo", {}) or {}
    sync_dev = ev.get("host", {}).get("sync_device", {}) or {}
    sync_ev = ev.get("host", {}).get("sync_event", {}) or {}
    sync_st = ev.get("host", {}).get("sync_stream", {}) or {}
    details = {}
    for r in ranks:
        tot = _step_totals(ev, r)
        free = _f(tot.get("free_ms", 0.0))
        stage = _f(tot.get("stage_ms", 0.0)) or session
        comp = _f(tot.get("computing_ms", 0.0))
        comm = _f(tot.get("communication_ms", 0.0))
        g = _f((gloo.get(str(r)) or {}).get("max_ms", 0.0))
        sd = _f((sync_dev.get(str(r)) or {}).get("max_ms", 0.0))
        se = _f((sync_ev.get(str(r)) or {}).get("max_ms", 0.0))
        ss = _f((sync_st.get(str(r)) or {}).get("max_ms", 0.0))
        details[r] = {
            "block_ms": g,
            "gloo_ms": g, "sync_device_ms": sd, "sync_event_ms": se,
            "sync_stream_ms": ss, "free_ms": free,
            "free_ratio": _ratio(free, stage),
            "compute_ratio": _ratio(comp, stage),
            "comm_ratio": _ratio(comm, stage),
        }
    max_block = max((d["block_ms"] for d in details.values()), default=0.0)
    floor = max(0.5 * max_block, 0.05 * session, 1000.0)
    blockers = [
        r for r in ranks
        if details[r]["block_ms"] >= floor
        and details[r]["free_ratio"] >= 0.1
        and details[r]["compute_ratio"] <= 0.1
    ]
    if not blockers:
        return None
    waiters = []
    for r in ranks:
        if r in blockers:
            continue
        tot = _step_totals(ev, r)
        stage = _f(tot.get("stage_ms", 0.0)) or session
        comm = _f(tot.get("communication_ms", 0.0))
        free = _f(tot.get("free_ms", 0.0))
        comm_ratio = _ratio(comm, stage)
        free_ratio = _ratio(free, stage)
        if comm_ratio >= 0.1 and free_ratio < 0.1:
            waiters.append(r)
    refs = []
    for r in blockers:
        refs.append(f"host.gloo.{r}.max_ms")
        refs.append(f"macro.step_trace.by_rank.{r}.totals.free_ms")
    max_gloo = max(details[r]["gloo_ms"] for r in blockers)
    free_max = max(details[r]["free_ms"] for r in blockers)
    comm_max = max((_f(_step_totals(ev, r).get("communication_ms", 0.0))
                    for r in waiters), default=0.0)
    return _finding(
        "F1", "host", "fired",
        f"rank {blockers} 的主机侧集合通信（Gloo）同步等待约 {max_gloo:.0f} ms，NPU 空闲约 {free_max:.0f} ms；"
        f"同时其他卡通信耗时约 {comm_max:.0f} ms，比空闲更长，说明 {blockers} 是在等通信完成，不是它们本身慢。",
        slow_side=blockers, baseline_side=waiters,
        waiters=waiters,
        metric="host.gloo.max_ms", value=max_gloo,
        confidence="high", evidence_refs=refs,
    )


def _f5_waiter(ev: Dict, ranks: List[int], th: Dict) -> Optional[Dict]:
    session = _f(ev.get("meta", {}).get("session_ms", 0.0))
    if session <= 0:
        session = max((_f(_step_totals(ev, r).get("stage_ms", 0.0)) for r in ranks), default=0.0)
    if session <= 0:
        return None
    waiters = []
    for r in ranks:
        tot = _step_totals(ev, r)
        stage = _f(tot.get("stage_ms", 0.0)) or session
        comm = _f(tot.get("communication_ms", 0.0))
        free = _f(tot.get("free_ms", 0.0))
        comm_ratio = _ratio(comm, stage)
        free_ratio = _ratio(free, stage)
        if comm_ratio >= 0.1 and free_ratio < 0.1:
            ops = (ev.get("macro", {}).get("comm_time", {}).get("by_rank", {})
                   .get(str(r), {}).get("ops", []) or [])
            if not ops:
                continue
            op = max(ops, key=lambda o: _f(o.get("elapse_ms", 0.0)))
            elapse = _f(op.get("elapse_ms", 0.0))
            if elapse <= 0:
                continue
            if (_f(op.get("transit_ms", 0.0)) <= 0.05 * elapse
                    and _f(op.get("idle_ms", 0.0)) >= 0.9 * elapse):
                waiters.append(r)
    if not waiters:
        return None
    elapse = 0.0
    transit = 0.0
    idle = 0.0
    for r in waiters:
        ops = (ev.get("macro", {}).get("comm_time", {}).get("by_rank", {})
               .get(str(r), {}).get("ops", []) or [])
        if not ops:
            continue
        op = max(ops, key=lambda o: _f(o.get("elapse_ms", 0.0)))
        elapse += _f(op.get("elapse_ms", 0.0))
        transit += _f(op.get("transit_ms", 0.0))
        idle += _f(op.get("idle_ms", 0.0))
    return _finding(
        "F5", "comm", "fired",
        f"rank {waiters} 的通信时间几乎全是等待（实际传输约 {transit:.0f} ms，空闲约 {idle:.0f} ms，"
        f"总耗时约 {elapse:.0f} ms），它们是在等慢卡，不是慢卡本身。",
        slow_side=[], baseline_side=waiters, waiters=waiters,
        metric="comm.transit/idle", value=0.0,
        confidence="high",
        evidence_refs=[f"macro.comm_time.by_rank.{r}.ops" for r in waiters],
    )


def _f4_slow_link(ev: Dict, ranks: List[int], th: Dict,
                   compute_dominant: bool = False) -> Optional[Dict]:
    if ev.get("meta", {}).get("level") == "level0":
        return None
    if compute_dominant:
        return None
    session = _f(ev.get("meta", {}).get("session_ms", 0.0))
    if session <= 0:
        session = max((_f(_step_totals(ev, r).get("stage_ms", 0.0)) for r in ranks), default=0.0)
    floor = max(0.01 * session, 1.0)

    # 1) transit 占比高（level1/cluster 才有 transit 分解）
    for r in ranks:
        ops = (ev.get("macro", {}).get("comm_time", {}).get("by_rank", {})
               .get(str(r), {}).get("ops", []) or [])
        for op in ops:
            elapse = _f(op.get("elapse_ms", 0.0))
            transit = _f(op.get("transit_ms", 0.0))
            if elapse <= 0 or transit < floor:
                continue
            ratio = _ratio(transit, elapse)
            if ratio >= 0.5:
                return _finding(
                    "F4", "comm", "fired",
                    f"rank {r} 的 {op.get('hccl_op_name')} transit/elapse={ratio:.2f}，是真实传输慢",
                    slow_side=[r], metric="comm.transit_ratio", value=ratio,
                    confidence="medium", derived=True,
                    evidence_refs=[f"macro.comm_time.by_rank.{r}.ops"],
                )

    # 2) 组间带宽差（仅 level1/cluster 的带宽口径可信）
    for domain in _domains(ev):
        groups = _group_of(ev, domain)
        if len(groups) < 2:
            continue
        for op in _all_ops(ev, groups):
            bws = []
            for g in groups:
                size = sum(_op_size_time(ev, r, _group_key(g), op)[0] for r in g)
                time = sum(_op_size_time(ev, r, _group_key(g), op)[1] for r in g)
                if size > 0 and time >= floor:
                    bws.append((_ratio(size, time), g, time))
            if len(bws) < 2:
                continue
            lo = min(bws, key=lambda x: x[0])
            hi = max(bws, key=lambda x: x[0])
            ratio = _ratio(hi[0], lo[0])
            if ratio >= th["comm"]:
                return _finding(
                    "F4", "comm", "fired",
                    f"{domain} 的 {op} 带宽组间差 {ratio:.2f}x（慢侧 {_group_key(sorted(lo[1]))}）",
                    slow_side=sorted(lo[1]), baseline_side=sorted(hi[1]),
                    metric=f"comm.{domain}.{op}.bandwidth_gbps",
                    value=lo[0], baseline_value=hi[0], ratio=ratio,
                    confidence="high", derived=True,
                    evidence_refs=[f"comm.by_rank_group.{r}.{_group_key(sorted(lo[1]))}.{op}.bandwidth_gbps"
                                   for r in sorted(lo[1])],
                )

    # 3) Matrix 慢链路（同一 transport / 同一 opType 内比较）
    for group, pairs in (ev.get("comm", {}).get("matrix", {}) or {}).items():
        for p in pairs:
            if _f(p.get("max_transit_ms", 0.0)) < floor:
                continue
            lo = _f(p.get("min_bandwidth_gbps", 0.0))
            hi = _f(p.get("max_bandwidth_gbps", 0.0))
            if lo <= 0 or hi <= 0:
                continue
            ratio = _ratio(hi, lo)
            if ratio >= th["comm"]:
                pair = p.get("slowest_pair") or [0, 0]
                return _finding(
                    "F4", "comm", "fired",
                    f"{group} 的 {p.get('hccl_op_name')}({p.get('transport_type')}) "
                    f"存在慢链路 {pair[0]}->{pair[1]}（{ratio:.2f}x）",
                    slow_side=[_i(pair[0])], baseline_side=[_i(pair[1])],
                    metric="comm.matrix.bandwidth_gbps", value=lo, baseline_value=hi,
                    ratio=ratio, confidence="medium", derived=True,
                    evidence_refs=[f"comm.matrix.{group}"],
                )
    return None


def _f6_freq(ev: Dict, ranks: List[int], th: Dict) -> Optional[Dict]:
    freq = ev.get("freq", {}).get("by_rank", {}) or {}
    vals = {}
    for r in ranks:
        d = freq.get(str(r), {}) or {}
        if d:
            vals[r] = _f(d.get("avg_mhz", 0.0))
    if len(vals) < 2:
        return None
    med = _median(list(vals.values()))
    if med <= 0:
        return None
    low = [r for r, v in vals.items() if v <= 0.9 * med]
    if not low:
        return None
    return _finding(
        "F6", "compute", "fired",
        f"频率异常：{low} 平均频率显著低于同组（{min(vals[r] for r in low):.0f} vs {med:.0f} MHz）",
        slow_side=sorted(low), baseline_side=[r for r in ranks if r not in low],
        metric="freq.avg_mhz", value=min(vals[r] for r in low), baseline_value=med,
        ratio=_ratio(med, min(vals[r] for r in low)),
        confidence="medium", derived=True,
        evidence_refs=[f"freq.by_rank.{r}.avg_mhz" for r in low],
    )
