"""证据层：把数据目录归一化成 evidence.json。

取数策略：analysis.db 优先，缺表/空表回退裸 profiler db；cluster db 单文件单独处理。
所有时间统一为 ms，带宽统一为 GB/s，容量统一为 MB。
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from . import db

# ---- 单位换算 ----
_NS_PER_MS = 1_000_000.0
_US_PER_MS = 1_000.0


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


def _ns_to_ms(v) -> float:
    return _f(v) / _NS_PER_MS


def _us_to_ms(v) -> float:
    return _f(v) / _US_PER_MS


def _median(values: List[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _ratio(a: float, b: float) -> float:
    if b > 0:
        return a / b
    return float("inf") if a > 0 else 1.0


def _op_type(name: str) -> str:
    """hcom_allReduce__503_0_1 → allreduce；非 hccl 记录返回原串。"""
    s = (name or "").strip()
    if "_" in s:
        return s.split("_")[1].lower()
    return s


def _parse_parallels(raw: Optional[str]) -> Dict[str, List[List[int]]]:
    """解析 META_DATA.parallel_group_info → {域: [[ranks], ...]}。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    out: Dict[str, List[List[int]]] = {}
    for info in data.values():
        if not isinstance(info, dict):
            continue
        pg = str(info.get("group_name") or "")
        ranks = info.get("global_ranks") or []
        if pg in ("pp", "embd", "intra_dist_opt_instance", "default_group"):
            continue
        try:
            rs = sorted(int(r) for r in ranks)
        except (TypeError, ValueError):
            continue
        if len(rs) >= 2:
            out.setdefault(pg, []).append(rs)
    return out


def _group_key(ranks: List[int]) -> str:
    return ",".join(str(r) for r in sorted(ranks))


def _parse_group_ranks(raw: Optional[str]) -> Dict[str, List[int]]:
    """解析 META_DATA.parallel_group_info → {group_name: [global_ranks]}。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    out: Dict[str, List[int]] = {}
    for gname, info in data.items():
        if not isinstance(info, dict):
            continue
        ranks = info.get("global_ranks") or []
        try:
            rs = sorted(int(r) for r in ranks)
        except (TypeError, ValueError):
            continue
        if len(rs) >= 2:
            out[str(gname)] = rs
    return out


def _op_group_map(profiler_db: str) -> Dict[str, str]:
    """opName -> groupName（用于把 analysis.db 的 hash group 映射回 group_name）。

    优先 COMMUNICATION_TASK_INFO（覆盖最全），再用 COMMUNICATION_OP 补。
    """
    out: Dict[str, str] = {}
    try:
        con = db.open_ro(profiler_db)
        sid = db.string_ids(con)
        for op, group in db.rows(con, "SELECT name, groupName FROM COMMUNICATION_TASK_INFO"):
            out[sid.get(op, str(op))] = sid.get(group, str(group))
        for op, group in db.rows(con, "SELECT opName, groupName FROM COMMUNICATION_OP"):
            out.setdefault(sid.get(op, str(op)), sid.get(group, str(group)))
        con.close()
    except Exception:
        pass
    return out


def _aggregate_calls(calls: List[Dict], keep_calls: bool = False) -> Dict:
    durs = [c["duration_ms"] for c in calls]
    out = {
        "count": len(calls),
        "sum_ms": sum(durs),
        "max_ms": max(durs) if durs else 0.0,
    }
    if keep_calls:
        out["calls"] = calls
    return out


def _detect_level(profiler_dbs: List[tuple]) -> str:
    for _rank, profiler_db in profiler_dbs:
        analysis_db = db.analysis_db_for(profiler_db)
        if analysis_db and _analysis_has_bandwidth(analysis_db):
            return "level1"
    for _rank, profiler_db in profiler_dbs:
        if _has_kernel_tasks(profiler_db):
            return "level0"
    return "levelnone"


def _analysis_has_bandwidth(path: str) -> bool:
    try:
        con = db.open_ro(path)
        ok = db.has_rows(con, "CommAnalyzerBandwidth")
        con.close()
        return ok
    except Exception:
        return False


def _has_kernel_tasks(path: str) -> bool:
    try:
        con = db.open_ro(path)
        row = db.one(
            con,
            "SELECT 1 FROM TASK t JOIN STRING_IDS s ON t.taskType=s.id "
            "WHERE s.value IN ('KERNEL_AICORE','KERNEL_AIVEC','MEMCPY_ASYNC') LIMIT 1",
        )
        con.close()
        return row is not None
    except Exception:
        return False


def _job_type(profiler_db: str, sid: Dict[int, str]) -> str:
    try:
        con = db.open_ro(profiler_db)
        for (name,) in db.rows(con, "SELECT name FROM PYTORCH_API"):
            n = (sid.get(name) or "").lower()
            if ".step" in n or "zero_grad" in n or "optimizer" in n:
                con.close()
                return "training"
        con.close()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# 每 rank：analysis.db 优先
# ---------------------------------------------------------------------------
def _step_trace_analysis(analysis_db: str) -> Dict:
    con = db.open_ro(analysis_db)
    rows = db.rows(
        con,
        "SELECT step, computing, communication, free, stage, bubble, preparing FROM StepTraceTime",
    )
    con.close()
    steps = []
    for step, comp, comm, free, stage, bubble, prep in rows:
        steps.append({
            "step": str(step),
            "computing_ms": _us_to_ms(comp),
            "communication_ms": _us_to_ms(comm),
            "free_ms": _us_to_ms(free),
            "stage_ms": _us_to_ms(stage),
            "bubble_ms": _us_to_ms(bubble),
            "preparing_ms": _us_to_ms(prep),
        })
    keys = ("computing_ms", "communication_ms", "free_ms", "stage_ms", "bubble_ms", "preparing_ms")
    totals = {k: steps[-1][k] for k in keys} if steps else {k: 0.0 for k in keys}
    return {"steps": steps, "totals": totals, "session_ms": max(
        (s["stage_ms"] for s in steps), default=0.0)}


def _step_trace_raw(profiler_db: str) -> Dict:
    """回退：用 TASK 间隙 + COMPUTE_TASK_INFO + COMMUNICATION_OP 估算。"""
    con = db.open_ro(profiler_db)
    sid = db.string_ids(con)
    session = db.one(con, "SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO")
    session_ms = 0.0
    if session:
        session_ms = _ns_to_ms(_i(session[1]) - _i(session[0]))

    task_rows = db.rows(con, "SELECT startNs, endNs FROM TASK ORDER BY startNs")
    intervals = []
    for s, e in task_rows:
        s_ms, e_ms = _ns_to_ms(s), _ns_to_ms(e)
        if e_ms > s_ms:
            intervals.append((s_ms, e_ms))
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = []
    for i in range(1, len(merged)):
        g = merged[i][0] - merged[i - 1][1]
        if g > 0:
            gaps.append((merged[i - 1][1], merged[i][0], g))
    free_ms = sum(g[2] for g in gaps)

    comp_ms = 0.0
    for _n, _cnt, ns in db.rows(
        con,
        "SELECT c.name, COUNT(*), SUM(t.endNs-t.startNs) "
        "FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
        "GROUP BY c.name",
    ):
        comp_ms += _ns_to_ms(ns)

    comm_ms = 0.0
    for _g, _o, _cnt, ns in db.rows(
        con,
        "SELECT o.groupName, o.opName, COUNT(*), SUM(o.endNs-o.startNs) "
        "FROM COMMUNICATION_OP o GROUP BY o.groupName, o.opName",
    ):
        comm_ms += _ns_to_ms(ns)

    span_ms = (merged[-1][1] - merged[0][0]) if merged else session_ms
    con.close()
    return {
        "steps": [],
        "totals": {
            "computing_ms": comp_ms,
            "communication_ms": comm_ms,
            "free_ms": free_ms,
            "stage_ms": span_ms or session_ms,
            "bubble_ms": 0.0,
            "preparing_ms": 0.0,
        },
        "free_windows": [
            {"start_ms": s, "end_ms": e, "duration_ms": d}
            for s, e, d in sorted(gaps, key=lambda x: -x[2])[:20]
        ],
        "source": "raw_fallback",
    }


def _comm_time_analysis(analysis_db: str, group_ranks: Optional[Dict[str, List[int]]] = None,
                        op_group: Optional[Dict[str, str]] = None) -> Dict:
    con = db.open_ro(analysis_db)
    rows = db.rows(
        con,
        "SELECT step, hccl_op_name, group_name, start_timestamp, elapse_time, "
        "transit_time, wait_time, synchronization_time, idle_time, type "
        "FROM CommAnalyzerTime",
    )
    con.close()
    agg: Dict[tuple, Dict] = {}
    for step, op, group, start, elapse, transit, wait, sync, idle, typ in rows:
        key = (str(group or ""), _op_type(str(op or "")))
        a = agg.setdefault(key, {
            "group_name": str(group or ""),
            "hccl_op_name": _op_type(str(op or "")),
            "type": str(typ or ""),
            "count": 0,
            "elapse_ms": 0.0,
            "transit_ms": 0.0,
            "wait_ms": 0.0,
            "sync_ms": 0.0,
            "idle_ms": 0.0,
            "first_start_ms": None,
            "last_end_ms": None,
        })
        e_ms = _f(elapse)
        a["count"] += 1
        a["elapse_ms"] += e_ms
        a["transit_ms"] += _f(transit)
        a["wait_ms"] += _f(wait)
        a["sync_ms"] += _f(sync)
        a["idle_ms"] += _f(idle)
        st = _f(start)
        if a["first_start_ms"] is None or st < a["first_start_ms"]:
            a["first_start_ms"] = st
        en = st + e_ms
        if a["last_end_ms"] is None or en > a["last_end_ms"]:
            a["last_end_ms"] = en
    return {"ops": list(agg.values()), "source": "analysis.db"}


def _comm_bandwidth_analysis(analysis_db: str, group_ranks: Optional[Dict[str, List[int]]] = None,
                            op_group: Optional[Dict[str, str]] = None) -> Dict:
    con = db.open_ro(analysis_db)
    rows = db.rows(
        con,
        "SELECT step, hccl_op_name, group_name, transport_type, transit_size, "
        "transit_time, bandwidth, count, total_duration FROM CommAnalyzerBandwidth",
    )
    con.close()
    agg: Dict[tuple, Dict] = {}
    for step, op, group, transport, size, transit, bw, cnt, total in rows:
        s, t = _f(size), _f(transit)
        if s <= 0 or t <= 0:
            continue
        gname = (op_group or {}).get(str(op or ""), str(group or ""))
        gkey = _group_key((group_ranks or {}).get(gname, [])) or gname
        key = (gkey, _op_type(str(op or "")), str(transport or ""))
        a = agg.setdefault(key, {
            "group_name": gkey,
            "hccl_op_name": _op_type(str(op or "")),
            "transport_type": str(transport or ""),
            "count": 0,
            "size_mb": 0.0,
            "transit_ms": 0.0,
        })
        a["count"] += _i(cnt)
        a["size_mb"] += s
        a["transit_ms"] += t
    for a in agg.values():
        a["bandwidth_gbps"] = _ratio(a["size_mb"], a["transit_ms"])
    return {"ops": list(agg.values()), "source": "analysis.db"}


def _matrix_analysis(analysis_db: str, group_ranks: Optional[Dict[str, List[int]]] = None,
                     op_group: Optional[Dict[str, str]] = None) -> Dict:
    con = db.open_ro(analysis_db)
    rows = db.rows(
        con,
        "SELECT step, hccl_op_name, group_name, src_rank, dst_rank, transport_type, "
        "transit_size, transit_time, bandwidth FROM CommAnalyzerMatrix",
    )
    con.close()
    agg: Dict[tuple, Dict] = {}
    for step, op, group, src, dst, transport, size, transit, bw in rows:
        s, t = _f(size), _f(transit)
        if s <= 0 or t <= 0:
            continue
        gname = (op_group or {}).get(str(op or ""), str(group or ""))
        gkey = _group_key((group_ranks or {}).get(gname, [])) or gname
        key = (gkey, _op_type(str(op or "")), str(transport or ""))
        a = agg.setdefault(key, {
            "group_name": gkey,
            "hccl_op_name": _op_type(str(op or "")),
            "transport_type": str(transport or ""),
            "count": 0,
            "min_bandwidth_gbps": None,
            "max_bandwidth_gbps": None,
            "slowest_pair": None,
        })
        b = _f(bw)
        a["count"] += 1
        if a["min_bandwidth_gbps"] is None or b < a["min_bandwidth_gbps"]:
            a["min_bandwidth_gbps"] = b
            a["slowest_pair"] = [int(_i(src)), int(_i(dst))]
        if a["max_bandwidth_gbps"] is None or b > a["max_bandwidth_gbps"]:
            a["max_bandwidth_gbps"] = b
    return {"pairs": list(agg.values()), "source": "analysis.db"}


# ---------------------------------------------------------------------------
# 每 rank：裸 profiler db
# ---------------------------------------------------------------------------
def _host_raw(profiler_db: str) -> Dict:
    con = db.open_ro(profiler_db)
    sid = db.string_ids(con)

    pytorch: Dict[str, List[Dict]] = {}
    for start, end, name in db.rows(con, "SELECT startNs, endNs, name FROM PYTORCH_API"):
        s = _i(start)
        e = _i(end)
        if e <= s:
            continue
        n = sid.get(name, str(name))
        pytorch.setdefault(n, []).append({
            "start_ms": s / _NS_PER_MS,
            "end_ms": e / _NS_PER_MS,
            "duration_ms": (e - s) / _NS_PER_MS,
        })

    pytorch_agg = {}
    for n, calls in pytorch.items():
        keep = "gloo" in n.lower()
        pytorch_agg[n] = _aggregate_calls(calls, keep_calls=keep)

    cann: Dict[str, List[Dict]] = {}
    for start, end, name in db.rows(con, "SELECT startNs, endNs, name FROM CANN_API"):
        s = _i(start)
        e = _i(end)
        if e <= s:
            continue
        n = sid.get(name, str(name))
        cann.setdefault(n, []).append({
            "start_ms": s / _NS_PER_MS,
            "end_ms": e / _NS_PER_MS,
            "duration_ms": (e - s) / _NS_PER_MS,
        })
    cann_agg = {n: _aggregate_calls(calls) for n, calls in cann.items()}

    session = db.one(con, "SELECT startTimeNs, endTimeNs FROM SESSION_TIME_INFO")
    session_ms = _ns_to_ms(_i(session[1]) - _i(session[0])) if session else 0.0
    session_start_ms = _ns_to_ms(_i(session[0])) if session else 0.0

    task_types = {}
    for task_type, cnt, ns in db.rows(
        con,
        "SELECT t.taskType, COUNT(*), SUM(t.endNs-t.startNs) FROM TASK t GROUP BY t.taskType",
    ):
        task_types[sid.get(task_type, str(task_type))] = {
            "count": _i(cnt),
            "sum_ms": _ns_to_ms(ns),
        }

    con.close()
    return {
        "pytorch_api": pytorch_agg,
        "cann_api": cann_agg,
        "task_types": task_types,
        "session_ms": session_ms,
        "session_start_ms": session_start_ms,
        "source": "profiler.db",
    }


def _task_gaps(profiler_db: str) -> Dict:
    con = db.open_ro(profiler_db)
    rows = db.rows(con, "SELECT startNs, endNs FROM TASK ORDER BY startNs")
    con.close()
    intervals = []
    for s, e in rows:
        s_ms, e_ms = _ns_to_ms(s), _ns_to_ms(e)
        if e_ms > s_ms:
            intervals.append((s_ms, e_ms))
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = []
    for i in range(1, len(merged)):
        g = merged[i][0] - merged[i - 1][1]
        if g > 0:
            gaps.append({
                "start_ms": merged[i - 1][1],
                "end_ms": merged[i][0],
                "duration_ms": g,
            })
    return {"gaps": sorted(gaps, key=lambda x: -x["duration_ms"]), "total_ms": sum(g["duration_ms"] for g in gaps)}


def _compute_raw(profiler_db: str) -> Dict:
    con = db.open_ro(profiler_db)
    rows = db.rows(
        con,
        "SELECT COALESCE(s.value, CAST(c.name AS TEXT)) AS op, COUNT(*), SUM(t.endNs-t.startNs) "
        "FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
        "LEFT JOIN STRING_IDS s ON s.id=c.name GROUP BY c.name",
    )
    con.close()
    ops = {}
    for op, cnt, ns in rows:
        c = _i(cnt)
        total = _ns_to_ms(ns)
        ops[str(op)] = {
            "count": c,
            "sum_ms": total,
            "avg_ms": (total / c) if c else 0.0,
        }
    return {"ops": ops, "source": "profiler.db"}


def _comm_raw(profiler_db: str, group_ranks: Optional[Dict[str, List[int]]] = None) -> Dict:
    con = db.open_ro(profiler_db)
    sid = db.string_ids(con)
    rows = db.rows(
        con,
        "SELECT o.groupName, o.opName, COUNT(*), SUM(o.endNs-o.startNs), SUM(o.count) "
        "FROM COMMUNICATION_OP o GROUP BY o.groupName, o.opName",
    )
    con.close()
    agg: Dict[tuple, Dict] = {}
    for group, op, cnt, ns, size in rows:
        gname = sid.get(group, str(group))
        g = _group_key((group_ranks or {}).get(gname, [])) or gname
        o = _op_type(sid.get(op, str(op)))
        key = (g, o)
        a = agg.setdefault(key, {
            "group_name": g,
            "hccl_op_name": o,
            "count": 0,
            "sum_ms": 0.0,
            "size_mb": 0.0,
        })
        a["count"] += _i(cnt)
        a["sum_ms"] += _ns_to_ms(ns)
        a["size_mb"] += _f(size) / 1e6
    for a in agg.values():
        a["bandwidth_gbps"] = _ratio(a["size_mb"], a["sum_ms"])
    return {"ops": list(agg.values()), "source": "profiler.db"}


def _freq_raw(profiler_db: str) -> Dict:
    con = db.open_ro(profiler_db)
    rows = db.rows(con, "SELECT deviceId, timestampNs, freq FROM AICORE_FREQ")
    con.close()
    by_dev: Dict[int, List[float]] = {}
    for dev, _ts, freq in rows:
        try:
            by_dev.setdefault(_i(dev), []).append(float(freq))
        except (TypeError, ValueError):
            continue
    out = {}
    for dev, vals in by_dev.items():
        if not vals:
            continue
        out[str(dev)] = {
            "samples": len(vals),
            "min_mhz": min(vals),
            "max_mhz": max(vals),
            "avg_mhz": sum(vals) / len(vals),
        }
    return {"by_device": out, "source": "profiler.db"}


def _rank_device_raw(profiler_db: str) -> Dict:
    con = db.open_ro(profiler_db)
    sid = db.string_ids(con)
    rank = db.one(con, "SELECT rankId, deviceId FROM RANK_DEVICE_MAP")
    host = db.one(con, "SELECT hostUid, hostName FROM HOST_INFO")
    npu = db.one(con, "SELECT id, name FROM NPU_INFO")
    con.close()
    out = {}
    if rank:
        out["rank_id"] = _i(rank[0])
        out["device_id"] = _i(rank[1])
    if host:
        out["host_uid"] = str(host[0])
        out["host_name"] = str(host[1])
    if npu:
        out["npu_name"] = str(npu[1])
    return out


# ---------------------------------------------------------------------------
# cluster db
# ---------------------------------------------------------------------------
def _build_cluster(cluster_db: str, full_cluster: bool = False) -> Dict:
    con = db.open_ro(cluster_db)

    g2pg: Dict[str, str] = {}
    g2ranks: Dict[str, List[int]] = {}
    parallels: Dict[str, List[List[int]]] = {}
    for gname, pg, rank_set in db.rows(
        con,
        "SELECT group_name, pg_name, rank_set FROM CommunicationGroupMapping "
        "WHERE type='collective' AND pg_name IS NOT NULL AND pg_name != ''",
    ):
        if pg in ("pp", "embd", "intra_dist_opt_instance", "default_group"):
            continue
        g2pg[str(gname)] = str(pg)
        ranks = sorted(int(x) for x in re.findall(r"\d+", str(rank_set or "")))
        if len(ranks) >= 2:
            g2ranks[str(gname)] = ranks
            parallels.setdefault(str(pg), []).append(ranks)

    step_trace: Dict[str, Dict] = {}
    for step, typ, idx, comp, comm, free, stage, bubble, prep in db.rows(
        con,
        "SELECT step, type, \"index\", computing, communication, free, stage, bubble, preparing "
        "FROM ClusterStepTraceTime WHERE type='rank'",
    ):
        r = str(_i(idx))
        step_trace.setdefault(r, {"steps": [], "totals": {}})
        step_trace[r]["steps"].append({
            "step": str(step),
            "computing_ms": _us_to_ms(comp),
            "communication_ms": _us_to_ms(comm),
            "free_ms": _us_to_ms(free),
            "stage_ms": _us_to_ms(stage),
            "bubble_ms": _us_to_ms(bubble),
            "preparing_ms": _us_to_ms(prep),
        })
    keys = ("computing_ms", "communication_ms", "free_ms", "stage_ms", "bubble_ms", "preparing_ms")
    for r in step_trace:
        steps = step_trace[r]["steps"]
        step_trace[r]["totals"] = (
            {k: steps[-1][k] for k in keys} if steps else {k: 0.0 for k in keys})
        step_trace[r]["session_ms"] = max(
            (s["stage_ms"] for s in steps), default=0.0)

    comm_time: Dict[str, Dict] = {}
    if full_cluster:
        op_type_sql = (
            "CASE WHEN instr(hccl_op_name,'_')>0 THEN "
            "substr(hccl_op_name, instr(hccl_op_name,'_')+1, "
            "CASE WHEN instr(substr(hccl_op_name,instr(hccl_op_name,'_')+1),'_')>0 "
            "THEN instr(substr(hccl_op_name,instr(hccl_op_name,'_')+1),'_')-1 "
            "ELSE length(hccl_op_name) END) ELSE hccl_op_name END"
        )
        sql = (
            "SELECT rank_id, group_name, " + op_type_sql + " AS op_type, "
            "COUNT(*), SUM(elapsed_time), SUM(transit_time), SUM(wait_time), "
            "SUM(synchronization_time), SUM(idle_time), "
            "MIN(start_timestamp), MAX(start_timestamp+elapsed_time) "
            "FROM ClusterCommunicationTime GROUP BY rank_id, group_name, op_type"
        )
        ct_agg: Dict[tuple, Dict] = {}
        for rank, group, op, cnt, elapse, transit, wait, sync, idle, first, last in db.rows(con, sql):
            r = str(_i(rank))
            gname = str(group or "")
            gkey = _group_key(g2ranks.get(gname, [])) or gname
            key = (r, gkey, _op_type(str(op or "")))
            a = ct_agg.setdefault(key, {
                "hccl_op_name": _op_type(str(op or "")),
                "group_name": gkey,
                "count": 0,
                "elapse_ms": 0.0,
                "transit_ms": 0.0,
                "wait_ms": 0.0,
                "sync_ms": 0.0,
                "idle_ms": 0.0,
                "first_start_ms": None,
                "last_end_ms": None,
            })
            a["count"] += _i(cnt)
            a["elapse_ms"] += _f(elapse)
            a["transit_ms"] += _f(transit)
            a["wait_ms"] += _f(wait)
            a["sync_ms"] += _f(sync)
            a["idle_ms"] += _f(idle)
            st = _f(first)
            en = _f(last)
            if a["first_start_ms"] is None or st < a["first_start_ms"]:
                a["first_start_ms"] = st
            if a["last_end_ms"] is None or en > a["last_end_ms"]:
                a["last_end_ms"] = en
        for (r, group, op), a in ct_agg.items():
            comm_time.setdefault(r, {"ops": [], "totals": {}})
            comm_time[r]["ops"].append(a)
        for r in comm_time:
            ops = comm_time[r]["ops"]
            comm_time[r]["totals"] = {
                k: sum(o[k] for o in ops)
                for k in ("elapse_ms", "transit_ms", "wait_ms", "sync_ms", "idle_ms")
            }

    comm_bandwidth: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    op_type_sql = (
        "CASE WHEN instr(hccl_op_name,'_')>0 THEN "
        "substr(hccl_op_name, instr(hccl_op_name,'_')+1, "
        "CASE WHEN instr(substr(hccl_op_name,instr(hccl_op_name,'_')+1),'_')>0 "
        "THEN instr(substr(hccl_op_name,instr(hccl_op_name,'_')+1),'_')-1 "
        "ELSE length(hccl_op_name) END) ELSE hccl_op_name END"
    )
    sql = (
        "SELECT rank_id, group_name, " + op_type_sql + " AS op_type, "
        "COUNT(*), SUM(transit_size), SUM(transit_time) "
        "FROM ClusterCommunicationBandwidth WHERE band_type='HCCS' "
        "GROUP BY rank_id, group_name, op_type"
    )
    for rank, group, op, cnt, size, transit in db.rows(con, sql):
        r = str(_i(rank))
        pg = g2pg.get(str(group or ""), str(group or ""))
        gkey = _group_key(g2ranks.get(str(group or ""), [])) or pg
        s, t = _f(size), _f(transit)
        if s <= 0 or t <= 0:
            continue
        g = comm_bandwidth.setdefault(r, {}).setdefault(gkey, {})
        a = g.setdefault(_op_type(str(op or "")), {
            "count": 0,
            "size_mb": 0.0,
            "transit_ms": 0.0,
        })
        a["count"] += _i(cnt)
        a["size_mb"] += s
        a["transit_ms"] += t
    for r in comm_bandwidth:
        for pg in comm_bandwidth[r]:
            for a in comm_bandwidth[r][pg].values():
                a["bandwidth_gbps"] = _ratio(a["size_mb"], a["transit_ms"])

    matrix: Dict[str, List[Dict]] = {}
    mat_agg: Dict[tuple, Dict] = {}
    for step, op, group, src, dst, transport, size, transit, bw in db.rows(
        con,
        "SELECT step, hccl_op_name, group_name, src_rank, dst_rank, transport_type, "
        "transit_size, transit_time, bandwidth FROM ClusterCommunicationMatrix",
    ):
        s, t = _f(size), _f(transit)
        if s <= 0 or t <= 0:
            continue
        pg = g2pg.get(str(group or ""), str(group or ""))
        gkey = _group_key(g2ranks.get(str(group or ""), [])) or pg
        key = (gkey, _op_type(str(op or "")))
        a = mat_agg.setdefault(key, {
            "hccl_op_name": _op_type(str(op or "")),
            "transport_type": str(transport or ""),
            "count": 0,
            "min_bandwidth_gbps": None,
            "max_bandwidth_gbps": None,
            "max_transit_ms": 0.0,
            "slowest_pair": None,
        })
        b = _f(bw) if bw is not None else _ratio(s, t)
        a["count"] += 1
        a["max_transit_ms"] = max(a["max_transit_ms"], t)
        if a["min_bandwidth_gbps"] is None or b < a["min_bandwidth_gbps"]:
            a["min_bandwidth_gbps"] = b
            a["slowest_pair"] = [_i(src), _i(dst)]
        if a["max_bandwidth_gbps"] is None or b > a["max_bandwidth_gbps"]:
            a["max_bandwidth_gbps"] = b
    for (pg, op), a in mat_agg.items():
        matrix.setdefault(pg, []).append(a)

    last_steps = []
    for r in step_trace:
        steps = step_trace[r].get("steps") or []
        if steps:
            last_steps.append(str(steps[-1]["step"]))
    ref_step = Counter(last_steps).most_common(1)[0][0] if last_steps else ""
    step_keys = ("computing_ms", "communication_ms", "free_ms", "stage_ms", "bubble_ms", "preparing_ms")
    for r in step_trace:
        steps = step_trace[r].get("steps") or []
        match = next((x for x in steps if str(x["step"]) == ref_step), None)
        if match:
            step_trace[r]["totals"] = {k: match[k] for k in step_keys}
        step_trace[r]["reference_step"] = ref_step

    rank_device = {}
    for rank, dev, host in db.rows(
        con, "SELECT rankId, deviceId, hostUid FROM RankDeviceMap"
    ):
        rank_device[str(_i(rank))] = {
            "device_id": _i(dev),
            "host_uid": str(host or ""),
        }

    host_info = {}
    for uid, name in db.rows(con, "SELECT hostUid, hostName FROM HostInfo"):
        host_info[str(uid)] = str(name or "")

    con.close()
    return {
        "parallels": parallels,
        "group_ranks": g2ranks,
        "reference_step": ref_step,
        "step_trace": step_trace,
        "comm_time": comm_time,
        "comm_bandwidth": comm_bandwidth,
        "matrix": matrix,
        "rank_device": rank_device,
        "host_info": host_info,
        "source": "cluster_analysis.db",
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_evidence(data_dir: str, detection: Optional[Dict] = None,
                   full_cluster: bool = False) -> Dict:
    data_dir = os.path.abspath(data_dir)
    missing: List[str] = []
    notes: List[str] = []

    if os.path.isfile(data_dir) and db.is_cluster_db(data_dir):
        cluster = _build_cluster(data_dir, full_cluster=full_cluster)
        ranks = sorted(
            int(r) for r in cluster["step_trace"].keys()
        ) if cluster["step_trace"] else sorted(
            int(r) for r in cluster["comm_time"].keys()
        )
        meta = {
            "data_dir": data_dir,
            "level": "cluster",
            "ranks": ranks,
            "has_analysis_db": False,
            "has_cluster_db": True,
            "parallels": cluster["parallels"],
            "group_ranks": cluster["group_ranks"],
            "rank_device": cluster["rank_device"],
            "host_info": cluster["host_info"],
            "job_type": "unknown",
            "reference_step": cluster.get("reference_step", ""),
            "session_ms": max(
                (t.get("session_ms", 0.0) for t in cluster["step_trace"].values()),
                default=0.0,
            ),
        }
        notes.append("cluster 单文件：无逐算子计算 count / Host API / AICORE_FREQ，F1/F2/F3/F6 不适用或仅宏观")
        if not full_cluster:
            missing.append("cluster 单文件默认跳过 ClusterCommunicationTime（F5 未评估；可用 full-cluster 打开）")
        return {
            "schema_version": "1.0",
            "meta": meta,
            "detection": detection or {},
            "macro": {
                "step_trace": {"by_rank": cluster["step_trace"], "source": "cluster_analysis.db"},
                "comm_time": {"by_rank": cluster["comm_time"], "source": "cluster_analysis.db"},
            },
            "host": {"gloo": {}, "sync_device": {}, "sync_event": {}, "sync_stream": {}, "free_windows": {}, "source": "none"},
            "compute": {"by_rank": {}, "source": "none"},
            "comm": {
                "by_rank_group": cluster["comm_bandwidth"],
                "matrix": cluster["matrix"],
                "source": "cluster_analysis.db",
            },
            "freq": {"by_rank": {}, "source": "none"},
            "coverage": {"missing": missing, "notes": notes},
        }

    profiler_dbs = db.find_profiler_dbs(data_dir)
    if not profiler_dbs:
        return {
            "schema_version": "1.0",
            "meta": {
                "data_dir": data_dir,
                "level": "levelnone",
                "ranks": [],
                "has_analysis_db": False,
                "has_cluster_db": False,
                "parallels": {},
                "rank_device": {},
                "host_info": {},
                "job_type": "unknown",
                "session_ms": 0.0,
            },
            "detection": detection or {},
            "macro": {"step_trace": {"by_rank": {}}, "comm_time": {"by_rank": {}}},
            "host": {"gloo": {}, "sync_device": {}, "sync_event": {}, "sync_stream": {}, "free_windows": {}},
            "compute": {"by_rank": {}},
            "comm": {"by_rank_group": {}, "matrix": {}},
            "freq": {"by_rank": {}},
            "coverage": {"missing": ["no profiler db"], "notes": notes},
        }

    level = _detect_level(profiler_dbs)

    parallels: Dict[str, List[List[int]]] = {}
    group_ranks: Dict[str, List[int]] = {}
    rank_device: Dict[str, Dict] = {}
    host_info: Dict[str, str] = {}
    job_type = "unknown"
    session_ms = 0.0
    first_con = db.open_ro(profiler_dbs[0][1])
    first_meta = db.meta(first_con)
    first_sid = db.string_ids(first_con)
    job_type = _job_type(profiler_dbs[0][1], first_sid)
    first_con.close()
    op_group: Dict[str, str] = {}

    # 合并所有 rank 的 parallel_group_info（每个 rank 只记录自己所属的组）
    for _rank, pdb in profiler_dbs:
        con = db.open_ro(pdb)
        raw = db.meta(con).get("parallel_group_info")
        con.close()
        op_group.update(_op_group_map(pdb))
        for pg, gs in _parse_parallels(raw).items():
            for g in gs:
                if g not in parallels.setdefault(pg, []):
                    parallels[pg].append(g)
        group_ranks.update(_parse_group_ranks(raw))

    step_trace: Dict[str, Dict] = {}
    comm_time: Dict[str, Dict] = {}
    comm_bandwidth: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    matrix: Dict[str, List[Dict]] = {}
    host: Dict[str, Dict] = {"gloo": {}, "sync_device": {}, "sync_event": {}, "sync_stream": {}, "free_windows": {}, "session_start_ms": {}}
    compute: Dict[str, Dict] = {}
    comm_raw: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    freq: Dict[str, Dict] = {}

    for rank, profiler_db in profiler_dbs:
        r = str(rank)
        analysis_db = db.analysis_db_for(profiler_db)

        if analysis_db:
            st = _step_trace_analysis(analysis_db)
            step_trace[r] = st
            ct = _comm_time_analysis(analysis_db, group_ranks, op_group)
            comm_time[r] = ct
            if level == "level1":
                grouped: Dict[str, Dict[str, Dict]] = {}
                for op in _comm_bandwidth_analysis(analysis_db, group_ranks, op_group).get("ops", []):
                    pg = op["group_name"]
                    grouped.setdefault(pg, {})[op["hccl_op_name"]] = op
                comm_bandwidth[r] = grouped
                for pair in _matrix_analysis(analysis_db, group_ranks, op_group).get("pairs", []):
                    matrix.setdefault(pair["group_name"], []).append(pair)
        else:
            st = _step_trace_raw(profiler_db)
            step_trace[r] = st
            comm_time[r] = {"ops": [], "totals": {}}
            missing.append(f"rank {rank}: analysis.db 缺失，StepTrace/CommTime 用裸 db 回退")

        raw_host = _host_raw(profiler_db)
        gaps = _task_gaps(profiler_db)
        host["gloo"][r] = raw_host["pytorch_api"].get("gloo:all_reduce", {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
        host["sync_device"][r] = raw_host["cann_api"].get("aclrtSynchronizeDeviceWithTimeout", {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
        host["sync_event"][r] = raw_host["cann_api"].get("aclrtSynchronizeEvent", {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
        host["sync_stream"][r] = raw_host["cann_api"].get("aclrtSynchronizeStreamWithTimeout", {"count": 0, "sum_ms": 0.0, "max_ms": 0.0})
        host["free_windows"][r] = {
            "gaps": gaps["gaps"][:10],
            "total_ms": gaps["total_ms"],
        }
        host.setdefault("task_types", {})[r] = raw_host["task_types"]
        host.setdefault("session_ms", {})[r] = raw_host["session_ms"]
        host.setdefault("session_start_ms", {})[r] = raw_host.get("session_start_ms", 0.0)
        session_ms = max(session_ms, raw_host["session_ms"])

        compute[r] = _compute_raw(profiler_db)
        if level != "level1":
            comm_raw[r] = _comm_raw(profiler_db, group_ranks)
        freq[r] = _freq_raw(profiler_db).get("by_device", {})
        rd = _rank_device_raw(profiler_db)
        if rd:
            rank_device[r] = rd
            if rd.get("host_uid"):
                host_info[rd["host_uid"]] = rd.get("host_name", "")

    # 选一个跨 rank 的共同 step（各 rank 最后一个 step 的众数），避免 step 编号错位
    last_steps = []
    for r in step_trace:
        steps = step_trace[r].get("steps") or []
        if steps:
            last_steps.append(str(steps[-1]["step"]))
    ref_step = Counter(last_steps).most_common(1)[0][0] if last_steps else ""
    step_keys = ("computing_ms", "communication_ms", "free_ms", "stage_ms", "bubble_ms", "preparing_ms")
    for r in step_trace:
        steps = step_trace[r].get("steps") or []
        match = next((x for x in steps if str(x["step"]) == ref_step), None)
        if match:
            step_trace[r]["totals"] = {k: match[k] for k in step_keys}
        step_trace[r]["reference_step"] = ref_step

    if level == "levelnone":
        missing.append("levelnone：无计算/搬运算子，compute 未评估")
    if level == "level0":
        notes.append("level0：CommAnalyzerBandwidth 为空，comm 带宽用 COMMUNICATION_OP 粗算；transit/idle 用 CommAnalyzerTime")
        for r, data in comm_raw.items():
            grouped: Dict[str, Dict[str, Dict]] = {}
            for op in data.get("ops", []):
                grouped.setdefault(op["group_name"], {})[op["hccl_op_name"]] = op
            comm_bandwidth[r] = grouped
    if not parallels:
        missing.append("parallel_group_info 缺失或为空，无法做组间对比")
    if not rank_device:
        missing.append("RANK_DEVICE_MAP/HOST_INFO 缺失，无法做物理设备定位")

    meta = {
        "data_dir": data_dir,
        "level": level,
        "ranks": [r for r, _ in profiler_dbs],
        "has_analysis_db": any(db.analysis_db_for(p) for _, p in profiler_dbs),
        "has_cluster_db": False,
        "parallels": parallels,
        "group_ranks": group_ranks,
        "rank_device": rank_device,
        "host_info": host_info,
        "job_type": job_type,
        "session_ms": session_ms,
        "reference_step": ref_step,
    }

    return {
        "schema_version": "1.0",
        "meta": meta,
        "detection": detection or {},
        "macro": {
            "step_trace": {"by_rank": step_trace, "source": "analysis.db"},
            "comm_time": {"by_rank": comm_time, "source": "analysis.db"},
        },
        "host": host,
        "compute": {"by_rank": compute, "source": "profiler.db"},
        "comm": {
            "by_rank_group": comm_bandwidth,
            "matrix": matrix,
            "source": "analysis.db" if level == "level1" else "profiler.db",
        },
        "freq": {"by_rank": freq, "source": "profiler.db"},
        "coverage": {"missing": missing, "notes": notes},
    }


def load_detection(data_dir: str, output_dir: Optional[str] = None) -> Dict:
    candidates = []
    if output_dir:
        candidates.append(os.path.join(output_dir, "straggler_detection_result.json"))
    candidates.append(os.path.join(data_dir, "straggler_detection_result.json"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(data_dir)),
                                  "straggler_detection_result.json"))
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                continue
    return {}
