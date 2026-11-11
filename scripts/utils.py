import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import summary_table

def parse_bandwidth_col(prefix: str, col: str):
    if not col.startswith(prefix):
        return None, 0
    rest = col[len(prefix):]
    if rest.startswith("_"):
        rest = rest[1:]
    idx = rest.rfind("_")
    if idx <= 0 or idx == len(rest) - 1:
        return None, 0
    op_type = rest[:idx]
    try:
        count = int(rest[idx + 1:])
    except ValueError:
        return None, 0
    if op_type in ("Duration", "Count"):
        return None, 0
    return op_type, count

def domain_bandwidth_cols(step_data: Dict[str, Dict[int, float]], domain: str) -> List[Any]:
    prefix = domain + "_"
    out = []
    for col in step_data:
        op_type, count = parse_bandwidth_col(prefix, col)
        if op_type is None:
            continue
        out.append((op_type, count, col))
    out.sort(key=lambda x: (x[0], x[1]))
    return out

def group_representative_bandwidth(
    step_data: Dict[str, Dict[int, float]], domain: str, group: List[int]
) -> List[Any]:
    by_op: Dict[str, Any] = {}
    for op_type, count, col in domain_bandwidth_cols(step_data, domain):
        by_rank = step_data.get(col) or {}
        v = None
        for r in group:
            rv = by_rank.get(r)
            if rv is not None and rv != -99999 and rv > 0:
                v = rv
                break
        if v is None:
            continue
        cur = by_op.get(op_type)
        if cur is None or count > cur[0]:
            by_op[op_type] = (count, v)
    return [(op_type, cnt, bw) for op_type, (cnt, bw) in sorted(by_op.items())]

def _rank_items(items: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for key, value in sorted(items.items(), key=lambda x: -x[1]):
        raw = str(key)
        if ":" in raw:
            raw = raw.split(":")[-1]
        try:
            rank = int(raw)
        except (TypeError, ValueError):
            continue
        out.append({"rank": rank, "degradation": value})
    return out


def _host_items(items: Dict[str, float]) -> List[Dict[str, Any]]:
    return [{"host": str(key), "degradation": value}
            for key, value in sorted(items.items(), key=lambda x: -x[1])]


def _comm_groups(items: Dict[str, float], parallels: Dict[str, List[List[int]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {"tp": [], "ep": []}
    for domain in (parallels or {}):
        if domain and domain not in groups:
            groups[domain] = []
    for key, value in sorted(items.items(), key=lambda x: -x[1]):
        domain, ranks_key = summary_table._split_domain_key(key)
        if not domain:
            domain = summary_table._domain_of_group(parallels or {}, ranks_key)
        domain = domain or "default_group"
        groups.setdefault(domain, [])
        try:
            ranks = [int(r) for r in ranks_key.split(",")]
        except (TypeError, ValueError):
            continue
        if len(ranks) == 1:
            target = f"rank{ranks[0]}"
        else:
            target = f"{domain}[{','.join(str(r) for r in ranks)}]"
        groups[domain].append({"target": target, "degradation": value})
    return groups


def _pp_items(items: Dict[str, float]) -> List[Dict[str, Any]]:
    """PP 点对点通信的数值是「等待占整个会话的占比」（0~1），不是劣化倍数。"""
    return [{"target": f"pp[{key}]", "degradation": value}
            for key, value in sorted(items.items(), key=lambda x: -x[1])]


_GROUP_NOTES = {
    "compute": "计算类异常",
    "comm": "通信类异常",
    "host": "Host类故障",
    "memcpy": "内存搬运类异常",
}
_METRIC_NOTES = {
    "collective": "集合通信",
    "computing": "cluster 单文件聚合值（cluster数据独有），含 MIX/AICPU 等，仅作混合负载粗筛。",
    "pp_comm": "pp点对点通信，劣化数值为pp通信的等待占整个会话的占比",
    "ZP_Host": "通信算子 + KERNEL_AICORE 的 Host 侧耗时均值",
    "free": "cluster 单文件 NPU 空闲时间（cluster数据独有），仅作 Host 粗筛",
}
# 各数据等级实际参与检测的指标：不参与该等级的指标键直接不出现。
_LEVEL_METRICS = {
    "level0": {
        "compute": ("KERNEL_AICORE", "KERNEL_AIVEC"),
        "comm": ("collective", "pp_comm"),
        "host": ("ZP_Host",),
        "memcpy": ("MEMCPY_ASYNC",),
    },
    "level1": {
        "compute": ("KERNEL_AICORE", "KERNEL_AIVEC"),
        "comm": ("collective",),
        "host": ("ZP_Host",),
        "memcpy": ("MEMCPY_ASYNC",),
    },
    "levelnone": {
        "comm": ("collective", "pp_comm"),
    },
    "cluster": {
        "compute": ("computing",),
        "comm": ("collective",),
        "host": ("free",),
    },
}


def _metric_entry(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    note = _METRIC_NOTES.get(name)
    if note:
        entry["note"] = note
    entry.update(body)
    return entry


def build_report_payload(final_result: Dict[str, Dict[str, float]],
                         parallels: Dict[str, List[List[int]]] = None,
                         level: str = "",
                         thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    fr = final_result or {}
    spec = _LEVEL_METRICS.get(level)
    th = dict(thresholds or {})

    def allowed(group: str, name: str) -> bool:
        return spec is None or name in spec.get(group, ())

    faults: Dict[str, Any] = {}

    compute_metrics: Dict[str, Any] = {}
    for out_name, src in (("KERNEL_AICORE", "KERNEL_AICORE"),
                          ("KERNEL_AIVEC", "kernel_aivec"),
                          ("computing", "computing_load")):
        if allowed("compute", out_name):
            compute_metrics[out_name] = _metric_entry(
                out_name, {"items": _rank_items(fr.get(src) or {})})
    if compute_metrics:
        faults["compute"] = {"note": _GROUP_NOTES["compute"], "metrics": compute_metrics}

    comm_metrics: Dict[str, Any] = {}
    if allowed("comm", "collective"):
        comm_metrics["collective"] = _metric_entry(
            "collective", {"groups": _comm_groups(fr.get("comm") or {}, parallels)})
    if allowed("comm", "pp_comm"):
        comm_metrics["pp_comm"] = _metric_entry(
            "pp_comm", {"items": _pp_items(fr.get("pp_comm") or {})})
    if comm_metrics:
        faults["comm"] = {"note": _GROUP_NOTES["comm"], "metrics": comm_metrics}

    host_metrics: Dict[str, Any] = {}
    for out_name in ("ZP_Host", "free"):
        if allowed("host", out_name):
            host_metrics[out_name] = _metric_entry(
                out_name, {"items": _host_items(fr.get("cpu") or {})})
    if host_metrics:
        faults["host"] = {"note": _GROUP_NOTES["host"], "metrics": host_metrics}

    if allowed("memcpy", "MEMCPY_ASYNC"):
        faults["memcpy"] = {
            "note": _GROUP_NOTES["memcpy"],
            "metrics": {"MEMCPY_ASYNC": _metric_entry(
                "MEMCPY_ASYNC", {"items": _rank_items(fr.get("memcpy_async") or {})})},
        }

    out_th: Dict[str, Any] = {}
    if "compute" in faults and "compute" in th:
        out_th["compute"] = th["compute"]
    if ("host" in faults or "memcpy" in faults) and "io" in th:
        out_th["io/host"] = th["io"]
    if "comm" in faults and "comm" in th:
        out_th["comm"] = th["comm"]
    if "pp_comm" in (faults.get("comm", {}).get("metrics") or {}) and "pp_comm" in th:
        out_th["pp_comm"] = th["pp_comm"]

    return {
        "schema_version": "2.0",
        "data_level": level,
        "thresholds": out_th,
        "faults": faults,
    }


def write_result(final_result: Dict[str, Dict[str, float]], parallels: Dict[str, List[List[int]]] = None):
    for category, items in final_result.items():
        if not items:
            continue
        kvs = sorted(((k, v) for k, v in items.items()), key=lambda x: x[1], reverse=True)
        is_group_category = category in ("comm", "pp_comm")
        rank_set_to_domain = {}
        if is_group_category and parallels:
            for domain_name, domain_groups in parallels.items():
                for group in domain_groups:
                    sorted_group = tuple(sorted(group))
                    rank_set_to_domain[sorted_group] = domain_name
        print(f"\n【{category}】全部异常项（共 {len(kvs)} 项）:")
        for i, (key, value) in enumerate(kvs):
            if category == "pp_comm":
                display_key = "[" + ", ".join(key.split(",")) + "]"
            elif is_group_category:
                key_domain, ranks_key = summary_table._split_domain_key(key)
                ranks = ranks_key.split(",")
                if not key_domain:
                    try:
                        key_domain = rank_set_to_domain.get(tuple(int(r) for r in ranks), "")
                    except ValueError:
                        key_domain = ""
                if key_domain:
                    display_key = f"{key_domain}[{', '.join(ranks)}]"
                else:
                    display_key = "[" + ", ".join(ranks) + "]"
            else:
                display_key = key
            if category == "pp_comm":
                print(f"  {i+1}. {display_key} → 等待占比 = {value * 100:.2f}%")
            else:
                print(f"  {i+1}. {display_key} → 劣化指数 = {value:.3f}")

    report = build_report_payload(
        final_result, parallels, config.get_profiling_level(),
        {
            "compute": config.get_compute_threshold(),
            "io": config.get_io_threshold(),
            "comm": config.get_comm_threshold(),
            "pp_comm": config.PP_WAIT_RATIO_THRESHOLD * 100.0,
        },
    )
    output_path = os.path.join(config.get_output_path(), "straggler_detection_result.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n检测结果已保存至：{output_path}")
    except Exception as e:
        print(f"[ERROR] 无法创建结果文件 {output_path}: {e}")

def _is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False

def _read_input(prompt: str) -> str:
    """读一行输入。stdin 不可读（/dev/null、nul、管道已关闭）时返回空串，让调用方走默认值。"""
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return ""

def confirm_clean(input_path: str) -> bool:
    items_to_clean = [
        "op_metric",
        "straggler_analysis_output",
        "straggler_detection_result.json",
        "straggler_detection_result_flat.json",
        "straggler_detection_result",
        "detection_report.log",
        "conclusion.md",
        "joint_failure_analysis.log"
    ]
    existing = []
    for item in items_to_clean:
        item_path = os.path.join(input_path, item)
        if os.path.exists(item_path):
            existing.append(item)
    if not existing:
        return True
    if not _is_interactive():
        print(f"\n[非交互] 发现已有数据（{', '.join(existing)}），默认保留、不删除")
        return False
    print(f"\n发现以下已有数据：{', '.join(existing)}")
    while True:
        ans = _read_input("是否删除已有数据并重新解析？(y/N): ").lower()
        if ans in ('y', 'yes'):
            return True
        elif ans in ('n', 'no', ''):
            return False
        print("请输入 y 或 n")

def _ask_threshold(name: str, default: float) -> float:
    while True:
        ans = _read_input(f"请输入{name}（默认 {default}，回车使用默认）: ")
        if ans == "":
            return default
        try:
            v = float(ans)
            if v > 0:
                return v
            print("请输入一个大于 0 的数")
        except ValueError:
            print("请输入一个数字")

def confirm_thresholds(thresholds: tuple, keys: tuple = ("compute", "io", "comm")) -> tuple:
    compute, io, comm = thresholds
    if not _is_interactive():
        return (compute, io, comm)
    print("检测阈值设置（回车使用默认值）：")
    if "compute" in keys:
        compute = _ask_threshold("计算类阈值（KERNEL_AICORE / kernel_aivec）", compute)
    if "io" in keys:
        io = _ask_threshold("IO/CPU 类阈值（cpu / memcpy_async）", io)
    if "comm" in keys:
        comm = _ask_threshold("通信类阈值（comm / pp_comm）", comm)
    return (compute, io, comm)

def confirm_no_align(default: bool = False) -> bool:
    if not _is_interactive():
        return default
    hint = "Y/n" if default else "y/N"
    while True:
        ans = _read_input(
            f"慢通信带宽是否关闭跨 rank 对齐（--no-align：用全体 rank 扁平池；{hint}）："
        ).lower()
        if ans == "":
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("请输入 y 或 n")

def clean_detection_outputs(input_path: str):
    items_to_clean = [
        "op_metric",
        "straggler_analysis_output",
        "straggler_detection_result.json",
        "straggler_detection_result_flat.json",
        "straggler_detection_result",
        "detection_report.log",
        "conclusion.md",
        "joint_failure_analysis.log"
    ]
    cleaned_count = 0
    for item in items_to_clean:
        item_path = os.path.join(input_path, item)
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
                cleaned_count += 1
                print(f"[CLEAN] 已删除目录：{item_path}")
            else:
                os.remove(item_path)
                cleaned_count += 1
                print(f"[CLEAN] 已删除文件：{item_path}")
    return cleaned_count

def read_file(file_path: str) -> bytes:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    with open(file_path, "rb") as f:
        return f.read()

def check_parallel_domain_is_exist(parallel: List[List[int]], cur_npus: int) -> bool:
    if not parallel:
        return False
    per_domain_nums = len(parallel[0]) if parallel else 0
    has_multi_card_group = False
    for domain in parallel:
        if len(domain) > 1:
            has_multi_card_group = True
        if len(domain) != per_domain_nums:
            print(f"[SLOWNODE ALGO] 通信域间卡的数量不一致:{parallel}")
            return False
    if not has_multi_card_group:
        return False
    return True

def write_batch_result(
    batch_results: Dict[str, Any],
    output_dir: str,
    total_cases: int,
    success_count: int,
    failed_count: int,
    skipped_count: int
) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"san_batch_detection_result_{date_str}.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print("\n" + "="*80)
        print("批量检测结果汇总")
        print("="*80)
        print(f"总案例数：{total_cases}")
        print(f"成功检测：{success_count} 个")
        print(f"检测失败：{failed_count} 个")
        print(f"已跳过：{skipped_count} 个")
        print(f"\n详细结果已保存至：{output_file}")
        print("="*80)
        return output_file
    except Exception as e:
        print(f"[ERROR] 无法创建结果文件 {output_file}: {e}")
        return ""
