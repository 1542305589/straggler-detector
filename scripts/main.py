#!/usr/bin/env python3
import os
import sys
import logging
from typing import Dict, Any, List
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import profilingdataparse
import nodelevel
import nodelevel_data_handler
import utils
import visualizer
import summary_table
logging.basicConfig(
    # 默认只把 ERROR 打到 stderr：过程日志（INFO/WARNING）不进用户可见输出，
    # 排查问题时可设环境变量 SLOWNODE_DEBUG=1 打开完整日志。
    level=logging.DEBUG if os.environ.get("SLOWNODE_DEBUG") else logging.ERROR,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger("[SLOWNODE ALGO]")

def parse_args(args: list) -> Dict[str, Any]:
    result = {
        'path': None,
        'compute': 1.3,
        'io': 2.5,
        'comm': 1.3,
        'pp_comm': 5.0,
        'no_align': False,
        'clean': 'ask'
    }
    def _bad(name, val, default):
        logger.warning(f"Invalid {name} value '{val}', using default {default}")
    for arg in args:
        if arg == 'no-align':
            result['no_align'] = True
            continue
        if '=' not in arg:
            continue
        parts = arg.split('=', 1)
        if len(parts) != 2:
            continue
        key, val = parts[0], parts[1]
        if key == 'path':
            result['path'] = val
        elif key in ('compute', 'io', 'comm', 'pp_comm', 'pp'):
            try:
                f = float(val)
                if f > 0:
                    result[key] = f
                else:
                    _bad(key, val, result[key])
            except ValueError:
                _bad(key, val, result[key])
        elif key == 'no-align':
            if val.lower() in ('true', 'yes', '1'):
                result['no_align'] = True
            elif val.lower() in ('false', 'no', '0'):
                result['no_align'] = False
            else:
                logger.warning(f"Invalid no-align value '{val}', using default false")
        elif key == 'clean':
            if val.lower() in ('yes', 'no', 'ask'):
                result['clean'] = val.lower()
            else:
                logger.warning(f"Invalid clean value '{val}', using default 'ask'")
    return result

def _find_job_subdirs(parent_path: str) -> List[str]:
    jobs = []
    try:
        for entry in sorted(os.listdir(parent_path)):
            sub = os.path.join(parent_path, entry)
            if not os.path.isdir(sub):
                continue
            try:
                names = os.listdir(sub)
            except OSError:
                continue
            if any(
                f.startswith("ascend_pytorch_profiler_") and f.endswith(".db")
                for f in names
            ):
                jobs.append(sub)
    except OSError as e:
        logger.warning(f"[SLOWNODE ALGO] 扫描父目录 job 子目录失败：{e}")
    return jobs

def _dir_has_db_directly(path: str) -> bool:
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(
        f.startswith("ascend_pytorch_profiler_") and f.endswith(".db")
        for f in names
    )

def _safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
        except Exception:
            sys.stdout.write(text.encode("ascii", errors="replace").decode("ascii"))

def _print_conclusion(job_path: str, detection: dict, thresholds: tuple, unavailable: dict = None):
    try:
        import config
        import conclusion
        out = config.get_output_path()
        if not out:
            out = os.path.dirname(job_path) if os.path.isfile(job_path) else job_path
        level = "unknown"
        try:
            level = nodelevel_data_handler.detect_profiling_level(job_path)
        except Exception:
            pass
        text = conclusion.render_from_dir(out, level, thresholds, source_path=job_path,
                                          unavailable=unavailable)
        _safe_print("")
        _safe_print("```text")
        _safe_print(text.rstrip())
        _safe_print("```")
        for line in (f"检测结果: {os.path.join(out, 'straggler_detection_result.json')}",
                     f"检测报告: {os.path.join(out, 'detection_report.log')}"):
            _safe_print(line)
    except Exception as e:
        logger.warning(f"[SLOWNODE ALGO] 结论文本生成失败：{e}")

def _backfill_slow_domain(job_path: str, db_files: list = None):
    try:
        profilingdataparse.backfill_slow_domain_bandwidth(job_path, db_files)
    except Exception as e:
        logger.warning(f"慢通信带宽回填失败：{e}")
    try:
        profilingdataparse.backfill_pp_wait(job_path, db_files)
    except Exception as e:
        logger.warning(f"PP 等待回填失败：{e}")

def _find_colocate_worlds(input_path: str) -> List[dict]:
    if not input_path or not os.path.isdir(input_path):
        return []
    try:
        import classify_role
    except Exception as e:
        logger.warning(f"[SLOWNODE ALGO] 导入 classify_role 失败，跳过 colocate 分组：{e}")
        return []
    try:
        worlds = classify_role.build_role_worlds(input_path)
    except Exception as e:
        logger.warning(f"[SLOWNODE ALGO] colocate 角色分组失败：{e}")
        return []
    if not worlds:
        return []
    roles = {w["role"] for w in worlds}
    if len(roles) < 2:
        return []
    if any(len(w["ranks"]) < 2 for w in worlds):
        logger.warning("[SLOWNODE ALGO] colocate 世界内 rank 数 <2，无法做跨卡检测，按普通流程处理")
        return []
    return worlds

def _process_colocate_root(input_path: str, worlds: List[dict], thresholds: tuple,
                           clean_mode: str, skip_parsing: bool,
                           skip_confirm: bool = False) -> Dict[str, Any]:
    config.set_file_path(input_path)
    if not skip_confirm:
        thresholds = utils.confirm_thresholds(thresholds)
    config.set_thresholds(*thresholds)
    if skip_parsing:
        clean_decision = False
    elif clean_mode == 'yes':
        clean_decision = True
    elif clean_mode == 'no':
        clean_decision = False
    else:
        clean_decision = utils.confirm_clean(input_path)
    results = {}
    for world in worlds:
        role = world["role"]
        out_path = os.path.join(input_path, "detection_output", role)
        logger.info(f"===== 处理 {role} 世界：{len(world['dbs'])} rank，ranks={world['ranks']} =====")
        if world.get("note"):
            logger.warning(f"[SLOWNODE ALGO] {role} 世界备注：{world['note']}")
        try:
            results[role] = _process_single_job(
                input_path, thresholds, clean_mode, output_path=out_path,
                db_files=world["dbs"], skip_confirm=True, clean_decision=clean_decision)
        except Exception as e:
            logger.error(f"[SLOWNODE ALGO] {role} 世界检测失败：{e}")
            results[role] = {}
    return results

def _threshold_ask_keys(level: str) -> tuple:
    if level == "cluster":
        return ("compute", "io", "comm")
    if level == "levelnone":
        return ("io", "comm")
    return ("compute", "io", "comm")


def _expected_categories(level: str) -> tuple:
    """该数据级别本应参与检测的类别。"""
    return {
        "level0": ("compute", "comm", "pp_comm", "host", "memcpy"),
        "level1": ("compute", "comm", "host", "memcpy"),
        "levelnone": ("comm", "pp_comm", "host"),
        "cluster": ("compute", "comm", "host"),
    }.get(level, ())


def _metric_availability(level: str, step_data: dict, comm, parallels: dict,
                         cluster_step_info: dict = None,
                         comm_metric: dict = None) -> dict:
    """返回「该级别应检测、但本次无法评估」的类别 -> 原因。

    用于区分"没有异常"和"没有数据"：只有可评估且未检出异常才是"无"。
    """
    unavailable = {}

    def _present(col: str) -> bool:
        return any(v is not None and v > 0 for v in (step_data.get(col) or {}).values())

    if level in ("level0", "level1"):
        has_aicore = _present("KERNEL_AICORE")
        has_aivec = _present("KERNEL_AIVEC")
        if not (has_aicore or has_aivec):
            unavailable["compute"] = "缺少 KERNEL_AICORE / KERNEL_AIVEC 指标"
        else:
            # 部分缺失时逐项说明（缺哪个标哪个）
            if not has_aicore:
                unavailable["KERNEL_AICORE"] = "缺少 KERNEL_AICORE 指标"
            if not has_aivec:
                unavailable["KERNEL_AIVEC"] = "缺少 KERNEL_AIVEC 指标"
        if not _present("MEMCPY_ASYNC"):
            unavailable["memcpy"] = "缺少 MEMCPY_ASYNC 指标"
        if not _present(nodelevel.zpHostDataColumn):
            unavailable["host"] = f"缺少 {nodelevel.zpHostDataColumn} 指标"
    elif level == "levelnone":
        if not _present(nodelevel.zpHostDataColumn):
            unavailable["host"] = f"缺少 {nodelevel.zpHostDataColumn} 指标"
    elif level == "cluster":
        rows = (cluster_step_info or {}).get("rows") or []
        has_computing = any((metrics.get("computing") or 0) > 0 for _s, _r, metrics in rows)
        has_free = any((metrics.get("free") or 0) > 0 for _s, _r, metrics in rows)
        if not has_computing:
            unavailable["compute"] = "缺少 ClusterStepTraceTime.computing"
        if not has_free:
            unavailable["host"] = "缺少 ClusterStepTraceTime.free"

    if level in ("level0", "levelnone") and not _present(config.PP_WAIT_GAP_COLUMN):
        unavailable["pp_comm"] = f"缺少 {config.PP_WAIT_GAP_COLUMN} 指标"

    if level in ("level1", "cluster"):
        if not (comm_metric or {}).get("card"):
            unavailable["comm"] = "缺少通信带宽数据"
    elif level in ("level0", "levelnone"):
        if not config.get_has_named_domain():
            unavailable["comm"] = "无通信域名，未检测通信"
        elif not any(utils.domain_bandwidth_cols(step_data, d) for d in (parallels or {}) if d):
            unavailable["comm"] = "缺少通信带宽列"
    return unavailable

def _partial_data_hint(level: str, missing_ranks, total: int) -> str:
    """部分 rank 缺数据时的提示。level1 的 comm 来自 analysis.db，不受 CSV 缺失影响。"""
    ranks = sorted(missing_ranks)
    if level == "level1":
        return (f"注意：{len(ranks)}/{total} 张卡的 op_metric 指标缺失（rank {ranks}），"
                f"计算 / Host / 内存搬运仅基于其余 rank；通信带宽来自 analysis.db，不受影响。")
    return (f"注意：{len(ranks)}/{total} 张卡缺少指标数据（rank {ranks}），"
            f"检测结果仅基于其余 rank。")


def _profiler_rank_dirs(job_path: str, db_files=None) -> Dict[int, str]:
    dirs: Dict[int, str] = {}
    if db_files:
        for path in db_files:
            rank = nodelevel_data_handler._rank_from_profiler_filename(path)
            if rank is not None:
                dirs[rank] = os.path.dirname(os.path.abspath(path))
        return dirs
    for root, _dirs, files in os.walk(job_path):
        for fn in files:
            if fn.startswith("ascend_pytorch_profiler_") and fn.endswith(".db"):
                rank = nodelevel_data_handler._rank_from_profiler_filename(os.path.join(root, fn))
                if rank is not None:
                    dirs[rank] = root
    return dirs


def _incomplete_cache_reasons(job_path: str, level: str, db_files=None) -> list:
    """检查 op_metric / 通信指标缓存是否完整；返回不完整原因（空列表表示完整）。"""
    rank_dirs = _profiler_rank_dirs(job_path, db_files)
    if not rank_dirs:
        return []
    metric_dir = os.path.join(config.get_output_path() or job_path, "op_metric")
    reasons = []
    for rank in sorted(rank_dirs):
        csv_path = os.path.join(metric_dir, f"global_rank_{rank}.csv")
        if not os.path.exists(csv_path):
            reasons.append(f"rank {rank} 缺少指标 CSV")
        else:
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.readline()
                    second = f.readline()
                if not second:
                    reasons.append(f"rank {rank} 的指标 CSV 为空")
            except OSError:
                reasons.append(f"rank {rank} 的指标 CSV 读取失败")
        if not os.path.exists(os.path.join(metric_dir, f"group_info_{rank}.json")):
            reasons.append(f"rank {rank} 缺少拓扑 JSON")
    if level == "level1":
        for rank, rank_dir in sorted(rank_dirs.items()):
            if not os.path.exists(os.path.join(rank_dir, "analysis.db")):
                reasons.append(f"rank {rank} 缺少 analysis.db")
        if not os.path.exists(os.path.join(metric_dir, "comm_metric.json")):
            reasons.append("缺少通信指标缓存")
    return reasons


def _process_single_job(job_path: str, thresholds: tuple, clean_mode: str, output_path: str = None,
                        db_files: list = None, skip_confirm: bool = False, clean_decision: bool = None):
    config.set_file_path(job_path)
    if output_path:
        config.set_output_path(output_path)
    level = nodelevel_data_handler.detect_profiling_level(job_path)
    config.set_profiling_level(level)
    if not skip_confirm:
        thresholds = utils.confirm_thresholds(thresholds, keys=_threshold_ask_keys(level))
    config.set_thresholds(*thresholds)
    is_cluster_input = (level == "cluster")
    if is_cluster_input:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(job_path)), "detection_output")
        config.set_output_path(out_dir)
    logger.info(f"开始慢节点检测 - 路径：{job_path}, 阈值(计算/IO/通信)：{thresholds}")
    logger.info(f"数据级别：{level}")
    logger.info(f"检测结果输出目录：{config.get_output_path()}")
    out_path = config.get_output_path()
    if is_cluster_input:
        parallels = nodelevel_data_handler.load_cluster_parallels(job_path)
        step_info = nodelevel_data_handler.load_cluster_step_data(job_path)
        comm_metric = nodelevel_data_handler.load_cluster_comm_metric(job_path, parallels)
        unavailable = _metric_availability("cluster", {}, None, parallels, step_info,
                                           comm_metric=comm_metric)
        expected = _expected_categories(level)
        if not parallels or (expected and all(cat in unavailable for cat in expected)):
            _safe_print("未评估：cluster 数据为空或缺少可用指标，本次未执行检测（不输出“无异常”）。")
            logger.error("cluster 数据为空或缺少可用指标，跳过检测")
            return None
        os.makedirs(out_dir, exist_ok=True)
        valid_ranks = sorted({int(r) for gs in parallels.values() for g in gs for r in g})
        if not valid_ranks:
            valid_ranks = sorted({int(r) for _s, r, _m in step_info.get("rows", [])})
        result = nodelevel.delimit_detection(
            {}, parallels, valid_ranks, cluster_step_info=step_info,
            comm_metric=comm_metric)
        utils.write_result(result, parallels, unavailable=unavailable)
        try:
            visualizer.run_visualization(
                {}, parallels, valid_ranks, out_dir,
                detection_result=result, data_path=job_path)
        except Exception as e:
            logger.warning(f"生成报告失败：{e}")
        try:
            summary = summary_table.build_summary_table(
                result, parallels, {}, unavailable=unavailable)
            _safe_print(summary)
        except Exception as e:
            logger.warning(f"生成汇总表失败：{e}")
        _print_conclusion(job_path, result, thresholds, unavailable=unavailable)
        return result
    if clean_decision is None:
        if clean_mode == 'yes':
            clean_decision = True
        elif clean_mode == 'no':
            clean_decision = False
        else:
            clean_decision = utils.confirm_clean(out_path)
    if not clean_decision:
        reasons = _incomplete_cache_reasons(job_path, level, db_files)
        if reasons:
            shown = "；".join(reasons[:3])
            if len(reasons) > 3:
                shown += f" 等共 {len(reasons)} 项"
            _safe_print(f"注意：检测缓存不完整（{shown}），将自动重新解析。")
            logger.warning(f"检测缓存不完整，自动重新解析：{shown}")
            clean_decision = True
    if clean_decision:
        logger.info("开始 Profiling 数据解析（清理旧 op_metric 后重新解析）...")
        utils.clean_detection_outputs(out_path)
        if db_files:
            profilingdataparse.data_parsing_paths(db_files, job_path)
        else:
            profilingdataparse.data_parsing(job_path)
        logger.info("数据解析完成")
        _backfill_slow_domain(job_path, db_files)
    else:
        logger.info("跳过清理和重新解析，直接使用已有的 op_metric 数据")
        profilingdataparse.load_host_info_only(job_path, db_files)
    parallels, valid_ranks = nodelevel_data_handler.get_cur_detection_info(job_path)
    if not parallels or not valid_ranks:
        logger.error("获取并行域/卡数失败")
        _safe_print("未评估：无法获取并行域 / 有效 rank，未执行检测（不输出“无异常”）。")
        return None
    logger.info(f"并行域：{list(parallels.keys())}")
    logger.info(f"有效 Ranks: {valid_ranks}")
    last_step_data, load_status = nodelevel_data_handler.get_cur_job_last_step_data(valid_ranks)
    comm_metric = None
    if level == "level1":
        comm_metric = nodelevel_data_handler.load_level1_comm_metric(job_path, parallels)
    unavailable = _metric_availability(level, last_step_data, None, parallels,
                                       comm_metric=comm_metric)
    expected = _expected_categories(level)
    if expected and all(cat in unavailable for cat in expected):
        _safe_print("未评估：未找到可用的指标数据，本次未执行检测（不输出“无异常”）。")
        _safe_print(f"  参与检测的 rank 数：{load_status.get('ranks_total', 0)}，"
                    f"成功读取指标的 rank 数：{len(load_status.get('ranks_with_data', []))}")
        if load_status.get("ranks_missing_csv"):
            _safe_print(f"  指标 CSV 缺失的 rank：{load_status['ranks_missing_csv']}")
        if load_status.get("ranks_empty_csv"):
            _safe_print(f"  指标 CSV 为空/无法解析的 rank：{load_status['ranks_empty_csv']}")
        logger.error("未找到可用的指标数据，跳过检测")
        return None
    result = nodelevel.delimit_detection(last_step_data, parallels, valid_ranks,
                                        comm_metric=comm_metric)
    utils.write_result(result, parallels, unavailable=unavailable)
    # 指标 CSV 全缺但通信仍有数据（如 level1 从 analysis.db 读取）时，
    # 仍然生成检测报告，避免结论里给出的报告路径实际不存在。
    visualizer.run_visualization(
        last_step_data, parallels, valid_ranks, config.get_output_path(),
        detection_result=result,
        data_path=job_path,
    )
    if not result:
        logger.info("未检测到异常节点")
    try:
        summary = summary_table.build_summary_table(
            result, parallels, last_step_data, unavailable=unavailable)
        _safe_print(summary)
    except Exception as e:
        logger.warning(f"生成汇总表失败：{e}")
    partial = sorted(set(load_status.get("ranks_missing_csv", []))
                     | set(load_status.get("ranks_empty_csv", [])))
    if partial:
        _safe_print(_partial_data_hint(level, partial, load_status.get("ranks_total", 0)))
    _print_conclusion(job_path, result, thresholds, unavailable=unavailable)
    return result

def main():
    args = sys.argv[1:]
    if len(args) < 1:
        logger.error("[SLOWNODE ALGO] Usage: python main.py path=<dir | cluster.db> [compute=1.3] [io=2.5] [comm=1.3] [no-align] [clean=ask]")
        sys.exit(1)
    parsed = parse_args(args)
    # 调用方（agent / 脚本）已显式给出全部阈值时，不再交互式追问
    explicit = {a.split("=", 1)[0] for a in args if "=" in a}
    skip_confirm = {"compute", "io", "comm"} <= explicit
    input_path = parsed['path']
    thresholds = (parsed['compute'], parsed['io'], parsed['comm'])
    clean_mode = parsed['clean']
    config.set_comm_no_align(parsed['no_align'])
    config.PP_WAIT_RATIO_THRESHOLD = parsed['pp_comm'] / 100.0
    if not input_path:
        logger.error("[SLOWNODE ALGO] Missing required parameter: path=/your/data/dir")
        sys.exit(1)
    if not os.path.isdir(input_path) and not (
        os.path.isfile(input_path) and nodelevel_data_handler.is_cluster_db(input_path)
    ):
        logger.error(f"[SLOWNODE ALGO] Invalid input (need a directory, or a cluster db file): {input_path}")
        sys.exit(1)
    for name, v in zip(("compute", "io", "comm"), thresholds):
        if v <= 0:
            logger.warning(f"[WARN] {name} threshold must be > 0. Please verify.")
    if os.path.isfile(input_path):
        config.set_output_path("")
        result = _process_single_job(input_path, thresholds, clean_mode, skip_confirm=skip_confirm)
        if result is None:
            sys.exit(1)
        return result
    colocate_worlds = _find_colocate_worlds(input_path)
    if colocate_worlds:
        logger.info(f"[SLOWNODE ALGO] 检测到 verl colocate：训练/rollout 分开检测 "
                    f"{[w['role'] for w in colocate_worlds]}")
        results = _process_colocate_root(input_path, colocate_worlds, thresholds, clean_mode,
                                         False, skip_confirm=skip_confirm)
        if results and all(v is None for v in results.values()):
            sys.exit(1)
        return results
    sub_jobs = _find_job_subdirs(input_path)
    parent_has_db = _dir_has_db_directly(input_path)
    if sub_jobs and not parent_has_db:
        logger.info(f"[SLOWNODE ALGO] 检测到 {len(sub_jobs)} 个子 job，逐个执行检测：{sub_jobs}")
        output_root = os.path.join(input_path, "detection_output")
        results = {}
        for job in sub_jobs:
            logger.info(f"===== 处理 job：{job} =====")
            job_output = os.path.join(output_root, os.path.basename(job))
            results[job] = _process_single_job(job, thresholds, clean_mode,
                                               output_path=job_output, skip_confirm=skip_confirm)
        if results and all(v is None for v in results.values()):
            sys.exit(1)
        return results
    config.set_output_path("")
    result = _process_single_job(input_path, thresholds, clean_mode, skip_confirm=skip_confirm)
    if result is None:
        sys.exit(1)
    return result

def run_detection(input_path: str, compute: float = 1.3, io: float = 2.5, comm: float = 1.3,
                  skip_parsing: bool = False, clean: str = 'ask',
                  no_align: bool = None) -> Dict[str, Dict[str, float]]:
    level = nodelevel_data_handler.detect_profiling_level(input_path)
    config.set_profiling_level(level)
    if no_align is not None:
        config.set_comm_no_align(bool(no_align))
    elif level in ("levelnone", "level0"):
        config.set_comm_no_align(utils.confirm_no_align())
    else:
        config.set_comm_no_align(False)
    thresholds = (compute, io, comm)
    colocate_worlds = _find_colocate_worlds(input_path)
    if colocate_worlds:
        logger.info(f"[SLOWNODE ALGO] 检测到 verl colocate：训练/rollout 分开检测 "
                    f"{[w['role'] for w in colocate_worlds]}")
        return _process_colocate_root(input_path, colocate_worlds, thresholds, clean, skip_parsing)
    return _process_single_job(
        input_path, thresholds, clean,
        skip_confirm=False,
        clean_decision=False if skip_parsing else None,
    )
if __name__ == "__main__":
    main()
