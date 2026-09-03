"""
Profiling 数据解析模块 - 对应 Go 代码中的 profilingdataparse/
将 SQLite profiling 数据库解析为 CSV/JSON 格式
"""

import sqlite3
import csv
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[DATA PROCESS]")


@dataclass
class StepTime:
    """Step 时间范围结构体"""
    id: int
    start_ns: int
    end_ns: int


@dataclass
class CommunicationOp:
    """通信算子结构体"""
    start_ns: int
    end_ns: int
    h_start_ns: int = 0
    h_end_ns: int = 0
    count: int = 0
    connection_id: int = 0
    domain_id: int = 0
    op_stream_index: int = 0


@dataclass
class HostOp:
    """Host 端算子结构体"""
    start_ns: int
    end_ns: int


@dataclass
class OpStat:
    """算子统计信息"""
    duration: int
    count: int


@dataclass
class PerformanceMetrics:
    """性能指标结构体"""
    step_index: int = 0
    step_duration: int = 0
    zp_device: int = 0
    zp_duration: int = 0
    zp_host: int = 0
    zp_bubble: int = 0
    zp_count: int = 0
    zp_kernel: int = 0
    memcpy_async: int = 0
    kernel_aivec: int = 0
    host_duration: int = 0      # 指标12：通信算子侧 Host 执行耗时均值
    data_loader: int = 0
    durations: Dict[str, int] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


def data_parsing(folder_path: str):
    """
    数据解析主入口
    对应 Go 代码中的 DataParsing 函数
    递归查找目录下的 ascend_pytorch_profiler_*.db 文件，不使用别人处理后的中间数据（如 analysis.db、cluster_analysis.db 等）
    """
    # 递归查找所有符合条件的原始数据库文件
    db_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只使用 ascend_pytorch_profiler_*.db 原始数据，不使用其他中间数据
            if file.startswith("ascend_pytorch_profiler_") and file.endswith(".db"):
                db_path = os.path.join(root, file)
                # 检查文件大小，跳过空文件
                if os.path.getsize(db_path) > 0:
                    db_files.append(db_path)

    if not db_files:
        logger.error(f"未找到数据库文件：{folder_path}")
        return

    # 清空节点映射，开始新一轮解析填充
    config.reset_host_rank_map()

    start_process(db_files, folder_path)


def start_process(db_files: List[str], output_folder: str):
    """
    并发处理数据库文件
    对应 Go 代码中的 StartProcess
    强制重新解析原始 db 数据，不使用已存在的中间数据

    db 从 input（folder_path）递归发现，op_metric 结果写入独立输出目录
    （config.get_output_path()，多 job 场景下与原始数据目录分离）。
    """
    # 解析结果写入独立输出目录；未设置时回退到输出入目录（单 job 场景）
    write_root = config.get_output_path() or output_folder

    # 删除已存在的 op_metric 目录，强制重新解析
    output_metric_dir = os.path.join(write_root, "op_metric")
    if os.path.exists(output_metric_dir):
        try:
            import shutil
            shutil.rmtree(output_metric_dir)
            logger.info(f"已删除旧的 op_metric 目录，重新解析：{output_metric_dir}")
        except Exception as e:
            logger.warning(f"删除 op_metric 目录失败：{e}")

    # Python 中简单串行处理（如需并发可用 threading）
    for db_file in db_files:
        try:
            process_database(db_file, write_root)
        except Exception as e:
            logger.error(f"处理数据库文件 {db_file} 时出错：{e}")


def _op_name_is_optimizer_update(op_name: str) -> bool:
    """
    通过结构模式判断一个算子名是否属于“优化器更新”。

    不依赖具体的优化器名称（不同模型可能用 AdamW/SGD/LAMB/Adafactor 等），
    而是通过 PyTorch 优化器统一的方法名后缀 .step / .zero_grad 来判断，
    因此换用不同优化器也能通用识别。

    参数:
        op_name: 算子名（来自 STRING_IDS.value / PYTORCH_API.name）

    返回:
        True - 该算子属于优化器更新；False - 不是
    """
    if not op_name:
        return False
    name = op_name.strip()
    # 结构模式：方法名以 .step 或 .zero_grad 结尾
    # 例：Optimizer.step#AdamW.step、AdamW.zero_grad、SGD.step、LAMB.step
    return name.endswith(".step") or name.endswith(".zero_grad")


def detect_job_type(conn: sqlite3.Connection) -> str:
    """
    判断当前 job 的类型（training / rollout）。

    通过扫描 PYTORCH_API 中的算子名（JOIN STRING_IDS），
    若存在优化器更新算子（.step / .zero_grad）则判定为 training，否则为 rollout。

    返回:
        "training" - 存在优化器更新算子
        "rollout"  - 不存在优化器更新算子
    """
    has_optimizer = False
    try:
        if table_exists(conn, "PYTORCH_API") and table_exists(conn, "STRING_IDS"):
            cursor = conn.execute(
                "SELECT DISTINCT s.value FROM PYTORCH_API p "
                "JOIN STRING_IDS s ON p.name = s.id "
                "WHERE s.value IS NOT NULL"
            )
            for (op_name,) in cursor:
                if _op_name_is_optimizer_update(op_name):
                    has_optimizer = True
                    break
    except Exception as e:
        logger.warning(f"检测优化器更新算子失败：{e}")

    job_type = "training" if has_optimizer else "rollout"
    logger.info(f"Job 类型判定：{job_type}（存在优化器更新算子={has_optimizer}）")
    return job_type


def process_database(db_file_path: str, output_dir: str) -> bool:
    """
    处理单个数据库文件并将结果保存到 CSV
    对应 Go 代码中的 ProcessDatabase
    """
    try:
        conn = sqlite3.connect(db_file_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        logger.error(f"无法打开数据库文件 {db_file_path}: {e}")
        return False

    try:
        # 启用 WAL 模式提升性能
        conn.execute("PRAGMA journal_mode=WAL;")

        # 创建索引
        conn.execute("CREATE INDEX IF NOT EXISTS idx_string_ids_value ON STRING_IDS(value);")
        # 注意：DEVICE_OP 表在某些数据库版本中不存在，跳过该索引创建
        # conn.execute("CREATE INDEX IF NOT EXISTS idx_device_op_time ON DEVICE_OP(startNs, endNs);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_time_type ON TASK(startNs, endNs, taskType);")

        # 判断并记录当前 job 类型（training/rollout）
        config.set_job_type(detect_job_type(conn))

        # 从文件名提取 global rank
        global_rank = extract_global_rank_from_filename(db_file_path)
        if global_rank is None:
            logger.error(f"无法从文件名提取 rank: {db_file_path}")
            return False

        # 创建输出目录
        output_metric_dir = os.path.join(output_dir, "op_metric")
        os.makedirs(output_metric_dir, exist_ok=True)

        output_file = os.path.join(output_metric_dir, f"global_rank_{global_rank}.csv")
        group_info_file = os.path.join(output_metric_dir, f"group_info_{global_rank}.json")

        # 获取并行域信息
        parallel_group_info = get_parallel_group_info(conn, group_info_file)
        if not parallel_group_info:
            logger.error("获取 parallel_group_info 失败")
            return False

        xp_to_group_name, group_name_to_global_ranks, group_name_to_id = create_group_name_dicts(parallel_group_info)

        logger.info("ParallelGroup Info:")
        for k, v in group_name_to_global_ranks.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Group Name to ID: {group_name_to_id}")

        # 获取该卡的节点信息（hostName），存入内存 config.HostRankMap
        get_host_info(conn, global_rank)

        # 获取所有 step 时间
        all_steps = get_all_step_times(conn)
        if not all_steps:
            logger.warning("未找到任何 step 数据")
            return True

        # 创建聚合 step：开始时间为所有 step 的最小值，结束时间为最大值
        min_start = min(s.start_ns for s in all_steps)
        max_end = max(s.end_ns for s in all_steps)
        aggregated_step = StepTime(id=0, start_ns=min_start, end_ns=max_end)

        # 使用聚合 step 进行检测
        pms = []
        time_diff = time_diff_for_step(conn, xp_to_group_name, aggregated_step)
        if time_diff is not None:
            time_diff.step_index = 0
            time_diff.step_duration = max_end - min_start

            # 保留数据（包括带有 -99999 标记的数据）
            # 只要 KERNEL_AICORE 有值或者是 -99999 标记的数据，都写入 CSV
            pms.append(time_diff)

        # 写入 CSV
        write_results_to_csv(output_file, pms)
        logger.info(f"成功写入 CSV 文件：{output_file}")
        return True

    finally:
        conn.close()


def extract_global_rank_from_filename(db_file_path: str) -> Optional[str]:
    """从文件名提取 global rank"""
    base_name = os.path.basename(db_file_path)
    prefix = "ascend_pytorch_profiler_"
    suffix = ".db"

    if not base_name.startswith(prefix) or not base_name.endswith(suffix):
        return None

    return base_name[len(prefix):-len(suffix)]


def get_parallel_group_info(conn: sqlite3.Connection, filename: str) -> Dict[str, Any]:
    """
    获取并行域信息
    对应 Go 代码中的 GetParallelGroupInfo
    """
    try:
        cursor = conn.execute("SELECT value FROM META_DATA WHERE name = 'parallel_group_info'")
        row = cursor.fetchone()
        if not row:
            return {}

        value = row[0]
        result = json.loads(value)

        # 写入文件（只写一次）
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"已成功写入 parallel_group_info: {filename}")

        return result
    except Exception as e:
        logger.error(f"获取 parallel_group_info 失败：{e}")
        return {}


def create_group_name_dicts(data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, int]]:
    """
    创建 group_name 字典
    对应 Go 代码中的 createGroupNameDicts

    返回:
        xp_to_group_name: {"dp": "group_name_115", ...}
        group_name_to_global_ranks: {"dp": [0, 2, 4, ...], ...}
        group_name_to_id: {"dp": 115, ...}  # 直接从键名中提取的数字 ID
    """
    xp_to_group_name = {}
    group_name_to_global_ranks = {}
    group_name_to_id = {}

    for key, v in data.items():
        if isinstance(v, dict) and "group_name" in v:
            group_name = v["group_name"]
            xp_to_group_name[group_name] = key
            group_name_to_global_ranks[group_name] = v.get("global_ranks", [])

            # 从键名中提取数字 ID，如 "group_name_115" -> 115
            if key.startswith("group_name_"):
                try:
                    group_id = int(key[len("group_name_"):])
                    group_name_to_id[group_name] = group_id
                except ValueError:
                    pass

    return xp_to_group_name, group_name_to_global_ranks, group_name_to_id


def get_host_info(conn: sqlite3.Connection, global_rank: str):
    """
    从 HOST_INFO 表获取该卡所属节点信息（hostUid），存入内存（config.HostRankMap）
    不再生成 node_hostname_map.json 文件
    供 CPU 检测按物理节点分组、无通信域名时节点分组回退使用
    """
    host_uid = None
    try:
        cursor = conn.execute("SELECT hostUid FROM HOST_INFO LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] is not None:
            host_uid = str(row[0])
    except Exception as e:
        logger.warning(f"读取 HOST_INFO 失败：{e}")

    config.set_host_rank_map(global_rank, host_uid)
    logger.info(f"节点信息：rank{global_rank} hostUid={host_uid}")


def get_all_step_times(conn: sqlite3.Connection) -> List[StepTime]:
    """
    获取所有 step 时间范围
    对应 Go 代码中的 GetAllStepTimes
    """
    # 优先从 STEP_TIME 表获取
    if table_exists(conn, "STEP_TIME"):
        return get_step_times_from_step_time(conn)

    # 尝试从 TASK 表获取
    step_times = get_step_times_from_task(conn)
    if step_times:
        return step_times

    # 返回默认值
    return [StepTime(id=-1, start_ns=float('-inf'), end_ns=float('inf'))]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """检查表是否存在"""
    cursor = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
        (table_name,)
    )
    return cursor.fetchone()[0] == 1


def get_step_times_from_step_time(conn: sqlite3.Connection) -> List[StepTime]:
    """从 STEP_TIME 表获取数据"""
    cursor = conn.execute(
        "SELECT id, startNs, endNs FROM STEP_TIME ORDER BY id DESC"
    )

    steps = []
    for row in cursor:
        steps.append(StepTime(id=row[0], start_ns=row[1], end_ns=row[2]))

    # 反转顺序
    steps.reverse()
    return steps


def get_step_times_from_task(conn: sqlite3.Connection) -> List[StepTime]:
    """从 TASK 表推导 step 时间"""
    re_pattern = re.compile(r'step\s+(\d+)')

    cursor = conn.execute("SELECT id, value FROM STRING_IDS")

    step_times = []
    for row in cursor:
        string_id, value = row[0], row[1]
        matches = re_pattern.search(value)
        if not matches:
            continue

        step_id = int(matches.group(1))

        # 查询 TASK 表
        cursor2 = conn.execute(
            "SELECT connectionId FROM MSTX_EVENTS WHERE message = ?",
            (str(string_id),)
        )
        conn_id_row = cursor2.fetchone()
        if not conn_id_row:
            continue

        conn_id = conn_id_row[0]

        cursor3 = conn.execute(
            "SELECT startNs, endNs FROM TASK WHERE connectionId = ?",
            (conn_id,)
        )
        task_row = cursor3.fetchone()
        if not task_row:
            continue

        step_times.append(StepTime(id=step_id, start_ns=task_row[0], end_ns=task_row[1]))

    # 按 ID 排序
    step_times.sort(key=lambda x: x.id)
    return step_times


def time_diff_for_step(
    conn: sqlite3.Connection,
    xp_to_group_name: Dict[str, str],
    step_time: StepTime
) -> Optional[PerformanceMetrics]:
    """
    计算 step 的时间差
    对应 Go 代码中的 TimeDiffForStep
    当无法获取数据时，使用 -99999 标记缺失字段
    """
    # 定义无效数据标记
    INVALID_MARKER = -99999

    if not xp_to_group_name:
        return PerformanceMetrics(durations={}, counts={})

    # 查询 DataLoader ID
    data_loader_id = query_data_loader_id(conn)

    # 初始化 metrics
    metrics = PerformanceMetrics(
        durations={xp: 0 for xp in xp_to_group_name},
        counts={xp: 0 for xp in xp_to_group_name}
    )

    # 获取 group_name → id 映射
    group_names = list(xp_to_group_name.values())
    if not group_names:
        return metrics

    placeholders = ",".join("?" * len(group_names))
    cursor = conn.execute(
        f"SELECT value, id FROM STRING_IDS WHERE value IN ({placeholders})",
        group_names
    )

    group_name_to_id = {row[0]: row[1] for row in cursor}

    # 构建 id → xp 反向映射
    id_to_xp = {}
    group_name_ids = []
    for xp, group_name in xp_to_group_name.items():
        if group_name in group_name_to_id:
            group_id = group_name_to_id[group_name]
            id_to_xp[group_id] = xp
            group_name_ids.append(group_id)

    if not group_name_ids:
        # 无通信域信息，但仍尝试从 KERNEL_AICORE 获取 Host 耗时
        kernel_host_durations = get_kernel_host_durations(conn, step_time)
        if kernel_host_durations:
            metrics.zp_host = calculate_mean(kernel_host_durations)
        else:
            metrics.zp_host = INVALID_MARKER

        kernel_duration = get_avg_kernel_task_duration(conn, step_time)
        if kernel_duration != 0:
            metrics.zp_kernel = kernel_duration
        # 标记通信相关字段为无效
        metrics.zp_device = INVALID_MARKER
        metrics.zp_duration = INVALID_MARKER
        metrics.zp_bubble = INVALID_MARKER
        return metrics

    # 获取 device ops
    device_ops = get_device_op_list(conn, group_name_ids, step_time)
    if not device_ops:
        # 无通信算子，但仍尝试从 KERNEL_AICORE 获取 Host 耗时
        kernel_host_durations = get_kernel_host_durations(conn, step_time)
        if kernel_host_durations:
            metrics.zp_host = calculate_mean(kernel_host_durations)
        else:
            metrics.zp_host = INVALID_MARKER

        kernel_duration = get_avg_kernel_task_duration(conn, step_time)
        if kernel_duration != 0:
            metrics.zp_kernel = kernel_duration
        # 标记通信相关字段为无效
        metrics.zp_device = INVALID_MARKER
        metrics.zp_duration = INVALID_MARKER
        metrics.zp_bubble = INVALID_MARKER
        return metrics

    # 收集 connection IDs
    connection_id_set = set(op.connection_id for op in device_ops)

    # 获取 Host 端时间
    cann_map = get_host_op_from_table(conn, "CANN_API", list(connection_id_set))
    mstx_map = get_host_op_from_table(conn, "MSTX_EVENTS", list(connection_id_set))

    # 填充 Host 时间
    host_durations = []
    bubble_durations = []
    comm_intervals = []
    ops_by_xp: Dict[str, List[CommunicationOp]] = {}

    for op in device_ops:
        conn_id = op.connection_id

        # 填充 Host 时间
        if conn_id in cann_map:
            host_op = cann_map[conn_id]
            op.h_start_ns = host_op.start_ns
            op.h_end_ns = host_op.end_ns
        elif conn_id in mstx_map:
            host_op = mstx_map[conn_id]
            op.h_start_ns = host_op.start_ns
            op.h_end_ns = host_op.end_ns

        # 收集 Host 耗时
        if op.h_start_ns > 0 and op.h_end_ns >= op.h_start_ns:
            host_durations.append(op.h_end_ns - op.h_start_ns)
            bubble = op.start_ns - op.h_end_ns
            if bubble > 0:
                bubble_durations.append(bubble)

        comm_intervals.append((op.start_ns, op.end_ns))

        # 按 xp 分组
        if op.domain_id in id_to_xp:
            xp = id_to_xp[op.domain_id]
            if xp not in ops_by_xp:
                ops_by_xp[xp] = []
            ops_by_xp[xp].append(op)

    # 计算 ZP_Host 和 ZP_Bubble
    # 可靠性设计：同时纳入 KERNEL_AICORE 的 host 耗时，确保无通信算子时也能获取 CPU 数据
    kernel_host_durations = get_kernel_host_durations(conn, step_time)
    all_host_durations = host_durations + kernel_host_durations
    metrics.zp_host = calculate_mean(all_host_durations)
    metrics.zp_bubble = calculate_mean(bubble_durations)
    # 指标12：HostDuration = 通信算子侧 host 执行耗时均值（不含 kernel，区别于 zp_host）
    metrics.host_duration = calculate_mean(host_durations)

    # 计算通信总时长
    total_comm_duration = merge_intervals_simple(comm_intervals)
    step_duration = step_time.end_ns - step_time.start_ns
    non_comm_time = step_duration - total_comm_duration

    if non_comm_time < 0:
        logger.warning(f"通信总耗时超过 step 总耗时 (step={step_duration}, comm={total_comm_duration})")
        non_comm_time = 0

    metrics.zp_device = non_comm_time
    metrics.zp_duration = total_comm_duration

    # 计算各 xp 组的通信时长
    valid_xp_groups = {"tp", "ep", "exp", "pp", "cp", "tp_exp", "dp_modulo_exp_cp", "embd", "mc2", "dp"}

    for xp, ops in ops_by_xp.items():
        xp_key = xp.lower()
        if xp_key not in valid_xp_groups or not ops:
            metrics.durations[xp] = 0
            metrics.counts[xp] = 0
            continue

        stats = [OpStat(duration=op.end_ns - op.start_ns, count=int(op.count) if isinstance(op.count, str) else op.count)
                 for op in ops if op.end_ns - op.start_ns >= 0 and (int(op.count) if isinstance(op.count, str) else op.count) >= 0]

        if stats:
            mean_dur, mean_cnt = calculate_mid_mean_pair(stats)
            metrics.durations[xp] = mean_dur
            metrics.counts[xp] = mean_cnt

    # DataLoader 和 Kernel
    metrics.data_loader = query_data_loader_duration(conn, data_loader_id, step_time)
    kernel_duration = get_avg_kernel_task_duration(conn, step_time)
    if kernel_duration != 0:
        metrics.zp_kernel = kernel_duration

    # 参照 KERNEL_AICORE 生成方式，额外统计 MEMCPY_ASYNC / KERNEL_AIVEC 的平均耗时
    memcpy_async_duration = get_avg_task_duration_by_name(conn, step_time, "MEMCPY_ASYNC")
    if memcpy_async_duration != 0:
        metrics.memcpy_async = memcpy_async_duration

    kernel_aivec_duration = get_avg_task_duration_by_name(conn, step_time, "KERNEL_AIVEC")
    if kernel_aivec_duration != 0:
        metrics.kernel_aivec = kernel_aivec_duration

    return metrics


def get_device_op_list(
    conn: sqlite3.Connection,
    group_name_ids: List[int],
    step_time: StepTime
) -> List[CommunicationOp]:
    """获取通信算子列表"""
    if not table_exists(conn, "COMMUNICATION_OP"):
        return []

    placeholders = ",".join("?" * len(group_name_ids))
    # 注意 SELECT 列顺序：0=opName, 1=startNs, 2=endNs, 3=connectionId,
    # 4=count, 5=_rowid_, 6=groupName。下面按此顺序取下标，与 Go 版
    # (rows.Scan(&OpName,&StartNs,&EndNs,&ConnectionID,&Count,&OpStreamIndex,&DomainID)) 保持一致。
    cursor = conn.execute(
        f"""
        SELECT opName, startNs, endNs, connectionId, count, _rowid_, groupName
        FROM COMMUNICATION_OP
        WHERE groupName IN ({placeholders})
          AND startNs >= ?
          AND endNs <= ?
        ORDER BY startNs ASC
        """,
        group_name_ids + [step_time.start_ns, step_time.end_ns]
    )

    rows = cursor.fetchall()

    device_ops = []
    for row in rows:
        device_ops.append(CommunicationOp(
            start_ns=row[1],           # startNs
            end_ns=row[2],             # endNs
            connection_id=row[3],      # connectionId
            count=int(row[4]) if isinstance(row[4], str) else row[4],  # count
            op_stream_index=row[5],    # _rowid_
            domain_id=row[6],          # groupName
        ))

    return device_ops


def get_host_op_from_table(
    conn: sqlite3.Connection,
    table_name: str,
    connection_ids: List[int]
) -> Dict[int, HostOp]:
    """从指定表获取 Host 端时间"""
    results = {}
    if not connection_ids:
        return results

    if not table_exists(conn, table_name):
        logger.warning(f"表 {table_name} 不存在")
        return results

    placeholders = ",".join("?" * len(connection_ids))
    cursor = conn.execute(
        f"SELECT startNs, endNs, connectionId FROM {table_name} WHERE connectionId IN ({placeholders})",
        connection_ids
    )

    for row in cursor:
        results[row[2]] = HostOp(start_ns=row[0], end_ns=row[1])

    return results


def merge_intervals_simple(intervals: List[Tuple[int, int]]) -> int:
    """合并区间并返回总覆盖时长"""
    if not intervals:
        return 0

    # 按 Start 排序
    intervals.sort(key=lambda x: x[0])

    total = 0
    current_end = intervals[0][0]

    for start, end in intervals:
        if start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end

    return total


def calculate_mean(values: List[int]) -> int:
    """计算均值（过滤负数）"""
    valid_values = [v for v in values if v >= 0]
    if not valid_values:
        return 0
    return int(sum(valid_values) / len(valid_values) + 0.5)


def calculate_mid_mean_pair(stats: List[OpStat]) -> Tuple[int, int]:
    """计算中间均值"""
    n = len(stats)
    if n == 0:
        return 0, 0

    sum_dur = sum(s.duration for s in stats)
    sum_cnt = sum(s.count for s in stats)

    mean_duration = int(sum_dur / n + 0.5)
    mean_count = int(sum_cnt / n + 0.5)

    return mean_duration, mean_count


def query_data_loader_id(conn: sqlite3.Connection) -> int:
    """查询 DataLoader ID"""
    cursor = conn.execute(
        "SELECT id FROM STRING_IDS WHERE value = ?",
        ("dataloader",)
    )
    row = cursor.fetchone()
    return row[0] if row else -1


def query_data_loader_duration(
    conn: sqlite3.Connection,
    data_loader_id: int,
    step_time: StepTime
) -> int:
    """查询 DataLoader 耗时"""
    if data_loader_id == -1:
        return 0

    cursor = conn.execute(
        """
        SELECT startNs, endNs FROM MSTX_EVENTS
        WHERE message = ? AND startNs >= ? AND endNs <= ?
        LIMIT 1
        """,
        (str(data_loader_id), step_time.start_ns, step_time.end_ns)
    )
    row = cursor.fetchone()
    if not row:
        return 0

    start_ns, end_ns = row
    if end_ns < start_ns:
        return 0

    return end_ns - start_ns


def get_avg_kernel_task_duration(conn: sqlite3.Connection, step_time: StepTime) -> int:
    """获取 Kernel 平均耗时"""
    cursor = conn.execute(
        """
        SELECT AVG(t.endNs - t.startNs)
        FROM TASK t
        INNER JOIN STRING_IDS s ON t.taskType = s.id
        WHERE s.value IN ('KERNEL_AICORE')
          AND t.startNs >= ?
          AND t.endNs <= ?
        """,
        (step_time.start_ns, step_time.end_ns)
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(round(row[0]))
    return 0


def get_avg_task_duration_by_name(conn: sqlite3.Connection, step_time: StepTime, name: str) -> int:
    """获取指定 taskType 名称算子的平均耗时，参照 KERNEL_AICORE（get_avg_kernel_task_duration）的生成方式"""
    cursor = conn.execute(
        """
        SELECT AVG(t.endNs - t.startNs)
        FROM TASK t
        INNER JOIN STRING_IDS s ON t.taskType = s.id
        WHERE s.value = ?
          AND t.startNs >= ?
          AND t.endNs <= ?
        """,
        (name, step_time.start_ns, step_time.end_ns)
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(round(row[0]))
    return 0


def get_kernel_host_durations(
    conn: sqlite3.Connection,
    step_time: StepTime
) -> List[int]:
    """获取 KERNEL_AICORE 类型算子的 Host 端耗时列表"""
    cursor = conn.execute(
        """
        SELECT t.connectionId
        FROM TASK t
        INNER JOIN STRING_IDS s ON t.taskType = s.id
        WHERE s.value IN ('KERNEL_AICORE')
          AND t.startNs >= ?
          AND t.endNs <= ?
        """,
        (step_time.start_ns, step_time.end_ns)
    )

    connection_ids = [row[0] for row in cursor if row[0] is not None]
    if not connection_ids:
        return []

    # 通过 connectionId 查 CANN_API / MSTX_EVENTS 获取 host 耗时
    cann_map = get_host_op_from_table(conn, "CANN_API", connection_ids)
    mstx_map = get_host_op_from_table(conn, "MSTX_EVENTS", connection_ids)

    host_durations = []
    for conn_id in connection_ids:
        if conn_id in cann_map:
            host_op = cann_map[conn_id]
        elif conn_id in mstx_map:
            host_op = mstx_map[conn_id]
        else:
            continue

        if host_op.start_ns > 0 and host_op.end_ns >= host_op.start_ns:
            host_durations.append(host_op.end_ns - host_op.start_ns)

    return host_durations


def write_results_to_csv(output_file: str, pms: List[PerformanceMetrics]):
    """将结果写入 CSV 文件"""
    if not pms:
        logger.warning("无数据可写入")
        return

    # 收集所有 XP keys
    xp_keys = set()
    for pm in pms:
        xp_keys.update(pm.durations.keys())
        xp_keys.update(pm.counts.keys())

    sorted_xp_keys = sorted(xp_keys)

    # 构造表头
    headers = [
        "StepIndex", "StepDuration", "ZP_Device", "ZP_Duration",
        "ZP_Host", "ZP_Bubble", "ZP_Count", "KERNEL_AICORE",
        "MEMCPY_ASYNC", "KERNEL_AIVEC", "HostDuration",
        "DataLoader"
    ]
    for xp in sorted_xp_keys:
        headers.extend([f"{xp}_Duration", f"{xp}_Count"])

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for pm in pms:
            record = [
                str(pm.step_index),
                str(pm.step_duration),
                str(pm.zp_device),
                str(pm.zp_duration),
                str(pm.zp_host),
                str(pm.zp_bubble),
                str(pm.zp_count),
                str(pm.zp_kernel),
                str(pm.memcpy_async),
                str(pm.kernel_aivec),
                str(pm.host_duration),
                str(pm.data_loader),
            ]
            for xp in sorted_xp_keys:
                record.extend([
                    str(pm.durations.get(xp, 0)),
                    str(pm.counts.get(xp, 0))
                ])
            writer.writerow(record)

    logger.info(f"成功写入 {len(pms)} 条记录到 {output_file}")
