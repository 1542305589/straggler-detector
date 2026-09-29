"""
Node Level 检测模块 - 对应 Go 代码中的 nodelevel/
慢节点检测核心逻辑
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import kmeans_detector
import utils
import nodelevel_data_handler  # 供 get_cal_detection_group 无命名域/未命中优先级时节点分组回退

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[SLOWNODE ALGO]")

# 常量定义（与 Go 代码一致）
minRanksInGroup = 2
zpDeviceColumn = "ZP_Device"
zpKernelColumn = "KERNEL_AICORE"
zpDurationColumn = "ZP_Duration"
zpHostDataColumn = "ZP_Host"
zpBubbleColumn = "ZP_Bubble"
dataLoaderDataColumn = "DataLoader"
ppParallelDomainName = "pp"
memcpyAsyncColumn = "MEMCPY_ASYNC"
kernelAivecColumn = "KERNEL_AIVEC"

# 通用检测（指标4-10）使用"选定的检测组"运行的指标列 → 类别映射
# 单卡级别检测，进入 kmeans_detector.general_anomaly_detection
GENERAL_METRIC_CATEGORIES = [
    (zpKernelColumn, "KERNEL_AICORE"),  # 指标7：KERNEL_AICORE 计算
    (kernelAivecColumn, "kernel_aivec"),  # 指标6：KERNEL_AIVEC
    (memcpyAsyncColumn, "memcpy_async"),  # 指标9：MEMCPY_ASYNC
]


def delimit_detection(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    单次定界检测（最新检测方式）
    对应 Go 代码中的 DelimitDetection 函数

    参数:
        step_data: 单个 step 的快照数据，如 {"ZP_Device": {0: 1.66e9, 1: 1.67e9, ...}, ...}
        parallels: 并行域信息，如 {"tp": [[0,1], [2,3], ...], "pp": [[0,8], ...]}
        valid_ranks: 有效 rank 列表

    返回:
        检测结果：{"KERNEL_AICORE": {"0": 1.5}, "comm": {"0,1": 1.8}, "cpu": {"5": 2.1}}
    """
    local_result = config.DegradationData()

    # 按照优先级获取检测组和对应的并行域名
    cal_detection_group_name, cal_detection_group = get_cal_detection_group(parallels, valid_ranks)

    # 检查检测组是否有效（注意：并行域名称可能为空字符串 ""）
    if not cal_detection_group or (cal_detection_group_name is None and cal_detection_group != []):
        logger.warning("获取检测组失败")
        return {}

    if not step_data:
        logger.warning("step 数据为空")
        return {}

    # ===== 指标 11：BubbleTime，单阈值检测（<5000ns，沿用） =====
    detection_zp_bubble_data(step_data.get(zpBubbleColumn, {}), local_result)

    # ===== 指标 7：慢计算卡 KERNEL_AICORE（原 cal） =====
    logger.info("\n慢计算卡检测 (KERNEL_AICORE):")
    get_slow_calculate_ranks(cal_detection_group, step_data, cal_detection_group_name, local_result)

    # ===== 指标 4-10：使用选定的检测组，进入通用检测算法 =====
    # 指标 6/9：KERNEL_AIVEC / MEMCPY_ASYNC
    logger.info("\n通用检测（选定的检测组）:")
    for column, category in GENERAL_METRIC_CATEGORIES:
        if column == zpKernelColumn:
            continue  # KERNEL_AICORE 已单独检测
        get_slow_metric_ranks(cal_detection_group, step_data, column, category, local_result)
        logger.info(f"  - {column} -> {category}")

    # ===== 慢通信域检测（带宽聚类，comm） =====
    # 无命名通信域（情况 A）时，无带宽列可检测，直接跳过；
    # 否则按域组带宽做 min 方向聚类（情况 B / 正常数据）。
    if config.get_has_named_domain():
        logger.info("\n慢通信域检测（带宽聚类）:")
        detect_slow_domain_by_bandwidth(parallels, step_data, local_result)
        logger.info("\nPP 流水线慢通信检测:")
        detect_pp_slow_domain(parallels, step_data, local_result)
    else:
        logger.info("[SKIP] 无通信域名，跳过慢通信域检测（comm）")

    # ===== CPU 资源卡检测（ZP_Host，集群整体拉齐） =====
    logger.info("\nCPU 资源卡检测（集群整体拉齐）:")
    get_slow_host_ranks_by_homogenize(valid_ranks, step_data.get(zpHostDataColumn, {}), local_result)

    return dict(local_result)


def get_detection_groups(tp_ranks: List[List[int]], node_global_rank: List[int]) -> Optional[List[List[int]]]:
    """
    通过并行域和当前节点侧任务级卡信息，获取检测组
    对应 Go 代码中的 getDetectionGroups 函数

    将 TP 域中的 rank 过滤，只保留本地节点上的 rank
    """
    rank_map = {rank: True for rank in node_global_rank}

    if not tp_ranks:
        logger.warning("[SLOWNODE ALGO] unexpected empty detection groups!")
        return None

    detection_groups = []
    for sub_rank_list in tp_ranks:
        valid_ranks = [rank for rank in sub_rank_list if rank in rank_map]
        if valid_ranks:
            detection_groups.append(valid_ranks)

    return detection_groups


def get_slow_calculate_ranks(
    detection_groups: List[List[int]],
    aligned_data: Dict[str, Dict[int, float]],
    detection_parallel: str,
    local_result: config.DegradationData
) -> bool:
    """
    获取通信域中的慢计算卡
    对应 Go 代码中的 getSlowCalculateRanks 函数
    """
    if not aligned_data or (len(aligned_data.get(zpDeviceColumn, {})) == 0 and
                            len(aligned_data.get(zpKernelColumn, {})) == 0):
        logger.warning("[SLOWNODE ALGO] empty aligned ZP_device map data")

    # 注意：detection_parallel 可能为空字符串 ""（无命名并行域场景）
    if detection_parallel is None:
        logger.warning("[SLOWNODE ALGO] unexpected detection parallel name!")

    for npu_group in detection_groups:
        abnormal_ranks, rank_deg_severitys = det_cal_for_one_group(aligned_data, npu_group)

        for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
            rank = abnormal_ranks[i]
            degradation = rank_deg_severitys[i]
            local_result.add_single("KERNEL_AICORE", rank, degradation)

    return True


def det_cal_for_one_group(
    aligned_data: Dict[str, Dict[int, float]],
    npu_group: List[int]
) -> Tuple[List[int], List[float]]:
    """
    对单个检测组进行慢计算检测
    对应 Go 代码中的 detCalForOneGroup 函数

    统一使用 KERNEL_AICORE 计算耗时，方向为 "max"（偏大异常），
    进入通用检测算法（kmeans_detector.general_anomaly_detection）。
    排除 0 和 -99999 标记的无效数据。
    """
    # 收集有效数据（排除 0 和 -99999）
    col = zpKernelColumn
    values = []
    ranks = []
    for npu_id in npu_group:
        val = aligned_data.get(col, {}).get(npu_id, 0)
        if val != 0 and val != -99999:
            values.append(val)
            ranks.append(npu_id)

    if len(ranks) < minRanksInGroup:
        return [], []

    # 调用通用检测算法（新核心），cal 属计算类 → 用计算类阈值
    return kmeans_detector.general_anomaly_detection(
        ranks, values, config.get_compute_threshold()
    )


def det_metric_for_one_group(
    aligned_data: Dict[str, Dict[int, float]],
    npu_group: List[int],
    column: str,
    multiplier: float,
) -> Tuple[List[int], List[float]]:
    """
    对单个检测组进行指定指标的慢卡检测
    参照 det_cal_for_one_group 的检测逻辑（KERNEL_AICORE 生成方式）
    排除 0 和 -99999 标记的无效数据，进入通用检测算法（max 方向）
    """
    values = []
    ranks = []
    for npu_id in npu_group:
        val = aligned_data.get(column, {}).get(npu_id, 0)
        if val != 0 and val != -99999:
            values.append(val)
            ranks.append(npu_id)

    if len(ranks) < minRanksInGroup:
        return [], []

    # 通用检测算法（新核心），倍率由 category 决定（memcpy_async 用通信类倍率）
    return kmeans_detector.general_anomaly_detection(
        ranks, values, multiplier
    )


def get_slow_metric_ranks(
    detection_groups: List[List[int]],
    aligned_data: Dict[str, Dict[int, float]],
    column: str,
    category: str,
    local_result: config.DegradationData
) -> bool:
    """
    对指定指标列（MEMCPY_ASYNC/KERNEL_AIVEC 等）检测慢卡
    参照 get_slow_calculate_ranks 的逻辑：在 cal 检测组内逐个组做齐次化聚类
    """
    multiplier = config.get_threshold_for_category(category)
    for npu_group in detection_groups:
        abnormal_ranks, rank_deg_severitys = det_metric_for_one_group(
            aligned_data, npu_group, column, multiplier)

        for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
            rank = abnormal_ranks[i]
            degradation = rank_deg_severitys[i]
            local_result.add_single(category, rank, degradation)

    return True


def detection_zp_bubble_data(npu_data: Dict[int, float], local_result: config.DegradationData):
    """
    检测 ZP bubble
    对应 Go 代码中的 detectionZpBubbleData 函数

    bubble < BUBBLE_THRESHOLD_NS（5000ns）视为异常
    排除 -99999 和 ≤0 的无效数据
    """
    if not npu_data:
        return

    for npu_id, value in npu_data.items():
        # 排除 -99999 标记的无效数据
        if value == -99999:
            continue
        # 排除 ≤0 的数据（数据缺失）
        if value <= 0:
            continue
        if value < config.BUBBLE_THRESHOLD_NS:
            local_result.add_single("npu_bubble", npu_id, value)


def process_cpu_data(ranks_data: List[float]):
    """
    按每 4 张卡为一组，对单个时刻的数据计算组内均值并覆盖原值
    对应 Go 代码中的 processCPUData 函数
    优化：去掉最大值、最小值后再计算均值
    """
    if not ranks_data:
        return

    group_size = 4
    n = len(ranks_data)
    i = 0

    while i < n:
        end = min(i + group_size, n)
        group_data = ranks_data[i:end]

        # 去掉最大值和最小值后计算均值
        if len(group_data) > 2:
            sorted_data = sorted(group_data)
            trimmed_data = sorted_data[1:-1]  # 去掉最小和最大
            mean = sum(trimmed_data) / len(trimmed_data)
        else:
            # 数据不足 3 个时，直接计算均值
            mean = sum(group_data) / len(group_data)

        for k in range(i, end):
            ranks_data[k] = mean
        i = end


def process_cpu_data_by_node(
    have_data_ranks: List[int],
    ranks_data: List[float]
):
    """
    按物理节点分组计算组内均值并覆盖原值
    节点信息取自内存 config.HostRankMap（解析阶段从 HOST_INFO 表填充，不生成文件）
    每节点组内使用与 process_cpu_data 相同的去首尾均值方法
    """
    node_map = config.get_host_rank_map()
    if not node_map:
        # 无节点信息时，回退到原有按 4 分组
        process_cpu_data(ranks_data)
        return

    # 按节点分组：收集每个节点下的卡及对应的数据值
    node_groups = {}
    for i, rank in enumerate(have_data_ranks):
        host = node_map.get(str(rank), str(rank))
        node_groups.setdefault(host, []).append((i, ranks_data[i]))

    for group in node_groups.values():
        group_data = [v for _, v in group]
        # 去掉最大值和最小值后计算均值
        if len(group_data) > 2:
            sorted_data = sorted(group_data)
            trimmed_data = sorted_data[1:-1]  # 去掉最小和最大
            mean = sum(trimmed_data) / len(trimmed_data)
        else:
            # 数据不足 3 个时，直接计算均值
            mean = sum(group_data) / len(group_data)

        # 覆盖该节点组内所有卡的值为组均值
        for idx, _ in group:
            ranks_data[idx] = mean


def get_slow_host_ranks_by_homogenize(
    npus: List[int],
    detection_data: Dict[int, float],
    local_result: config.DegradationData
) -> List[int]:
    """
    获取慢 CPU ranks
    对应 Go 代码中的 getSlowHostRanksByHomogenize 函数
    排除 -99999 标记的无效数据
    """
    have_data_ranks = []
    ranks_data = []

    for npu in npus:
        if npu in detection_data:
            val = detection_data[npu]
            # 排除 -99999 标记的无效数据
            if val != -99999:
                have_data_ranks.append(npu)
                ranks_data.append(val)

    # 按物理节点分组计算组内均值（取代固定按 4 分组）—— 集群整体拉齐
    process_cpu_data_by_node(have_data_ranks, ranks_data)

    # cpu 属 IO/CPU 类阈值
    abnormal_ranks, rank_deg_severitys = kmeans_detector.general_anomaly_detection(
        have_data_ranks, ranks_data, config.get_io_threshold()
    )

    for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
        rank = abnormal_ranks[i]
        degradation = rank_deg_severitys[i]
        local_result.add_single("cpu", rank, degradation)

    return abnormal_ranks


def get_cal_detection_group(
    parallels: Dict[str, List[List[int]]],
    cur_npus: List[int]
) -> Tuple[str, List[List[int]]]:
    """
    选择用于检测的并行域，以及对应的检测组
    对应 Go 代码中的 GetCalDetectionGroup 函数

    优先级：tp → exp → ep → tp_exp → cp → cp2 → cp_ulysses → cp_ring → dp → dp_cp → dp_modulo_exp_cp

    集群数据（Case A）：返回优先级域的完整集群分组，不做节点过滤；
    非集群数据（Case B）：返回按本地节点过滤后的分组，或 "" 节点回退分组。
    """
    if not parallels or not cur_npus:
        return "", []

    is_cluster = config.get_is_cluster_data()

    # 并行域检测优先级（按优先级从高到低）
    detection_priority = [
        "tp", "exp", "ep", "tp_exp", "cp", "cp2", "cp_ulysses", "cp_ring",
        "dp", "dp_cp", "dp_modulo_exp_cp"
    ]

    for domain in detection_priority:
        if domain in parallels and parallels[domain]:
            parallel_info = parallels[domain]
            if not parallel_info:
                continue

            logger.info(f"[SLOWNODE ALGO] use {domain} parallel detection: {parallel_info}")
            if is_cluster:
                # Case A：完整集群分组，不做节点过滤
                detection_groups = list(parallel_info)
            else:
                detection_groups = get_detection_groups(parallel_info, cur_npus)
            return domain, detection_groups

    # 如果并行域名称为空字符串（无命名并行域场景，情况 A），使用空字符串作为 key 的节点分组
    if "" in parallels and parallels[""]:
        parallel_info = parallels[""]
        if parallel_info:
            logger.info(f"[SLOWNODE ALGO] use '' (unnamed/node fallback) parallel detection: {parallel_info}")
            if is_cluster:
                detection_groups = list(parallel_info)
            else:
                detection_groups = get_detection_groups(parallel_info, cur_npus)
            return "", detection_groups

    # 未命中任何优先级域（情况 B：有命名域，但都不在检测优先级内）：
    # 检测组退化为按 hostUid 的物理节点分组，单卡指标在节点组内检测。
    # 这里不做短路 return "", []，避免丢失单卡指标检测。
    node_groups = nodelevel_data_handler._build_node_fallback_groups(cur_npus)
    if node_groups:
        logger.info(f"[SLOWNODE ALGO] 未命中检测优先级，回退按物理节点分组检测: {node_groups}")
        return "", node_groups

    logger.warning("[SLOWNODE ALGO] no valid parallel domain found for detection")
    return "", []


def detect_slow_domain_by_bandwidth(
    parallels: Dict[str, List[List[int]]],
    step_data: Dict[str, Dict[int, float]],
    local_result: config.DegradationData,
):
    """
    按带宽聚类检测慢通信组（对应 Go DetectSlowDomainByBandwidth）。

    数据来源：带宽回填写进 CSV 的动态列 "<domain>_<opType>_<count>"。
    对每个集合通信域、每个 opType：
      1. 每组取 count 最大的条目作为代表（count 越大带宽越准）；
      2. 保留 count >= max×0.5 且 > SLOW_COMM_COUNT_FLOOR 的组（滤 count 噪声）；
      3. 用通用检测（min 方向，带宽越小越慢）聚类代表带宽，阈值 = COMM_THRESHOLD。
    只报告在每个 opType 上都异常的组，劣化指数取各 opType 最大值。
    """
    ratio = config.get_comm_threshold()
    if ratio <= 0:
        ratio = 1.3

    for domain, groups in parallels.items():
        if domain == ppParallelDomainName or domain == "embd":
            continue
        if len(groups) < 2:
            continue

        group_bws = [_bw_set_for_group(domain, group, step_data) for group in groups]
        op_types = _collect_op_types(group_bws)
        if not op_types:
            continue

        anomalous: Dict[int, Dict[str, float]] = {}

        for op_type in op_types:
            # 每组取 count 最大的条目作为代表
            reps = []  # (groupIdx, count, bw)
            for gi, bws in enumerate(group_bws):
                max_c = -1
                max_bw = 0.0
                for e in bws:
                    if e["op_type"] != op_type or e["count"] <= max_c:
                        continue
                    max_c = e["count"]
                    max_bw = e["bw"]
                if max_c >= 0:
                    reps.append((gi, max_c, max_bw))

            if len(reps) < 2:
                continue

            max_count = max(r[1] for r in reps)
            half = max_count * 0.5
            kept = [r for r in reps if r[1] >= half and r[1] > config.SLOW_COMM_COUNT_FLOOR]
            if len(kept) < 2:
                continue

            kept_gidxs = [r[0] for r in kept]
            bw_values = [r[2] for r in kept]
            abnormal_gidxs, degradations = kmeans_detector.general_anomaly_detection(
                kept_gidxs, bw_values, ratio, high_is_anomaly=False
            )
            for gi, deg in zip(abnormal_gidxs, degradations):
                anomalous.setdefault(gi, {})[op_type] = deg

        # 只报告在每个 opType 上都异常的组，劣化指数取各 opType 最大值
        for gi, op_degs in anomalous.items():
            if len(op_degs) != len(op_types):
                continue
            max_deg = max(op_degs.values())
            local_result.add_group("comm", groups[gi], max_deg)


def _bw_set_for_group(
    domain: str, group: List[int], step_data: Dict[str, Dict[int, float]]
) -> List[Dict[str, Any]]:
    """收集某通信组在各带宽列上的值（组内所有 rank 共享同一回填带宽，取第一个有值的）。"""
    prefix = domain + "_"
    out = []
    for col, by_rank in step_data.items():
        op_type, count = _parse_bandwidth_col(prefix, col)
        if op_type is None:
            continue
        v = None
        for r in group:
            if r in by_rank:
                v = by_rank[r]
                break
        if v is None:
            continue
        out.append({"op_type": op_type, "count": count, "bw": v})
    return out


def _parse_bandwidth_col(prefix: str, col: str):
    """解析带宽列名 "<opType>_<count>"（给定域前缀）。非数字尾/诊断列返回 None。"""
    if not col.startswith(prefix):
        return None, 0
    rest = col[len(prefix):]
    if rest.startswith("_"):
        rest = rest[1:]
    idx = rest.rfind("_")
    if idx <= 0 or idx == len(rest) - 1:
        return None, 0
    op_type = rest[:idx]
    count_str = rest[idx + 1:]
    try:
        count = int(count_str)
    except ValueError:
        return None, 0
    if op_type == "Duration" or op_type == "Count":
        return None, 0
    return op_type, count


def _collect_op_types(group_bws: List[List[Dict[str, Any]]]) -> set:
    s = set()
    for bws in group_bws:
        for e in bws:
            s.add(e["op_type"])
    return s


def detect_pp_slow_domain(
    parallels: Dict[str, List[List[int]]],
    step_data: Dict[str, Dict[int, float]],
    local_result: config.DegradationData,
):
    """
    PP 流水线慢通信检测（另一方案，对应 PP 域）。

    数据来源：PP 链路重叠回填写进 CSV 的动态列 "PP_Overlap"（某卡作为收方时，其入边
    链路 Send/Recv 时间窗重叠时长）。PP 组取自 parallel_group_info 的 pp 项；若无 PP
    分组（parallels 无 "pp"），则不检测（避免把 CP/Ring Attention 误当 PP）。

    检测：按 PP 组内 stage 位置分组，把各 PP 组相邻两 stage 组成的链路(s->r)的
    重叠时长放到同一组内做通用检测（max 方向，重叠越长→该链路传输越慢）；异常按
    "发送方->接收方"（组键 [s, r]）写入类别 "pp_comm"。
    """
    groups = parallels.get(ppParallelDomainName)
    if not groups or len(groups) < 2:
        return

    overlap = step_data.get(config.PP_OVERLAP_COLUMN, {})
    if not overlap:
        return

    # 组内升序视为 stage 顺序（rank = pp_stage * tp_size + tp_rank）
    sorted_groups = [sorted(g) for g in groups]
    max_len = max(len(g) for g in sorted_groups)

    for t in range(max_len - 1):
        links = []  # (sender, receiver, value)
        for g in sorted_groups:
            if len(g) <= t + 1:
                continue
            s, r = g[t], g[t + 1]
            v = overlap.get(r)
            if v is None or v == -99999 or v <= 0:
                continue
            links.append((s, r, v))
        if len(links) < 2:
            continue

        recv_ranks = [r for _, r, _ in links]
        values = [v for _, _, v in links]
        abnormal_recv, degs = kmeans_detector.general_anomaly_detection(
            recv_ranks, values, config.get_comm_threshold(), high_is_anomaly=True
        )
        for rk, deg in zip(abnormal_recv, degs):
            for s, r, _ in links:
                if r == rk:
                    local_result.add_group("pp_comm", [s, r], deg)
                    break
