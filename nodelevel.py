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
cpuDegradationPercent = 2.0
memcpyAsyncColumn = "MEMCPY_ASYNC"
kernelAivecColumn = "KERNEL_AIVEC"
stepDurationColumn = "StepDuration"  # 指标1：DB 数据总时间间隔
hostDurationColumn = "HostDuration"  # 指标12：Host 端执行耗时均值

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

    # ===== 指标 1/2/3：通信域组间对比（step_duration / comm / xp_count） =====
    # 无命名通信域（情况 A）时，通信域指标无法向用户解释对应 tp/ep，直接跳过；
    # 否则正常检测（情况 B / 正常数据）。
    if config.get_has_named_domain():
        logger.info("\n通信域组间对比检测:")
        detection_all_communication_parallel(parallels, cal_detection_group, valid_ranks, step_data, local_result)
    else:
        logger.info("[SKIP] 无通信域名，跳过通信域组间对比检测（comm/step_duration/xp_count）")

    # ===== 指标 12 + 兼容：HostDuration / ZP_Host，集群整体拉齐 =====
    logger.info("\nCPU / Host 资源卡检测（集群整体拉齐）:")
    get_slow_host_ranks_by_homogenize(valid_ranks, step_data.get(zpHostDataColumn, {}), local_result)
    _get_slow_host_metric_ranks(valid_ranks, step_data.get(hostDurationColumn, {}), "host_duration", local_result)

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

    # 调用通用检测算法（新核心），cal 属计算类 → 用计算类倍率
    return kmeans_detector.general_anomaly_detection(
        ranks, values, config.get_compute_multiplier()
    )


def det_metric_for_one_group(
    aligned_data: Dict[str, Dict[int, float]],
    npu_group: List[int],
    column: str
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

    # 通用检测算法（新核心），计算/IO 类 → 用计算类倍率
    return kmeans_detector.general_anomaly_detection(
        ranks, values, config.get_compute_multiplier()
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
    for npu_group in detection_groups:
        abnormal_ranks, rank_deg_severitys = det_metric_for_one_group(aligned_data, npu_group, column)

        for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
            rank = abnormal_ranks[i]
            degradation = rank_deg_severitys[i]
            local_result.add_single(category, rank, degradation)

    return True


def detection_zp_bubble_data(npu_data: Dict[int, float], local_result: config.DegradationData):
    """
    检测 ZP bubble
    对应 Go 代码中的 detectionZpBubbleData 函数

    bubble < 5000ns 视为异常
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
        if value < 5000:
            local_result.add_single("npu_bubble", npu_id, value)


def check_parallel_domain_is_exist(parallel: List[List[int]], cur_npus: int) -> bool:
    """
    检查并行域是否有效
    对应 Go 代码中的 checkParallelDomainIsExist 函数

    注意：不再要求域中总卡数等于本节点卡数（cur_npus），
    因为 group_info 是全集群拓扑，可能包含其他节点的卡。
    下游检测函数会通过 -99999 过滤来处理数据缺失的情况。
    """
    if not parallel:
        return False

    per_domain_nums = len(parallel[0]) if parallel else 0
    has_multi_card_group = False

    for domain in parallel:
        if len(domain) > 1:
            has_multi_card_group = True

        # 各个域卡数参差不齐
        if len(domain) != per_domain_nums:
            logger.warning(f"[SLOWNODE ALGO] 通信域间卡的数量不一致:{parallel}")
            return False

    # 并行域中只有卡本身：不存在卡间并行域
    if not has_multi_card_group:
        return False

    return True


def get_pp_slow_communicate_domains(
    slow_send_ranks: List[int],
    pp_parallel: List[List[int]]
) -> List[List[int]]:
    """
    通过慢 send ranks 检测慢 PP 通信域
    对应 Go 代码中的 getPpSlowCommunicateDomains 函数
    """
    if not pp_parallel or not slow_send_ranks:
        logger.warning("[SLOWNODE ALGO] detection without pp parallel or slow send ranks!")
        return []

    ret = []
    for slow_send_rank in slow_send_ranks:
        for pp_domain in pp_parallel:
            if slow_send_rank in pp_domain:
                ret.append(pp_domain)
                break

    return ret


def homogenization_for_slow_communication(
    detection_domains: List[List[int]],
    detection_data: Dict[int, float],
    degradation_percent: float,
    pp_stage_num: int
) -> Tuple[List[List[int]], List[float]]:
    """
    慢通信域聚类方法
    对应 Go 代码中的 HomogenizationForSlowCommunication 函数
    排除 -99999 标记的无效数据

    入参为并行域，同时也是检测组
    """
    slow_comm_domains = []
    slow_comm_domain_severitys = []

    if not detection_domains or not detection_data:
        logger.warning("[SLOWNODE ALGO] slow communication domains detection data is empty!")
        return [], []

    # 1. 对每个子域内部排序
    for domain in detection_domains:
        domain.sort()

    # 2. 对整个 detection_domains 按字典序排序
    detection_domains.sort()

    # 3. 找出每个通信域中耗时最短的卡（排除 -99999）
    detection_cards = []
    rank2_groups = {}

    for domain in detection_domains:
        # 过滤掉 -99999 的卡，找有效数据中的最小值
        valid_cards = [card for card in domain if detection_data.get(card, 0) != -99999]
        if not valid_cards:
            continue
        min_card = min(valid_cards, key=lambda x: detection_data.get(x, float('inf')))
        detection_cards.append(min_card)
        rank2_groups[min_card] = domain

    if not detection_cards:
        return [], []

    # 4. 按 pp_size 划分成几份分别进行聚类
    detection_card_groups = []
    interval = len(detection_cards) // pp_stage_num

    for i in range(pp_stage_num):
        start = i * interval
        end = start + interval
        if start < len(detection_cards):
            detection_card_groups.append(detection_cards[start:end])

    # 5. 对每个组进行聚类检测（排除 -99999）
    for detection_ranks in detection_card_groups:
        # 过滤掉 -99999 的 rank 和对应的数据
        valid_ranks = []
        valid_datas = []
        for rank in detection_ranks:
            val = detection_data.get(rank, 0)
            if val != -99999 and val != 0:
                valid_ranks.append(rank)
                valid_datas.append(val)

        if len(valid_ranks) < minRanksInGroup:
            continue

        abnormal_ranks, rank_deg_severitys = kmeans_detector.general_anomaly_detection(
            valid_ranks, valid_datas, config.get_comm_multiplier()
        )

        # 将异常 rank 映射回对应的通信域
        for i, rank in enumerate(abnormal_ranks):
            if rank in rank2_groups:
                slow_comm_domains.append(rank2_groups[rank])
                if i < len(rank_deg_severitys):
                    slow_comm_domain_severitys.append(rank_deg_severitys[i])

    return slow_comm_domains, slow_comm_domain_severitys


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

    abnormal_ranks, rank_deg_severitys = kmeans_detector.general_anomaly_detection(
        have_data_ranks, ranks_data, config.get_compute_multiplier()
    )

    for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
        rank = abnormal_ranks[i]
        degradation = rank_deg_severitys[i]
        local_result.add_single("cpu", rank, degradation)

    return abnormal_ranks


def _get_slow_host_metric_ranks(
    npus: List[int],
    detection_data: Dict[int, float],
    category: str,
    local_result: config.DegradationData
) -> List[int]:
    """
    对某个 host 侧指标列做“集群整体拉齐”检测：
    先按物理节点分组求组内均值（覆盖组内卡值），再对整个检测列做通用检测。
    用于指标12 HostDuration 与兼容列 ZP_Host(cpu)。
    """
    have_data_ranks = []
    ranks_data = []
    for npu in npus:
        if npu in detection_data:
            val = detection_data[npu]
            if val != -99999:
                have_data_ranks.append(npu)
                ranks_data.append(val)

    if len(have_data_ranks) < minRanksInGroup:
        return []

    process_cpu_data_by_node(have_data_ranks, ranks_data)

    abnormal_ranks, rank_deg_severitys = kmeans_detector.general_anomaly_detection(
        have_data_ranks, ranks_data, config.get_compute_multiplier()
    )

    for i in range(min(len(abnormal_ranks), len(rank_deg_severitys))):
        rank = abnormal_ranks[i]
        degradation = rank_deg_severitys[i]
        local_result.add_single(category, rank, degradation)

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


def get_slow_communication_detection_data(
    parallel_name: str,
    all_data: Dict[str, Dict[int, float]]
) -> Dict[int, float]:
    """
    获取慢通信检测数据
    对应 Go 代码中的 getSlowCommunicationDetectionData 函数
    排除 -99999 标记的无效数据
    """
    ret = {}
    duration_label = f"{parallel_name}_Duration"
    count_label = f"{parallel_name}_Count"

    if not all_data or not all_data.get(duration_label) or not all_data.get(count_label):
        logger.warning(f"[SLOWNODE ALGO] slow {parallel_name} detection data is empty!")
        return ret

    count_data = all_data[count_label]
    duration_data = all_data[duration_label]

    # 卡数不对齐情况
    if len(count_data) != len(duration_data):
        logger.warning(f"[SLOWNODE ALGO] {parallel_name} detection cards data not aligned!")
        return ret

    for npu_id in count_data:
        if npu_id in duration_data:
            val = duration_data[npu_id]
            # 排除 -99999 标记的无效数据
            if val != -99999:
                ret[npu_id] = val

    return ret


def _detect_comm_group_metric(
    parallels: Dict[str, List[List[int]]],
    cal_detection_group: List[List[int]],
    valid_ranks: List[int],
    data: Dict[str, Dict[int, float]],
    local_result: config.DegradationData,
    metric_column: str,
    category: str,
):
    """
    对单个通信组间指标（指标 1/2/3：StepDuration / {xp}_Duration / {xp}_Count）做通信域组间对比。

    规则：
    - 对每个并行域，取其各分组中"代表卡"（组内该指标值最小的有效卡），再跨组做通用检测，
      慢组映射回整个通信域组。
    - 该指标值/计数用 metric_column 取数（区别于 always 用 {xp}_Duration）。

    返回:
        True 固定（便于调用方 continue）
    """
    pp_stage_num = 1

    for name, parallel in parallels.items():
        if name == ppParallelDomainName:
            # PP 域：在 cal_detection_group（如 tp_exp 组）内比较该指标，找出慢 rank 后映射回 PP 域
            xp_detection_data = _collect_metric_data(metric_column, data)
            slow_domains, severities = get_pp_slow_communication_domains(
                parallel, cal_detection_group, xp_detection_data, config.get_comm_multiplier()
            )
            for i in range(min(len(slow_domains), len(severities))):
                group = slow_domains[i]
                if not group:
                    continue
                local_result.add_group(category, group, severities[i])
            continue

        if name == "embd":
            continue

        if not check_parallel_domain_is_exist(parallel, len(valid_ranks)):
            continue

        xp_detection_data = _collect_metric_data(metric_column, data)

        slow_domains, severities = homogenization_for_slow_communication(
            parallel, xp_detection_data, config.get_comm_multiplier(), pp_stage_num
        )

        for i in range(min(len(slow_domains), len(severities))):
            group = slow_domains[i]
            if not group:
                continue
            local_result.add_group(category, group, severities[i])

    return True


def _collect_metric_data(
    metric_column: str,
    all_data: Dict[str, Dict[int, float]]
) -> Dict[int, float]:
    """
    收集通信组间检测所用指标数据：{rank: value}。
    取 metric_column（StepDuration / {xp}_Duration / {xp}_Count）并按 -99999、0 过滤无效值。
    """
    ret = {}
    col_data = all_data.get(metric_column, {})
    if not col_data:
        return ret
    for npu_id, val in col_data.items():
        if val != -99999 and val != 0:
            ret[npu_id] = val
    return ret


def detection_all_communication_parallel(
    parallels: Dict[str, List[List[int]]],
    cal_detection_group: List[List[int]],
    valid_ranks: List[int],
    data: Dict[str, Dict[int, float]],
    local_result: config.DegradationData
) -> bool:
    """
    对所有通信域做组间对比检测，覆盖指标 1/2/3：
    - 指标2：{xp}_Duration → comm（慢通信域，沿用旧类别名）
    - 指标1：StepDuration → step_duration
    - 指标3：{xp}_Count → xp_count
    对应 Go 代码中的 detectionAllCommunicationParallel 函数（扩展到 3 个指标）。
    """
    if not parallels:
        return True

    # 双保险：无命名通信域（情况 A）时，不检测任何通信域组间指标
    if not config.get_has_named_domain():
        logger.info("[SKIP] 无通信域名，跳过通信域组间对比检测")
        return True

    # 指标2：{xp}_Duration → comm
    for name in parallels:
        if not name:
            continue
        _detect_comm_group_metric(
            {name: parallels[name]}, cal_detection_group, valid_ranks, data,
            local_result, f"{name}_Duration", "comm",
        )

    # 指标1：StepDuration → step_duration
    _detect_comm_group_metric(
        parallels, cal_detection_group, valid_ranks, data,
        local_result, stepDurationColumn, "step_duration",
    )

    # 指标3：{xp}_Count → xp_count
    for name in parallels:
        if not name:
            continue
        _detect_comm_group_metric(
            {name: parallels[name]}, cal_detection_group, valid_ranks, data,
            local_result, f"{name}_Count", "xp_count",
        )

    return True


def get_pp_slow_communication_domains(
    pp_parallel_domains: List[List[int]],
    zp_parallels: List[List[int]],
    detection_data: Dict[int, float],
    degradation_percent: float
) -> Tuple[List[List[int]], List[float]]:
    """
    获取慢 PP 通信域
    对应 Go 代码中的 getPpSlowCommunicationDomains 函数
    排除 -99999 标记的无效数据
    """
    if not zp_parallels or len(zp_parallels) == 0 or len(zp_parallels[0]) == 1:
        logger.warning("[SLOWNODE ALGO] 此时只存在 PP 通信域，无法进行同一个 stage 的聚类检测!")
        return [], []

    if not detection_data:
        logger.warning("[SLOWNODE ALGO] slow communication domains detection data is empty!")
        return [], []

    # 遍历 TP 并行域，获取慢 PP send ranks
    slow_pp_send_ranks = []
    deg_levels = []

    for zp_parallel in zp_parallels:
        # 过滤掉 -99999 的 rank，只保留有效数据
        valid_ranks = []
        valid_datas = []
        for rank in zp_parallel:
            val = detection_data.get(rank, 0)
            if val != -99999 and val != 0:
                valid_ranks.append(rank)
                valid_datas.append(val)

        # 如果有效数据不足，跳过该组
        if len(valid_ranks) < minRanksInGroup:
            continue

        slow_pp_send_ranks_tmp, rank_deg_severitys = kmeans_detector.general_anomaly_detection(
            valid_ranks, valid_datas, config.get_comm_multiplier()
        )

        slow_pp_send_ranks.extend(slow_pp_send_ranks_tmp)
        deg_levels.extend(rank_deg_severitys)

    logger.info(f"[SLOWNODE ALGO] 慢 PP 通信的 Rank: {slow_pp_send_ranks}")

    if not slow_pp_send_ranks:
        return [], []

    # 获取慢 PP 通信域
    slow_pp_communications = get_pp_slow_communicate_domains(slow_pp_send_ranks, pp_parallel_domains)

    # 返回每个通信域的最大劣化值
    degradations = find_domain_max_degradations(slow_pp_communications, slow_pp_send_ranks, deg_levels)

    return slow_pp_communications, degradations


def find_domain_max_degradations(
    slow_pp_communications: List[List[int]],
    slow_pp_send_ranks: List[int],
    deg_levels: List[float]
) -> List[float]:
    """
    返回每个通信域的最大劣化值
    对应 Go 代码中的 findDomainMaxDegradations 函数
    """
    # 构建 rank -> degradation 的映射
    rank_to_deg = {}
    for i, rank in enumerate(slow_pp_send_ranks):
        if i < len(deg_levels):
            rank_to_deg[rank] = deg_levels[i]
        else:
            logger.warning(f"警告：rank {rank} 没有对应的劣化值")
            rank_to_deg[rank] = 0.0

    result = []
    for domain in slow_pp_communications:
        if not domain:
            result.append(0.0)
            continue

        max_deg = -1.0
        found = False
        for rank in domain:
            if rank in rank_to_deg:
                found = True
                max_deg = max(max_deg, rank_to_deg[rank])

        if not found:
            max_deg = 0.0

        result.append(max_deg)

    return result
