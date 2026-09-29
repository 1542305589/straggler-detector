"""
配置模块 - 对应 Go 代码中的 config/config.go
"""

# 全局变量
FilePath = ""
OutputPath = ""  # 检测结果输出目录；为空时回退到 FilePath（单 job 场景）

# 常量
ZP_BUBBLE_ABNORMAL_BOUNDARY = 50000  # 50us

# ---- 检测阈值（skill 调用时逐组询问用户；已删除“阈值基数”设定）----
# 异常判据：簇均值 > 基线簇均值 × 阈值。每组共用一个固定倍率阈值：
#   计算类（KERNEL_AICORE / kernel_aivec） → COMPUTE_THRESHOLD（默认 1.3）
#   IO/CPU 类（cpu / memcpy_async）        → IO_THRESHOLD（默认 2.5）
#   通信类（comm / pp_comm）               → COMM_THRESHOLD（默认 1.3）
#   npu_bubble                            → 固定硬阈值 BUBBLE_THRESHOLD_NS（< 5000ns，不询问）
COMPUTE_THRESHOLD = 1.3      # 计算类阈值
IO_THRESHOLD = 2.5           # IO/CPU 类阈值
COMM_THRESHOLD = 1.3         # 通信类阈值
BUBBLE_THRESHOLD_NS = 5000   # npu_bubble 固定硬阈值（ns）

MAX_K = 10                  # 肘部法最大簇数上限
MAX_ITERATIONS = 300        # Lloyd 迭代轮数上限
RECURSION_DEPTH = 10        # 异常递归检测深度上限
CONVERGENCE_EPS = 1e-9      # 质心收敛位移阈值

# ---- 慢通信带宽检测（对应 Go DetectSlowDomainByBandwidth）----
SLOW_COMM_MIN_COUNT = 1000       # 带宽回填时算子 count 的最小值
SLOW_COMM_COUNT_FLOOR = 10240    # 检测时代表 count 的绝对下限（低于视为噪声）

# ---- PP 流水线慢通信检测（另一方案：PP 链路 Send/Recv 重叠时间）----
# PP 组从 parallel_group_info 的 pp 项读取（读不到则不检测 PP，避免把 CP/Ring Attention 误当 PP）。
# 对每个 stage 位置，把各 PP 组相邻两 stage 配成链路(s->r)；链路的指标 = 该链路收方 Recv
# 与发方 Send 算子的时间窗重叠时长（跨所有 step 求和）。跨链路做 kmeans（max 方向，
# 重叠越长→传输越慢），异常按 "发送方->接收方" 报告；阈值用 COMM_THRESHOLD。
PP_OVERLAP_COLUMN = "PP_Overlap"   # 每卡"其入边链路 Send/Recv 重叠时长"的动态列名

# 集群数据标志：由 nodelevel_data_handler 在检测时判定（Case A 集群 / Case B 非集群）
IsClusterData = False

# 是否有命名通信域名标志：由 nodelevel_data_handler 在检测时判定。
# True  → 存在命名通信域，通信域组间指标（comm）正常检测（情况 B）；
# False → 无命名通信域，通信域组间指标直接跳过，检测组退化为按 hostUid 的物理节点分组（情况 A）。
HasNamedDomain = False

# 节点信息映射：rank -> hostName（内存存储，解析阶段填充，不生成文件）
# 供 CPU 检测按物理节点分组使用
HostRankMap = {}

# 物理设备映射：rank -> {"host_name": ..., "npu_id": ...}（内存存储，解析阶段填充）
# 供报告"物理设备"列展示：hostName:Device{npu_id}
RankDeviceMap = {}

# Job 类型：training（含优化器更新）/ rollout（不含）
# 由 profilingdataparse 解析阶段根据 PYTORCH_API 中的优化器更新算子（.step/.zero_grad）判断
JobType = "unknown"


def set_host_rank_map(rank: int, host_name):
    """记录某张卡的节点 hostName"""
    HostRankMap[str(rank)] = host_name


def get_host_rank_map() -> dict:
    """获取节点映射 {rank: hostName}，可能为空"""
    return HostRankMap


def reset_host_rank_map():
    """清空节点映射（开始新一次解析前调用）"""
    HostRankMap.clear()


def set_rank_device_map(rank, host_name, npu_id):
    """记录某张卡的物理设备信息：hostName + NPU 设备 id"""
    RankDeviceMap[str(rank)] = {"host_name": host_name, "npu_id": npu_id}


def get_rank_device_map() -> dict:
    """获取物理设备映射 {rank: {"host_name": ..., "npu_id": ...}}，可能为空"""
    return RankDeviceMap


def reset_rank_device_map():
    """清空物理设备映射（开始新一次解析前调用）"""
    RankDeviceMap.clear()


def get_node_name(rank) -> str:
    """节点显示名：优先 hostName（RankDeviceMap），退化 hostUid（HostRankMap），再退化 rank{n}。"""
    info = RankDeviceMap.get(str(rank))
    if info and info.get("host_name"):
        return info["host_name"]
    uid = HostRankMap.get(str(rank))
    if uid:
        return uid
    return f"rank{rank}"


def get_node_ranks_map() -> dict:
    """返回 {节点显示名: [rank, ...]}（由 HostRankMap + RankDeviceMap 现算，不落盘）。"""
    ranks = set(HostRankMap.keys()) | set(RankDeviceMap.keys())
    m = {}
    for r in ranks:
        try:
            m.setdefault(get_node_name(int(r)), []).append(int(r))
        except (TypeError, ValueError):
            continue
    return {k: sorted(v) for k, v in m.items()}


def set_job_type(job_type: str):
    """设置 Job 类型（training/rollout）"""
    global JobType
    JobType = job_type


def get_job_type() -> str:
    """获取 Job 类型"""
    return JobType


def set_is_cluster_data(is_cluster: bool):
    """设置是否为集群数据（Case A 集群 / Case B 非集群）"""
    global IsClusterData
    IsClusterData = is_cluster


def get_is_cluster_data() -> bool:
    """获取是否为集群数据标志"""
    return IsClusterData


def set_has_named_domain(v: bool):
    """设置是否存在命名通信域标志"""
    global HasNamedDomain
    HasNamedDomain = v


def get_has_named_domain() -> bool:
    """获取是否存在命名通信域标志"""
    return HasNamedDomain


class DegradationData(dict):
    """
    劣化数据类 - 对应 Go 中的 DegradationData 类型
    结构：map[string]map[string]float64
    例如：{"KERNEL_AICORE": {"0": 1.5, "1": 2.0}, "comm": {"0,1": 1.8}}
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _single_key(rank: int) -> str:
        """将单个 rank 转为字符串 key"""
        return str(rank)

    @staticmethod
    def _group_key(ranks: list) -> str:
        """将 rank 列表转为排序后的字符串 key"""
        if not ranks:
            return ""
        sorted_ranks = sorted(ranks)
        return ",".join(str(r) for r in sorted_ranks)

    def add_single(self, category: str, rank: int, degradation: float):
        """添加单个 rank 的劣化数据"""
        if category not in self:
            self[category] = {}
        key = self._single_key(rank)
        self[category][key] = degradation

    def add_group(self, category: str, ranks: list, degradation: float):
        """添加一组 rank 的劣化数据"""
        if not ranks:
            return
        if category not in self:
            self[category] = {}
        key = self._group_key(ranks)
        # 如果已存在，保留较大的劣化值
        if key in self[category]:
            self[category][key] = max(self[category][key], degradation)
        else:
            self[category][key] = degradation


def set_file_path(path: str):
    """设置文件路径（输入数据目录）"""
    global FilePath
    FilePath = path


def get_file_path() -> str:
    """获取输入数据目录"""
    return FilePath


def set_output_path(path: str):
    """设置检测结果输出目录（多 job 场景下独立于输入目录）"""
    global OutputPath
    OutputPath = path


def get_output_path() -> str:
    """
    获取检测结果输出目录。
    未显式设置时回退到输入数据目录（FilePath），保证单 job 场景向后兼容。
    """
    return OutputPath if OutputPath else FilePath


def set_thresholds(compute: float = 1.3, io: float = 2.5, comm: float = 1.3):
    """设置检测阈值（skill 调用时询问用户后调用）"""
    global COMPUTE_THRESHOLD, IO_THRESHOLD, COMM_THRESHOLD
    COMPUTE_THRESHOLD = compute
    IO_THRESHOLD = io
    COMM_THRESHOLD = comm


def get_compute_threshold() -> float:
    """计算类（KERNEL_AICORE / kernel_aivec）阈值"""
    return COMPUTE_THRESHOLD


def get_io_threshold() -> float:
    """IO/CPU 类（cpu / memcpy_async）阈值"""
    return IO_THRESHOLD


def get_comm_threshold() -> float:
    """通信类（comm / pp_comm）阈值"""
    return COMM_THRESHOLD


# 类别 → 阈值组
COMPUTE_CATEGORIES = ("KERNEL_AICORE", "kernel_aivec")
IO_CATEGORIES = ("cpu", "memcpy_async")
COMM_CATEGORIES = ("comm", "pp_comm")


def get_threshold_for_category(category: str) -> float:
    """返回某检测类别所属组的阈值（npu_bubble 走固定硬阈值，不在此函数内）。"""
    if category in IO_CATEGORIES:
        return IO_THRESHOLD
    if category in COMM_CATEGORIES:
        return COMM_THRESHOLD
    return COMPUTE_THRESHOLD
