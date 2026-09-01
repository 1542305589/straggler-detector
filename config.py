"""
配置模块 - 对应 Go 代码中的 config/config.go
"""

# 全局变量
FilePath = ""
OutputPath = ""  # 检测结果输出目录；为空时回退到 FilePath（单 job 场景）

# 常量
ZP_BUBBLE_ABNORMAL_BOUNDARY = 50000  # 50us

# ---- 新核心检测算法（KMeans + Z-score + 肘部法）参数 ----
# 异常判据：簇均值 > 基线簇均值 × 倍率（倍率 = 1 + 放缩倍数 × degradation）。
# 不再使用固定 2.0 倍率；degradation 由运行时提问用户确定。
Degradation = 0.3            # 劣化阈值基础值（运行时提问，未提问时用默认 0.3）

# 放缩倍数分组：
# - 计算/IO/Host 类指标（KERNEL_AICORE, kernel_aivec, memcpy_async, cpu, host_duration）→ 倍率 = 1 + degradation
# - 通信域类指标（step_duration, comm, xp_count）→ 倍率 = 1 + 5*degradation
Utilization_ComputeMultiplier = 0.0   # 计算类倍率 = 1 + 1*degradation（运行时 set_thresholds 计算）
Utilization_CommMultiplier = 0.0      # 通信类倍率 = 1 + 5*degradation（运行时 set_thresholds 计算）
CALC_MULTIPLIER_BASE = 1.0    # 计算/IO/Host 类放缩倍数基数
COMM_MULTIPLIER_BASE = 5.0    # 通信类放缩倍数基数

MAX_K = 10                  # 肘部法最大簇数上限
MAX_ITERATIONS = 300        # Lloyd 迭代轮数上限
RECURSION_DEPTH = 10        # 异常递归检测深度上限
CONVERGENCE_EPS = 1e-9      # 质心收敛位移阈值

# 集群数据标志：由 nodelevel_data_handler 在检测时判定（Case A 集群 / Case B 非集群）
IsClusterData = False

# 是否有命名通信域名标志：由 nodelevel_data_handler 在检测时判定。
# True  → 存在命名通信域，通信域组间指标（comm/step_duration/xp_count）正常检测（情况 B）；
# False → 无命名通信域，通信域组间指标直接跳过，检测组退化为按 hostUid 的物理节点分组（情况 A）。
HasNamedDomain = False

# 节点信息映射：rank -> hostName（内存存储，解析阶段填充，不生成文件）
# 供 CPU 检测按物理节点分组使用
HostRankMap = {}

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


def set_thresholds(degradation: float):
    """根据 degradation 设置阈值"""
    global Degradation, Utilization_ComputeMultiplier, Utilization_CommMultiplier
    Degradation = degradation
    Utilization_ComputeMultiplier = 1 + CALC_MULTIPLIER_BASE * degradation
    Utilization_CommMultiplier = 1 + COMM_MULTIPLIER_BASE * degradation


def get_compute_multiplier() -> float:
    """计算/IO/Host 类指标的异常倍率（1 + degradation）"""
    return Utilization_ComputeMultiplier if Utilization_ComputeMultiplier > 0 else 1 + CALC_MULTIPLIER_BASE * Degradation


def get_comm_multiplier() -> float:
    """通信域类指标的异常倍率（1 + 5*degradation）"""
    return Utilization_CommMultiplier if Utilization_CommMultiplier > 0 else 1 + COMM_MULTIPLIER_BASE * Degradation
