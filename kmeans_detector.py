"""
通用检测算法模块 - 新核心
对应慢节点检测算法第六节描述的 KMeans + Z-score + 肘部法通用检测算法。

取代旧 spacedetector.py 的递归间隙聚类（齐次化 / homogenization）。

算法流程：
1. 过滤 ≤0（及 -99999）的值；过滤后不足 2 个 → 无异常退出。
2. Z-score 标准化；标准差 ≈ 0 → 强制置 1 避免除零。
3. 肘部法选最优 K：K=2..min(n,MAX_K)，算 inertia（簇内平方和），取二阶差分最大者。
4. KMeans++ 初始化质心：首质心 = data[0]，后续 D² 加权随机采样（种子 RNG 保复现）。
5. Lloyd 迭代 ≤MAX_ITERATIONS 轮：最近质心分配 → 质心=簇均值；
   空簇质心放到离其分配质心最远的样本；收敛 = 质心位移 < eps 且无分配变化。
6. 识别异常簇：按原始值均值降序，基线 = 最小均值簇；簇均值 > 基线×倍率 → 该簇异常。
7. 无异常簇 → 无异常退出。
8. 逐轮剥离：把本轮异常簇数据**剔除**，对**剩余数据**回到步骤 2（轮数 ≤ MAX_DEPTH）；
   各轮按**当轮基线**判断是否异常（剥掉大值后基线单调不增，逐轮可检出更细微的离群点）。
9. 返回全部轮的异常簇（映射回 rank）。检测判断各用当轮基线；
   劣化指数统一用**最后一次得到的基线簇**（最严格地板）作为分母，
   degradation = value / 最后基线均值，使所有轮检出的异常劣化在同一刻度上可比。

纯 Python 实现（仅 math/random），保持模块零依赖、可复现。
"""

from typing import List, Tuple, Optional
import math
import random


INVALID_MARKER = -99999


def general_anomaly_detection(
    ranks: List[int],
    values: List[float],
    anomaly_multiplier: float = 2.0,
    max_k: int = 10,
    max_iter: int = 300,
    max_depth: int = 10,
    convergence_eps: float = 1e-9,
    seed: int = 42,
) -> Tuple[List[int], List[float]]:
    """
    通用慢节点检测算法（新核心，始终 max 方向：值偏大视为异常）。

    参数:
        ranks: 卡的 Rank 号列表，如 [1, 5, 9, 13]
        values: 与 ranks 对应的指标值列表，如 [10, 20, 10, 100]
        anomaly_multiplier: 异常倍率（默认 2.0）
        max_k: 肘部法最大簇数
        max_iter: Lloyd 最大迭代轮数
        max_depth: 递归深度上限
        convergence_eps: 收敛位移阈值
        seed: 随机种子（保复现）

    返回:
        (异常 rank 列表, 各异常 rank 的劣化程度列表)

    劣化指数说明：
        异常判断各轮用当轮基线（剥掉大值后基线单调不增，逐步检出更细微离群点）；
        但劣化指数统一用“最后一次得到的基线簇”（最严格地板）作为分母，
        使所有轮检出的异常劣化在同一刻度上可比。
    """
    if not values or len(values) < 2:
        return [], []

    # 确保 ranks 与 values 等长
    if len(ranks) != len(values):
        n = min(len(ranks), len(values))
        ranks = list(ranks[:n])
        values = list(values[:n])

    rng = random.Random(seed)

    # 递归逐轮剥离异常簇，携带原始索引（映射回 ranks）。
    # 每轮中先将样本归一为“携带原始索引”，每轮返回 (当轮基线均值, 当轮异常原始索引)；
    # 累积全部轮结果，最后映射回 ranks 并计算劣化程度。
    rounds = _recurse_anomaly(
        list(values), list(range(len(values))), 0,
        anomaly_multiplier, max_k, max_iter, max_depth, convergence_eps, rng,
    )

    if not rounds:
        return [], []

    # 劣化指数统一用“最后一次得到的基线簇”（最严格地板）作为分母。
    # 剥离只删大值簇，各轮基线均值单调不增，故 rounds[-1][0] 即最小、最严格的基线。
    # 这样所有轮检出的异常劣化都在同一刻度上可比。
    global_denom = rounds[-1][0]
    if not global_denom or global_denom <= 0:
        global_denom = 1.0

    anomaly_ranks = []
    degradations = []
    for _, anomaly_indices in rounds:
        for idx in anomaly_indices:
            if 0 <= idx < len(ranks):
                anomaly_ranks.append(ranks[idx])
                value = values[idx]
                degradations.append(value / global_denom)
            else:
                degradations.append(1.0)

    return anomaly_ranks, degradations


def _recurse_anomaly(
    data: List[float],
    indices: List[int],
    depth: int,
    anomaly_multiplier: float,
    max_k: int,
    max_iter: int,
    max_depth: int,
    convergence_eps: float,
    rng: random.Random,
) -> List[Tuple[float, List[int]]]:
    """
    递归异常检测核心（逐轮剥离异常簇）。

    每轮：对当前样本做一次 KMeans 检出异常簇，然后**剔除**这些异常数据，
    对**剩余数据**再次 KMeans 聚类，直到剩余数据无异常或达到轮数上限（max_depth）。

    返回:
        每轮结果列表，元素为 (该轮基线簇均值, 该轮异常原始索引列表)，累积全部轮。
    """
    # 1. 过滤 ≤0 及 -99999
    valid_data = []
    valid_indices = []
    for v, i in zip(data, indices):
        if v > 0 and v != INVALID_MARKER:
            valid_data.append(v)
            valid_indices.append(i)

    if len(valid_data) < 2:
        # 剩余样本不足 2 → 无法继续聚类，无更多异常
        return []

    # 一次 KMeans 检测：返回异常数据值及其原始索引
    anomaly_vals, anomaly_ids, baseline_mean = _kmeans_anomaly_detect(
        valid_data, valid_indices, anomaly_multiplier, max_k, max_iter, convergence_eps, rng,
    )

    if not anomaly_vals:
        # 7. 无异常簇 → 无异常退出
        return []

    # 本轮异常
    rounds = [(baseline_mean, list(anomaly_ids))]

    # 达到轮数上限 → 不再向下剥离
    if depth >= max_depth:
        return rounds

    # 8. 剔除异常簇数据，对剩余数据再次聚类
    anom_set = set(anomaly_ids)
    remaining_data = [v for v, i in zip(valid_data, valid_indices) if i not in anom_set]
    remaining_indices = [i for i in valid_indices if i not in anom_set]

    if len(remaining_data) >= 2:
        rounds.extend(_recurse_anomaly(
            remaining_data, remaining_indices, depth + 1,
            anomaly_multiplier, max_k, max_iter, max_depth, convergence_eps, rng,
        ))

    return rounds


def _kmeans_anomaly_detect(
    data: List[float],
    indices: List[int],
    anomaly_multiplier: float,
    max_k: int,
    max_iter: int,
    convergence_eps: float,
    rng: random.Random,
) -> Tuple[List[float], List[int], float]:
    """
    单轮 KMeans 聚类并识别异常簇。

    返回:
        (异常数据值列表, 异常数据对应的原始索引列表, 基线簇均值)
    """
    n = len(data)
    if n < 2:
        return [], [], 0.0

    # 2. Z-score 标准化
    zdata = _zscore(data, convergence_eps)

    # 3. 肘部法选 K
    k = _elbow_optimal_k(zdata, max_k, max_iter, convergence_eps, rng)

    # 4-5. KMeans++ 初始化 + Lloyd 迭代
    labels, centers = _kmeans(zdata, k, max_iter, convergence_eps, rng)

    # 6. 识别异常簇
    #    计算每个簇的原始值均值，按降序排列，基线 = 最小均值簇
    cluster_means = {}
    for c in range(k):
        members = [data[j] for j in range(n) if labels[j] == c]
        if members:
            cluster_means[c] = sum(members) / len(members)

    if not cluster_means:
        return [], [], 0.0

    baseline_mean = min(cluster_means.values())

    # 按簇均值降序遍历，遇到第一个不满足(>基线×倍率)则停止
    anomaly_clusters = []
    for c in sorted(cluster_means, key=lambda x: cluster_means[x], reverse=True):
        if cluster_means[c] > baseline_mean * anomaly_multiplier:
            anomaly_clusters.append(c)
        else:
            break

    if not anomaly_clusters:
        return [], [], baseline_mean

    anomaly_vals = []
    anomaly_ids = []
    for j in range(n):
        if labels[j] in anomaly_clusters:
            anomaly_vals.append(data[j])
            anomaly_ids.append(indices[j])

    return anomaly_vals, anomaly_ids, baseline_mean


def _zscore(data: List[float], eps: float) -> List[float]:
    """Z-score 标准化；标准差 ≈ 0 → 强制置 1 避免除零。"""
    if not data:
        return []
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / len(data)
    std = math.sqrt(var)
    if std < eps:
        std = 1.0
    return [(x - mean) / std for x in data]


def _elbow_optimal_k(
    data: List[float],
    max_k: int,
    max_iter: int,
    convergence_eps: float,
    rng: random.Random,
) -> int:
    """肘部法选最优 K：K=2..min(n,max_k)，取二阶差分最大的 K；退化为 2。"""
    n = len(data)
    upper = min(n, max_k)
    if upper < 2:
        return 2

    inertias = {}
    for k in range(2, upper + 1):
        _, _, inertia = _kmeans(data, k, max_iter, convergence_eps, rng, want_inertia=True)
        inertias[k] = inertia

    if len(inertias) < 2:
        return 2

    # 二阶差分最大
    ks = sorted(inertias.keys())
    scores = {}
    for i in range(1, len(ks) - 1):
        prev, cur, nxt = ks[i - 1], ks[i], ks[i + 1]
        d1 = inertias[cur] - inertias[prev]
        d2 = inertias[nxt] - inertias[cur]
        scores[cur] = d1 - d2

    if not scores:
        return 2

    return max(scores, key=scores.get)


def _kmeans(
    data: List[float],
    k: int,
    max_iter: int,
    convergence_eps: float,
    rng: random.Random,
    want_inertia: bool = False,
):
    """
    KMeans 聚类（KMeans++ 初始化 + Lloyd 迭代）。

    返回:
        (labels, centers) 或 (labels, centers, inertia)（want_inertia=True）
    """
    n = len(data)
    if k <= 1 or n < 2:
        labels = [0] * n
        centers = [sum(data) / n]
        if want_inertia:
            return labels, centers, 0.0
        return labels, centers
    if k > n:
        k = n

    # KMeans++ 初始化质心索引
    init_idx = _kmeans_pp_seed(data, k, rng)
    centers = [data[i] for i in init_idx]

    labels = [0] * n
    for _ in range(max_iter):
        # 分配最近质心
        new_labels = []
        assign_changed = False
        for x in data:
            best = 0
            best_d = float('inf')
            for c_idx, c in enumerate(centers):
                d = (x - c) ** 2
                if d < best_d:
                    best_d = d
                    best = c_idx
            new_labels.append(best)
            if best != labels[len(new_labels) - 1]:
                assign_changed = True

        # 更新质心 = 簇均值
        sums = [0.0] * k
        cnts = [0] * k
        for j, x in enumerate(data):
            c = new_labels[j]
            sums[c] += x
            cnts[c] += 1

        new_centers = []
        for c in range(k):
            if cnts[c] > 0:
                new_centers.append(sums[c] / cnts[c])
            else:
                # 空簇：质心放到离其分配质心最远的样本
                farthest_j = -1
                farthest_d = -1.0
                for j, x in enumerate(data):
                    d = (x - centers[c]) ** 2
                    if d > farthest_d:
                        farthest_d = d
                        farthest_j = j
                if farthest_j >= 0:
                    new_centers.append(data[farthest_j])
                else:
                    new_centers.append(centers[c])

        # 收敛判断：质心位移 < eps 且无分配变化
        shift = max(abs(new_centers[c] - centers[c]) for c in range(k))
        centers = new_centers
        labels = new_labels
        if shift < convergence_eps and not assign_changed:
            break

    if want_inertia:
        inertia = 0.0
        for j, x in enumerate(data):
            inertia += (x - centers[labels[j]]) ** 2
        return labels, centers, inertia

    return labels, centers


def _kmeans_pp_seed(data: List[float], k: int, rng: random.Random) -> List[int]:
    """
    KMeans++ 初始化：首个质心 = data[0]，后续通过 D² 加权随机采样。
    返回质心在 data 中的索引列表。
    """
    n = len(data)
    indices = [0]  # 首质心 = data[0]（规范）
    centers = [data[0]]

    while len(indices) < k:
        # 计算每个点到最近质心的距离平方
        dist_sq = []
        total = 0.0
        for x in data:
            min_d = min((x - c) ** 2 for c in centers)
            dist_sq.append(min_d)
            total += min_d

        if total <= 0:
            # 所有点已与质心重合，随便选一个未选过的点
            for i in range(n):
                if i not in indices:
                    indices.append(i)
                    centers.append(data[i])
                    break
            else:
                break
            continue

        # D² 加权随机采样
        r = rng.random() * total
        acc = 0.0
        chosen = -1
        for i, d in enumerate(dist_sq):
            acc += d
            if acc >= r:
                chosen = i
                break
        if chosen < 0:
            chosen = n - 1

        if chosen not in indices:
            indices.append(chosen)
            centers.append(data[chosen])

    return indices[:k]
