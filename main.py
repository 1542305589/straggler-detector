#!/usr/bin/env python3
"""
Slow Node Detection - Python 版本主入口
对应 Go 代码中的 main.go 和 export_interface.go

用法:
    python main.py path=/xxx degradation=0.3
    或作为 skill 被 Claude Code 调用
"""

import os
import sys
import logging
from typing import Dict, Any, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import profilingdataparse
import nodelevel
import nodelevel_data_handler
import utils
import visualizer
import joint_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger("[SLOWNODE ALGO]")


def parse_args(args: list) -> Dict[str, Any]:
    """
    解析命令行参数
    对应 Go 代码中的 main 函数参数解析逻辑

    支持的参数:
        path=xxx          - 数据目录路径（必需）
        degradation=xxx   - 劣化阈值（可选，默认 0.3）
    """
    result = {
        'path': None,
        'degradation': 0.3,  # 默认值
        'clean': 'ask'       # ask / yes / no
    }

    for arg in args:
        if '=' not in arg:
            continue  # 忽略无效参数

        parts = arg.split('=', 1)
        if len(parts) != 2:
            continue

        key, val = parts[0], parts[1]

        if key == 'path':
            result['path'] = val
        elif key == 'degradation':
            try:
                f = float(val)
                if f > 0:
                    result['degradation'] = f
                else:
                    logger.warning(f"Invalid degradation value '{val}', using default 0.3")
            except ValueError:
                logger.warning(f"Invalid degradation value '{val}', using default 0.3")
        elif key == 'clean':
            if val.lower() in ('yes', 'no', 'ask'):
                result['clean'] = val.lower()
            else:
                logger.warning(f"Invalid clean value '{val}', using default 'ask'")

    return result


def _find_job_subdirs(parent_path: str) -> List[str]:
    """
    查找父目录下直接包含 ascend_pytorch_profiler_*.db 文件的子目录（作为独立 job）。

    用于“父目录含多 job 子目录”场景，逐个子目录执行检测。

    返回:
        满足条件的子目录绝对路径列表（不包含父目录本身直接含 db 的情况）
    """
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
    """判断目录自身是否直接包含 ascend_pytorch_profiler_*.db 文件"""
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(
        f.startswith("ascend_pytorch_profiler_") and f.endswith(".db")
        for f in names
    )


def _safe_print(text: str):
    """向 stdout 打印汇总表（兼容 Windows GBK 等非 UTF-8 控制台，避免 UnicodeEncodeError）"""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
        except Exception:
            sys.stdout.write(text.encode("ascii", errors="replace").decode("ascii"))


def _process_single_job(job_path: str, degradation: float, clean_mode: str, output_path: str = None):
    """
    对单个 job 目录执行完整检测流程（解析 → 并行域 → 定界检测 → 结果 → 报告 → 可视化）。
    对应原 main() 的主体逻辑，供多 job 场景逐个调用。

    参数:
        job_path: 输入数据目录（含原始 db）
        output_path: 检测结果输出目录；为 None 时输出到 job_path 自身（单 job 场景）
    """
    # 设置全局配置（含 job 类型检测所需的 FilePath）
    config.set_file_path(job_path)
    if output_path:
        config.set_output_path(output_path)

    # 总是向用户提问劣化阈值 degradation（传入值作为默认/回退）
    degradation = utils.confirm_degradation(degradation)
    config.set_thresholds(degradation)

    logger.info(f"开始慢节点检测 - 路径：{job_path}, 劣化阈值：{degradation}")
    logger.info(f"检测结果输出目录：{config.get_output_path()}")

    # 检测结果清理与输出目录（output_path），而非原始数据目录
    out_path = config.get_output_path()

    # === 步骤 1: 清理/解析（根据 clean 模式） ===
    if clean_mode == 'yes':
        logger.info("开始 Profiling 数据解析（clean=yes，强制重新解析）...")
        utils.clean_detection_outputs(out_path)
        profilingdataparse.data_parsing(job_path)
        logger.info("数据解析完成")
    elif clean_mode == 'no':
        logger.info("跳过清理和重新解析（clean=no），直接使用已有的 op_metric 数据")
    else:
        # clean_mode == 'ask': 交互式询问
        should_clean = utils.confirm_clean(out_path)
        if should_clean:
            logger.info("开始 Profiling 数据解析...")
            utils.clean_detection_outputs(out_path)
            profilingdataparse.data_parsing(job_path)
            logger.info("数据解析完成")
        else:
            logger.info("跳过清理和重新解析，直接使用已有的 op_metric 数据")

    # === 步骤 2: 获取并行域和有效 ranks ===
    parallels, valid_ranks = nodelevel_data_handler.get_cur_detection_info(job_path)

    if not parallels or not valid_ranks:
        logger.error("获取并行域/卡数失败")
        return {}

    logger.info(f"并行域：{list(parallels.keys())}")
    logger.info(f"有效 Ranks: {valid_ranks}")

    # === 步骤 3: 获取最新 step 数据 ===
    last_step_data = nodelevel_data_handler.get_cur_job_last_step_data(valid_ranks)

    # === 步骤 4: 执行定界检测 ===
    result = nodelevel.delimit_detection(last_step_data, parallels, valid_ranks)

    # === 步骤 5: 输出结果 ===
    # 始终输出 JSON 结果文件（即使没有异常）
    utils.write_result(result, parallels)

    # === 步骤 5.1: 故障联合分析报告 ===
    joint_analysis.generate_joint_report(result, parallels, last_step_data, config.get_output_path())

    # === 步骤 6: 可视化 ===
    if last_step_data:
        visualizer.run_visualization(
            last_step_data, parallels, valid_ranks, config.get_output_path(),
            detection_result=result, degradation=degradation,
            data_path=job_path,
        )

    if not result:
        logger.info("未检测到异常节点")

    # === 步骤 7: 最终输出逐类别汇总表（渲染到调用方 agent 的 stdout） ===
    try:
        summary = joint_analysis.build_summary_table(
            result, parallels, last_step_data, degradation)
        _safe_print(summary)
    except Exception as e:  # 汇总表为附加展示，失败不应中断检测
        logger.warning(f"生成汇总表失败：{e}")

    return result


def main():
    """
    主入口函数
    对应 Go 代码中的 main 函数
    """
    args = sys.argv[1:]

    if len(args) < 1:
        logger.error("[SLOWNODE ALGO] Usage: python main.py path=/your/data/dir [degradation=0.3]")
        sys.exit(1)

    # 解析参数
    parsed = parse_args(args)
    input_path = parsed['path']
    degradation = parsed['degradation']
    clean_mode = parsed['clean']

    if not input_path:
        logger.error("[SLOWNODE ALGO] Missing required parameter: path=/your/data/dir")
        sys.exit(1)

    # 校验目录是否存在
    if not os.path.isdir(input_path):
        logger.error(f"[SLOWNODE ALGO] Invalid directory: {input_path}")
        sys.exit(1)

    # 校验 degradation 范围
    if degradation < 0:
        logger.warning("[WARN] Degradation threshold cannot be negative. Reset to default 0.3.")
        degradation = 0.3
    elif degradation > 1:
        logger.warning("[WARN] Degradation threshold is greater than 1. Please verify if this is intentional.")

    # === 判断是否为“父目录含多 job 子目录”场景 ===
    sub_jobs = _find_job_subdirs(input_path)
    parent_has_db = _dir_has_db_directly(input_path)

    if sub_jobs and not parent_has_db:
        logger.info(f"[SLOWNODE ALGO] 检测到 {len(sub_jobs)} 个子 job，逐个执行检测：{sub_jobs}")
        # 检测结果统一输出到父目录下的独立结果目录，原始数据目录（含 db）不写入任何结果
        output_root = os.path.join(input_path, "detection_output")
        results = {}
        for job in sub_jobs:
            logger.info(f"===== 处理 job：{job} =====")
            job_output = os.path.join(output_root, os.path.basename(job))
            results[job] = _process_single_job(job, degradation, clean_mode, output_path=job_output)
        return results

    # === 默认：单个 job 目录直接处理（输出到输入目录自身，向后兼容） ===
    config.set_output_path("")
    return _process_single_job(input_path, degradation, clean_mode)


def run_detection(input_path: str, degradation: float = 0.3, skip_parsing: bool = False, clean: str = 'ask') -> Dict[str, Dict[str, float]]:
    """
    作为 skill 被调用的入口函数

    参数:
        input_path: 数据目录路径
        degradation: 劣化阈值（默认 0.3）
        skip_parsing: 是否跳过清理和重新解析，直接使用已有的 op_metric 数据（默认 False）
                     当 skip_parsing=True 时相当于 clean='no'
        clean: 清理模式（'ask' / 'yes' / 'no'），仅在 skip_parsing=False 时生效
               'ask' - 交互式询问（默认，需终端 TTY）
               'yes' - 强制清理并重新解析
               'no'  - 跳过清理，使用已有数据

    返回:
        检测结果：{"KERNEL_AICORE": {"0": 1.5}, "comm": {"0,1": 1.8}, "cpu": {"5": 2.1}}
    """
    # 设置全局配置（先设置，后续步骤可能用到）
    config.set_file_path(input_path)
    # 总是向用户提问劣化阈值 degradation（传入值作为默认/回退）
    degradation = utils.confirm_degradation(degradation)
    config.set_thresholds(degradation)

    if skip_parsing:
        logger.info("跳过数据解析，直接使用已有的 op_metric 数据")
    else:
        if clean == 'yes':
            # 强制清理并重新解析
            utils.clean_detection_outputs(input_path)
            profilingdataparse.data_parsing(input_path)
        elif clean == 'no':
            logger.info("跳过清理和重新解析（clean=no），直接使用已有的 op_metric 数据")
        else:
            # clean == 'ask': 交互式询问
            should_clean = utils.confirm_clean(input_path)
            if should_clean:
                utils.clean_detection_outputs(input_path)
                profilingdataparse.data_parsing(input_path)
            else:
                logger.info("跳过清理和重新解析，直接使用已有的 op_metric 数据")

    logger.info(f"开始慢节点检测 - 路径：{input_path}, 劣化阈值：{degradation}")

    # 步骤 2: 获取并行域和有效 ranks
    parallels, valid_ranks = nodelevel_data_handler.get_cur_detection_info(input_path)

    if not parallels or not valid_ranks:
        logger.error("获取并行域/卡数失败")
        return {}

    # 步骤 3: 获取最新 step 数据
    last_step_data = nodelevel_data_handler.get_cur_job_last_step_data(valid_ranks)

    # 步骤 4: 执行定界检测
    result = nodelevel.delimit_detection(last_step_data, parallels, valid_ranks)

    # 步骤 5: 输出结果
    # 始终输出 JSON 结果文件（即使没有异常）
    utils.write_result(result, parallels)

    # 步骤 5.1: 故障联合分析报告
    joint_analysis.generate_joint_report(result, parallels, last_step_data, input_path)

    # 步骤 6: 可视化
    if last_step_data:
        visualizer.run_visualization(
            last_step_data, parallels, valid_ranks, input_path,
            detection_result=result, degradation=degradation,
            data_path=input_path,
        )

    if not result:
        logger.info("未检测到异常节点")

    # 步骤 7: 最终输出逐类别汇总表（渲染到调用方 agent 的 stdout）
    try:
        summary = joint_analysis.build_summary_table(
            result, parallels, last_step_data, degradation)
        _safe_print(summary)
    except Exception as e:  # 汇总表为附加展示，失败不应中断检测
        logger.warning(f"生成汇总表失败：{e}")

    return result


if __name__ == "__main__":
    main()
