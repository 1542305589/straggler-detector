# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#!/usr/bin/env python3
"""结果分析层入口：读数据目录 → evidence.json → formal_result.json + diagnosis_report.md。

用法:
    python diagnose.py path=/path/to/data [out=/path/to/output]
                        [compute=1.3] [io=2.5] [comm=1.3] [full-cluster] [json]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnosis import evidence, pipeline, render  # noqa: E402


def parse_args(args):
    out = {
        "path": None,
        "out": None,
        "compute": 1.3,
        "io": 2.5,
        "comm": 1.3,
        "json": False,
        "full_cluster": False,
        "clean": "ask",
        "no_align": False,
        "no_ask": False,
    }
    for arg in args:
        if arg == "json":
            out["json"] = True
            continue
        if arg == "full-cluster":
            out["full_cluster"] = True
            continue
        if arg == "no-align":
            out["no_align"] = True
            continue
        if arg == "no-ask":
            out["no_ask"] = True
            continue
        if "=" not in arg:
            print(f"[WARN] 忽略未知参数: {arg}", file=sys.stderr)
            continue
        k, v = arg.split("=", 1)
        if k in ("path", "out"):
            out[k] = v
        elif k == "clean" and v in ("yes", "no", "ask"):
            out[k] = v
        elif k == "no-align" and v in ("true", "yes", "1"):
            out["no_align"] = True
        elif k in ("compute", "io", "comm"):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def _threshold_ask_keys(level: str) -> tuple:
    if level == "cluster":
        return ("comm",)
    if level == "levelnone":
        return ("io", "comm")
    return ("compute", "io", "comm")


def _ensure_detection(path: str, out_dir: str, thresholds: tuple, detection: dict,
                      clean_decision: bool = False, force: bool = False) -> dict:
    """没有检测结果（或用户选择清理）时先跑完整检测（复用 main 的检测流程）。"""
    if detection and not force:
        return detection
    try:
        import shutil
        import config
        import nodelevel_data_handler
        import main as detect_main
        level = nodelevel_data_handler.detect_profiling_level(path)
        if level == "cluster":
            return detection
        os.makedirs(out_dir, exist_ok=True)
        config.set_file_path(path)
        # 检测的 op_metric 在数据目录；检测结果写到数据目录后再复制到 out_dir
        config.set_output_path("")
        config.set_profiling_level(level)
        config.set_thresholds(*thresholds)
        try:
            detect_main._process_single_job(
                path, thresholds, "no", output_path=None,
                skip_confirm=True, clean_decision=clean_decision)
        except Exception:
            detect_main._process_single_job(
                path, thresholds, "no", output_path=None,
                skip_confirm=True, clean_decision=True)
        src = os.path.join(path, "straggler_detection_result.json")
        dst = os.path.join(out_dir, "straggler_detection_result.json")
        # out_dir 默认就是数据目录本身，同路径时不能 copy2（会抛 SameFileError）
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return evidence.load_detection(path, out_dir)
    except Exception as e:
        print(f"[WARN] 检测未运行：{e}", file=sys.stderr)
        return {}


def main():
    args = parse_args(sys.argv[1:])
    path = args["path"]
    if not path:
        print("用法: python diagnose.py path=<数据目录> [out=<输出目录>] [compute=1.3] [io=2.5] [comm=1.3] [clean=yes|no|ask] [no-align] [no-ask] [full-cluster]", file=sys.stderr)
        return 1
    if not os.path.exists(path):
        print(f"路径不存在: {path}", file=sys.stderr)
        return 1

    # 输出与其余产物同级（默认落到数据目录自身，不再单建 analysis_result/ 子目录）
    out_dir = args["out"] or (
        path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    )
    import config
    import nodelevel_data_handler
    import utils

    level = nodelevel_data_handler.detect_profiling_level(path)
    thresholds = (args["compute"], args["io"], args["comm"])
    if not args["no_ask"]:
        thresholds = utils.confirm_thresholds(
            thresholds, keys=_threshold_ask_keys(level))
        clean_decision = utils.confirm_clean(path)
        no_align = (utils.confirm_no_align()
                    if level in ("levelnone", "level0") else False)
    else:
        clean_decision = args["clean"] == "yes"
        no_align = args["no_align"]

    config.set_file_path(path)
    config.set_output_path(out_dir)
    config.set_profiling_level(level)
    config.set_thresholds(*thresholds)
    config.set_comm_no_align(no_align)

    detection = evidence.load_detection(path, out_dir)
    detection = _ensure_detection(
        path, out_dir, thresholds, detection,
        clean_decision=clean_decision, force=clean_decision and bool(detection))
    print("## 检测结果")
    print(json.dumps(detection, ensure_ascii=False, indent=2))
    print("")
    ev, result, paths, errors, warnings = pipeline.run_diagnosis(
        path, out_dir,
        {"compute": thresholds[0], "io": thresholds[1], "comm": thresholds[2]},
        detection=detection, full_cluster=args["full_cluster"],
    )
    for w in warnings:
        print(f"[WARN] {w}", file=sys.stderr)
    for e in errors:
        print(f"[ERROR] {e}", file=sys.stderr)

    if args["json"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render.console_evidence(ev, result))
        print("")
        det_result = next((p for p in (
            os.path.join(out_dir, "straggler_detection_result.json"),
            os.path.join(path, "straggler_detection_result.json"),
        ) if os.path.isfile(p)), "")
        det_report = next((p for p in (
            os.path.join(out_dir, "detection_report.log"),
            os.path.join(path, "detection_report.log"),
            os.path.join(out_dir, "analysis_result", "detection_report.log"),  # 旧布局兜底
            os.path.join(path, "analysis_result", "detection_report.log"),
        ) if os.path.isfile(p)), "")
        for line in render.format_artifacts([
            ("检测结果", det_result),
            ("检测报告", det_report),
            ("诊断报告", paths["report"]),
            ("完整证据", paths["evidence"]),
            ("诊断结论", paths["result"]),
        ]):
            print(line)
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())