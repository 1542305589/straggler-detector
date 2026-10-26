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
"""诊断编排：取证 → 规则 → 校验门 → 落盘。

`main.py`（检测后自动诊断）与 `diagnose.py`（独立入口）共用这一份编排，
避免两处各写一遍——此前校验门只在 `diagnose.py` 那条路径上生效。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import evidence, render, rules, validate


def run_diagnosis(data_dir: str, out_dir: str, thresholds: Dict[str, float],
                  detection: Optional[Dict] = None,
                  full_cluster: bool = False
                  ) -> Tuple[Dict, Dict, Dict[str, str], list, list]:
    """执行一次完整诊断。

    参数：
        data_dir：数据目录，或 cluster 单文件 db。
        out_dir：产物输出目录（`evidence.json` / `formal_result.json` /
            `diagnosis_report.md` 均写在这里）。
        thresholds：`{"compute": .., "io": .., "comm": ..}`。
        detection：调用方已持有的检测结果。磁盘上的 `straggler_detection_result.json`
            优先，读不到时回退到该值；两者都为空则按空检测结果走 analysis_fallback。
        full_cluster：cluster 单文件是否做逐 op 等待分解（较慢）。

    返回：
        (evidence, formal_result, paths, errors, warnings)。

    副作用：校验门结果写入 `formal_result["validation"]`；`errors` 非空表示
    正式结论不可直接采信（调用方应显式提示，不要静默丢弃）。
    """
    det = evidence.load_detection(data_dir, out_dir) or detection or {}
    ev = evidence.build_evidence(data_dir, detection=det, full_cluster=full_cluster)
    result = rules.evaluate(ev, thresholds)
    errors, warnings = validate.validate(ev, result)
    result["validation"] = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    paths = render.write_outputs(ev, result, out_dir)
    return ev, result, paths, errors, warnings
