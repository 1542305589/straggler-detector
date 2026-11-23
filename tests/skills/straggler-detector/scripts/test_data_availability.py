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
"""数据可用性 / 评估状态的回归测试。

覆盖：
1. op_metric 缺文件、空表、部分 rank 缺失时，load_status 必须能区分状态；
2. 指标数据缺失时，结论与汇总表必须输出"未评估"，不能退化成"无异常"；
3. cluster 的 computing/free 可用性按实际值判断；
4. 评估状态写入结果 JSON 后，回读结论仍能区分"无数据"与"无异常"。

运行（仓库根目录）：
    python tests/skills/straggler-detector/scripts/test_data_availability.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "skills", "profiler", "straggler-detector", "scripts",
    ),
)
import conclusion  # noqa: E402
import summary_table  # noqa: E402


class LoadStatusTest(unittest.TestCase):
    """缺文件 / 空表 / 部分 rank 缺失：状态必须能区分"没数据"和"没异常"。"""

    def setUp(self):
        import config
        self._config = config
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_load_")
        self.metric_dir = os.path.join(self.tmpdir, "op_metric")
        os.makedirs(self.metric_dir)
        self._old_output = config.get_output_path()
        config.set_output_path(self.tmpdir)

    def tearDown(self):
        self._config.set_output_path(self._old_output)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, rank, rows):
        path = os.path.join(self.metric_dir, f"global_rank_{rank}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("StepIndex,ZP_Host,KERNEL_AICORE\n")
            for row in rows:
                f.write(row + "\n")

    def test_missing_csv_is_reported(self):
        import nodelevel_data_handler
        data, status = nodelevel_data_handler.get_cur_job_last_step_data([0, 1])
        self.assertEqual(data, {})
        self.assertEqual(status["ranks_missing_csv"], [0, 1])
        self.assertEqual(status["ranks_empty_csv"], [])
        self.assertEqual(status["ranks_with_data"], [])

    def test_header_only_csv_is_empty(self):
        import nodelevel_data_handler
        self._write_csv(0, [])
        data, status = nodelevel_data_handler.get_cur_job_last_step_data([0])
        self.assertEqual(data, {})
        self.assertEqual(status["ranks_missing_csv"], [])
        self.assertEqual(status["ranks_empty_csv"], [0])

    def test_partial_rank_missing(self):
        import nodelevel_data_handler
        self._write_csv(0, ["0,3.0,1.2", "1,3.1,1.3"])
        data, status = nodelevel_data_handler.get_cur_job_last_step_data([0, 1])
        self.assertTrue(data, "有数据的 rank 仍应产出检测数据")
        self.assertEqual(status["ranks_total"], 2)
        self.assertEqual(status["ranks_with_data"], [0])
        self.assertEqual(status["ranks_missing_csv"], [1])


class NoDataVsNoAnomalyTest(unittest.TestCase):
    def test_availability_marks_missing_metrics(self):
        # 空数据：level0 的五类指标都不可评估
        import main
        unavailable = main._metric_availability("level0", {}, None, {})
        for cat in ("compute", "memcpy", "host", "pp_comm", "comm"):
            self.assertIn(cat, unavailable)

    def test_availability_empty_when_data_present(self):
        import config
        import main
        step_data = {
            "KERNEL_AICORE": {0: 1.0, 1: 1.0},
            "KERNEL_AIVEC": {0: 1.0, 1: 1.0},
            "MEMCPY_ASYNC": {0: 1.0},
            "ZP_Host": {0: 1.0},
            "PP_WaitGap": {0: 0.01},
            "tp_allgather_1": {0: 1.0},
        }
        old = config.get_has_named_domain()
        config.set_has_named_domain(True)
        try:
            unavailable = main._metric_availability("level0", step_data, None, {"tp": [[0, 1]]})
        finally:
            config.set_has_named_domain(old)
        self.assertEqual(unavailable, {})

    def test_availability_marks_partial_compute(self):
        """只缺 KERNEL_AIVEC 时，应逐项标注，而不是把整个计算类标成未评估。"""
        import config
        import main
        step_data = {
            "KERNEL_AICORE": {0: 1.0},
            "MEMCPY_ASYNC": {0: 1.0},
            "ZP_Host": {0: 1.0},
            "PP_WaitGap": {0: 0.01},
            "tp_allgather_1": {0: 1.0},
        }
        old = config.get_has_named_domain()
        config.set_has_named_domain(True)
        try:
            unavailable = main._metric_availability("level0", step_data, None, {"tp": [[0, 1]]})
        finally:
            config.set_has_named_domain(old)
        self.assertNotIn("compute", unavailable)
        self.assertIn("KERNEL_AIVEC", unavailable)

    def test_summary_table_says_unevaluated_not_normal(self):
        table = summary_table.build_summary_table(
            {}, None, {}, unavailable={"comm": "缺少通信带宽数据"})
        self.assertIn("未评估", table)
        self.assertNotIn("无异常", table)

    def test_conclusion_says_unevaluated_not_normal(self):
        text = conclusion.render(
            {}, "level0", (1.3, 2.5, 1.3),
            unavailable={"compute": "缺少 KERNEL_AICORE / KERNEL_AIVEC 指标"})
        self.assertIn("未评估（缺少 KERNEL_AICORE / KERNEL_AIVEC 指标）", text)
        self.assertNotIn("1、计算类故障检测结果：\n无", text)

    def test_comm_unavailable_noted_even_when_pp_has_anomaly(self):
        text = conclusion.render(
            {"pp_comm": [{"display_key": "[0, 4]", "metric_value": 0.083}]},
            "level0", (1.3, 2.5, 1.3),
            unavailable={"comm": "缺少通信带宽数据"})
        self.assertIn("PP点对点通信等待", text)
        self.assertIn("缺少通信带宽数据", text)

    def test_cluster_availability_uses_computing_free_values(self):
        """cluster 有 step rows 但缺 computing/free 时，仍应标记未评估。"""
        import main
        step_info = {"rows": [(0, 0, {"communication": 1.0}),
                              (0, 1, {"communication": 2.0})]}
        unavailable = main._metric_availability("cluster", {}, None, {}, step_info)
        self.assertIn("compute", unavailable)
        self.assertIn("host", unavailable)

        step_info_ok = {"rows": [(0, 0, {"computing": 1.0, "free": 2.0}),
                                 (0, 1, {"computing": 1.0, "free": 2.0})]}
        unavailable_ok = main._metric_availability("cluster", {}, None, {}, step_info_ok)
        self.assertNotIn("compute", unavailable_ok)
        self.assertNotIn("host", unavailable_ok)

    def test_unevaluated_state_persisted_in_result_json(self):
        """结果 JSON 落盘后，回读结论仍能区分"无数据"和"无异常"。"""
        import utils
        tmpdir = tempfile.mkdtemp(prefix="straggler_persist_")
        try:
            payload = utils.build_report_payload(
                {}, None, "level0",
                {"compute": 1.3, "io": 2.5, "comm": 1.3, "pp_comm": 5.0},
                unavailable={"comm": "缺少通信带宽数据"})
            with open(os.path.join(tmpdir, "straggler_detection_result.json"),
                      "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            text = conclusion.render_from_dir(tmpdir, "level0", (1.3, 2.5, 1.3))
            self.assertIn("未评估", text)
            self.assertIn("缺少通信带宽数据", text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
