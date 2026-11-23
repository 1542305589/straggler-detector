#!/usr/bin/env python3
"""只读访问与"无数据 vs 无异常"的回归测试。

覆盖两类回归：
1. 解析入口不得修改用户的原始 Profiling DB（journal_mode / 索引 / 文件内容）；
2. 指标数据缺失时，输出必须是"未评估"，不能变成"无异常"。

运行：python scripts/test_readonly_db.py
"""
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db_utils  # noqa: E402
import conclusion  # noqa: E402
import summary_table  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = sorted(
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        )
    finally:
        conn.close()
    return {
        "journal_mode": journal_mode,
        "indexes": indexes,
        "sha256": _sha256(db_path),
        "size": os.path.getsize(db_path),
    }


class ReadOnlyDbTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_ro_")
        self.db_path = os.path.join(self.tmpdir, "sample.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE TASK (startNs INTEGER, endNs INTEGER, taskType INTEGER)")
        conn.execute("CREATE TABLE STRING_IDS (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("CREATE INDEX idx_existing ON STRING_IDS(value)")
        conn.executemany("INSERT INTO STRING_IDS(id, value) VALUES (?, ?)",
                         [(1, "KERNEL_AICORE"), (2, "MEMCPY_ASYNC")])
        conn.executemany("INSERT INTO TASK(startNs, endNs, taskType) VALUES (?, ?, ?)",
                         [(0, 100, 1), (100, 200, 2)])
        conn.commit()
        conn.close()

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            path = os.path.join(self.tmpdir, name)
            os.chmod(path, 0o666)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reading_does_not_modify_db(self):
        before = _snapshot(self.db_path)
        conn = db_utils.open_readonly_db(self.db_path)
        try:
            conn.execute("SELECT count(*) FROM TASK").fetchone()
            conn.execute("SELECT value FROM STRING_IDS WHERE value='KERNEL_AICORE'").fetchone()
        finally:
            conn.close()
        after = _snapshot(self.db_path)
        self.assertEqual(before, after, "只读访问后 db 的日志模式/索引/内容不应变化")
        self.assertFalse(os.path.exists(self.db_path + "-wal"), "不应产生 -wal 文件")
        self.assertFalse(os.path.exists(self.db_path + "-shm"), "不应产生 -shm 文件")

    def test_write_is_rejected(self):
        conn = db_utils.open_readonly_db(self.db_path)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("CREATE INDEX idx_new ON STRING_IDS(value)")
        finally:
            conn.close()
        self.assertNotIn("idx_new", _snapshot(self.db_path)["indexes"])

    def test_readonly_file_can_still_be_read(self):
        os.chmod(self.db_path, 0o444)
        conn = db_utils.open_readonly_db(self.db_path)
        try:
            row = conn.execute("SELECT count(*) FROM TASK").fetchone()
            self.assertEqual(row[0], 2)
        finally:
            conn.close()


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
