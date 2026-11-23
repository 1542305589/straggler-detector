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
"""原始 Profiling DB 只读访问的回归测试。

约束：解析入口不得修改用户提供的原始 db —— journal_mode、索引、文件内容
都必须保持不变；只读文件也应能正常打开读取。

运行（仓库根目录）：
    python tests/skills/straggler-detector/scripts/test_readonly_db.py
"""
import hashlib
import os
import shutil
import sqlite3
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
import db_utils  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
