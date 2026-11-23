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
"""level1 / cluster 通信数据缓存的回归测试。

覆盖：缓存命中、源文件变化后失效、缓存损坏自动重建。

运行（仓库根目录）：
    python tests/skills/straggler-detector/scripts/test_comm_cache.py
"""
import json
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
import config  # noqa: E402
import nodelevel_data_handler as handler  # noqa: E402


def _touch_later(path: str) -> None:
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))


class Level1CommCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_l1cache_")
        self.analysis_db = os.path.join(self.tmpdir, "analysis.db")
        conn = sqlite3.connect(self.analysis_db)
        conn.execute(
            "CREATE TABLE CommAnalyzerBandwidth (hccl_op_name TEXT, count INTEGER, "
            "transit_size REAL, transit_time REAL, transport_type TEXT, step INTEGER)")
        conn.execute("INSERT INTO CommAnalyzerBandwidth VALUES "
                     "('allreduce', 10, 1000.0, 100.0, 'HCCS', 1)")
        conn.execute("INSERT INTO CommAnalyzerBandwidth VALUES "
                     "('allreduce', 20, 2000.0, 200.0, 'HCCS', 2)")
        conn.commit()
        conn.close()
        self.cache_path = os.path.join(self.tmpdir, "level1_comm_rank_0.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_and_invalidation(self):
        rows = handler._read_bandwidth_rows(self.analysis_db, self.cache_path)
        self.assertEqual(len(rows), 1)
        self.assertTrue(os.path.exists(self.cache_path))

        # 改写缓存内的数据：指纹未变，应命中缓存
        data = json.load(open(self.cache_path, encoding="utf-8"))
        data["rows"] = [["sentinel", 1, 1.0, 1.0]]
        json.dump(data, open(self.cache_path, "w", encoding="utf-8"))

        rows_hit = handler._read_bandwidth_rows(self.analysis_db, self.cache_path)
        self.assertEqual(rows_hit[0][0], "sentinel")

        # 源文件变化 -> 缓存失效，重新读取
        _touch_later(self.analysis_db)
        rows_fresh = handler._read_bandwidth_rows(self.analysis_db, self.cache_path)
        self.assertEqual(rows_fresh[0][0], "allreduce")

    def test_corrupt_cache_is_rebuilt(self):
        handler._read_bandwidth_rows(self.analysis_db, self.cache_path)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write("{not a json")
        rows = handler._read_bandwidth_rows(self.analysis_db, self.cache_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "allreduce")
        with open(self.cache_path, encoding="utf-8") as f:
            json.load(f)  # 应重新写成合法 JSON


class ClusterCommCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_clcache_")
        self.db_path = os.path.join(self.tmpdir, "cluster.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE CommunicationGroupMapping "
                     "(group_name TEXT, pg_name TEXT, rank_set TEXT, type TEXT)")
        conn.execute("CREATE TABLE ClusterCommunicationBandwidth "
                     "(group_name TEXT, rank_id INTEGER, hccl_op_name TEXT, count INTEGER, "
                     "transit_size REAL, transit_time REAL, band_type TEXT)")
        conn.execute("INSERT INTO CommunicationGroupMapping VALUES ('g0', 'dp', '[0,1]', 'collective')")
        for rank in (0, 1):
            conn.execute("INSERT INTO ClusterCommunicationBandwidth VALUES "
                         "('g0', ?, 'allreduce', 10, 1000.0, 100.0, 'HCCS')", (rank,))
        conn.commit()
        conn.close()
        self._old_output = config.get_output_path()
        config.set_output_path(self.tmpdir)
        self.cache_path = os.path.join(self.tmpdir, "op_metric", "cluster_comm.json")

    def tearDown(self):
        config.set_output_path(self._old_output)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_and_invalidation(self):
        parallels, comm = handler.load_cluster_comm(self.db_path)
        self.assertEqual(len(comm), 2)
        self.assertTrue(os.path.exists(self.cache_path))

        data = json.load(open(self.cache_path, encoding="utf-8"))
        data["comm"] = [["sentinel", "op", 1, 0, 1.0, 1.0]]
        json.dump(data, open(self.cache_path, "w", encoding="utf-8"))

        _parallels_hit, comm_hit = handler.load_cluster_comm(self.db_path)
        self.assertEqual(comm_hit[0][0], "sentinel")

        _touch_later(self.db_path)
        _parallels_fresh, comm_fresh = handler.load_cluster_comm(self.db_path)
        self.assertEqual(comm_fresh[0][0], "dp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
