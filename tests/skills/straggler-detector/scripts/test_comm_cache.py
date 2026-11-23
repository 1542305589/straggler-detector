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
"""level1 / cluster 通信指标缓存的回归测试。

覆盖：缓存命中、源 db 变化后失效、拓扑变化后失效、缓存损坏自动重建。

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


def _make_analysis_db(path: str, count: int, size: float, transit: float) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE CommAnalyzerBandwidth (hccl_op_name TEXT, count INTEGER, "
        "transit_size REAL, transit_time REAL, transport_type TEXT, step INTEGER)")
    conn.execute("INSERT INTO CommAnalyzerBandwidth VALUES "
                 "('hcom_allReduce_1', ?, ?, ?, 'HCCS', 1)", (count, size, transit))
    conn.commit()
    conn.close()


class Level1CommMetricCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_l1metric_")
        self.data_dir = os.path.join(self.tmpdir, "data")
        os.makedirs(self.data_dir)
        self.analysis_dbs = []
        for rank in (0, 1):
            rank_dir = os.path.join(self.data_dir, "worker%d" % rank)
            os.makedirs(rank_dir)
            open(os.path.join(rank_dir, "ascend_pytorch_profiler_%d.db" % rank), "w").close()
            analysis_db = os.path.join(rank_dir, "analysis.db")
            _make_analysis_db(analysis_db, 10, 1000.0 * (rank + 1), 100.0)
            self.analysis_dbs.append(analysis_db)
        self.out_dir = os.path.join(self.tmpdir, "out")
        self._old_output = config.get_output_path()
        config.set_output_path(self.out_dir)
        self.parallels = {"tp": [[0, 1]]}
        self.cache_path = os.path.join(self.out_dir, "op_metric", "comm_metric.json")

    def tearDown(self):
        config.set_output_path(self._old_output)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_and_invalidation(self):
        metric = handler.load_level1_comm_metric(self.data_dir, self.parallels)
        self.assertIn("0", metric["card"]["tp"])
        self.assertIn("1", metric["card"]["tp"])
        self.assertTrue(os.path.exists(self.cache_path))

        data = json.load(open(self.cache_path, encoding="utf-8"))
        data["metric"] = {"card": {"tp": {"0": 0.5}}, "group": {}, "singleton": []}
        json.dump(data, open(self.cache_path, "w", encoding="utf-8"))

        cached = handler.load_level1_comm_metric(self.data_dir, self.parallels)
        self.assertEqual(cached["card"]["tp"]["0"], 0.5)

        _touch_later(self.analysis_dbs[0])
        fresh = handler.load_level1_comm_metric(self.data_dir, self.parallels)
        self.assertNotEqual(fresh["card"]["tp"]["0"], 0.5)

    def test_parallels_change_invalidates(self):
        handler.load_level1_comm_metric(self.data_dir, self.parallels)
        data = json.load(open(self.cache_path, encoding="utf-8"))
        data["metric"] = {"card": {"tp": {"0": 0.5}}, "group": {}, "singleton": []}
        json.dump(data, open(self.cache_path, "w", encoding="utf-8"))

        other = {"ep": [[0, 1]]}
        fresh = handler.load_level1_comm_metric(self.data_dir, other)
        values = [v for domain in fresh["card"].values() for v in domain.values()]
        self.assertTrue(values and all(v != 0.5 for v in values))

    def test_corrupt_cache_is_rebuilt(self):
        handler.load_level1_comm_metric(self.data_dir, self.parallels)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write("{not a json")
        metric = handler.load_level1_comm_metric(self.data_dir, self.parallels)
        self.assertIn("0", metric["card"]["tp"])
        with open(self.cache_path, encoding="utf-8") as f:
            json.load(f)


class ClusterCommMetricCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="straggler_clmetric_")
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
                         "('g0', ?, 'hcom_allReduce_1', 10, 1000.0, 100.0, 'HCCS')", (rank,))
        conn.commit()
        conn.close()
        self.out_dir = os.path.join(self.tmpdir, "out")
        self._old_output = config.get_output_path()
        config.set_output_path(self.out_dir)
        self.parallels = handler.load_cluster_parallels(self.db_path)
        self.cache_path = os.path.join(self.out_dir, "comm_metric.json")

    def tearDown(self):
        config.set_output_path(self._old_output)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_and_invalidation(self):
        metric = handler.load_cluster_comm_metric(self.db_path, self.parallels)
        self.assertIn("0", metric["card"]["dp"])
        self.assertTrue(os.path.exists(self.cache_path))

        data = json.load(open(self.cache_path, encoding="utf-8"))
        data["metric"] = {"card": {"dp": {"0": 0.5}}, "group": {}, "singleton": []}
        json.dump(data, open(self.cache_path, "w", encoding="utf-8"))

        cached = handler.load_cluster_comm_metric(self.db_path, self.parallels)
        self.assertEqual(cached["card"]["dp"]["0"], 0.5)

        _touch_later(self.db_path)
        fresh = handler.load_cluster_comm_metric(self.db_path, self.parallels)
        self.assertNotEqual(fresh["card"]["dp"]["0"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
