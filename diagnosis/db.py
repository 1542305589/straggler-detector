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
"""只读取数工具：定位 db、只读连接、表/列探测。

所有函数都不写库、不建表、不改数据。
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

PROFILER_PREFIX = "ascend_pytorch_profiler_"
CLUSTER_TABLES = ("CommunicationGroupMapping", "ClusterCommunicationBandwidth")


def open_ro(path: str) -> sqlite3.Connection:
    """以只读 URI 打开 sqlite（不会锁库、不会写入）。"""
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def columns(con: sqlite3.Connection, name: str) -> List[str]:
    try:
        return [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
    except sqlite3.Error:
        return []


def has_rows(con: sqlite3.Connection, name: str) -> bool:
    if not table_exists(con, name):
        return False
    try:
        return bool(con.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone())
    except sqlite3.Error:
        return False


def rows(con: sqlite3.Connection, sql: str, params: tuple = ()) -> List[tuple]:
    """安全查询：表/列不存在时返回空列表，不抛异常。"""
    try:
        return con.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []


def rank_from_profiler(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if not (base.startswith(PROFILER_PREFIX) and base.endswith(".db")):
        return None
    try:
        return int(base[len(PROFILER_PREFIX):-len(".db")])
    except ValueError:
        return None


def find_profiler_dbs(root: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if os.path.isfile(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.startswith(PROFILER_PREFIX) and fn.endswith(".db"):
                p = os.path.join(dirpath, fn)
                r = rank_from_profiler(p)
                if r is not None:
                    out.append((r, p))
    return sorted(out)


def analysis_db_for(profiler_db: str) -> Optional[str]:
    p = os.path.join(os.path.dirname(profiler_db), "analysis.db")
    return p if os.path.isfile(p) else None


def is_cluster_db(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        con = open_ro(path)
        ok = all(has_rows(con, t) for t in CLUSTER_TABLES)
        con.close()
        return ok
    except sqlite3.Error:
        return False


def find_cluster_db(root: str) -> Optional[str]:
    if os.path.isfile(root):
        return root if is_cluster_db(root) else None
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".db"):
                p = os.path.join(dirpath, fn)
                if is_cluster_db(p):
                    return p
    return None


def string_ids(con: sqlite3.Connection) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for i, v in rows(con, "SELECT id, value FROM STRING_IDS"):
        try:
            out[int(i)] = v
        except (TypeError, ValueError):
            continue
    return out


def meta(con: sqlite3.Connection) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in rows(con, "SELECT name, value FROM META_DATA"):
        out[str(k)] = v
    return out


def one(con: sqlite3.Connection, sql: str, params: tuple = ()):
    r = rows(con, sql, params)
    return r[0] if r else None
