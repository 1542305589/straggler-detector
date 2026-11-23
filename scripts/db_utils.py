"""输入 Profiling DB 的只读访问。

约束：本 skill 只读用户提供的原始 Profiling 数据，不修改 db 文件本身的
日志模式、索引和内容。所有对输入 db 的访问都必须通过 open_readonly_db()。
"""
import json
import os
import sqlite3
from urllib.parse import quote

# 解析缓存（level1 / cluster 的通信数据）的格式版本
CACHE_SCHEMA = 1


def open_readonly_db(path: str) -> sqlite3.Connection:
    """以只读模式打开输入 db。

    使用 SQLite URI 的 mode=ro，并额外打开 query_only，确保不会产生任何写入
    （包括 journal_mode 切换、建索引、临时落盘）。路径中的特殊字符会被转义。

    注意：若 db 本身已经是 WAL 模式，SQLite 在只读打开时仍会生成 -shm/-wal
    旁路文件（这是 SQLite 的固有行为），但不会修改 db 文件内容；本 skill 也
    不会把 db 切换成 WAL。
    """
    abspath = os.path.abspath(path).replace("\\", "/")
    uri = "file:" + quote(abspath, safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def load_cache(cache_path: str, source_path: str):
    """读取解析缓存；source 的 mtime/size 不匹配、schema 不匹配或文件损坏时返回 None。"""
    if not cache_path or not source_path:
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA:
        return None
    try:
        fp = _fingerprint(source_path)
    except OSError:
        return None
    src = data.get("source") or {}
    if src.get("mtime_ns") != fp["mtime_ns"] or src.get("size") != fp["size"]:
        return None
    return data


def save_cache(cache_path: str, source_path: str, kind: str, payload: dict) -> None:
    """原子写解析缓存；失败时静默忽略（缓存是尽力而为，不影响正确性）。"""
    if not cache_path:
        return
    try:
        fp = _fingerprint(source_path)
    except OSError:
        return
    data = {"schema": CACHE_SCHEMA, "kind": kind, "source": fp}
    data.update(payload or {})
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    except OSError:
        pass
