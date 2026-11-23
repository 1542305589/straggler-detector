"""输入 Profiling DB 的只读访问。

约束：本 skill 只读用户提供的原始 Profiling 数据，不修改 db 文件本身的
日志模式、索引和内容。所有对输入 db 的访问都必须通过 open_readonly_db()。
"""
import os
import sqlite3
from urllib.parse import quote


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
