"""验证：Level-1 单卡慢通信是否带上域名（"ep:5" → "ep[5]"）。

不依赖任何 profiling 数据，纯内存检查；在能跑 Python 的机器上：
    python verify_comm_domain_key.py

全部 PASS 说明修复生效且未破坏既有行为。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import markdown_viz
import summary_table
import utils
from diagnosis import rules

FAILS = []


def check(name, got, want):
    ok = got == want
    print(("PASS  " if ok else "FAIL  ") + f"{name}\n       got ={got!r}\n       want={want!r}")
    if not ok:
        FAILS.append(name)


print("== 1. key 生成（config.SlowNodeResult.add_single） ==")
res = config.SlowNodeResult()
res.add_single("comm", 5, 2.5, "ep")
res.add_single("comm", 5, 1.8, "tp")          # 同一 rank、不同域：不得互相覆盖
check("同 rank 不同域共存", sorted(res["comm"].keys()), ["ep:5", "tp:5"])
res.add_single("comm", 7, 1.4)                # 无域名：保持旧形式
check("无域名保持旧形式", sorted(res["comm"].keys()), ["7", "ep:5", "tp:5"])

print("\n== 2. key 拆分 / rank 解析 ==")
check("_split_domain_key('ep:5')", summary_table._split_domain_key("ep:5"), ("ep", "5"))
check("_split_domain_key('0,1,2')", summary_table._split_domain_key("0,1,2"), ("", "0,1,2"))
check("_split_domain_key('5')", summary_table._split_domain_key("5"), ("", "5"))
check("_parse_ranks_from_key('ep:5')", summary_table._parse_ranks_from_key("ep:5"), [5])
check("_parse_ranks_from_key('0,1,2')", summary_table._parse_ranks_from_key("0,1,2"), [0, 1, 2])

print("\n== 3. 汇总表单元格（summary_table._summary_cells） ==")
parallels = {"tp": [[0, 1, 2, 3]], "ep": [[0, 1, 2, 3, 4, 5, 6, 7]]}
cards, degs, _ = summary_table._summary_cells("comm", {"ep:5": 2.5}, parallels)
check("单卡带域名 → 异常卡列", cards, "ep[5]")
check("单卡带域名 → 劣化指数列", degs, "ep[5]:2.5")
cards, degs, _ = summary_table._summary_cells("comm", {"0,1,2,3": 1.7}, parallels)
check("组键仍反查出域名", cards, "tp[0, 1, 2, 3]")

print("\n== 4. straggler_detection_result.json（utils.write_result） ==")
tmp = tempfile.mkdtemp(prefix="verify_comm_")
config.set_output_path(tmp)
buf = []
_stdout = sys.stdout
sys.stdout = type("_T", (), {"write": lambda self, s: buf.append(s), "flush": lambda self: None})()
try:
    utils.write_result({"comm": {"ep:5": 2.5, "0,1,2,3": 1.7}}, parallels)
finally:
    sys.stdout = _stdout
with open(os.path.join(tmp, "straggler_detection_result.json"), encoding="utf-8") as f:
    data = json.load(f)
keys = sorted(it["display_key"] for it in data["comm"])
check("JSON 中 comm 的 display_key", keys, ["ep[5]", "tp[0, 1, 2, 3]"])

print("\n== 5. 报告回退分支（markdown_viz._comm_two_level_section） ==")
config.set_comm_two_level_detail(None)        # 强制走「仅列异常项」回退分支
section = markdown_viz._comm_two_level_section({"comm": {"ep:5": 2.5}}, parallels)
check("报告中单卡带域名", "ep[5]" in section, True)
check("报告中不再出现裸 rank 5", "rank 5" in section, False)

print("\n== 6. 域名含数字的边界（diagnosis.rules._ranks_from_key） ==")
check("'mc2[5]' 只取到 rank 5", rules._ranks_from_key("mc2[5]"), [5])
check("'mc2[0, 1]' 不把 2 当 rank", rules._ranks_from_key("mc2[0, 1]"), [0, 1])
check("'tp[0, 1, 2, 3]'", rules._ranks_from_key("tp[0, 1, 2, 3]"), [0, 1, 2, 3])

print("\n" + "=" * 60)
if FAILS:
    print(f"FAILED {len(FAILS)} 项：")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("全部 PASS")
