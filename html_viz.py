"""
HTML 报告生成模块 - 为慢节点检测结果生成自包含 HTML 报告（内嵌 matplotlib 图表）。

与 markdown_viz 生成的纯文本 ``detection_report.log`` 同源，额外产出一份
``detection_report.html``，把各指标排序柱状图用 matplotlib 绘制并内嵌为 base64，
直接在浏览器打开即可查看。

依赖:
    matplotlib（可选）。未安装时本模块优雅降级为仅写一张纯 HTML 表格（不含图），
    不影响 skill 主流程与零依赖承诺（仅在用户显式要求图形报告时才使用）。

功能:
    1. 概览卡片（有效 Rank、劣化阈值、异常类别/卡数、Job 类型）
    2. 检测结果摘要表
    3. 各单卡指标排序柱状图（异常卡标红）
    4. 总通信耗时排序柱状图
    5. 各并行域通信耗时柱状图（组粒度 + 逐卡粒度）
"""

import base64
import io
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 添加父目录到路径以便导入 config / markdown_viz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import markdown_viz

logger = logging.getLogger("[HTML REPORT]")

# ---- matplotlib 相关（可选依赖，导入失败则图表降级） ----
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，仅渲染图片
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    _MPL_AVAILABLE = True
except Exception as e:  # pragma: no cover - 依赖缺失时降级
    logger.warning(f"[HTML REPORT] matplotlib 不可用，HTML 报告将不含图表：{e}")


# 颜色（现代仪表盘配色）
_C_NORMAL = "#4c7df0"
_C_ABNORMAL = "#e5484d"
_C_GRID = "#e5e7eb"
_C_TEXT = "#30313a"


def _setup_chinese_font():
    """为 matplotlib 配置中文字体（Windows 常见的微软雅黑/黑体优先）。"""
    if not _MPL_AVAILABLE:
        return
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    for cand in candidates:
        if cand in available:
            plt.rcParams["font.sans-serif"] = [cand] + list(plt.rcParams["font.sans-serif"])
            break
    plt.rcParams["axes.unicode_minus"] = False


def _fmt_ns(value: float) -> str:
    return markdown_viz._fmt_ns(value)


def _filter_valid(data) -> Dict[int, float]:
    return markdown_viz._filter_valid(data)


# ---------------------------------------------------------------------------
# matplotlib 图表
# ---------------------------------------------------------------------------

def _ns_axis_formatter():
    """把坐标轴刻度（纳秒）格式化为可读单位。"""
    def _fmt(x, _pos):
        return _fmt_ns(x)
    return FuncFormatter(_fmt)


def _bar_chart_svg(ranks: List, values: List[float], title: str,
                   abnormal: Optional[set] = None,
                   xlabel: str = "耗时"):
    """
    生成一张水平柱状图（按值降序），异常项标红，返回内嵌 SVG 的字符串。

    SVG 为矢量格式，任意缩放不模糊（替代原 PNG 位图方案）。

    abnormal: 异常标签集合（字符串；单卡图为 rank 字符串，组图为 "0,1" 形式）。
    返回 None 表示 matplotlib 不可用。
    """
    if not _MPL_AVAILABLE or not ranks or not values:
        return None

    abnormal_labels = {str(x) for x in (abnormal or set())}
    pairs = sorted(zip(ranks, values), key=lambda p: p[1], reverse=True)
    labels = [str(r) for r, _ in pairs]
    vals = [v for _, v in pairs]

    fig, ax = plt.subplots(figsize=(9, max(2.2, 0.32 * len(pairs) + 0.6)))
    ypos = list(range(len(labels)))

    colors = [_C_ABNORMAL if lb in abnormal_labels else _C_NORMAL for lb in labels]

    bars = ax.barh(ypos, vals, color=colors, height=0.62, alpha=0.92)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=12)
    ax.xaxis.set_major_formatter(_ns_axis_formatter())
    ax.grid(axis="x", color=_C_GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=_C_TEXT)

    # 在条末端标注具体耗时
    max_v = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                _fmt_ns(v), va="center", ha="left", fontsize=8, color=_C_TEXT)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue().decode("utf-8")
    # 让 SVG 随容器自适应宽度，避免因固定像素尺寸而显示过小/过糊
    svg = svg.replace("<svg ", '<svg style="max-width:100%;height:auto" ', 1)
    return svg


# ---------------------------------------------------------------------------
# HTML 构建
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #f4f5f7; --card: #ffffff; --ink: #30313a; --muted: #6b7280;
  --line: #e5e7eb; --accent: #4c7df0; --bad: #e5484d; --good: #19a974;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 28px 22px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 22px; }
.card {
  flex: 1 1 160px; background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 14px 16px; min-width: 140px;
}
.card .k { font-size: 12px; color: var(--muted); }
.card .v { font-size: 22px; font-weight: 700; margin-top: 4px; }
.card .v.bad { color: var(--bad); }
.card .v.good { color: var(--good); }
section {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 20px; margin-bottom: 18px;
}
section h2 { font-size: 16px; margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; background: #fafafa; }
td.bad, .bad { color: var(--bad); font-weight: 600; }
.tag {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600;
}
.tag.bad { background: #fdeaea; color: var(--bad); }
.tag.ok { background: #e8f7f0; color: var(--good); }
.figure { margin: 10px 0 4px; text-align: center; }
.figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.note { color: var(--muted); font-size: 12px; }
.footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 8px; }
.chips { margin: 6px 0 0; }
.empty { color: var(--muted); font-size: 13px; padding: 6px 0; }
"""


def _esc(txt) -> str:
    return (str(txt).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _type_names():
    names = {
        "KERNEL_AICORE": "慢计算 (KERNEL_AICORE)",
        "kernel_aivec": "矢量计算 (kernel_aivec)",
        "memcpy_async": "内存搬运 (memcpy_async)",
        "comm": "慢通信 (comm)",
        "cpu": "慢CPU (cpu)",
        "npu_bubble": "Bubble (npu_bubble)",
    }
    return names


def _abnormal_rank_set(detection_result: Dict, category: str) -> set:
    items = (detection_result or {}).get(category) or {}
    out = set()
    for key in items:
        for r in key.split(","):
            try:
                out.add(int(r))
            except ValueError:
                pass
    return out


def _single_metric_section(metric_name: str, data: Dict[int, float],
                           abnormal: Optional[set], cat_label: str, note: str = ""):
    filtered = _filter_valid(data)
    if not filtered:
        return ""
    ranks = sorted(filtered.keys(), key=lambda r: filtered[r], reverse=True)
    values = [filtered[r] for r in ranks]
    svg = _bar_chart_svg(ranks, values, f"{cat_label} · {metric_name} 耗时排序", abnormal)
    parts = [f"<section><h2>{_esc(cat_label)} · <code>{_esc(metric_name)}</code> 耗时排序</h2>"]
    if note:
        parts.append(f'<p class="note">{_esc(note)}</p>')
    if svg:
        parts.append(f'<div class="figure">{svg}</div>')
    else:
        parts.append('<div class="note">（matplotlib 不可用，图表省略）</div>')
    parts.append("</section>")
    return "".join(parts)


def _init_html_render():
    if _MPL_AVAILABLE:
        _setup_chinese_font()


def generate_html_report(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    input_path: str = "",
    degradation: float = 0.3,
) -> str:
    """
    生成自包含 HTML 报告字符串（图表内嵌 base64）。

    参数与 markdown_viz.generate_report 一致。
    """
    _init_html_render()
    detection_result = detection_result or {}
    type_names = _type_names()

    # ---- 概览卡片 ----
    abnormal_categories = [c for c, items in detection_result.items() if items]
    abnormal_items_total = sum(len(items) for items in detection_result.values())

    cards = [
        ("有效 Rank 数", f"{len(valid_ranks)}", ""),
        ("劣化阈值", f"{degradation}", ""),
        ("Job 类型", config.get_job_type(), ""),
        ("异常类别", f"{len(abnormal_categories)}", "bad" if abnormal_categories else "good"),
        ("异常项数", f"{abnormal_items_total}", "bad" if abnormal_items_total else "good"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v {cls}">{_esc(v)}</div></div>'
        for k, v, cls in cards
    )

    body = [f'<div class="cards">{cards_html}</div>']

    # ---- 检测结果摘要 ----
    body.append("<section><h2>检测结果摘要</h2>")
    if not any(detection_result.values()):
        body.append('<div class="empty">未检测到异常节点</div>')
    else:
        body.append('<table><thead><tr><th>检测类型</th><th>状态</th>'
                    '<th>异常项数</th><th>劣化指数</th></tr></thead><tbody>')
        order = ["KERNEL_AICORE", "kernel_aivec", "memcpy_async", "comm",
                 "cpu", "npu_bubble"]
        order += [c for c in detection_result if c not in order]
        for key in order:
            items = detection_result.get(key) or {}
            name = type_names.get(key, key)
            if items:
                details = "，".join(f'<span class="tag bad">{_esc(rk)}({v:.2f}×)</span>'
                                    for rk, v in sorted(items.items(), key=lambda x: -x[1]))
                body.append(f'<tr><td>{_esc(name)}</td><td><span class="tag bad">异常</span></td>'
                            f'<td>{len(items)}</td><td>{details}</td></tr>')
            else:
                body.append(f'<tr><td>{_esc(name)}</td><td><span class="tag ok">正常</span></td>'
                            f'<td>0</td><td>-</td></tr>')
        body.append("</tbody></table>")
    body.append("</section>")

    # ---- 单卡指标排序柱状图 ----
    cat_to_metric = {
        "KERNEL_AICORE": ("KERNEL_AICORE", "慢计算"),
        "kernel_aivec": ("KERNEL_AIVEC", "矢量计算"),
        "memcpy_async": ("MEMCPY_ASYNC", "内存搬运"),
        "cpu": ("ZP_Host", "慢CPU"),
        "npu_bubble": ("ZP_Bubble", "NPU空泡"),
    }
    rendered_cols = set()
    for cat, (col, label) in cat_to_metric.items():
        if col not in step_data or col in rendered_cols:
            continue
        rendered_cols.add(col)
        abnormal = _abnormal_rank_set(detection_result, cat)
        body.append(_single_metric_section(col, step_data[col], abnormal, label))

    # ---- 各并行域通信耗时 ----
    if parallels:
        comm_abnormal_ranks = _abnormal_rank_set(detection_result, "comm")

        # 组粒度（按组内最小值）
        for domain_name, domain_groups in parallels.items():
            if not domain_name or not domain_groups:
                continue
            duration_key = f"{domain_name}_Duration"
            if duration_key not in step_data:
                continue
            group_labels, group_vals = [], []
            group_abnormal = set()
            for g in domain_groups:
                vals = [v for v in (step_data[duration_key].get(r, -99999) for r in g)
                        if v != -99999 and v > 0]
                if not vals:
                    continue
                key = ",".join(str(r) for r in g)
                group_labels.append(key)
                group_vals.append(min(vals))
                if any(r in comm_abnormal_ranks for r in g):
                    group_abnormal.add(key)
            if group_labels:
                svg = _bar_chart_svg(group_labels, group_vals,
                                     f"{domain_name} 并行域 · 集合通信耗时（组 min）",
                                     abnormal=group_abnormal)
                fig_html = (f'<div class="figure">{svg}</div>' if svg
                            else '<div class="note">（matplotlib 不可用，图表省略）</div>')
                body.append(f'<section><h2>{_esc(domain_name)} 并行域 · 集合通信耗时（组 min）</h2>'
                            + fig_html + "</section>")

        # 逐卡粒度的各域通信耗时
        for domain_name in parallels:
            if not domain_name:
                continue
            duration_key = f"{domain_name}_Duration"
            if duration_key not in step_data:
                continue
            filtered = _filter_valid(step_data[duration_key])
            if not filtered:
                continue
            body.append(_single_metric_section(duration_key, filtered, comm_abnormal_ranks,
                                               f"{domain_name} 通信",
                                               note=f"仅展示{domain_name}通信耗时分布，不参与检测，通信异常检测以通信组({{xp}}_Duration)为单位"))

    # ---- 总通信耗时排序（置于最后，仅展示、不参与检测） ----
    if parallels:
        total_section = _comm_total_html(step_data, parallels, detection_result)
        if total_section:
            body.append(total_section)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>慢节点检测报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1>慢节点检测报告</h1>
<div class="sub">
  数据目录：{_esc(input_path or "-")} &nbsp;·&nbsp; 生成时间：{_esc(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
  &nbsp;·&nbsp; 有效 Rank：{len(valid_ranks)}
</div>
{''.join(body)}
<div class="footer">straggler-detector · 自动生成</div>
</div>
</body>
</html>"""
    return html


def _comm_total_html(step_data, parallels, detection_result) -> str:
    """总通信耗时排序（各域 Duration 求和），复刻 markdown_viz._comm_total_section 口径。"""
    parallel_keys = list(parallels.keys()) if parallels else []
    parallel_names = [n for n in parallel_keys if n]
    domain_keys = [f"{name}_Duration" for name in parallel_names]
    has_domain = any(dk in step_data and any(v > 0 and v != -99999 for v in step_data[dk].values())
                     for dk in domain_keys)
    if "" in parallel_keys:
        if "_Duration" in step_data:
            domain_keys.append("_Duration")

    comm_totals: Dict[int, float] = {}
    subtitle = ""
    if has_domain:
        for dk in domain_keys:
            if dk not in step_data:
                continue
            for rank, val in step_data[dk].items():
                if val != -99999 and val > 0:
                    comm_totals[rank] = comm_totals.get(rank, 0) + val
        included = [dk.replace("_Duration", "") for dk in domain_keys
                    if dk in step_data and any(v > 0 and v != -99999 for v in step_data[dk].values())]
        subtitle = "各域通信耗时求和: " + ", ".join(included)
    else:
        zp = step_data.get("ZP_Duration", {})
        comm_totals = {k: v for k, v in zp.items() if v != -99999 and v > 0}
        subtitle = "无域通信数据，使用 ZP_Duration（总通信耗时）"

    if not comm_totals:
        return ""
    abnormal = _abnormal_rank_set(detection_result, "comm")
    ranks = sorted(comm_totals.keys(), key=lambda r: comm_totals[r], reverse=True)
    vals = [comm_totals[r] for r in ranks]
    svg = _bar_chart_svg(ranks, vals, "总通信耗时排序", abnormal)
    fig_html = (f'<div class="figure">{svg}</div>' if svg
                else '<div class="note">（matplotlib 不可用，图表省略）</div>')
    note = (f"仅展示总通信耗时分布，不参与检测，通信异常检测以通信组({{xp}}_Duration)为单位"
            + (f" · {_esc(subtitle)}" if subtitle else ""))
    html = (f'<section><h2>总通信耗时排序</h2><p class="note">{note}</p>'
            + fig_html + "</section>")
    return html


def write_html_report(
    step_data: Dict[str, Dict[int, float]],
    parallels: Dict[str, List[List[int]]],
    valid_ranks: List[int],
    output_dir: str,
    detection_result: Optional[Dict[str, Dict[str, float]]] = None,
    input_path: str = "",
    degradation: float = 0.3,
) -> str:
    """
    生成并写入 HTML 报告（analysis_result/detection_report.html）。

    返回: 报告文件路径；matplotlib 不可用时仍会写出 HTML（图表省略）。
    """
    html = generate_html_report(
        step_data, parallels, valid_ranks, output_dir,
        detection_result=detection_result,
        input_path=input_path,
        degradation=degradation,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "detection_report.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        logger.warning(f"[HTML REPORT] 写入 HTML 报告失败：{e}")
        return ""

    logger.info(f"HTML 报告已保存: {path}")
    print(f"HTML 报告已保存: {path}")
    return path