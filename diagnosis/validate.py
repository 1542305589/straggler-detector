"""校验门：正式结论必须带 rule_id，证据引用必须能在 evidence.json 解析。"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class _Missing:
    pass


MISSING = _Missing()


def resolve(ev: Dict, ref: str) -> Any:
    cur: Any = ev
    for part in str(ref).split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return MISSING
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return MISSING
        else:
            return MISSING
    return cur


def _flatten_numbers(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_numbers(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_flatten_numbers(v))
    return out


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= 1e-6


def validate(ev: Dict, result: Dict) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    findings = result.get("findings", []) or []
    if not findings:
        warnings.append("findings 为空")

    for f in findings:
        rid = f.get("rule_id")
        if not rid:
            errors.append(f"finding 缺少 rule_id：{f}")
            continue
        if f.get("status") not in ("fired", "conflict", "not_evaluated",
                                   "authoritative", "auxiliary"):
            errors.append(f"{rid}: 非法 status {f.get('status')}")

        nums = _flatten_numbers([f.get("value"), f.get("baseline_value"), f.get("ratio")])
        resolved_nums: List[float] = []
        for ref in f.get("evidence_refs", []) or []:
            v = resolve(ev, ref)
            if v is MISSING:
                errors.append(f"{rid}: evidence_ref 无法解析 `{ref}`")
                continue
            resolved_nums.extend(_flatten_numbers(v))

        for n in nums:
            if f.get("derived"):
                break
            if not math.isfinite(n):
                warnings.append(f"{rid}: 非有限数值 {n}")
                continue
            if resolved_nums and not any(_close(n, x) for x in resolved_nums):
                warnings.append(f"{rid}: 数值 {n} 未在 evidence_refs 中直接命中（可能为派生比值）")

    if result.get("primary", {}).get("rule_id"):
        if (result["primary"]["rule_id"] not in (result.get("rules_fired") or [])
                and result.get("verdict") != "insufficient_evidence"):
            warnings.append("primary.rule_id 不在 rules_fired 中")

    formal_text = " ".join(str(result.get(k, "")) for k in ("verdict", "primary", "actions"))
    for s in result.get("speculations", []) or []:
        hyp = str(s.get("hypothesis", ""))
        if hyp and hyp in formal_text:
            errors.append("推测内容出现在正式结论区")

    return errors, warnings
