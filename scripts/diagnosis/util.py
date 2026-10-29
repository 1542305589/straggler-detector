from __future__ import annotations
from typing import List
_NS_PER_MS = 1_000_000.0
_US_PER_MS = 1_000.0

def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _i(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default

def _ns_to_ms(v) -> float:
    return _f(v) / _NS_PER_MS

def _us_to_ms(v) -> float:
    return _f(v) / _US_PER_MS

def _median(values: List[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0

def _ratio(a: float, b: float) -> float:
    if b > 0:
        return a / b
    return float("inf") if a > 0 else 1.0

def _group_key(ranks: List[int]) -> str:
    return ",".join(str(r) for r in sorted(ranks))
