from __future__ import annotations
from typing import Dict, Optional, Tuple
from . import evidence, render, rules, validate

def run_diagnosis(data_dir: str, out_dir: str, thresholds: Dict[str, float],
                  detection: Optional[Dict] = None,
                  full_cluster: bool = False
                  ) -> Tuple[Dict, Dict, Dict[str, str], list, list]:
    det = evidence.load_detection(data_dir, out_dir) or detection or {}
    ev = evidence.build_evidence(data_dir, detection=det, full_cluster=full_cluster)
    result = rules.evaluate(ev, thresholds)
    errors, warnings = validate.validate(ev, result)
    result["validation"] = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    paths = render.write_outputs(ev, result, out_dir)
    return ev, result, paths, errors, warnings
