from __future__ import annotations

from typing import Dict


def grade_from_kpis(kpis: Dict[str, float]) -> float:
    """Deterministic 0..1 grading wrapper."""
    return float(max(0.0, min(1.0, kpis.get("task_score", 0.0))))

