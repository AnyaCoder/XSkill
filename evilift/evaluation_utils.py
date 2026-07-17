"""Shared deterministic evaluation helpers for Evilift benchmarks."""

from __future__ import annotations

import math
import re
from typing import Any


ANSWER_PATTERN = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE)
VALID_ANSWERS = frozenset("ABCDE")


def extract_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    matches = ANSWER_PATTERN.findall(value)
    if matches:
        return matches[-1].upper()
    stripped = value.strip().upper()
    return stripped if stripped in VALID_ANSWERS else None


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return float(ordered[index])
