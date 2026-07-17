"""Summarize DiCoBench accuracy and BudgetGate efficiency metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ANSWER_PATTERN = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE)


def extract_answer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = ANSWER_PATTERN.search(value)
    if match:
        return match.group(1).upper()
    stripped = value.strip().upper()
    return stripped if stripped in set("ABCDE") else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def calculate_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    task_totals: dict[str, int] = defaultdict(int)
    task_correct: dict[str, int] = defaultdict(int)
    model_calls: list[float] = []
    tool_calls: list[float] = []
    direct_answers = 0
    budget_violations = 0
    null_tp = null_fp = null_fn = 0

    for row in rows:
        predicted = extract_answer(row.get("final_answer"))
        expected = extract_answer(row.get("ground_truth"))
        task = row.get("task_signature") or row.get("data_source") or "unknown"
        correct = predicted is not None and predicted == expected
        task_totals[task] += 1
        task_correct[task] += int(correct)

        model_calls.append(float(row.get("model_call_count", 0)))
        tool_calls.append(float(row.get("tool_call_count", 0)))
        direct_answers += int(bool(row.get("direct_answer")))
        budget_violations += int(bool(row.get("budget_violation")))

        if predicted == "E" and expected == "E":
            null_tp += 1
        elif predicted == "E" and expected != "E":
            null_fp += 1
        elif predicted != "E" and expected == "E":
            null_fn += 1

    total = len(rows)
    total_correct = sum(task_correct.values())
    precision = null_tp / (null_tp + null_fp) if null_tp + null_fp else 0.0
    recall = null_tp / (null_tp + null_fn) if null_tp + null_fn else 0.0
    null_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "total_samples": total,
        "accuracy": round(total_correct / total, 4) if total else 0.0,
        "task_accuracy": {
            task: round(task_correct[task] / count, 4)
            for task, count in sorted(task_totals.items())
        },
        "average_model_calls": round(sum(model_calls) / total, 4) if total else 0.0,
        "p95_model_calls": _percentile(model_calls, 0.95),
        "average_tool_calls": round(sum(tool_calls) / total, 4) if total else 0.0,
        "direct_answer_rate": round(direct_answers / total, 4) if total else 0.0,
        "budget_violation_rate": round(budget_violations / total, 4) if total else 0.0,
        "null_option": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(null_f1, 4),
        },
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = calculate_metrics(load_results(args.results))
    serialized = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

