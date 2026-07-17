"""Evaluate V* answers, localized evidence search, and inference efficiency."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evilift.evaluation_utils import extract_answer, percentile


Box = tuple[float, float, float, float]


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of records in {path}")
    return payload


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and (value := _literal_number(node.operand)) is not None
    ):
        return -value
    return None


def _root_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _root_name(node.value)
    if isinstance(node, ast.Call):
        return _root_name(node.func)
    return None


def extract_literal_crop_regions(code: str) -> tuple[list[Box], int, int]:
    """Return original-image crop boxes, total crop calls, and unscorable calls."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return [], 0, 1

    regions: list[Box] = []
    crop_calls = 0
    unscorable = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "crop":
            continue

        crop_calls += 1
        root_name = _root_name(node.func.value)
        argument = node.args[0] if node.args else None
        if (
            root_name not in {"original_image", "image"}
            or not isinstance(argument, (ast.Tuple, ast.List))
            or len(argument.elts) != 4
        ):
            unscorable += 1
            continue

        values = [_literal_number(element) for element in argument.elts]
        if any(value is None for value in values):
            unscorable += 1
            continue
        left, top, right, bottom = (float(value) for value in values)
        if right <= left or bottom <= top:
            unscorable += 1
            continue
        regions.append((left, top, right, bottom))
    return regions, crop_calls, unscorable


def _xywh_to_xyxy(box: Iterable[float]) -> Box:
    left, top, width, height = (float(value) for value in box)
    return left, top, left + width, top + height


def target_coverage(crop: Box, target: Box) -> float:
    left = max(crop[0], target[0])
    top = max(crop[1], target[1])
    right = min(crop[2], target[2])
    bottom = min(crop[3], target[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    return intersection / target_area if target_area > 0 else 0.0


def evidence_hit(
    crop_regions: list[Box],
    bboxes_xywh: Iterable[Iterable[float]],
    threshold: float = 0.5,
) -> bool:
    targets = [_xywh_to_xyxy(box) for box in bboxes_xywh]
    return bool(targets) and all(
        any(target_coverage(crop, target) >= threshold for crop in crop_regions)
        for target in targets
    )


def _trajectory_path(trajectories_dir: Path, row: dict[str, Any]) -> Path:
    sample_dir = trajectories_dir / str(row["question_id"])
    rollout_index = row.get("_rollout_idx")
    if rollout_index is not None:
        return sample_dir / f"rollout_{rollout_index}" / "traj.jsonl"
    return sample_dir / "traj.jsonl"


def _trajectory_metrics(path: Path) -> dict[str, Any]:
    regions: list[Box] = []
    zoom_calls = 0
    tool_calls = 0
    model_calls = 0
    crop_calls = 0
    unscorable_crops = 0

    if not path.exists():
        return {
            "regions": regions,
            "zoom_calls": 0,
            "tool_calls": 0,
            "model_calls": 0,
            "crop_calls": 0,
            "unscorable_crops": 0,
            "trajectory_missing": True,
        }

    for event in load_records(path):
        if "text_output" in event:
            model_calls += 1
        tool_call = event.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        tool_calls += 1
        if tool_call.get("tool_name") != "zoom":
            continue

        zoom_calls += 1
        parameters = tool_call.get("parameters", {})
        code = parameters.get("code", "") if isinstance(parameters, dict) else ""
        event_regions, event_crop_calls, event_unscorable = extract_literal_crop_regions(code)
        regions.extend(event_regions)
        crop_calls += event_crop_calls
        if not event_regions and event_crop_calls == 0:
            event_unscorable += 1
        unscorable_crops += event_unscorable

    return {
        "regions": regions,
        "zoom_calls": zoom_calls,
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "crop_calls": crop_calls,
        "unscorable_crops": unscorable_crops,
        "trajectory_missing": False,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    zoom_rows = [row for row in rows if row["zoom_used"]]
    crop_attempts = sum(max(row["crop_calls"], row["zoom_calls"]) for row in rows)
    return {
        "total_samples": total,
        "accuracy": round(sum(row["correct"] for row in rows) / total, 4) if total else 0.0,
        "valid_answer_rate": round(sum(row["predicted"] is not None for row in rows) / total, 4) if total else 0.0,
        "zoom_rate": round(len(zoom_rows) / total, 4) if total else 0.0,
        "crop_hit_rate": round(sum(row["crop_hit"] for row in zoom_rows) / len(zoom_rows), 4) if zoom_rows else 0.0,
        "crop_hit_rate_all": round(sum(row["crop_hit"] for row in rows) / total, 4) if total else 0.0,
        "average_model_calls": round(sum(row["model_calls"] for row in rows) / total, 4) if total else 0.0,
        "p95_model_calls": percentile([row["model_calls"] for row in rows], 0.95),
        "average_tool_calls": round(sum(row["tool_calls"] for row in rows) / total, 4) if total else 0.0,
        "unscorable_crop_rate": round(
            sum(row["unscorable_crops"] for row in rows)
            / crop_attempts,
            4,
        ) if crop_attempts else 0.0,
        "missing_trajectory_count": sum(row["trajectory_missing"] for row in rows),
    }


def calculate_metrics(
    benchmark: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    trajectories_dir: Path,
    coverage_threshold: float = 0.5,
) -> dict[str, Any]:
    samples = {str(sample["doc_id"]): sample for sample in benchmark}
    evaluated: list[dict[str, Any]] = []

    for result in results:
        question_id = str(result.get("question_id", ""))
        if question_id not in samples:
            raise KeyError(f"Result question_id not found in benchmark: {question_id}")
        sample = samples[question_id]
        trajectory = _trajectory_metrics(_trajectory_path(trajectories_dir, result))
        predicted = extract_answer(result.get("final_answer"))
        expected = extract_answer(sample.get("solution"))
        model_calls = float(result.get("model_call_count", trajectory["model_calls"]))
        tool_calls = float(result.get("tool_call_count", trajectory["tool_calls"]))
        zoom_used = trajectory["zoom_calls"] > 0
        hit = zoom_used and evidence_hit(
            trajectory["regions"],
            sample.get("evidence", {}).get("bboxes_xywh", []),
            coverage_threshold,
        )
        evaluated.append(
            {
                "question_id": question_id,
                "task": sample.get("data_source", "unknown"),
                "predicted": predicted,
                "expected": expected,
                "correct": predicted is not None and predicted == expected,
                "zoom_used": zoom_used,
                "crop_hit": hit,
                "model_calls": model_calls,
                "tool_calls": tool_calls,
                **{
                    key: value
                    for key, value in trajectory.items()
                    if key not in {"regions", "model_calls", "tool_calls"}
                },
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        grouped[row["task"]].append(row)

    metrics = _aggregate(evaluated)
    metrics["coverage_threshold"] = coverage_threshold
    metrics["by_task"] = {task: _aggregate(rows) for task, rows in sorted(grouped.items())}
    metrics["samples"] = evaluated
    return metrics


COORDINATE_SEQUENCE = re.compile(
    r"[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]"
)


def audit_skill(skill_text: str, training_samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    known_boxes: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
    target_phrases: dict[str, list[str]] = defaultdict(list)
    for sample in training_samples:
        question_id = str(sample["doc_id"])
        evidence = sample.get("evidence", {})
        for box in evidence.get("bboxes_xywh", []):
            if len(box) == 4:
                known_boxes[tuple(int(value) for value in box)].append(question_id)
        for target in evidence.get("target_objects", []):
            normalized = str(target).strip().lower()
            if normalized:
                target_phrases[normalized].append(question_id)

    leaked_coordinates = []
    for match in COORDINATE_SEQUENCE.finditer(skill_text):
        box = tuple(int(value) for value in match.groups())
        if box in known_boxes:
            leaked_coordinates.append({"bbox_xywh": list(box), "sample_ids": known_boxes[box]})

    lowered_skill = skill_text.lower()
    mentioned_targets = [
        {"target": target, "sample_ids": sample_ids}
        for target, sample_ids in sorted(target_phrases.items())
        if target in lowered_skill
    ]
    return {
        "coordinate_leak_count": len(leaked_coordinates),
        "coordinate_leaks": leaked_coordinates,
        "target_phrase_mentions": mentioned_targets,
        "passed": not leaked_coordinates,
    }


def compare_with_baseline(
    baseline: dict[str, Any], candidate: dict[str, Any], min_crop_hit_gain: float = 0.1
) -> dict[str, Any]:
    tolerance = 1e-9
    checks = {
        "accuracy_non_regression": candidate["accuracy"] >= baseline["accuracy"],
        "crop_hit_gain": (
            candidate["crop_hit_rate"] - baseline["crop_hit_rate"]
            >= min_crop_hit_gain - tolerance
        ),
        "tool_calls_non_increasing": candidate["average_tool_calls"] <= baseline["average_tool_calls"],
        "leakage_passed": candidate.get("skill_audit", {}).get("passed", True),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "deltas": {
            "accuracy": round(candidate["accuracy"] - baseline["accuracy"], 4),
            "crop_hit_rate": round(candidate["crop_hit_rate"] - baseline["crop_hit_rate"], 4),
            "average_tool_calls": round(candidate["average_tool_calls"] - baseline["average_tool_calls"], 4),
        },
    }


def select_best_candidate(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    comparisons = []
    for order, candidate in enumerate(candidates):
        comparison = compare_with_baseline(baseline, candidate)
        comparisons.append(
            {
                "order": order,
                "metrics_path": candidate.get("metrics_path"),
                "skill_path": candidate.get("skill_path"),
                **comparison,
                "accuracy": candidate["accuracy"],
                "crop_hit_rate": candidate["crop_hit_rate"],
                "average_tool_calls": candidate["average_tool_calls"],
            }
        )

    eligible = [row for row in comparisons if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["accuracy"],
            -row["crop_hit_rate"],
            row["average_tool_calls"],
            row["order"],
        )
    )
    return {
        "selected": eligible[0] if eligible else None,
        "candidate_count": len(comparisons),
        "eligible_count": len(eligible),
        "comparisons": comparisons,
    }


def validate_pilot(
    metrics: dict[str, Any],
    skill_text: str,
    experiences: dict[str, Any],
    expected_rollouts: int,
) -> dict[str, Any]:
    experience_items = experiences.get("experiences", {})
    valid_answers = round(metrics.get("valid_answer_rate", 0.0) * expected_rollouts)
    checks = {
        "rollout_count": metrics.get("total_samples") == expected_rollouts,
        "at_least_15_valid_answers": valid_answers >= expected_rollouts - 1,
        "all_trajectories_present": metrics.get("missing_trajectory_count") == 0,
        "skill_frontmatter": skill_text.startswith("---\n") and "\n---\n" in skill_text[4:],
        "experiences_nonempty": isinstance(experience_items, dict) and bool(experience_items),
        "no_coordinate_leak": metrics.get("skill_audit", {}).get("passed", False),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "valid_answer_count": valid_answers,
        "expected_rollouts": expected_rollouts,
        "experience_count": len(experience_items) if isinstance(experience_items, dict) else 0,
    }


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


def _score(args: argparse.Namespace) -> None:
    benchmark = load_records(args.benchmark)
    metrics = calculate_metrics(
        benchmark,
        load_records(args.results),
        args.trajectories,
        args.coverage_threshold,
    )
    metrics["benchmark_path"] = str(args.benchmark)
    metrics["results_path"] = str(args.results)
    if args.skill:
        metrics["skill_path"] = str(args.skill)
        audit_benchmark = (
            load_records(args.audit_benchmark) if args.audit_benchmark else benchmark
        )
        metrics["skill_audit"] = audit_skill(
            args.skill.read_text(encoding="utf-8"), audit_benchmark
        )
    _write_json(metrics, args.output)


def _select(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidates = []
    for path in args.candidates:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["metrics_path"] = str(path)
        candidates.append(metrics)
    _write_json(select_best_candidate(baseline, candidates), args.output)


def _pilot_gate(args: argparse.Namespace) -> None:
    result = validate_pilot(
        json.loads(args.metrics.read_text(encoding="utf-8")),
        args.skill.read_text(encoding="utf-8"),
        json.loads(args.experiences.read_text(encoding="utf-8")),
        args.expected_rollouts,
    )
    _write_json(result, args.output)
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score one V* inference run")
    score.add_argument("--benchmark", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--trajectories", type=Path, required=True)
    score.add_argument("--coverage-threshold", type=float, default=0.5)
    score.add_argument("--skill", type=Path)
    score.add_argument("--audit-benchmark", type=Path)
    score.add_argument("--output", type=Path)
    score.set_defaults(handler=_score)

    select = subparsers.add_parser("select", help="Select a promoted dev Skill")
    select.add_argument("--baseline", type=Path, required=True)
    select.add_argument("--candidates", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path)
    select.set_defaults(handler=_select)

    pilot_gate = subparsers.add_parser(
        "pilot-gate", help="Validate the fixed V* pilot acceptance gate"
    )
    pilot_gate.add_argument("--metrics", type=Path, required=True)
    pilot_gate.add_argument("--skill", type=Path, required=True)
    pilot_gate.add_argument("--experiences", type=Path, required=True)
    pilot_gate.add_argument("--expected-rollouts", type=int, default=16)
    pilot_gate.add_argument("--output", type=Path)
    pilot_gate.set_defaults(handler=_pilot_gate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
