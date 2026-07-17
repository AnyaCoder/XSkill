"""Prepare leakage-safe V* splits in XSkill's benchmark format."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


SPLIT_RATIOS = (("train", 0.6), ("dev", 0.2), ("test", 0.2))
DEFAULT_PILOT_SIZE = 8


def _stable_bucket(image_path: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{image_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _split_name(image_path: str, seed: int) -> str:
    bucket = _stable_bucket(image_path, seed)
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS:
        cumulative += ratio
        if bucket < cumulative:
            return name
    return SPLIT_RATIOS[-1][0]


def _format_problem(sample: dict[str, Any]) -> str:
    if sample.get("text"):
        return f"<image>\n{sample['text'].strip()}"

    option_lines = [
        f"({chr(65 + index)}) {option}"
        for index, option in enumerate(sample.get("options", []))
    ]
    return "\n".join(
        [
            "<image>",
            sample["question"].strip(),
            *option_lines,
            "Answer with the option's letter from the given choices directly.",
        ]
    )


def convert_sample(sample: dict[str, Any], index: int) -> dict[str, Any]:
    image_path = sample["input_image"]
    return {
        "doc_id": f"vstar_{index:04d}",
        "images": [f"vstar/{image_path}"],
        "problem": _format_problem(sample),
        "solution": sample.get("label"),
        "data_source": f"vstar_{sample.get('test_type', 'unknown')}",
        "evidence": {
            "target_objects": sample.get("target_object", []),
            "bboxes_xywh": sample.get("bbox", []),
            "source_image": image_path,
        },
    }


def _source_digest(input_path: Path) -> str:
    return hashlib.sha256(input_path.read_bytes()).hexdigest()


def _largest_bbox_area_ratio(sample: dict[str, Any], dataset_root: Path) -> float:
    bboxes = sample.get("bbox", [])
    if not bboxes:
        return 0.0

    try:
        with Image.open(dataset_root / sample["input_image"]) as image:
            image_area = image.width * image.height
    except (KeyError, OSError):
        image_area = 0

    largest_bbox = max(float(width) * float(height) for _, _, width, height in bboxes)
    return largest_bbox / image_area if image_area > 0 else largest_bbox


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]

    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def select_pilot_samples(
    raw_samples: list[dict[str, Any]],
    converted_samples: list[dict[str, Any]],
    dataset_root: Path,
    pilot_size: int = DEFAULT_PILOT_SIZE,
) -> list[dict[str, Any]]:
    """Select a deterministic task- and target-scale-stratified training pilot."""
    if pilot_size <= 0 or not converted_samples:
        return []

    raw_by_id = {
        f"vstar_{index:04d}": sample for index, sample in enumerate(raw_samples)
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for converted in converted_samples:
        raw = raw_by_id[converted["doc_id"]]
        ranked = dict(converted)
        ranked["_area_ratio"] = _largest_bbox_area_ratio(raw, dataset_root)
        by_task[converted["data_source"]].append(ranked)

    task_names = sorted(by_task)
    allocation = {task: pilot_size // len(task_names) for task in task_names}
    for task in task_names[: pilot_size % len(task_names)]:
        allocation[task] += 1

    selected: list[dict[str, Any]] = []
    for task in task_names:
        ranked = sorted(by_task[task], key=lambda row: (row["_area_ratio"], row["doc_id"]))
        for row in _evenly_spaced(ranked, min(allocation[task], len(ranked))):
            row.pop("_area_ratio", None)
            selected.append(row)
    return sorted(selected, key=lambda row: row["doc_id"])


def _task_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(sample["data_source"] for sample in samples).items()))


def prepare_vstar(
    input_path: Path,
    output_dir: Path,
    seed: int = 42,
    pilot_size: int = DEFAULT_PILOT_SIZE,
) -> dict[str, int]:
    samples = json.loads(input_path.read_text(encoding="utf-8"))
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, sample in enumerate(samples):
        split = _split_name(sample["input_image"], seed)
        splits[split].append(convert_sample(sample, index))

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, _ in SPLIT_RATIOS:
        split_samples = splits[split_name]
        counts[split_name] = len(split_samples)
        (output_dir / f"{split_name}.json").write_text(
            json.dumps(split_samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    pilot_samples = select_pilot_samples(
        samples,
        splits["train"],
        input_path.parent,
        pilot_size=pilot_size,
    )
    (output_dir / "pilot.json").write_text(
        json.dumps(pilot_samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source": str(input_path),
        "source_sha256": _source_digest(input_path),
        "seed": seed,
        "split_strategy": "sha256(image_path, seed)",
        "counts": counts,
        "task_counts": {
            split_name: _task_counts(splits[split_name])
            for split_name, _ in SPLIT_RATIOS
        },
        "sample_ids": {
            split_name: [sample["doc_id"] for sample in splits[split_name]]
            for split_name, _ in SPLIT_RATIOS
        },
        "pilot": {
            "size": len(pilot_samples),
            "selection": "task-stratified target-area quartiles from train",
            "task_counts": _task_counts(pilot_samples),
            "sample_ids": [sample["doc_id"] for sample in pilot_samples],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_vstar(args.input, args.output_dir, args.seed, args.pilot_size)
        )
    )


if __name__ == "__main__":
    main()
