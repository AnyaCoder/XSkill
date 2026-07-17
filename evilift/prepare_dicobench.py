"""Convert DiCoBench into XSkill's multi-image benchmark format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_NAMES = {
    "vstar_questions_from_common_category.json": "Commonality.Category",
    "vstar_questions_from_common_instance.json": "Commonality.Instance",
    "vstar_questions_from_common_reason.json": "Commonality.Reasoning",
    "vstar_questions_from_common_spatial.json": "Commonality.Spatial",
    "vstar_questions_from_difference_attribute.json": "Difference.Attribute",
    "vstar_questions_from_difference_object.json": "Difference.Entity",
    "vstar_questions_from_difference_reason.json": "Difference.Reasoning",
    "vstar_questions_from_difference_spatial.json": "Difference.Spatial",
}


def _format_problem(sample: dict[str, Any], task_name: str) -> str:
    options = [f"({letter}) {sample[letter]}" for letter in "ABCDE"]
    return "\n".join(
        [
            "<image>",
            "<image>",
            f"[Task: {task_name}]",
            sample["question"].strip(),
            *options,
            "Answer with the option's letter from the given choices directly.",
        ]
    )


def convert_sample(
    sample: dict[str, Any], task_name: str, source_file: str, index: int
) -> dict[str, Any]:
    task_id = task_name.lower().replace(".", "_")
    return {
        "doc_id": f"{task_id}_{index:04d}",
        "images": [
            f"dicobench/{sample['image']}",
            f"dicobench/{sample['output_image']}",
        ],
        "problem": _format_problem(sample, task_name),
        "solution": sample["answer"],
        "data_source": task_id,
        "task_signature": task_name,
        "source_file": source_file,
    }


def prepare_dicobench(dataset_root: Path, output_dir: Path) -> dict[str, int]:
    index = json.loads((dataset_root / "dataset_index.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for dataset in index["datasets"]:
        source_file = dataset["file"]
        task_name = TASK_NAMES[source_file]
        source_path = dataset_root / "questions" / source_file
        source_samples = json.loads(source_path.read_text(encoding="utf-8"))
        converted = [
            convert_sample(sample, task_name, source_file, sample_index)
            for sample_index, sample in enumerate(source_samples)
        ]
        counts[task_name] = len(converted)
        all_samples.extend(converted)
        destination = output_dir / f"{task_name.lower().replace('.', '_')}.json"
        destination.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (output_dir / "all.json").write_text(
        json.dumps(all_samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({"counts": counts, "total": len(all_samples)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_dicobench(args.dataset_root, args.output_dir)))


if __name__ == "__main__":
    main()

