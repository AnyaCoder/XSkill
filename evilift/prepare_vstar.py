"""Prepare leakage-safe V* splits in XSkill's benchmark format."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_RATIOS = (("train", 0.6), ("dev", 0.2), ("test", 0.2))


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


def prepare_vstar(input_path: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
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

    manifest = {
        "source": str(input_path),
        "seed": seed,
        "split_strategy": "sha256(image_path, seed)",
        "counts": counts,
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
    args = parser.parse_args()
    print(json.dumps(prepare_vstar(args.input, args.output_dir, args.seed)))


if __name__ == "__main__":
    main()

