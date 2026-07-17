import json
import tempfile
import unittest
from pathlib import Path

from evilift.prepare_vstar import prepare_vstar


class PrepareVStarTest(unittest.TestCase):
    def test_image_groups_do_not_cross_splits(self):
        samples = []
        for index in range(20):
            samples.append(
                {
                    "input_image": f"direct_attributes/image_{index // 2}.jpg",
                    "question": "What color is the target?",
                    "text": "What color is the target?\n(A) red\n(B) blue",
                    "label": "A",
                    "test_type": "direct_attributes",
                    "target_object": ["target"],
                    "bbox": [[1, 2, 3, 4]],
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "splits"
            source.write_text(json.dumps(samples), encoding="utf-8")
            counts = prepare_vstar(source, output, seed=42)

            self.assertEqual(sum(counts.values()), len(samples))
            image_splits = {}
            for split in ("train", "dev", "test"):
                records = json.loads((output / f"{split}.json").read_text())
                for record in records:
                    image = record["evidence"]["source_image"]
                    previous_split = image_splits.setdefault(image, split)
                    self.assertEqual(previous_split, split)

    def test_output_uses_xskill_schema_and_preserves_evidence(self):
        sample = {
            "input_image": "direct_attributes/sample.jpg",
            "question": "What is visible?",
            "text": "What is visible?\n(A) comb\n(B) brush",
            "label": "A",
            "test_type": "direct_attributes",
            "target_object": ["comb"],
            "bbox": [[10, 20, 30, 40]],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "splits"
            source.write_text(json.dumps([sample]), encoding="utf-8")
            prepare_vstar(source, output)

            records = []
            for split in ("train", "dev", "test"):
                records.extend(json.loads((output / f"{split}.json").read_text()))

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["images"], ["vstar/direct_attributes/sample.jpg"])
            self.assertEqual(record["solution"], "A")
            self.assertEqual(record["evidence"]["bboxes_xywh"], [[10, 20, 30, 40]])


if __name__ == "__main__":
    unittest.main()

