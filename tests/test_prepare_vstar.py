import json
import tempfile
import unittest
from pathlib import Path

from evilift.prepare_vstar import convert_sample, prepare_vstar, select_pilot_samples


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

            pilot = json.loads((output / "pilot.json").read_text())
            train_ids = {
                record["doc_id"]
                for record in json.loads((output / "train.json").read_text())
            }
            self.assertLessEqual(len(pilot), 8)
            self.assertTrue({record["doc_id"] for record in pilot} <= train_ids)

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

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertIn("sample_ids", manifest)

    def test_pilot_is_stratified_by_task_and_target_scale(self):
        samples = []
        for task in ("direct_attributes", "relative_position"):
            for index in range(8):
                samples.append(
                    {
                        "input_image": f"{task}/image_{index}.jpg",
                        "question": "Where is the target?",
                        "text": "Where is the target?\n(A) left\n(B) right",
                        "label": "A",
                        "test_type": task,
                        "target_object": ["target"],
                        "bbox": [[0, 0, index + 1, index + 1]],
                    }
                )

        converted = [convert_sample(sample, index) for index, sample in enumerate(samples)]
        pilot = select_pilot_samples(samples, converted, Path("/missing"), pilot_size=8)

        task_counts = {}
        selected_areas = {}
        for record in pilot:
            task = record["data_source"]
            task_counts[task] = task_counts.get(task, 0) + 1
            box = record["evidence"]["bboxes_xywh"][0]
            selected_areas.setdefault(task, []).append(box[2] * box[3])

        self.assertEqual(task_counts["vstar_direct_attributes"], 4)
        self.assertEqual(task_counts["vstar_relative_position"], 4)
        self.assertEqual(min(selected_areas["vstar_direct_attributes"]), 1)
        self.assertEqual(max(selected_areas["vstar_direct_attributes"]), 64)


if __name__ == "__main__":
    unittest.main()
