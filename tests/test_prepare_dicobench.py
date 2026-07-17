import json
import tempfile
import unittest
from pathlib import Path

from evilift.prepare_dicobench import TASK_NAMES, prepare_dicobench


class PrepareDiCoBenchTest(unittest.TestCase):
    def test_conversion_excludes_hidden_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions"
            questions.mkdir()
            source_file = "vstar_questions_from_difference_attribute.json"
            sample = {
                "image": "images/a.jpg",
                "output_image": "images/b.jpg",
                "question": "What is the difference?",
                "A": "red",
                "B": "blue",
                "C": "green",
                "D": "yellow",
                "E": "no visible difference",
                "answer": "A",
                "mask": [1, 2, 3, 4],
                "expanded_mask": [0, 0, 5, 5],
                "instruction": "hidden edit instruction",
            }
            (questions / source_file).write_text(json.dumps([sample]), encoding="utf-8")
            (root / "dataset_index.json").write_text(
                json.dumps({"datasets": [{"file": source_file}]}), encoding="utf-8"
            )

            output = root / "converted"
            counts = prepare_dicobench(root, output)
            records = json.loads((output / "all.json").read_text())

            self.assertEqual(counts[TASK_NAMES[source_file]], 1)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(
                record["images"], ["dicobench/images/a.jpg", "dicobench/images/b.jpg"]
            )
            self.assertEqual(record["problem"].count("<image>"), 2)
            serialized = json.dumps(record)
            self.assertNotIn("mask", serialized)
            self.assertNotIn("instruction", serialized)


if __name__ == "__main__":
    unittest.main()
