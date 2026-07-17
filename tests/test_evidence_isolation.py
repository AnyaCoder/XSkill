import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_ROOT))

qwen_vl_utils = types.ModuleType("qwen_vl_utils")
qwen_vl_utils.fetch_image = lambda value: value
sys.modules.setdefault("qwen_vl_utils", qwen_vl_utils)

from engine.api_processors import _initialize_sample_and_image  # noqa: E402


class EvidenceIsolationTest(unittest.TestCase):
    def test_offline_evidence_is_logged_but_not_sent_to_model(self):
        sample = {
            "doc_id": "sample_1",
            "problem": "Choose the visible color.\n(A) red\n(B) blue",
            "solution": "A",
            "evidence": {
                "target_objects": ["PRIVATE_TARGET"],
                "bboxes_xywh": [[10, 20, 30, 40]],
            },
        }
        args = SimpleNamespace(
            image_folder="/unused",
            max_pixels=2_000_000,
            min_pixels=40_000,
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _initialize_sample_and_image(sample, args, directory)
            api_user_content = result[-1]
            trajectory = json.loads(
                (Path(directory) / "traj.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertIn("evidence", trajectory)
        self.assertNotIn("PRIVATE_TARGET", json.dumps(api_user_content))
        self.assertNotIn("10, 20, 30, 40", json.dumps(api_user_content))


if __name__ == "__main__":
    unittest.main()
