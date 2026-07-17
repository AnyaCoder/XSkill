import json
import tempfile
import unittest
from pathlib import Path

from evilift.evaluate_vstar import (
    audit_skill,
    calculate_metrics,
    compare_with_baseline,
    evidence_hit,
    extract_literal_crop_regions,
    select_best_candidate,
    validate_pilot,
)


class EvaluateVStarTest(unittest.TestCase):
    def test_literal_crop_extraction_rejects_dynamic_coordinates(self):
        regions, crop_calls, unscorable = extract_literal_crop_regions(
            "display(original_image.crop((10, 20, 30, 40)))\n"
            "box = (1, 2, 3, 4)\noriginal_image.crop(box)"
        )

        self.assertEqual(regions, [(10.0, 20.0, 30.0, 40.0)])
        self.assertEqual(crop_calls, 2)
        self.assertEqual(unscorable, 1)

    def test_evidence_hit_requires_every_annotated_target(self):
        targets = [[10, 10, 10, 10], [50, 50, 10, 10]]
        self.assertFalse(evidence_hit([(5, 5, 25, 25)], targets))
        self.assertTrue(
            evidence_hit([(5, 5, 25, 25), (45, 45, 65, 65)], targets)
        )

    def test_metrics_join_results_with_trajectory(self):
        benchmark = [
            {
                "doc_id": "sample_1",
                "solution": "A",
                "data_source": "vstar_direct_attributes",
                "evidence": {"bboxes_xywh": [[10, 10, 10, 10]]},
            }
        ]
        results = [
            {
                "question_id": "sample_1",
                "final_answer": "<answer>A</answer>",
                "model_call_count": 2,
                "tool_call_count": 1,
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            trajectories = Path(directory)
            sample_dir = trajectories / "sample_1"
            sample_dir.mkdir()
            events = [
                {"turn_idx": 0, "text_output": "zoom"},
                {
                    "turn_idx": 0,
                    "tool_call": {
                        "tool_name": "zoom",
                        "parameters": {
                            "code": "display(original_image.crop((5, 5, 25, 25)))"
                        },
                    },
                },
                {"turn_idx": 1, "text_output": "<answer>A</answer>"},
            ]
            (sample_dir / "traj.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            metrics = calculate_metrics(benchmark, results, trajectories)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["crop_hit_rate"], 1.0)
        self.assertEqual(metrics["average_model_calls"], 2.0)
        self.assertEqual(metrics["average_tool_calls"], 1.0)

    def test_skill_audit_and_promotion_gate(self):
        training = [
            {
                "doc_id": "sample_1",
                "evidence": {
                    "bboxes_xywh": [[10, 20, 30, 40]],
                    "target_objects": ["silver watch"],
                },
            }
        ]
        audit = audit_skill(
            "Inspect the silver watch with crop coordinates (10, 20, 30, 40).",
            training,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["coordinate_leak_count"], 1)
        self.assertEqual(audit["target_phrase_mentions"][0]["target"], "silver watch")

        baseline = {
            "accuracy": 0.6,
            "crop_hit_rate": 0.4,
            "average_tool_calls": 1.0,
        }
        passing = {
            "accuracy": 0.6,
            "crop_hit_rate": 0.5,
            "average_tool_calls": 1.0,
            "skill_path": "early.md",
            "skill_audit": {"passed": True},
        }
        better = {
            "accuracy": 0.7,
            "crop_hit_rate": 0.6,
            "average_tool_calls": 1.0,
            "skill_path": "later.md",
            "skill_audit": {"passed": True},
        }
        self.assertTrue(compare_with_baseline(baseline, passing)["eligible"])
        selection = select_best_candidate(baseline, [passing, better])
        self.assertEqual(selection["selected"]["skill_path"], "later.md")

    def test_pilot_gate_validates_artifacts(self):
        result = validate_pilot(
            {
                "total_samples": 16,
                "valid_answer_rate": 15 / 16,
                "missing_trajectory_count": 0,
                "skill_audit": {"passed": True},
            },
            "---\nname: evidence-search\n---\n# Evidence Search\n",
            {"experiences": {"E0": "When details are tiny, zoom first."}},
            16,
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
