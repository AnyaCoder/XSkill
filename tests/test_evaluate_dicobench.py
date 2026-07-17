import unittest

from evilift.evaluate_dicobench import calculate_metrics, extract_answer


class EvaluateDiCoBenchTest(unittest.TestCase):
    def test_extract_answer(self):
        self.assertEqual(extract_answer("Evidence. <answer>b</answer>"), "B")
        self.assertEqual(extract_answer(" E "), "E")
        self.assertIsNone(extract_answer("unknown"))

    def test_metrics_include_budget_and_null_scores(self):
        metrics = calculate_metrics(
            [
                {
                    "final_answer": "<answer>A</answer>",
                    "ground_truth": "A",
                    "task_signature": "Difference.Attribute",
                    "model_call_count": 1,
                    "tool_call_count": 0,
                    "direct_answer": True,
                    "budget_violation": False,
                },
                {
                    "final_answer": "<answer>E</answer>",
                    "ground_truth": "E",
                    "task_signature": "Difference.Attribute",
                    "model_call_count": 2,
                    "tool_call_count": 1,
                    "direct_answer": False,
                    "budget_violation": False,
                },
            ]
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["average_model_calls"], 1.5)
        self.assertEqual(metrics["p95_model_calls"], 2.0)
        self.assertEqual(metrics["null_option"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

