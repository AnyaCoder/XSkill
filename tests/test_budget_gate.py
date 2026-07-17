import unittest

from eval.utils.budget_gate import BudgetState


class BudgetStateTest(unittest.TestCase):
    def test_direct_answer_uses_one_call(self):
        state = BudgetState(enabled=True, max_tool_turns=1, configured_max_turns=16)
        state.record_model_call()
        self.assertEqual(state.max_model_turns, 2)
        self.assertTrue(state.metrics()["direct_answer"])

    def test_tool_turn_disables_tools_for_final_call(self):
        state = BudgetState(enabled=True, max_tool_turns=1, configured_max_turns=16)
        state.record_model_call()
        self.assertTrue(state.allow_tools)
        state.record_tool_call()
        self.assertFalse(state.allow_tools)
        state.record_model_call()
        self.assertEqual(state.metrics()["model_call_count"], 2)
        self.assertEqual(state.metrics()["tool_call_count"], 1)

    def test_disabled_gate_preserves_original_turn_limit(self):
        state = BudgetState(enabled=False, max_tool_turns=1, configured_max_turns=16)
        state.record_tool_call()
        self.assertEqual(state.max_model_turns, 16)
        self.assertTrue(state.allow_tools)


if __name__ == "__main__":
    unittest.main()
