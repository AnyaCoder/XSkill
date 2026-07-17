"""State tracking for evidence-gated model and tool call budgets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetState:
    enabled: bool
    max_tool_turns: int
    configured_max_turns: int
    model_call_count: int = 0
    tool_call_count: int = 0
    budget_violation: bool = False

    @property
    def max_model_turns(self) -> int:
        if not self.enabled:
            return self.configured_max_turns
        return min(self.configured_max_turns, self.max_tool_turns + 1)

    @property
    def allow_tools(self) -> bool:
        return not self.enabled or self.tool_call_count < self.max_tool_turns

    def record_model_call(self) -> None:
        self.model_call_count += 1

    def record_tool_call(self) -> None:
        self.tool_call_count += 1

    def record_violation(self) -> None:
        self.budget_violation = True

    def metrics(self) -> dict[str, int | bool]:
        return {
            "model_call_count": self.model_call_count,
            "tool_call_count": self.tool_call_count,
            "direct_answer": self.model_call_count == 1 and self.tool_call_count == 0,
            "budget_violation": self.budget_violation,
        }

