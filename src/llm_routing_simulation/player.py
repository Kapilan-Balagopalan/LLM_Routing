"""Player interfaces for history-based LLM routing algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True)
class PlayerDecision:
    action: int
    diagnostics: Any = None


class HistoryBasedPlayer(ABC):
    """Bandit-style player: act from context, then receive partial feedback."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = context_dim
        self.actions: list[int] = []
        self.contexts: list[np.ndarray] = []
        self.outcomes: list[int | None] = []
        self._revealed_count = 0
        self._revealed_outcome_counts = [0, 0]
        self._pending_context: np.ndarray | None = None
        self._pending_action: int | None = None

    @abstractmethod
    def next_action(self, context: np.ndarray) -> PlayerDecision:
        """Return 0 for weak or 1 for strong."""

    def update(
        self, action: int, context: np.ndarray, outcome: int | None
    ) -> None:
        """Commit one environment transition to the player's history."""
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError(f"Expected context dimension {self.context_dim}")
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        if outcome is not None and outcome not in (0, 1):
            raise ValueError("Outcome must be 0, 1, or None")
        if action == 0 and outcome is not None:
            raise ValueError("Weak action must not reveal disagreement feedback")
        if self._pending_action != action or self._pending_context is None:
            raise RuntimeError("update must follow next_action for the same action")
        if not np.array_equal(self._pending_context, x):
            raise RuntimeError("update context differs from the acted-on context")
        self.actions.append(action)
        self.contexts.append(x.copy())
        self.outcomes.append(outcome)
        if outcome is not None:
            self._revealed_count += 1
            self._revealed_outcome_counts[outcome] += 1
        self._pending_context = None
        self._pending_action = None

    def get_history(
        self,
    ) -> tuple[list[int], list[np.ndarray], list[int | None]]:
        return self.actions.copy(), [x.copy() for x in self.contexts], self.outcomes.copy()
