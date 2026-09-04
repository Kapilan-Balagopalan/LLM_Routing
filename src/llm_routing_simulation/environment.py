"""Partial-monitoring environment for an LLM cascade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CascadeRound:
    example_id: str
    prompt: str
    context: np.ndarray
    weak_answer: str
    strong_answer: str
    gold_answer: str
    outcome_override: int | None = None

    def __post_init__(self) -> None:
        if self.outcome_override not in (0, 1, None):
            raise ValueError("outcome_override must be 0, 1, or None")

    @property
    def routing_outcome(self) -> int:
        """Return the environment outcome, synthetic when explicitly supplied."""
        if self.outcome_override is not None:
            return self.outcome_override
        return int(self.weak_answer != self.strong_answer)


@dataclass(frozen=True)
class EnvironmentObservation:
    t: int
    example_id: str
    prompt: str
    context: np.ndarray
    weak_answer: str
    gold_answer: str


@dataclass(frozen=True)
class EnvironmentTransition:
    t: int
    example_id: str
    action: int
    outcome: int | None
    revealed_strong_answer: str | None


class LLMCascadeEnvironment:
    """Own the LLM-derived stream and reveal feedback according to action."""

    def __init__(self, rounds: list[CascadeRound]) -> None:
        if not rounds:
            raise ValueError("Environment needs at least one round")
        self.rounds = rounds
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.rounds)

    def observe(self) -> EnvironmentObservation:
        if self.done:
            raise RuntimeError("Environment is finished")
        item = self.rounds[self.index]
        return EnvironmentObservation(
            self.index + 1,
            item.example_id,
            item.prompt,
            item.context.copy(),
            item.weak_answer,
            item.gold_answer,
        )

    def step(self, action: int) -> EnvironmentTransition:
        if self.done:
            raise RuntimeError("Environment is finished")
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        item = self.rounds[self.index]
        if action == 1:
            outcome = item.routing_outcome
            revealed_strong = item.strong_answer
        else:
            outcome = None
            revealed_strong = None
        transition = EnvironmentTransition(
            t=self.index + 1,
            example_id=item.example_id,
            action=action,
            outcome=outcome,
            revealed_strong_answer=revealed_strong,
        )
        self.index += 1
        return transition
