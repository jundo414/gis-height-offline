"""
Self Model — Representation of self within the world.

This module occupies Layer 3 of the cognitive architecture.  It
provides the agent with:

* **Body schema** — a representation of the agent's physical form,
  location, orientation, and energy level.
* **Capabilities** — what the agent can and cannot do.
* **Autobiographical memory** — a narrative record of significant
  past experiences.
* **Metacognition** — awareness of the agent's own cognitive
  states (confidence, confusion, surprise, etc.).

The Self Model answers the question: *"Where and what am I in this
world?"*  Without it, the World Model would be a third-person map
with no anchor point.
"""

from __future__ import annotations

import logging
from typing import Any

from cognitive_architecture.core.types import (
    BroadcastMessage,
    EmotionalState,
    EmotionType,
    Goal,
    GoalStatus,
    MemoryRecord,
    Thought,
)

logger = logging.getLogger(__name__)


class SelfModel:
    """Layer 3 — The agent's model of itself.

    Tracks physical state, cognitive state, goals, capabilities,
    and autobiographical memory.
    """

    MODULE_NAME = "self_model"

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

        # Physical self-representation
        self.location: tuple[float, float] = (50.0, 50.0)
        self.orientation: float = 0.0  # degrees
        self.energy: float = 1.0       # 0.0 – 1.0
        self.health: float = 1.0       # 0.0 – 1.0

        # Capabilities the agent knows it has
        self.capabilities: set[str] = {
            "move", "grasp", "release", "speak", "look", "listen", "think",
        }

        # Goal stack
        self.goals: list[Goal] = []

        # Autobiographical memory
        self.autobiographical_memory: list[MemoryRecord] = []

        # Metacognitive state
        self.metacognition: dict[str, float] = {
            "confidence": 0.5,
            "confusion": 0.0,
            "curiosity": 0.3,
            "fatigue": 0.0,
        }

        # Self-narrative (a running story of "who I am")
        self.narrative: list[str] = [f"I am {self.name}. I have just come into being."]

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def add_goal(self, description: str, priority: float = 0.5) -> Goal:
        """Create and register a new goal."""
        goal = Goal(description=description, priority=priority)
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        self.narrative.append(f"New goal: {description} (priority {priority:.2f}).")
        logger.debug("Added goal: %s", goal)
        return goal

    def achieve_goal(self, goal_id: str) -> None:
        """Mark a goal as achieved."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.ACHIEVED
                self.narrative.append(f"Achieved goal: {goal.description}.")
                self._record_memory(
                    f"Successfully achieved: {goal.description}",
                    emotional_valence=0.5,
                    importance=0.7,
                )
                break

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        """Mark a goal as failed."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.FAILED
                self.narrative.append(f"Failed goal: {goal.description}. Reason: {reason}")
                self._record_memory(
                    f"Failed: {goal.description} — {reason}",
                    emotional_valence=-0.5,
                    importance=0.6,
                )
                break

    @property
    def active_goals(self) -> list[Goal]:
        """Return currently active goals, sorted by priority."""
        return [g for g in self.goals if g.status == GoalStatus.ACTIVE]

    @property
    def top_goal(self) -> Goal | None:
        """Return the highest-priority active goal, or None."""
        active = self.active_goals
        return active[0] if active else None

    # ------------------------------------------------------------------
    # Physical state
    # ------------------------------------------------------------------

    def update_location(self, new_location: tuple[float, float]) -> None:
        """Move the agent's self-representation to a new location."""
        old = self.location
        self.location = new_location
        self.energy = max(0.0, self.energy - 0.02)  # movement costs energy
        self.metacognition["fatigue"] = 1.0 - self.energy
        logger.debug("Moved from %s to %s (energy=%.2f)", old, new_location, self.energy)

    def consume_energy(self, amount: float) -> None:
        """Decrease energy by *amount* (clamped to [0, 1])."""
        self.energy = max(0.0, min(1.0, self.energy - amount))
        self.metacognition["fatigue"] = 1.0 - self.energy

    def recover_energy(self, amount: float) -> None:
        """Increase energy by *amount* (clamped to [0, 1])."""
        self.energy = max(0.0, min(1.0, self.energy + amount))
        self.metacognition["fatigue"] = 1.0 - self.energy

    # ------------------------------------------------------------------
    # Metacognition
    # ------------------------------------------------------------------

    def assess_confidence(self, beliefs_count: int, prediction_errors: int) -> float:
        """Re-evaluate overall confidence based on cognitive performance."""
        if beliefs_count == 0:
            self.metacognition["confidence"] = 0.3
        else:
            error_rate = prediction_errors / max(beliefs_count, 1)
            self.metacognition["confidence"] = max(0.1, 1.0 - error_rate)
        return self.metacognition["confidence"]

    def assess_emotional_impact(self, emotional_state: EmotionalState) -> None:
        """Update metacognitive awareness based on current emotional state."""
        if emotional_state.arousal > 0.7:
            self.metacognition["confusion"] = min(
                1.0, self.metacognition["confusion"] + 0.1
            )
        if emotional_state.primary_emotion == EmotionType.CURIOSITY:
            self.metacognition["curiosity"] = min(
                1.0, self.metacognition["curiosity"] + 0.15
            )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def recall(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        """Retrieve the most relevant autobiographical memories.

        Simple keyword match ranked by importance × recency.
        """
        scored: list[tuple[float, MemoryRecord]] = []
        query_words = set(query.lower().split())
        for mem in self.autobiographical_memory:
            overlap = len(query_words & set(mem.content.lower().split()))
            score = overlap * mem.importance
            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_self_summary(self) -> dict[str, Any]:
        """Return a structured summary of the agent's self-knowledge."""
        return {
            "name": self.name,
            "location": self.location,
            "energy": round(self.energy, 2),
            "health": round(self.health, 2),
            "capabilities": sorted(self.capabilities),
            "active_goals": [
                {"description": g.description, "priority": g.priority}
                for g in self.active_goals
            ],
            "metacognition": {k: round(v, 2) for k, v in self.metacognition.items()},
            "memory_count": len(self.autobiographical_memory),
            "narrative_length": len(self.narrative),
        }

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    def generate_thought(self) -> Thought:
        """Generate a self-model thought for the workspace competition."""
        top = self.top_goal
        meta = self.metacognition

        if self.energy < 0.2:
            content = f"I am running low on energy ({self.energy:.0%}). I should rest."
            salience = 0.8
        elif meta["confusion"] > 0.6:
            content = "I am confused and need to re-orient myself."
            salience = 0.7
        elif top is not None:
            content = (
                f"I am {self.name} at {self.location}. "
                f"My primary goal is: {top.description}."
            )
            salience = 0.5 + top.priority * 0.3
        else:
            content = f"I am {self.name} at {self.location}, with no active goals."
            salience = 0.3

        return Thought(
            content=content,
            source_module=self.MODULE_NAME,
            salience=salience,
        )

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def on_broadcast(self, topic: str, data: Any) -> None:
        """Handle a broadcast from the Global Workspace."""
        if isinstance(data, BroadcastMessage):
            self.assess_emotional_impact(data.emotional_context)
            # Record significant broadcasts as memories
            if data.thought.salience > 0.6:
                self._record_memory(
                    f"Conscious moment: {data.thought.content}",
                    emotional_valence=data.emotional_context.valence,
                    importance=data.thought.salience,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_memory(
        self,
        content: str,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
    ) -> None:
        record = MemoryRecord(
            content=content,
            memory_type="episodic",
            emotional_valence=emotional_valence,
            importance=importance,
        )
        self.autobiographical_memory.append(record)

        if len(self.autobiographical_memory) > 500:
            self.autobiographical_memory.sort(
                key=lambda m: m.importance, reverse=True
            )
            self.autobiographical_memory = self.autobiographical_memory[:500]
