"""
Global Workspace — Conscious integration of all modules.

This module occupies Layer 6 of the cognitive architecture and
implements Baars' **Global Workspace Theory** (GWT).  The core idea:

1. **Specialist modules** (LLM, World Model, Self Model, Emotion,
   Embodied AI) each generate candidate *Thoughts*.
2. These thoughts **compete** for access to the workspace based on
   their *salience* — a composite score influenced by relevance,
   emotional charge, goal alignment, and novelty.
3. The **winning thought** is promoted to consciousness: it is
   wrapped in a :class:`BroadcastMessage` and broadcast to *all*
   modules via the :class:`EventBus`.
4. Modules **respond** to broadcasts by updating their internal
   state, generating new thoughts, and potentially initiating
   actions.

This creates a cyclic process — the *cognitive cycle* — that
unfolds over discrete time steps and produces a stream of
conscious experience.
"""

from __future__ import annotations

import logging
from typing import Any

from cognitive_architecture.core.event_bus import EventBus
from cognitive_architecture.core.types import (
    BroadcastMessage,
    EmotionalState,
    Goal,
    Thought,
)

logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """Layer 6 — The theatre of consciousness.

    Collects thoughts from all specialist modules, selects the most
    salient one, and broadcasts it to the entire system.
    """

    MODULE_NAME = "global_workspace"
    BROADCAST_TOPIC = "workspace.broadcast"

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.current_broadcast: BroadcastMessage | None = None
        self.broadcast_history: list[BroadcastMessage] = []
        self._cycle: int = 0

        # Attention parameters
        self._salience_threshold: float = 0.1
        self._recency_bonus: float = 0.05
        self._emotional_amplification: float = 0.3

    # ------------------------------------------------------------------
    # Core cognitive cycle
    # ------------------------------------------------------------------

    def compete_and_broadcast(
        self,
        candidate_thoughts: list[Thought],
        emotional_state: EmotionalState,
        active_goals: list[Goal],
        world_summary: dict[str, Any] | None = None,
    ) -> BroadcastMessage | None:
        """Run one cognitive cycle.

        1. Score and rank candidate thoughts.
        2. Select the winner.
        3. Build a :class:`BroadcastMessage`.
        4. Publish the broadcast to all subscribers.

        Returns the broadcast message, or ``None`` if no thought
        exceeded the salience threshold.
        """
        self._cycle += 1

        if not candidate_thoughts:
            logger.debug("Cycle %d: no candidate thoughts", self._cycle)
            return None

        scored = self._score_thoughts(candidate_thoughts, emotional_state, active_goals)

        if not scored:
            return None

        winner = scored[0][1]
        logger.info(
            "Cycle %d: winner = %r (score=%.3f)",
            self._cycle,
            winner.content[:60],
            scored[0][0],
        )

        broadcast = BroadcastMessage(
            thought=winner,
            emotional_context=emotional_state,
            active_goals=list(active_goals),
            world_summary=world_summary or {},
            cycle=self._cycle,
        )

        self.current_broadcast = broadcast
        self.broadcast_history.append(broadcast)
        self._trim_history()

        # Publish to all modules
        self.event_bus.publish(self.BROADCAST_TOPIC, broadcast)

        return broadcast

    # ------------------------------------------------------------------
    # Attention & scoring
    # ------------------------------------------------------------------

    def _score_thoughts(
        self,
        thoughts: list[Thought],
        emotional_state: EmotionalState,
        active_goals: list[Goal],
    ) -> list[tuple[float, Thought]]:
        """Score each thought and return a descending-sorted list.

        The score combines:
        - Base salience (set by the originating module)
        - Emotional amplification (high arousal boosts salience)
        - Goal relevance (thoughts related to active goals get a bonus)
        - Recency bonus (newer thoughts are slightly preferred)
        """
        scored: list[tuple[float, Thought]] = []

        for thought in thoughts:
            score = thought.salience

            # Emotional amplification
            score += emotional_state.arousal * self._emotional_amplification

            # Goal relevance bonus
            goal_bonus = self._goal_relevance(thought, active_goals)
            score += goal_bonus

            # Recency bonus (higher cycle → more recent)
            score += self._recency_bonus

            # Clamp
            score = max(0.0, min(2.0, score))

            if score >= self._salience_threshold:
                scored.append((score, thought))

        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    @staticmethod
    def _goal_relevance(thought: Thought, goals: list[Goal]) -> float:
        """Compute how much *thought* relates to active goals."""
        if not goals:
            return 0.0
        thought_words = set(thought.content.lower().split())
        best = 0.0
        for goal in goals:
            goal_words = set(goal.description.lower().split())
            overlap = len(thought_words & goal_words)
            if overlap > 0:
                relevance = min(0.3, overlap * 0.1) * goal.priority
                best = max(best, relevance)
        return best

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        """Return the current cycle number."""
        return self._cycle

    def get_consciousness_stream(self, last_n: int = 10) -> list[str]:
        """Return the last *n* conscious thoughts as strings."""
        return [
            f"[cycle {b.cycle}] {b.thought.content}"
            for b in self.broadcast_history[-last_n:]
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_history(self, max_size: int = 300) -> None:
        if len(self.broadcast_history) > max_size:
            self.broadcast_history = self.broadcast_history[-max_size:]
