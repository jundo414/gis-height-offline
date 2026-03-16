"""
Emotion Module — Priority and value assignment.

This module occupies Layer 4 of the cognitive architecture.  Emotions
are *not* noise to be filtered out; they are the mechanism by which
the agent assigns **value** and **urgency** to events, goals, and
actions.  The module implements:

* **Appraisal** — evaluating events along dimensions of relevance,
  congruence with goals, novelty, and controllability.
* **Dimensional model** — the PAD (Pleasure-Arousal-Dominance) space
  provides a continuous representation of emotional state.
* **Discrete labelling** — PAD coordinates are mapped to Ekman-style
  emotion categories for interpretability.
* **Mood** — a slow-moving baseline emotional state that biases
  appraisal over time.
* **Emotion regulation** — gradual decay toward equilibrium.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from cognitive_architecture.core.types import (
    BroadcastMessage,
    EmotionalState,
    EmotionType,
    Goal,
    Percept,
    Thought,
)

logger = logging.getLogger(__name__)


# Mapping from PAD regions to discrete emotion labels.
_PAD_EMOTION_MAP: list[tuple[EmotionType, float, float, float]] = [
    #                       valence  arousal  dominance
    (EmotionType.JOY,          0.8,    0.5,    0.6),
    (EmotionType.SADNESS,     -0.7,    0.2,    0.3),
    (EmotionType.ANGER,       -0.6,    0.8,    0.7),
    (EmotionType.FEAR,        -0.7,    0.8,    0.2),
    (EmotionType.SURPRISE,     0.2,    0.9,    0.4),
    (EmotionType.DISGUST,     -0.5,    0.5,    0.6),
    (EmotionType.TRUST,        0.5,    0.3,    0.5),
    (EmotionType.ANTICIPATION, 0.4,    0.6,    0.5),
    (EmotionType.CURIOSITY,    0.3,    0.6,    0.5),
    (EmotionType.NEUTRAL,      0.0,    0.0,    0.5),
]


class EmotionModule:
    """Layer 4 — Emotional processing and value assignment.

    Maintains both a *current emotion* (fast, event-driven) and a
    *mood* (slow-moving average).  The current emotion influences
    attention, goal priority, and action selection.
    """

    MODULE_NAME = "emotion"

    def __init__(self) -> None:
        self.current_state = EmotionalState()
        self.mood = EmotionalState()  # slow baseline
        self.emotion_history: list[EmotionalState] = []
        self._decay_rate: float = 0.05
        self._mood_inertia: float = 0.95  # how slowly mood changes

    # ------------------------------------------------------------------
    # Core appraisal
    # ------------------------------------------------------------------

    def appraise_percept(
        self,
        percept: Percept,
        active_goals: list[Goal] | None = None,
    ) -> EmotionalState:
        """Evaluate a percept and update the emotional state.

        Appraisal dimensions:
        - **relevance**: how much does this percept matter?
        - **goal_congruence**: does it help or hinder active goals?
        - **novelty**: is it unexpected?
        - **controllability**: can the agent do something about it?
        """
        relevance = self._assess_relevance(percept, active_goals or [])
        congruence = self._assess_goal_congruence(percept, active_goals or [])
        novelty = self._assess_novelty(percept)
        controllability = self._assess_controllability(percept)

        # Map appraisal dimensions to PAD
        valence = congruence * relevance
        arousal = (novelty + relevance) / 2.0
        dominance = controllability

        self._update_state(valence, arousal, dominance)
        return self.current_state

    def appraise_thought(self, thought: Thought) -> EmotionalState:
        """Evaluate an internal thought and update emotional state."""
        content_lower = thought.content.lower()

        valence_shift = 0.0
        arousal_shift = 0.0

        # Positive content
        for word in ("success", "achieve", "good", "happy", "safe", "love"):
            if word in content_lower:
                valence_shift += 0.2
        # Negative content
        for word in ("fail", "danger", "pain", "lost", "threat", "error"):
            if word in content_lower:
                valence_shift -= 0.2
        # Arousing content
        for word in ("urgent", "surprise", "sudden", "new", "unknown"):
            if word in content_lower:
                arousal_shift += 0.15

        self._update_state(
            self.current_state.valence + valence_shift,
            min(1.0, self.current_state.arousal + arousal_shift),
            self.current_state.dominance,
        )
        return self.current_state

    def appraise_prediction_error(self, error_magnitude: float) -> EmotionalState:
        """React to a prediction error from the World Model.

        Large errors trigger surprise/fear; small errors may trigger
        curiosity.
        """
        if error_magnitude > 0.7:
            self._update_state(-0.3, 0.9, 0.2)  # fear / surprise
        elif error_magnitude > 0.4:
            self._update_state(0.1, 0.6, 0.5)   # curiosity
        else:
            self._update_state(0.05, 0.3, 0.6)  # mild interest

        return self.current_state

    # ------------------------------------------------------------------
    # Goal-emotion interaction
    # ------------------------------------------------------------------

    def modulate_goal_priority(self, goals: list[Goal]) -> list[Goal]:
        """Re-weight goal priorities based on current emotional state.

        Fear → boost safety goals.  Joy → boost exploration goals.
        Low energy (high fatigue) → boost rest goals.
        """
        emotion = self.current_state.primary_emotion
        for goal in goals:
            desc = goal.description.lower()
            if emotion == EmotionType.FEAR:
                if any(w in desc for w in ("safety", "escape", "avoid", "protect")):
                    goal.priority = min(1.0, goal.priority + 0.2)
            elif emotion == EmotionType.JOY:
                if any(w in desc for w in ("explore", "play", "learn", "discover")):
                    goal.priority = min(1.0, goal.priority + 0.15)
            elif emotion == EmotionType.CURIOSITY:
                if any(w in desc for w in ("explore", "investigate", "learn")):
                    goal.priority = min(1.0, goal.priority + 0.2)
            elif emotion == EmotionType.SADNESS:
                if any(w in desc for w in ("rest", "recover", "comfort")):
                    goal.priority = min(1.0, goal.priority + 0.15)

        goals.sort(key=lambda g: g.priority, reverse=True)
        return goals

    # ------------------------------------------------------------------
    # Regulation & decay
    # ------------------------------------------------------------------

    def regulate(self) -> None:
        """Gradually decay the emotional state toward baseline (mood).

        This models natural emotion regulation — intense emotions
        fade over time unless re-triggered.
        """
        # Decay current state toward mood
        self.current_state.valence += (
            self.mood.valence - self.current_state.valence
        ) * self._decay_rate
        self.current_state.arousal += (
            self.mood.arousal - self.current_state.arousal
        ) * self._decay_rate
        self.current_state.dominance += (
            self.mood.dominance - self.current_state.dominance
        ) * self._decay_rate

        # Slowly update mood toward current state
        self.mood.valence += (
            self.current_state.valence - self.mood.valence
        ) * (1.0 - self._mood_inertia)
        self.mood.arousal += (
            self.current_state.arousal - self.mood.arousal
        ) * (1.0 - self._mood_inertia)

        self.current_state.primary_emotion = self._classify_emotion(
            self.current_state.valence,
            self.current_state.arousal,
            self.current_state.dominance,
        )

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    def generate_thought(self) -> Thought:
        """Generate an emotion-driven thought for the workspace competition."""
        state = self.current_state
        emotion_name = state.primary_emotion.name.lower()

        if abs(state.valence) < 0.1 and state.arousal < 0.2:
            content = "Emotionally calm — no strong feelings right now."
            salience = 0.15
        else:
            intensity = "strongly" if state.arousal > 0.6 else "mildly"
            content = (
                f"I {intensity} feel {emotion_name} "
                f"(valence={state.valence:+.2f}, arousal={state.arousal:.2f})."
            )
            salience = 0.3 + state.arousal * 0.4

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
            self.appraise_thought(data.thought)
            self.regulate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_state(self, valence: float, arousal: float, dominance: float) -> None:
        self.current_state.valence = max(-1.0, min(1.0, valence))
        self.current_state.arousal = max(0.0, min(1.0, arousal))
        self.current_state.dominance = max(0.0, min(1.0, dominance))
        self.current_state.primary_emotion = self._classify_emotion(
            self.current_state.valence,
            self.current_state.arousal,
            self.current_state.dominance,
        )
        self.emotion_history.append(
            EmotionalState(
                valence=self.current_state.valence,
                arousal=self.current_state.arousal,
                dominance=self.current_state.dominance,
                primary_emotion=self.current_state.primary_emotion,
            )
        )
        if len(self.emotion_history) > 200:
            self.emotion_history = self.emotion_history[-200:]

    @staticmethod
    def _classify_emotion(valence: float, arousal: float, dominance: float) -> EmotionType:
        best_emotion = EmotionType.NEUTRAL
        best_distance = float("inf")
        for etype, ev, ea, ed in _PAD_EMOTION_MAP:
            dist = math.sqrt(
                (valence - ev) ** 2 + (arousal - ea) ** 2 + (dominance - ed) ** 2
            )
            if dist < best_distance:
                best_distance = dist
                best_emotion = etype
        return best_emotion

    @staticmethod
    def _assess_relevance(percept: Percept, goals: list[Goal]) -> float:
        base = percept.intensity * 0.5
        if goals:
            base += 0.3
        return min(1.0, base)

    @staticmethod
    def _assess_goal_congruence(percept: Percept, goals: list[Goal]) -> float:
        if not goals:
            return 0.0
        content_str = str(percept.content).lower()
        for goal in goals:
            desc = goal.description.lower()
            if any(word in content_str for word in desc.split()):
                return 0.5
        return 0.0

    @staticmethod
    def _assess_novelty(percept: Percept) -> float:
        return min(1.0, percept.intensity * 0.7)

    @staticmethod
    def _assess_controllability(percept: Percept) -> float:
        if percept.source == "self":
            return 0.8
        return 0.4
