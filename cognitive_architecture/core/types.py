"""
Shared types and data structures for the cognitive architecture.

These types form the common vocabulary used across all modules,
enabling them to communicate through the Global Workspace.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Modality(Enum):
    """Sensory modality for percepts."""
    VISUAL = auto()
    AUDITORY = auto()
    TACTILE = auto()
    PROPRIOCEPTIVE = auto()
    INTEROCEPTIVE = auto()
    LINGUISTIC = auto()


class EmotionType(Enum):
    """Basic emotion categories (Ekman + extensions)."""
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    SURPRISE = auto()
    DISGUST = auto()
    TRUST = auto()
    ANTICIPATION = auto()
    CURIOSITY = auto()
    NEUTRAL = auto()


class ActionType(Enum):
    """Types of actions the embodied agent can perform."""
    MOVE = auto()
    GRASP = auto()
    RELEASE = auto()
    SPEAK = auto()
    LOOK = auto()
    LISTEN = auto()
    WAIT = auto()
    THINK = auto()


class GoalStatus(Enum):
    """Status of a goal."""
    ACTIVE = auto()
    ACHIEVED = auto()
    FAILED = auto()
    SUSPENDED = auto()


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Percept:
    """A unit of sensory input from the environment.

    Percepts are the raw building blocks of experience — they arrive
    from the environment (or from the body itself) and feed into the
    World Model and the Emotion Module.
    """
    modality: Modality
    content: dict[str, Any]
    intensity: float = 1.0  # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    source: str = "environment"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Percept({self.modality.name}, {self.content}, intensity={self.intensity:.2f})"


@dataclass
class Entity:
    """An object or agent in the world model.

    Entities populate the internal world representation.  Each has
    a unique id, a type tag, a set of properties, and a location
    expressed as (x, y) coordinates.
    """
    id: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    location: tuple[float, float] = (0.0, 0.0)

    def __repr__(self) -> str:
        return f"Entity({self.id!r}, type={self.entity_type!r}, loc={self.location})"


@dataclass
class Belief:
    """An internal proposition the agent holds about the world.

    Beliefs are derived from percepts, reasoning, or communication.
    They carry a confidence score that may be revised over time.
    """
    proposition: str
    confidence: float = 1.0  # 0.0 – 1.0
    source: str = "perception"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Belief({self.proposition!r}, conf={self.confidence:.2f})"


@dataclass
class Goal:
    """A desired state the agent is trying to achieve.

    Goals are prioritized by the Emotion Module and pursued by the
    Embodied AI module through action planning.
    """
    description: str
    priority: float = 0.5  # 0.0 – 1.0
    status: GoalStatus = GoalStatus.ACTIVE
    parent_goal: str | None = None
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Goal({self.description!r}, pri={self.priority:.2f}, {self.status.name})"


@dataclass
class EmotionalState:
    """The agent's current emotional state, represented in three dimensions.

    Uses the PAD (Pleasure-Arousal-Dominance) model:
    - valence:   pleasure ↔ displeasure  (−1 .. +1)
    - arousal:   calm ↔ excited          (0 .. 1)
    - dominance: submissive ↔ dominant   (0 .. 1)

    The discrete `primary_emotion` label is derived from the
    dimensional values for interpretability.
    """
    valence: float = 0.0      # −1.0 … +1.0
    arousal: float = 0.0      # 0.0 … 1.0
    dominance: float = 0.5    # 0.0 … 1.0
    primary_emotion: EmotionType = EmotionType.NEUTRAL
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"EmotionalState({self.primary_emotion.name}, "
            f"v={self.valence:+.2f}, a={self.arousal:.2f}, d={self.dominance:.2f})"
        )


@dataclass
class Action:
    """A motor command issued by the Embodied AI module.

    Actions are the agent's interface with the external world.
    They can succeed, fail, or be partially executed.
    """
    action_type: ActionType
    parameters: dict[str, Any] = field(default_factory=dict)
    energy_cost: float = 0.1  # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Action({self.action_type.name}, {self.parameters})"


@dataclass
class Thought:
    """A unit of conscious processing in the Global Workspace.

    Thoughts compete for access to the workspace; winners are
    broadcast to all specialist modules.
    """
    content: str
    source_module: str
    salience: float = 0.5  # 0.0 – 1.0
    associations: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Thought({self.content!r}, from={self.source_module}, sal={self.salience:.2f})"


@dataclass
class BroadcastMessage:
    """A message broadcast through the Global Workspace to all modules.

    When a thought wins the competition for consciousness, it is
    wrapped in a BroadcastMessage and sent to every specialist.
    """
    thought: Thought
    emotional_context: EmotionalState
    active_goals: list[Goal] = field(default_factory=list)
    world_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    cycle: int = 0

    def __repr__(self) -> str:
        return (
            f"Broadcast(cycle={self.cycle}, thought={self.thought.content!r}, "
            f"emotion={self.emotional_context.primary_emotion.name})"
        )


@dataclass
class MemoryRecord:
    """A record stored in episodic or working memory.

    Memory records are tagged with emotional valence so that
    emotionally significant events are recalled more easily.
    """
    content: str
    memory_type: str  # "episodic", "semantic", "working"
    emotional_valence: float = 0.0  # −1.0 … +1.0
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    associations: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        return f"Memory({self.content!r}, type={self.memory_type}, imp={self.importance:.2f})"
