"""
Embodied Agent — Action and physicality.

This module occupies Layer 5 of the cognitive architecture.  It
bridges the gap between cognition and the physical world by providing:

* **Body simulation** — the agent has a body with position,
  orientation, energy reserves, and an inventory of held objects.
* **Motor control** — translating high-level action intents into
  concrete motor commands with physical constraints (range, energy,
  collision).
* **Proprioception** — generating internal percepts about the body's
  own state (fatigue, posture, contact).
* **Action feedback** — after executing an action the module returns
  a result percept that the World Model can integrate.

The embodied layer ensures that cognition is always *situated*: the
agent cannot "think" its way out of physical constraints.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from cognitive_architecture.core.types import (
    Action,
    ActionType,
    BroadcastMessage,
    Goal,
    Modality,
    Percept,
    Thought,
)

logger = logging.getLogger(__name__)


class EmbodiedAgent:
    """Layer 5 — Physical body and motor system.

    Simulates a simple agent body on a 2-D grid with energy
    management, object manipulation, and locomotion.
    """

    MODULE_NAME = "embodied"

    def __init__(
        self,
        position: tuple[float, float] = (50.0, 50.0),
        orientation: float = 0.0,
        max_speed: float = 2.0,
        grid_bounds: tuple[float, float] = (100.0, 100.0),
    ) -> None:
        self.position = position
        self.orientation = orientation  # degrees
        self.max_speed = max_speed
        self.grid_bounds = grid_bounds

        self.energy: float = 1.0
        self.inventory: list[str] = []  # ids of held objects
        self.max_inventory: int = 5
        self.action_history: list[Action] = []

        # Body state
        self.is_moving: bool = False
        self.contact_objects: list[str] = []

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute(self, action: Action) -> Percept:
        """Execute *action* and return a proprioceptive feedback percept.

        Each action type is dispatched to a dedicated handler.  All
        handlers enforce energy constraints and spatial boundaries.
        """
        if self.energy <= 0:
            return self._feedback("exhausted", {"reason": "no energy"}, intensity=0.9)

        handler = {
            ActionType.MOVE: self._do_move,
            ActionType.GRASP: self._do_grasp,
            ActionType.RELEASE: self._do_release,
            ActionType.SPEAK: self._do_speak,
            ActionType.LOOK: self._do_look,
            ActionType.LISTEN: self._do_listen,
            ActionType.WAIT: self._do_wait,
            ActionType.THINK: self._do_think,
        }.get(action.action_type)

        if handler is None:
            return self._feedback(
                "unknown_action",
                {"action_type": action.action_type.name},
                intensity=0.3,
            )

        result = handler(action)
        self.energy = max(0.0, self.energy - action.energy_cost)
        self.action_history.append(action)
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

        return result

    def plan_action(self, goal: Goal) -> Action | None:
        """Propose an action that advances *goal*.

        Simple keyword-based planning — a real system would use a
        proper planner.
        """
        desc = goal.description.lower()

        if any(w in desc for w in ("move", "go", "walk", "reach", "approach")):
            target = goal.description.split("to")[-1].strip() if "to" in goal.description else ""
            return Action(
                action_type=ActionType.MOVE,
                parameters={"direction": target or "forward", "speed": 1.0},
                energy_cost=0.1,
            )
        if any(w in desc for w in ("pick", "grab", "take", "grasp", "get")):
            return Action(
                action_type=ActionType.GRASP,
                parameters={"target": desc},
                energy_cost=0.15,
            )
        if any(w in desc for w in ("say", "speak", "tell", "communicate")):
            return Action(
                action_type=ActionType.SPEAK,
                parameters={"utterance": goal.description},
                energy_cost=0.05,
            )
        if any(w in desc for w in ("look", "observe", "see", "find", "search")):
            return Action(
                action_type=ActionType.LOOK,
                parameters={"direction": "around"},
                energy_cost=0.03,
            )
        if any(w in desc for w in ("listen", "hear")):
            return Action(
                action_type=ActionType.LISTEN,
                parameters={},
                energy_cost=0.02,
            )
        if any(w in desc for w in ("rest", "wait", "pause")):
            return Action(
                action_type=ActionType.WAIT,
                parameters={},
                energy_cost=0.0,
            )
        if any(w in desc for w in ("think", "plan", "consider", "reflect")):
            return Action(
                action_type=ActionType.THINK,
                parameters={"topic": goal.description},
                energy_cost=0.02,
            )

        # Default: think about the goal
        return Action(
            action_type=ActionType.THINK,
            parameters={"topic": goal.description},
            energy_cost=0.02,
        )

    # ------------------------------------------------------------------
    # Proprioception
    # ------------------------------------------------------------------

    def get_body_state(self) -> dict[str, Any]:
        """Return current body state as a dict."""
        return {
            "position": self.position,
            "orientation": self.orientation,
            "energy": round(self.energy, 2),
            "is_moving": self.is_moving,
            "inventory": list(self.inventory),
            "contact_objects": list(self.contact_objects),
        }

    def generate_proprioceptive_percept(self) -> Percept:
        """Generate a percept representing the body's current state."""
        return Percept(
            modality=Modality.PROPRIOCEPTIVE,
            content=self.get_body_state(),
            intensity=0.3 if self.energy > 0.3 else 0.7,
            source="self",
        )

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    def generate_thought(self) -> Thought:
        """Generate a body-awareness thought for the workspace competition."""
        if self.energy < 0.15:
            content = f"My body is nearly exhausted (energy={self.energy:.0%})."
            salience = 0.85
        elif self.is_moving:
            content = f"I am moving through position {self.position}."
            salience = 0.4
        elif self.inventory:
            content = f"I am holding: {', '.join(self.inventory)}."
            salience = 0.35
        else:
            content = f"My body is at {self.position}, orientation {self.orientation:.0f}."
            salience = 0.2

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
            # Recover a tiny bit of energy each cycle when not acting
            if not self.is_moving:
                self.energy = min(1.0, self.energy + 0.005)

    # ------------------------------------------------------------------
    # Motor handlers
    # ------------------------------------------------------------------

    def _do_move(self, action: Action) -> Percept:
        params = action.parameters
        speed = min(params.get("speed", 1.0), self.max_speed)
        direction = params.get("direction", "forward")

        dx, dy = self._direction_to_delta(direction, speed)
        new_x = max(0.0, min(self.grid_bounds[0], self.position[0] + dx))
        new_y = max(0.0, min(self.grid_bounds[1], self.position[1] + dy))

        self.position = (new_x, new_y)
        self.is_moving = True

        return self._feedback(
            "moved",
            {"new_position": self.position, "direction": direction, "speed": speed},
        )

    def _do_grasp(self, action: Action) -> Percept:
        target = action.parameters.get("target", "unknown")
        if len(self.inventory) >= self.max_inventory:
            return self._feedback(
                "grasp_failed",
                {"reason": "inventory full", "target": target},
                intensity=0.6,
            )
        self.inventory.append(target)
        return self._feedback("grasped", {"target": target})

    def _do_release(self, action: Action) -> Percept:
        target = action.parameters.get("target")
        if target and target in self.inventory:
            self.inventory.remove(target)
            return self._feedback("released", {"target": target})
        if self.inventory:
            released = self.inventory.pop()
            return self._feedback("released", {"target": released})
        return self._feedback("release_failed", {"reason": "nothing to release"}, intensity=0.4)

    def _do_speak(self, action: Action) -> Percept:
        utterance = action.parameters.get("utterance", "...")
        return self._feedback("spoke", {"utterance": utterance}, modality=Modality.AUDITORY)

    def _do_look(self, action: Action) -> Percept:
        direction = action.parameters.get("direction", "forward")
        return self._feedback(
            "looked",
            {"direction": direction, "position": self.position},
            modality=Modality.VISUAL,
        )

    def _do_listen(self, action: Action) -> Percept:
        return self._feedback(
            "listened",
            {"position": self.position},
            modality=Modality.AUDITORY,
            intensity=0.3,
        )

    def _do_wait(self, action: Action) -> Percept:
        self.is_moving = False
        self.energy = min(1.0, self.energy + 0.05)  # resting recovers energy
        return self._feedback("waited", {"energy_recovered": 0.05}, intensity=0.1)

    def _do_think(self, action: Action) -> Percept:
        topic = action.parameters.get("topic", "nothing in particular")
        return self._feedback(
            "thought_about",
            {"topic": topic},
            modality=Modality.INTEROCEPTIVE,
            intensity=0.2,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _direction_to_delta(self, direction: str, speed: float) -> tuple[float, float]:
        direction_map: dict[str, tuple[float, float]] = {
            "forward": (0.0, speed),
            "backward": (0.0, -speed),
            "left": (-speed, 0.0),
            "right": (speed, 0.0),
            "north": (0.0, speed),
            "south": (0.0, -speed),
            "east": (speed, 0.0),
            "west": (-speed, 0.0),
        }
        if direction.lower() in direction_map:
            return direction_map[direction.lower()]
        # Default: move in the direction of current orientation
        rad = math.radians(self.orientation)
        return (speed * math.sin(rad), speed * math.cos(rad))

    @staticmethod
    def _feedback(
        description: str,
        details: dict[str, Any],
        modality: Modality = Modality.PROPRIOCEPTIVE,
        intensity: float = 0.5,
    ) -> Percept:
        return Percept(
            modality=modality,
            content={"action_result": description, **details},
            intensity=intensity,
            source="self",
        )
