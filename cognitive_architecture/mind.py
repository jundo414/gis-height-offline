"""
Mind — Top-level integration of all cognitive modules.

The :class:`Mind` class wires together every layer of the cognitive
architecture and drives the cognitive cycle:

    ┌─────────────────────────────────────────────────┐
    │              Global Workspace (Layer 6)         │
    │   ┌─────┐ ┌───────┐ ┌──────┐ ┌───────┐ ┌────┐ │
    │   │ LLM │ │ World │ │ Self │ │Emotion│ │Body│ │
    │   │ (1) │ │  (2)  │ │ (3)  │ │  (4)  │ │(5) │ │
    │   └─────┘ └───────┘ └──────┘ └───────┘ └────┘ │
    └─────────────────────────────────────────────────┘

Each call to :meth:`step` executes one cognitive cycle:

1. Gather candidate thoughts from every module.
2. The Global Workspace selects the most salient thought.
3. The winning thought is broadcast to all modules.
4. Modules update their internal state in response.
5. If an active goal exists, the Embodied module proposes and
   executes an action.
6. The resulting percept is fed back into the World Model.

Usage::

    mind = Mind(agent_name="Kokoro")
    mind.add_goal("explore the environment")
    mind.receive_input("There is a forest ahead.")

    for _ in range(10):
        report = mind.step()
        print(report["conscious_thought"])
"""

from __future__ import annotations

import logging
from typing import Any

from cognitive_architecture.core.event_bus import EventBus
from cognitive_architecture.core.types import Percept, Thought
from cognitive_architecture.embodied.embodied_agent import EmbodiedAgent
from cognitive_architecture.emotion.emotion_module import EmotionModule
from cognitive_architecture.global_workspace.workspace import GlobalWorkspace
from cognitive_architecture.llm.language_module import LanguageModule
from cognitive_architecture.self_model.self_model import SelfModel
from cognitive_architecture.world_model.world_model import WorldModel

logger = logging.getLogger(__name__)


class Mind:
    """Unified cognitive system integrating all six layers.

    Parameters
    ----------
    agent_name:
        A human-readable name for the agent.
    grid_size:
        Size of the simulated 2-D world.
    """

    def __init__(
        self,
        agent_name: str = "Agent",
        grid_size: tuple[int, int] = (100, 100),
    ) -> None:
        # Infrastructure
        self.event_bus = EventBus()

        # Layer 1 – Language
        self.language = LanguageModule()

        # Layer 2 – World Model
        self.world_model = WorldModel(grid_size=grid_size)

        # Layer 3 – Self Model
        self.self_model = SelfModel(name=agent_name)

        # Layer 4 – Emotion
        self.emotion = EmotionModule()

        # Layer 5 – Embodied AI
        self.body = EmbodiedAgent(
            position=self.self_model.location,
            grid_bounds=(float(grid_size[0]), float(grid_size[1])),
        )

        # Layer 6 – Global Workspace
        self.workspace = GlobalWorkspace(event_bus=self.event_bus)

        # Wire modules to the broadcast bus
        self.event_bus.subscribe(GlobalWorkspace.BROADCAST_TOPIC, self.language.on_broadcast)
        self.event_bus.subscribe(GlobalWorkspace.BROADCAST_TOPIC, self.world_model.on_broadcast)
        self.event_bus.subscribe(GlobalWorkspace.BROADCAST_TOPIC, self.self_model.on_broadcast)
        self.event_bus.subscribe(GlobalWorkspace.BROADCAST_TOPIC, self.emotion.on_broadcast)
        self.event_bus.subscribe(GlobalWorkspace.BROADCAST_TOPIC, self.body.on_broadcast)

        # Cycle counter
        self._cycle: int = 0

        logger.info("Mind '%s' initialised.", agent_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> dict[str, Any]:
        """Execute one cognitive cycle and return a status report.

        Returns a dict with keys:

        - ``cycle`` — cycle number
        - ``conscious_thought`` — the winning thought (or None)
        - ``emotional_state`` — current emotional state
        - ``body_state`` — body position, energy, etc.
        - ``self_summary`` — self-model snapshot
        - ``world_summary`` — world-model snapshot
        - ``action_taken`` — action performed this cycle (or None)
        - ``action_result`` — percept from the action (or None)
        """
        self._cycle += 1

        # 1. Gather candidate thoughts from every module
        candidates = self._collect_thoughts()

        # 2. Emotion regulates goal priorities
        self.emotion.modulate_goal_priority(self.self_model.goals)

        # 3. Global Workspace: compete & broadcast
        broadcast = self.workspace.compete_and_broadcast(
            candidate_thoughts=candidates,
            emotional_state=self.emotion.current_state,
            active_goals=self.self_model.active_goals,
            world_summary=self.world_model.get_world_summary(),
        )

        # 4. Action execution (if a goal exists)
        action_taken = None
        action_result = None
        top_goal = self.self_model.top_goal
        if top_goal is not None:
            planned = self.body.plan_action(top_goal)
            if planned is not None:
                action_result = self.body.execute(planned)
                action_taken = planned
                # Feed action result back into the world model
                self.world_model.process_percept(action_result)
                # Emotion appraises the result
                self.emotion.appraise_percept(
                    action_result, self.self_model.active_goals
                )
                # Sync body position to self model
                self.self_model.update_location(self.body.position)

        # 5. Proprioceptive feedback
        proprio = self.body.generate_proprioceptive_percept()
        self.world_model.process_percept(proprio)

        # 6. Emotion regulation
        self.emotion.regulate()

        # Build report
        conscious = (
            broadcast.thought.content if broadcast is not None else None
        )

        return {
            "cycle": self._cycle,
            "conscious_thought": conscious,
            "emotional_state": str(self.emotion.current_state),
            "body_state": self.body.get_body_state(),
            "self_summary": self.self_model.get_self_summary(),
            "world_summary": self.world_model.get_world_summary(),
            "action_taken": str(action_taken) if action_taken else None,
            "action_result": str(action_result) if action_result else None,
        }

    def receive_input(self, text: str) -> dict[str, Any]:
        """Receive natural-language input and integrate it.

        The text is:
        1. Parsed by the Language Module.
        2. Converted into a linguistic percept.
        3. Integrated into the World Model as beliefs.
        4. Appraised by the Emotion Module.

        Returns the comprehension result.
        """
        percept = self.language.text_to_percept(text)
        self.world_model.process_percept(percept)
        self.emotion.appraise_percept(percept, self.self_model.active_goals)

        meaning = percept.content
        logger.info("Received input: %s → intent=%s", text, meaning.get("intent"))
        return meaning

    def add_goal(self, description: str, priority: float = 0.5) -> str:
        """Add a new goal and return its id."""
        goal = self.self_model.add_goal(description, priority)
        return goal.id

    def receive_percept(self, percept: Percept) -> None:
        """Inject a raw percept into the system."""
        self.world_model.process_percept(percept)
        self.emotion.appraise_percept(percept, self.self_model.active_goals)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stream_of_consciousness(self, last_n: int = 10) -> list[str]:
        """Return recent conscious thoughts."""
        return self.workspace.get_consciousness_stream(last_n)

    def inner_speech_log(self) -> list[str]:
        """Return the LLM inner-speech history."""
        return list(self.language.inner_speech)

    def emotional_history(self) -> list[str]:
        """Return formatted emotional history."""
        return [str(e) for e in self.emotion.emotion_history[-20:]]

    def autobiography(self) -> list[str]:
        """Return the agent's self-narrative."""
        return list(self.self_model.narrative)

    def full_report(self) -> dict[str, Any]:
        """Return a comprehensive snapshot of the entire mind."""
        return {
            "agent_name": self.self_model.name,
            "cycle": self._cycle,
            "consciousness_stream": self.stream_of_consciousness(),
            "inner_speech": self.inner_speech_log(),
            "emotional_state": str(self.emotion.current_state),
            "mood": str(self.emotion.mood),
            "body_state": self.body.get_body_state(),
            "self_summary": self.self_model.get_self_summary(),
            "world_summary": self.world_model.get_world_summary(),
            "autobiography": self.autobiography()[-10:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_thoughts(self) -> list[Thought]:
        """Ask every module to produce a candidate thought."""
        thoughts: list[Thought] = []

        thoughts.append(self.world_model.generate_thought())
        thoughts.append(self.self_model.generate_thought())
        thoughts.append(self.emotion.generate_thought())
        thoughts.append(self.body.generate_thought())

        # LLM generates inner speech only if there is a prior broadcast
        if self.workspace.current_broadcast is not None:
            thoughts.append(
                self.language.inner_monologue(self.workspace.current_broadcast)
            )

        return thoughts
