"""Tests for the cognitive architecture."""

from __future__ import annotations

from cognitive_architecture import Mind
from cognitive_architecture.core.event_bus import EventBus
from cognitive_architecture.core.types import (
    Action,
    ActionType,
    Belief,
    EmotionalState,
    EmotionType,
    Entity,
    Goal,
    GoalStatus,
    MemoryRecord,
    Modality,
    Percept,
    Thought,
)
from cognitive_architecture.embodied.embodied_agent import EmbodiedAgent
from cognitive_architecture.emotion.emotion_module import EmotionModule
from cognitive_architecture.global_workspace.workspace import GlobalWorkspace
from cognitive_architecture.llm.language_module import LanguageModule
from cognitive_architecture.self_model.self_model import SelfModel
from cognitive_architecture.world_model.world_model import WorldModel

# ======================================================================
# Core types
# ======================================================================

class TestCoreTypes:
    def test_percept_creation(self) -> None:
        p = Percept(modality=Modality.VISUAL, content={"color": "red"})
        assert p.modality == Modality.VISUAL
        assert p.content["color"] == "red"
        assert 0.0 <= p.intensity <= 1.0

    def test_entity_creation(self) -> None:
        e = Entity(id="rock", entity_type="object", location=(10.0, 20.0))
        assert e.id == "rock"
        assert e.location == (10.0, 20.0)

    def test_belief_creation(self) -> None:
        b = Belief(proposition="The sky is blue", confidence=0.9)
        assert b.proposition == "The sky is blue"
        assert b.confidence == 0.9

    def test_goal_status(self) -> None:
        g = Goal(description="find food", priority=0.8)
        assert g.status == GoalStatus.ACTIVE

    def test_emotional_state_defaults(self) -> None:
        e = EmotionalState()
        assert e.valence == 0.0
        assert e.primary_emotion == EmotionType.NEUTRAL

    def test_action_creation(self) -> None:
        a = Action(action_type=ActionType.MOVE, parameters={"direction": "north"})
        assert a.action_type == ActionType.MOVE

    def test_thought_creation(self) -> None:
        t = Thought(content="I wonder why", source_module="test")
        assert t.source_module == "test"

    def test_memory_record(self) -> None:
        m = MemoryRecord(content="saw a cat", memory_type="episodic", importance=0.8)
        assert m.memory_type == "episodic"


# ======================================================================
# EventBus
# ======================================================================

class TestEventBus:
    def test_publish_subscribe(self) -> None:
        bus = EventBus()
        received: list = []
        bus.subscribe("test.topic", lambda t, d: received.append(d))
        bus.publish("test.topic", "hello")
        assert received == ["hello"]

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list = []
        handler = lambda t, d: received.append(d)  # noqa: E731
        bus.subscribe("t", handler)
        bus.unsubscribe("t", handler)
        bus.publish("t", "x")
        assert received == []

    def test_history(self) -> None:
        bus = EventBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        assert len(bus.history) == 2


# ======================================================================
# Language Module
# ======================================================================

class TestLanguageModule:
    def test_comprehend_greeting(self) -> None:
        lm = LanguageModule()
        result = lm.comprehend("Hello there!")
        assert result["intent"] == "greeting"

    def test_comprehend_question(self) -> None:
        lm = LanguageModule()
        result = lm.comprehend("Where is the forest?")
        assert result["intent"] == "question"

    def test_sentiment_positive(self) -> None:
        lm = LanguageModule()
        result = lm.comprehend("This is a wonderful good day")
        assert result["sentiment"] > 0.0

    def test_sentiment_negative(self) -> None:
        lm = LanguageModule()
        result = lm.comprehend("This is a terrible horrible day")
        assert result["sentiment"] < 0.0

    def test_text_to_percept(self) -> None:
        lm = LanguageModule()
        p = lm.text_to_percept("Look at that tree")
        assert p.modality == Modality.LINGUISTIC

    def test_generate(self) -> None:
        lm = LanguageModule()
        t = Thought(content="I observe a bird", source_module="test")
        text = lm.generate(t)
        assert isinstance(text, str)
        assert len(text) > 0


# ======================================================================
# World Model
# ======================================================================

class TestWorldModel:
    def test_add_entity(self) -> None:
        wm = WorldModel()
        wm.add_entity(Entity(id="tree", entity_type="plant", location=(10.0, 10.0)))
        assert "tree" in wm.entities

    def test_remove_entity(self) -> None:
        wm = WorldModel()
        wm.add_entity(Entity(id="rock", entity_type="object", location=(5.0, 5.0)))
        wm.remove_entity("rock")
        assert "rock" not in wm.entities

    def test_process_visual_percept(self) -> None:
        wm = WorldModel()
        p = Percept(
            modality=Modality.VISUAL,
            content={"entity_id": "bird", "entity_type": "animal", "location": (20.0, 30.0)},
        )
        beliefs = wm.process_percept(p)
        assert len(beliefs) > 0
        assert "bird" in wm.entities

    def test_predict(self) -> None:
        wm = WorldModel()
        pred = wm.predict("move to the north")
        assert "agent_location_change" in pred["predicted_changes"]

    def test_entities_near(self) -> None:
        wm = WorldModel()
        wm.add_entity(Entity(id="a", entity_type="x", location=(10.0, 10.0)))
        wm.add_entity(Entity(id="b", entity_type="x", location=(50.0, 50.0)))
        nearby = wm.entities_near((11.0, 11.0), radius=5.0)
        assert len(nearby) == 1
        assert nearby[0].id == "a"

    def test_generate_thought(self) -> None:
        wm = WorldModel()
        t = wm.generate_thought()
        assert t.source_module == "world_model"


# ======================================================================
# Self Model
# ======================================================================

class TestSelfModel:
    def test_add_goal(self) -> None:
        sm = SelfModel(name="Test")
        goal = sm.add_goal("find water", 0.7)
        assert goal.description == "find water"
        assert len(sm.active_goals) == 1

    def test_achieve_goal(self) -> None:
        sm = SelfModel()
        g = sm.add_goal("rest")
        sm.achieve_goal(g.id)
        assert g.status == GoalStatus.ACHIEVED
        assert len(sm.active_goals) == 0

    def test_energy_management(self) -> None:
        sm = SelfModel()
        sm.consume_energy(0.3)
        assert sm.energy < 1.0
        sm.recover_energy(0.1)
        assert sm.energy > 0.7 - 0.01  # float tolerance

    def test_metacognition(self) -> None:
        sm = SelfModel()
        conf = sm.assess_confidence(beliefs_count=10, prediction_errors=2)
        assert 0.0 <= conf <= 1.0

    def test_narrative(self) -> None:
        sm = SelfModel(name="Kokoro")
        assert any("Kokoro" in n for n in sm.narrative)

    def test_generate_thought(self) -> None:
        sm = SelfModel()
        t = sm.generate_thought()
        assert t.source_module == "self_model"


# ======================================================================
# Emotion Module
# ======================================================================

class TestEmotionModule:
    def test_initial_neutral(self) -> None:
        em = EmotionModule()
        assert em.current_state.primary_emotion == EmotionType.NEUTRAL

    def test_appraise_percept(self) -> None:
        em = EmotionModule()
        p = Percept(modality=Modality.VISUAL, content={"threat": True}, intensity=0.9)
        state = em.appraise_percept(p)
        assert isinstance(state, EmotionalState)

    def test_appraise_prediction_error(self) -> None:
        em = EmotionModule()
        state = em.appraise_prediction_error(0.8)
        assert state.arousal > 0.5

    def test_regulation(self) -> None:
        em = EmotionModule()
        em.appraise_prediction_error(0.9)
        initial_arousal = em.current_state.arousal
        for _ in range(20):
            em.regulate()
        assert em.current_state.arousal < initial_arousal

    def test_goal_modulation(self) -> None:
        em = EmotionModule()
        em.appraise_prediction_error(0.9)  # trigger fear
        goals = [
            Goal(description="explore the cave", priority=0.5),
            Goal(description="find safety shelter", priority=0.5),
        ]
        em.modulate_goal_priority(goals)
        safety_goal = next(g for g in goals if "safety" in g.description)
        assert safety_goal.priority > 0.5

    def test_generate_thought(self) -> None:
        em = EmotionModule()
        t = em.generate_thought()
        assert t.source_module == "emotion"


# ======================================================================
# Embodied Agent
# ======================================================================

class TestEmbodiedAgent:
    def test_move(self) -> None:
        agent = EmbodiedAgent(position=(50.0, 50.0))
        action = Action(
            action_type=ActionType.MOVE, parameters={"direction": "north", "speed": 1.0}
        )
        result = agent.execute(action)
        assert agent.position != (50.0, 50.0)
        assert result.modality == Modality.PROPRIOCEPTIVE

    def test_grasp(self) -> None:
        agent = EmbodiedAgent()
        action = Action(action_type=ActionType.GRASP, parameters={"target": "apple"})
        agent.execute(action)
        assert "apple" in agent.inventory

    def test_release(self) -> None:
        agent = EmbodiedAgent()
        agent.inventory.append("key")
        action = Action(action_type=ActionType.RELEASE, parameters={"target": "key"})
        agent.execute(action)
        assert "key" not in agent.inventory

    def test_energy_depletion(self) -> None:
        agent = EmbodiedAgent()
        for _ in range(20):
            action = Action(
                action_type=ActionType.MOVE, parameters={"direction": "north"}, energy_cost=0.1
            )
            agent.execute(action)
        assert agent.energy < 0.1

    def test_plan_action(self) -> None:
        agent = EmbodiedAgent()
        goal = Goal(description="move to the forest")
        action = agent.plan_action(goal)
        assert action is not None
        assert action.action_type == ActionType.MOVE

    def test_body_state(self) -> None:
        agent = EmbodiedAgent()
        state = agent.get_body_state()
        assert "position" in state
        assert "energy" in state


# ======================================================================
# Global Workspace
# ======================================================================

class TestGlobalWorkspace:
    def test_compete_and_broadcast(self) -> None:
        bus = EventBus()
        received: list = []
        bus.subscribe("workspace.broadcast", lambda t, d: received.append(d))

        gw = GlobalWorkspace(event_bus=bus)
        thoughts = [
            Thought(content="low priority", source_module="a", salience=0.1),
            Thought(content="high priority", source_module="b", salience=0.9),
        ]
        broadcast = gw.compete_and_broadcast(
            thoughts, EmotionalState(), []
        )
        assert broadcast is not None
        assert "high priority" in broadcast.thought.content
        assert len(received) == 1

    def test_empty_candidates(self) -> None:
        gw = GlobalWorkspace(event_bus=EventBus())
        result = gw.compete_and_broadcast([], EmotionalState(), [])
        assert result is None

    def test_consciousness_stream(self) -> None:
        gw = GlobalWorkspace(event_bus=EventBus())
        for i in range(5):
            gw.compete_and_broadcast(
                [Thought(content=f"thought {i}", source_module="t", salience=0.5)],
                EmotionalState(),
                [],
            )
        stream = gw.get_consciousness_stream(3)
        assert len(stream) == 3


# ======================================================================
# Mind (integration)
# ======================================================================

class TestMind:
    def test_creation(self) -> None:
        mind = Mind(agent_name="TestAgent")
        assert mind.self_model.name == "TestAgent"

    def test_receive_input(self) -> None:
        mind = Mind()
        result = mind.receive_input("Hello world")
        assert "intent" in result

    def test_add_goal(self) -> None:
        mind = Mind()
        gid = mind.add_goal("find water", 0.7)
        assert len(mind.self_model.active_goals) == 1
        assert gid is not None

    def test_step(self) -> None:
        mind = Mind()
        mind.add_goal("explore")
        report = mind.step()
        assert "cycle" in report
        assert "conscious_thought" in report
        assert report["cycle"] == 1

    def test_multiple_cycles(self) -> None:
        mind = Mind()
        mind.add_goal("explore the world", 0.7)
        mind.receive_input("There is a mountain ahead.")
        for _ in range(10):
            report = mind.step()
        assert report["cycle"] == 10

    def test_full_report(self) -> None:
        mind = Mind(agent_name="Kokoro")
        mind.add_goal("learn")
        mind.step()
        full = mind.full_report()
        assert full["agent_name"] == "Kokoro"
        assert "consciousness_stream" in full

    def test_stream_of_consciousness(self) -> None:
        mind = Mind()
        mind.add_goal("think")
        for _ in range(5):
            mind.step()
        stream = mind.stream_of_consciousness()
        assert len(stream) > 0

    def test_percept_injection(self) -> None:
        mind = Mind()
        p = Percept(
            modality=Modality.VISUAL,
            content={"entity_id": "cat", "entity_type": "animal", "location": (10.0, 10.0)},
            intensity=0.9,
        )
        mind.receive_percept(p)
        assert "cat" in mind.world_model.entities
