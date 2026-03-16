#!/usr/bin/env python3
"""
Cognitive Architecture Demo — 心のシミュレーション

This demo creates a Mind agent and runs it through a short scenario,
showing how the six layers (LLM, World Model, Self Model, Emotion,
Embodied AI, Global Workspace) interact to produce a stream of
conscious experience.

Usage::

    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognitive_architecture import Mind
from cognitive_architecture.core.types import Entity, Modality, Percept


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_report(report: dict) -> None:
    print(f"  Cycle           : {report['cycle']}")
    print(f"  Conscious thought: {report['conscious_thought']}")
    print(f"  Emotional state  : {report['emotional_state']}")
    body = report["body_state"]
    print(f"  Position         : {body['position']}")
    print(f"  Energy           : {body['energy']}")
    if report["action_taken"]:
        print(f"  Action taken     : {report['action_taken']}")
    print()


def main() -> None:
    divider("心のアーキテクチャ — Cognitive Architecture Demo")

    # ------------------------------------------------------------------
    # 1. Create the Mind
    # ------------------------------------------------------------------
    print("Creating mind: 'Kokoro' (心) ...\n")
    mind = Mind(agent_name="Kokoro", grid_size=(100, 100))

    # ------------------------------------------------------------------
    # 2. Populate the world
    # ------------------------------------------------------------------
    divider("Phase 1: Perceiving the World")

    # Add entities to the world model
    mind.world_model.add_entity(
        Entity(id="forest", entity_type="terrain", location=(30.0, 70.0),
               properties={"description": "dense forest", "traversable": True})
    )
    mind.world_model.add_entity(
        Entity(id="river", entity_type="terrain", location=(60.0, 50.0),
               properties={"description": "flowing river", "traversable": False})
    )
    mind.world_model.add_entity(
        Entity(id="apple_tree", entity_type="food_source", location=(35.0, 65.0),
               properties={"description": "apple tree with ripe fruit", "food_available": True})
    )

    print("World populated with: forest, river, apple_tree")
    print(f"  Entities: {len(mind.world_model.entities)}")
    print()

    # ------------------------------------------------------------------
    # 3. Language input
    # ------------------------------------------------------------------
    divider("Phase 2: Receiving Language Input")

    inputs = [
        "I see a beautiful forest ahead with tall trees.",
        "There seems to be danger near the river.",
        "I feel hungry and there might be food nearby.",
    ]

    for text in inputs:
        meaning = mind.receive_input(text)
        print(f"  Input   : {text}")
        print(f"  Intent  : {meaning['intent']}")
        print(f"  Sentiment: {meaning['sentiment']:.2f}")
        print()

    # ------------------------------------------------------------------
    # 4. Set goals
    # ------------------------------------------------------------------
    divider("Phase 3: Setting Goals")

    goals = [
        ("explore the forest", 0.6),
        ("find food to eat", 0.8),
        ("avoid danger near the river", 0.9),
    ]

    for desc, priority in goals:
        gid = mind.add_goal(desc, priority)
        print(f"  Goal: {desc!r} (priority={priority}, id={gid})")
    print()

    # ------------------------------------------------------------------
    # 5. Run cognitive cycles
    # ------------------------------------------------------------------
    divider("Phase 4: Cognitive Cycles")

    for i in range(15):
        report = mind.step()
        print(f"--- Cycle {report['cycle']} ---")
        print_report(report)

        # Inject a surprise event mid-way
        if i == 5:
            divider("SURPRISE EVENT: Loud noise!")
            surprise = Percept(
                modality=Modality.AUDITORY,
                content={"description": "a sudden loud roar from the forest"},
                intensity=0.95,
                source="environment",
            )
            mind.receive_percept(surprise)
            mind.emotion.appraise_prediction_error(0.8)
            print("  → Injected high-intensity auditory percept\n")

        if i == 10:
            divider("DISCOVERY: Found food!")
            food_percept = Percept(
                modality=Modality.VISUAL,
                content={
                    "entity_id": "apple",
                    "entity_type": "food",
                    "location": (36.0, 64.0),
                    "properties": {"edible": True, "appeal": "high"},
                },
                intensity=0.8,
                source="environment",
            )
            mind.receive_percept(food_percept)
            print("  → Found an apple nearby\n")

    # ------------------------------------------------------------------
    # 6. Final report
    # ------------------------------------------------------------------
    divider("Final Report")

    full = mind.full_report()
    print(f"  Agent            : {full['agent_name']}")
    print(f"  Total cycles     : {full['cycle']}")
    print(f"  Emotional state  : {full['emotional_state']}")
    print(f"  Mood             : {full['mood']}")
    print(f"  Body             : {full['body_state']}")
    print()

    print("  Self-Model Summary:")
    for key, val in full["self_summary"].items():
        print(f"    {key}: {val}")
    print()

    print("  World Summary:")
    ws = full["world_summary"]
    print(f"    Entities: {ws['entity_count']}")
    print(f"    Beliefs : {ws['belief_count']}")
    print()

    print("  Stream of Consciousness (last 10):")
    for line in full["consciousness_stream"]:
        print(f"    {line}")
    print()

    print("  Inner Speech (last 5):")
    for line in full["inner_speech"][-5:]:
        print(f"    {line}")
    print()

    print("  Autobiography (last 5):")
    for line in full["autobiography"][-5:]:
        print(f"    {line}")
    print()

    divider("Demo Complete — ありがとうございました")


if __name__ == "__main__":
    main()
