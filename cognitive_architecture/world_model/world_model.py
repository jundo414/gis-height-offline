"""
World Model — Internal simulation of world changes.

This module occupies Layer 2 of the cognitive architecture.  It
maintains an internal representation of the external environment and
supports:

* **State tracking** — entities, their properties, and spatial
  relationships are continuously updated from percepts.
* **Prediction** — the model can simulate "what happens if …"
  scenarios by projecting the current state forward.
* **Anomaly detection** — discrepancies between predicted and
  actual percepts are flagged, raising salience and triggering
  surprise in the Emotion Module.

The world model is a *generative* model: it tries to predict future
percepts and updates its beliefs when predictions are violated.
This implements a simple form of *predictive processing*.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from cognitive_architecture.core.types import (
    Belief,
    BroadcastMessage,
    Entity,
    Modality,
    Percept,
    Thought,
)

logger = logging.getLogger(__name__)


class WorldModel:
    """Layer 2 — Internal simulation of the external world.

    The model stores entities on a 2-D grid and maintains a set of
    beliefs derived from perception.  It supports simple spatial
    reasoning and change prediction.
    """

    MODULE_NAME = "world_model"

    def __init__(self, grid_size: tuple[int, int] = (100, 100)) -> None:
        self.grid_size = grid_size
        self.entities: dict[str, Entity] = {}
        self.beliefs: list[Belief] = []
        self.spatial_map: dict[tuple[int, int], list[str]] = {}
        self._prediction_errors: list[dict[str, Any]] = []
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Register a new entity in the world model."""
        self.entities[entity.id] = entity
        self._update_spatial_map(entity)
        self.beliefs.append(
            Belief(
                proposition=f"{entity.entity_type} '{entity.id}' exists at {entity.location}",
                confidence=1.0,
                source="perception",
            )
        )
        logger.debug("Added entity: %s", entity)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the world model."""
        entity = self.entities.pop(entity_id, None)
        if entity is not None:
            gx, gy = int(entity.location[0]), int(entity.location[1])
            cell = self.spatial_map.get((gx, gy), [])
            if entity_id in cell:
                cell.remove(entity_id)

    def update_entity(self, entity_id: str, **properties: Any) -> None:
        """Update properties of an existing entity."""
        entity = self.entities.get(entity_id)
        if entity is None:
            logger.warning("Entity %r not found for update", entity_id)
            return

        old_location = entity.location
        for key, value in properties.items():
            if key == "location":
                entity.location = value
            else:
                entity.properties[key] = value

        if entity.location != old_location:
            # Rebuild spatial index for this entity
            gx, gy = int(old_location[0]), int(old_location[1])
            cell = self.spatial_map.get((gx, gy), [])
            if entity_id in cell:
                cell.remove(entity_id)
            self._update_spatial_map(entity)

    # ------------------------------------------------------------------
    # Perception integration
    # ------------------------------------------------------------------

    def process_percept(self, percept: Percept) -> list[Belief]:
        """Integrate a percept into the world model.

        Returns any new or updated beliefs that resulted from the
        percept.
        """
        new_beliefs: list[Belief] = []

        if percept.modality == Modality.VISUAL:
            new_beliefs.extend(self._process_visual(percept))
        elif percept.modality == Modality.AUDITORY:
            new_beliefs.extend(self._process_auditory(percept))
        elif percept.modality == Modality.LINGUISTIC:
            new_beliefs.extend(self._process_linguistic(percept))
        else:
            new_beliefs.append(
                Belief(
                    proposition=f"Perceived {percept.modality.name}: {percept.content}",
                    confidence=percept.intensity,
                    source="perception",
                )
            )

        self.beliefs.extend(new_beliefs)
        self._trim_beliefs()
        return new_beliefs

    # ------------------------------------------------------------------
    # Prediction & simulation
    # ------------------------------------------------------------------

    def predict(self, action_description: str) -> dict[str, Any]:
        """Simulate the outcome of a hypothetical action.

        Returns a dict describing the predicted changes to the world
        state.  This is a simplified *forward model*.
        """
        prediction: dict[str, Any] = {
            "action": action_description,
            "predicted_changes": [],
            "confidence": 0.5,
        }

        lower = action_description.lower()
        if "move" in lower:
            prediction["predicted_changes"].append("agent_location_change")
            prediction["confidence"] = 0.8
        if "grasp" in lower or "take" in lower or "pick" in lower:
            prediction["predicted_changes"].append("object_possession_change")
            prediction["confidence"] = 0.7
        if "speak" in lower or "say" in lower:
            prediction["predicted_changes"].append("social_state_change")
            prediction["confidence"] = 0.6

        return prediction

    def check_prediction(self, predicted: dict[str, Any], actual: Percept) -> float:
        """Compare a prediction with an actual percept.

        Returns a *prediction error* score in [0, 1].  High values
        indicate surprise and trigger belief revision.
        """
        if not predicted.get("predicted_changes"):
            return 0.5

        error = 1.0 - predicted.get("confidence", 0.5)
        self._prediction_errors.append({
            "predicted": predicted,
            "actual": actual.content,
            "error": error,
        })

        if error > 0.5:
            logger.info("Prediction error %.2f — updating beliefs", error)
            self._revise_beliefs(predicted, actual)

        return error

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def entities_near(
        self, location: tuple[float, float], radius: float = 5.0
    ) -> list[Entity]:
        """Return all entities within *radius* of *location*."""
        results: list[Entity] = []
        for entity in self.entities.values():
            dist = math.dist(entity.location, location)
            if dist <= radius:
                results.append(entity)
        return results

    def get_world_summary(self) -> dict[str, Any]:
        """Return a compact summary of the current world state."""
        return {
            "entity_count": len(self.entities),
            "entities": {
                eid: {
                    "type": e.entity_type,
                    "location": e.location,
                    "properties": e.properties,
                }
                for eid, e in self.entities.items()
            },
            "belief_count": len(self.beliefs),
            "recent_beliefs": [b.proposition for b in self.beliefs[-5:]],
            "prediction_errors": len(self._prediction_errors),
            "tick": self._tick,
        }

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def on_broadcast(self, topic: str, data: Any) -> None:
        """Handle a broadcast from the Global Workspace."""
        if isinstance(data, BroadcastMessage):
            self._tick += 1

    def generate_thought(self) -> Thought:
        """Generate a world-model thought for the workspace competition."""
        summary = self.get_world_summary()
        if self._prediction_errors:
            last_err = self._prediction_errors[-1]
            content = f"Prediction error detected: {last_err['error']:.2f}"
            salience = 0.4 + last_err["error"] * 0.4
        elif summary["entity_count"] > 0:
            recent = summary["recent_beliefs"][-1] if summary["recent_beliefs"] else "nothing new"
            content = f"World state: {summary['entity_count']} entities. Latest: {recent}"
            salience = 0.3
        else:
            content = "The world is empty — no entities detected."
            salience = 0.2

        return Thought(
            content=content,
            source_module=self.MODULE_NAME,
            salience=salience,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_spatial_map(self, entity: Entity) -> None:
        gx, gy = int(entity.location[0]), int(entity.location[1])
        self.spatial_map.setdefault((gx, gy), []).append(entity.id)

    def _process_visual(self, percept: Percept) -> list[Belief]:
        beliefs: list[Belief] = []
        content = percept.content

        if "entity_id" in content:
            eid = content["entity_id"]
            etype = content.get("entity_type", "object")
            loc = content.get("location", (0.0, 0.0))
            if eid not in self.entities:
                self.add_entity(Entity(id=eid, entity_type=etype, location=loc))
            else:
                self.update_entity(eid, location=loc, **content.get("properties", {}))
            beliefs.append(
                Belief(
                    proposition=f"Visually confirmed {etype} '{eid}' at {loc}",
                    confidence=percept.intensity,
                    source="visual_perception",
                )
            )
        else:
            beliefs.append(
                Belief(
                    proposition=f"Visual observation: {content}",
                    confidence=percept.intensity * 0.8,
                    source="visual_perception",
                )
            )
        return beliefs

    def _process_auditory(self, percept: Percept) -> list[Belief]:
        content = percept.content
        proposition = f"Heard: {content.get('description', content)}"
        return [
            Belief(
                proposition=proposition,
                confidence=percept.intensity * 0.7,
                source="auditory_perception",
            )
        ]

    def _process_linguistic(self, percept: Percept) -> list[Belief]:
        content = percept.content
        proposition = (
            content.get("raw", str(content)) if isinstance(content, dict) else str(content)
        )
        return [
            Belief(
                proposition=f"Understood: {proposition}",
                confidence=0.9,
                source="linguistic_perception",
            )
        ]

    def _revise_beliefs(self, predicted: dict[str, Any], actual: Percept) -> None:
        for belief in self.beliefs:
            for change in predicted.get("predicted_changes", []):
                if change in belief.proposition:
                    belief.confidence *= 0.5

    def _trim_beliefs(self, max_beliefs: int = 200) -> None:
        if len(self.beliefs) > max_beliefs:
            self.beliefs.sort(key=lambda b: b.confidence, reverse=True)
            self.beliefs = self.beliefs[:max_beliefs]
