"""
Language Module — LLM-based language processing.

This module occupies Layer 1 of the cognitive architecture.  It handles:

* **Comprehension** — parsing natural-language input into structured
  meaning representations that other modules can consume.
* **Generation** — producing natural-language output from internal
  thoughts, beliefs, and intentions.
* **Inner speech** — an internal narrative stream that feeds the
  Global Workspace, enabling language-mediated reasoning.

Implementation note
-------------------
The current implementation uses lightweight pattern matching and
template-based generation so that the package has **zero external
dependencies**.  It can be extended with a real LLM backend (e.g.
OpenAI, HuggingFace) by subclassing `LanguageModule` and overriding
`comprehend()` and `generate()`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cognitive_architecture.core.types import (
    BroadcastMessage,
    Modality,
    Percept,
    Thought,
)

logger = logging.getLogger(__name__)


class LanguageModule:
    """Layer 1 — Language pattern processing.

    The module maintains a short *conversation buffer* (working
    memory for dialogue) and an *inner speech* log that records the
    agent's internal monologue.
    """

    MODULE_NAME = "llm"

    def __init__(self, vocabulary_size: int = 10_000) -> None:
        self.vocabulary_size = vocabulary_size
        self.conversation_buffer: list[dict[str, str]] = []
        self.inner_speech: list[str] = []
        self._patterns: list[tuple[re.Pattern[str], str]] = self._build_patterns()
        self._templates: dict[str, str] = self._build_templates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def comprehend(self, text: str) -> dict[str, Any]:
        """Parse *text* into a structured meaning representation.

        Returns a dict with keys ``intent``, ``entities``,
        ``sentiment``, and ``raw``.
        """
        intent = self._extract_intent(text)
        entities = self._extract_entities(text)
        sentiment = self._analyze_sentiment(text)

        meaning = {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "raw": text,
        }

        self.conversation_buffer.append({"role": "user", "content": text})
        self._trim_buffer()

        logger.debug("Comprehended: %s → %s", text, meaning)
        return meaning

    def generate(self, thought: Thought, context: dict[str, Any] | None = None) -> str:
        """Generate a natural-language utterance from *thought*.

        An optional *context* dict can supply additional information
        (e.g. emotional state, active goals) that shapes the output.
        """
        text = self._thought_to_text(thought, context or {})
        self.conversation_buffer.append({"role": "assistant", "content": text})
        self._trim_buffer()

        logger.debug("Generated: %s", text)
        return text

    def inner_monologue(self, broadcast: BroadcastMessage) -> Thought:
        """Produce an inner-speech thought in response to a workspace broadcast.

        Inner speech is the language-mediated reasoning that runs
        alongside perception and emotion, providing a narrative
        overlay on conscious experience.
        """
        narration = self._narrate(broadcast)
        self.inner_speech.append(narration)

        return Thought(
            content=narration,
            source_module=self.MODULE_NAME,
            salience=0.4 + 0.3 * abs(broadcast.emotional_context.valence),
        )

    def text_to_percept(self, text: str) -> Percept:
        """Wrap raw *text* into a linguistic :class:`Percept`."""
        meaning = self.comprehend(text)
        return Percept(
            modality=Modality.LINGUISTIC,
            content=meaning,
            intensity=0.8,
            source="language_input",
        )

    # ------------------------------------------------------------------
    # Event handler (called by Global Workspace)
    # ------------------------------------------------------------------

    def on_broadcast(self, topic: str, data: Any) -> None:
        """Handle a broadcast event from the Global Workspace."""
        if isinstance(data, BroadcastMessage):
            self.inner_monologue(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_patterns() -> list[tuple[re.Pattern[str], str]]:
        return [
            (re.compile(r"\b(hello|hi|hey|greetings)\b", re.I), "greeting"),
            (re.compile(r"\b(what|where|when|who|how|why)\b.*\?", re.I), "question"),
            (re.compile(r"\b(please|could you|can you|would you)\b", re.I), "request"),
            (re.compile(r"\b(danger|threat|warning|careful)\b", re.I), "alert"),
            (re.compile(r"\b(go|move|walk|run|come)\b", re.I), "movement"),
            (re.compile(r"\b(see|look|observe|notice)\b", re.I), "observation"),
            (re.compile(r"\b(feel|emotion|happy|sad|angry|afraid)\b", re.I), "emotion_expression"),
            (re.compile(r"\b(think|believe|consider|wonder)\b", re.I), "reflection"),
            (re.compile(r"\b(eat|drink|hungry|thirsty)\b", re.I), "need"),
            (re.compile(r"\b(help|assist|support)\b", re.I), "help_request"),
        ]

    @staticmethod
    def _build_templates() -> dict[str, str]:
        return {
            "observation": "I notice {content}.",
            "reflection": "I think about {content}.",
            "emotion_response": "I feel {emotion} because {content}.",
            "action_intent": "I want to {content}.",
            "greeting_response": "Hello! I am here.",
            "question_response": "That's an interesting question about {content}.",
            "narration": "Right now, {content}.",
            "default": "{content}",
        }

    def _extract_intent(self, text: str) -> str:
        for pattern, intent in self._patterns:
            if pattern.search(text):
                return intent
        return "statement"

    @staticmethod
    def _extract_entities(text: str) -> list[dict[str, str]]:
        entities: list[dict[str, str]] = []
        # Simple proper-noun extraction (capitalized words)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            entities.append({"text": match.group(), "type": "noun_phrase"})
        return entities

    @staticmethod
    def _analyze_sentiment(text: str) -> float:
        positive = {"good", "great", "happy", "love", "wonderful", "beautiful",
                    "excellent", "joy", "nice", "amazing", "like", "enjoy"}
        negative = {"bad", "terrible", "sad", "hate", "awful", "ugly",
                    "horrible", "pain", "angry", "fear", "danger", "hurt"}
        words = set(text.lower().split())
        pos = len(words & positive)
        neg = len(words & negative)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _thought_to_text(self, thought: Thought, context: dict[str, Any]) -> str:
        template_key = "default"
        if "observe" in thought.content.lower() or "see" in thought.content.lower():
            template_key = "observation"
        elif "think" in thought.content.lower() or "consider" in thought.content.lower():
            template_key = "reflection"
        elif "feel" in thought.content.lower():
            template_key = "emotion_response"
            emotion = context.get("emotion", "something")
            return self._templates[template_key].format(
                emotion=emotion, content=thought.content
            )
        elif "want" in thought.content.lower() or "goal" in thought.content.lower():
            template_key = "action_intent"

        template = self._templates.get(template_key, self._templates["default"])
        return template.format(content=thought.content)

    def _narrate(self, broadcast: BroadcastMessage) -> str:
        emotion = broadcast.emotional_context.primary_emotion.name.lower()
        content = broadcast.thought.content

        parts: list[str] = []
        parts.append(f"I am aware of: {content}")
        if emotion != "neutral":
            parts.append(f"I feel {emotion}")
        if broadcast.active_goals:
            goal_desc = broadcast.active_goals[0].description
            parts.append(f"My current focus is: {goal_desc}")

        return ". ".join(parts) + "."

    def _trim_buffer(self, max_size: int = 50) -> None:
        if len(self.conversation_buffer) > max_size:
            self.conversation_buffer = self.conversation_buffer[-max_size:]
