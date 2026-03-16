"""
Event bus for inter-module communication.

The EventBus provides a publish-subscribe mechanism that allows
cognitive modules to communicate without direct coupling.  Each
module can subscribe to specific event types and publish events
for others to consume.  The Global Workspace uses this bus as
its primary communication backbone.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventHandler = Callable[[str, Any], None]


class EventBus:
    """Publish-subscribe event bus for cognitive module communication.

    Modules register handlers for named event topics.  When an event
    is published, all subscribed handlers are invoked synchronously in
    registration order.

    Example::

        bus = EventBus()
        bus.subscribe("percept.new", lambda topic, data: print(data))
        bus.publish("percept.new", percept)
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[tuple[str, Any]] = []
        self._max_history: int = 500

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register *handler* for events on *topic*."""
        self._subscribers[topic].append(handler)
        logger.debug("Subscribed handler %s to topic %r", handler, topic)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove *handler* from *topic* (no-op if not subscribed)."""
        handlers = self._subscribers.get(topic)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def publish(self, topic: str, data: Any = None) -> None:
        """Publish an event to all handlers registered for *topic*."""
        self._record(topic, data)
        for handler in self._subscribers.get(topic, []):
            try:
                handler(topic, data)
            except Exception:
                logger.exception(
                    "Handler %s raised an exception for topic %r", handler, topic
                )

    def clear(self) -> None:
        """Remove all subscriptions and history."""
        self._subscribers.clear()
        self._history.clear()

    @property
    def history(self) -> list[tuple[str, Any]]:
        """Return the recent event history (read-only copy)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, topic: str, data: Any) -> None:
        self._history.append((topic, data))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
