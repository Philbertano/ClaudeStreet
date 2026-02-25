"""Base agent — infrastructure-agnostic business logic.

Agents are pure processors: they receive an Event, access Memory
for state, and return zero or more child Events. They have no
knowledge of Lambda, EventBridge, or DynamoDB — those concerns
live in the handler layer.

This makes agents:
  - Unit-testable without AWS mocks
  - Portable across runtimes (Lambda, Fargate, local)
  - Easy to reason about (input → output)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from claudestreet.core.memory import DynamoMemory
from claudestreet.models.events import Event, EventType, EventPriority

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all ClaudeStreet agents."""

    agent_id: str = "base"
    description: str = "Base agent"

    def __init__(self, memory: DynamoMemory, config: dict) -> None:
        self.memory = memory
        self.config = config

    @abstractmethod
    def process(self, event: Event) -> list[Event]:
        """Process an event and return child events to publish.

        This is the core method each agent implements. Must be
        a pure function of (event, memory state) → events.
        """
        ...

    def heartbeat(self) -> list[Event]:
        """Called on schedule by EventBridge Scheduler.

        Override in subclasses for proactive behavior.
        """
        return []

    # ── Helpers ──

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        parent: Event | None = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Event:
        """Create an event sourced from this agent."""
        if parent:
            return parent.spawn(event_type, self.agent_id, payload, priority)
        return Event(
            type=event_type,
            source=self.agent_id,
            payload=payload or {},
            priority=priority,
        )
