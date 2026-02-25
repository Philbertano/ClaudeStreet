"""Lambda handler factory — wires agents to AWS infrastructure.

Each agent Lambda follows the same pattern:
  1. Parse EventBridge event → our Event model
  2. Initialize agent with DynamoDB memory
  3. Dispatch to process() or heartbeat()
  4. Publish output events to EventBridge
  5. Return success/failure

This factory eliminates handler boilerplate and ensures
consistent error handling, logging, and metrics across
all agent Lambdas.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Callable, Type

import boto3

from claudestreet.agents.base import BaseAgent
from claudestreet.core.config import load_config
from claudestreet.core.event_bus import EventBridgeClient
from claudestreet.core.memory import DynamoMemory
from claudestreet.models.events import Event, EventType

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Module-level singletons (reused across warm Lambda invocations)
_memory: DynamoMemory | None = None
_eb: EventBridgeClient | None = None
_config: dict | None = None


def _get_infra() -> tuple[DynamoMemory, EventBridgeClient, dict]:
    """Lazy-init shared infrastructure (survives warm starts)."""
    global _memory, _eb, _config
    if _memory is None:
        _config = load_config()
        _memory = DynamoMemory()
        _eb = EventBridgeClient()
    return _memory, _eb, _config


def create_handler(agent_class: Type[BaseAgent]) -> Callable:
    """Create a Lambda handler function for an agent class.

    The returned handler supports both:
      - EventBridge event routing (agent.process)
      - EventBridge Scheduler heartbeats (agent.heartbeat)
    """

    def handler(event: dict, context) -> dict:
        start = time.time()
        memory, eb, config = _get_infra()
        agent = agent_class(memory=memory, config=config)

        try:
            # Determine if this is a heartbeat or a routed event
            is_heartbeat = EventBridgeClient.is_heartbeat(event)

            if is_heartbeat:
                logger.info("[%s] Heartbeat triggered", agent.agent_id)
                output_events = agent.heartbeat()
            else:
                parsed = EventBridgeClient.from_eventbridge(event)
                logger.info(
                    "[%s] Processing %s from %s",
                    agent.agent_id, parsed.type.value, parsed.source,
                )
                output_events = agent.process(parsed)

            # Publish output events to EventBridge
            published = 0
            if output_events:
                published = eb.put_events(output_events)

            elapsed = (time.time() - start) * 1000
            logger.info(
                "[%s] Done: %d events published in %.0fms",
                agent.agent_id, published, elapsed,
            )

            return {
                "statusCode": 200,
                "agent": agent.agent_id,
                "events_published": published,
                "duration_ms": round(elapsed),
            }

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(
                "[%s] FAILED after %.0fms: %s\n%s",
                agent.agent_id, elapsed, str(e), traceback.format_exc(),
            )

            # Publish error event for observability
            try:
                eb.put_event(Event(
                    type=EventType.RISK_ALERT,
                    source=agent.agent_id,
                    payload={
                        "alert_type": "agent_error",
                        "agent": agent.agent_id,
                        "error": str(e),
                    },
                ))
            except Exception:
                pass

            return {
                "statusCode": 500,
                "agent": agent.agent_id,
                "error": str(e),
                "duration_ms": round(elapsed),
            }

    handler.__name__ = f"{agent_class.agent_id}_handler"
    handler.__qualname__ = f"{agent_class.agent_id}_handler"
    return handler
