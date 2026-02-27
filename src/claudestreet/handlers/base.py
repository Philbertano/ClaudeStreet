"""Lambda handler factory — wires agents to AWS infrastructure.

Each agent Lambda follows the same pattern:
  1. Parse EventBridge/SQS event → our Event model
  2. Initialize agent with DynamoDB memory
  3. Dispatch to process() or heartbeat()
  4. Publish output events to EventBridge
  5. Return success/failure with CloudWatch EMF metrics

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


def _emit_emf_metrics(
    agent_id: str,
    duration_ms: float,
    events_published: int,
    success: bool,
    trades_executed: int = 0,
) -> None:
    """Emit CloudWatch Embedded Metrics Format (EMF) for trading-specific metrics."""
    emf = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "ClaudeStreet",
                    "Dimensions": [["Agent"]],
                    "Metrics": [
                        {"Name": "HandlerDurationMs", "Unit": "Milliseconds"},
                        {"Name": "EventsPublished", "Unit": "Count"},
                        {"Name": "HandlerSuccess", "Unit": "Count"},
                        {"Name": "HandlerError", "Unit": "Count"},
                        {"Name": "TradesExecuted", "Unit": "Count"},
                    ],
                }
            ],
        },
        "Agent": agent_id,
        "HandlerDurationMs": round(duration_ms, 1),
        "EventsPublished": events_published,
        "HandlerSuccess": 1 if success else 0,
        "HandlerError": 0 if success else 1,
        "TradesExecuted": trades_executed,
    }
    # EMF requires printing to stdout as a single JSON line
    print(json.dumps(emf))


def _extract_events_from_sqs(raw_event: dict) -> list[dict]:
    """Extract all EventBridge events from SQS envelope if present."""
    records = raw_event.get("Records", [])
    if not records or "body" not in records[0]:
        return [raw_event]

    events = []
    for record in records:
        body = record.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)
        events.append(body)
    return events


def create_handler(agent_class: Type[BaseAgent]) -> Callable:
    """Create a Lambda handler function for an agent class.

    The returned handler supports both:
      - EventBridge event routing via SQS (agent.process)
      - EventBridge Scheduler heartbeats (agent.heartbeat)
    """

    def handler(event: dict, context) -> dict:
        start = time.time()
        memory, eb, config = _get_infra()
        agent = agent_class(memory=memory, config=config)

        try:
            # Unwrap SQS envelope — may contain multiple records
            actual_events = _extract_events_from_sqs(event)

            output_events: list[Event] = []
            for actual_event in actual_events:
                # Determine if this is a heartbeat or a routed event
                is_heartbeat = EventBridgeClient.is_heartbeat(actual_event)

                if is_heartbeat:
                    logger.info("[%s] Heartbeat triggered", agent.agent_id)
                    output_events.extend(agent.heartbeat())
                else:
                    parsed = EventBridgeClient.from_eventbridge(actual_event)
                    logger.info(
                        "[%s] Processing %s from %s",
                        agent.agent_id, parsed.type.value, parsed.source,
                    )
                    output_events.extend(agent.process(parsed))

            # Publish output events to EventBridge
            published = 0
            if output_events:
                published = eb.put_events(output_events)

            elapsed = (time.time() - start) * 1000

            # Count trades executed in this invocation
            trades_executed = sum(
                1 for e in (output_events or [])
                if e.type == EventType.TRADE_EXECUTED
            )

            logger.info(
                "[%s] Done: %d events published in %.0fms",
                agent.agent_id, published, elapsed,
            )

            # Emit CloudWatch EMF metrics
            _emit_emf_metrics(
                agent_id=agent.agent_id,
                duration_ms=elapsed,
                events_published=published,
                success=True,
                trades_executed=trades_executed,
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

            # Emit error metrics
            _emit_emf_metrics(
                agent_id=agent.agent_id,
                duration_ms=elapsed,
                events_published=0,
                success=False,
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
