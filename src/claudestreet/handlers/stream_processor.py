"""DynamoDB Streams processor — reacts to trade table changes.

Lambda triggered by DynamoDB Stream on the trades table.
On trade close (status change open→closed): triggers fitness recalculation.
Forwards all changes for analytics pipeline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def handler(event: dict, context) -> dict:
    """Process DynamoDB Stream events from the trades table."""
    records = event.get("Records", [])
    processed = 0
    fitness_triggers = 0

    for record in records:
        event_name = record.get("eventName", "")
        dynamodb_data = record.get("dynamodb", {})

        # Only process MODIFY events (status changes)
        if event_name != "MODIFY":
            processed += 1
            continue

        old_image = dynamodb_data.get("OldImage", {})
        new_image = dynamodb_data.get("NewImage", {})

        old_status = _extract_string(old_image.get("status", {}))
        new_status = _extract_string(new_image.get("status", {}))

        # Trade closed: open → closed
        if old_status == "open" and new_status == "closed":
            trade_id = _extract_string(new_image.get("trade_id", {}))
            strategy_id = _extract_string(new_image.get("strategy_id", {}))
            pnl = _extract_number(new_image.get("pnl", {}))

            logger.info(
                "Trade closed: %s strategy=%s pnl=%.2f",
                trade_id, strategy_id, pnl,
            )

            # Emit fitness recalculation event via EventBridge
            _emit_fitness_trigger(trade_id, strategy_id, pnl)
            fitness_triggers += 1

        processed += 1

    logger.info(
        "Stream processor: %d records processed, %d fitness triggers",
        processed, fitness_triggers,
    )

    return {
        "statusCode": 200,
        "processed": processed,
        "fitness_triggers": fitness_triggers,
    }


def _extract_string(dynamo_value: dict) -> str:
    """Extract string from DynamoDB Stream format {'S': 'value'}."""
    return dynamo_value.get("S", "")


def _extract_number(dynamo_value: dict) -> float:
    """Extract number from DynamoDB Stream format {'N': '123.45'}."""
    return float(dynamo_value.get("N", "0"))


def _emit_fitness_trigger(trade_id: str, strategy_id: str, pnl: float) -> None:
    """Emit event to trigger fitness recalculation."""
    try:
        from claudestreet.core.event_bus import EventBridgeClient
        from claudestreet.models.events import Event, EventType, EventPriority

        eb = EventBridgeClient()
        eb.put_event(Event(
            type=EventType.POSITION_CLOSED,
            source="stream_processor",
            payload={
                "trade_id": trade_id,
                "strategy_id": strategy_id,
                "pnl": pnl,
                "trigger": "dynamodb_stream",
            },
            priority=EventPriority.NORMAL,
        ))
    except Exception:
        logger.exception("Failed to emit fitness trigger for trade %s", trade_id)
