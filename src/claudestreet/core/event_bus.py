"""EventBridge client — the nervous system of ClaudeStreet on AWS.

Wraps Amazon EventBridge for publishing and replaying events.
All inter-agent communication flows through EventBridge rules
that route events to the correct Lambda function.

EventBridge gives us:
  - Serverless pub/sub with content-based routing
  - Schema registry for event validation
  - Archive + replay for debugging and backtesting
  - CloudWatch metrics on every event rule
  - Cross-account / cross-region event forwarding
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from claudestreet.models.events import Event, EventType

logger = logging.getLogger(__name__)

# EventBridge limits: 256KB per event, 10 entries per PutEvents call
_MAX_BATCH_SIZE = 10
_SOURCE_PREFIX = "claudestreet"


class EventBridgeClient:
    """Publishes and replays events via Amazon EventBridge."""

    def __init__(
        self,
        bus_name: str | None = None,
        region: str | None = None,
    ) -> None:
        self.bus_name = bus_name or os.environ.get(
            "CLAUDESTREET_EVENT_BUS", "claudestreet"
        )
        self._client = boto3.client(
            "events",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        )

    def put_event(self, event: Event) -> str | None:
        """Publish a single event to EventBridge.

        Returns the EventBridge entry ID on success, None on failure.
        """
        try:
            entry = self._to_eb_entry(event)
            response = self._client.put_events(Entries=[entry])

            failed = response.get("FailedEntryCount", 0)
            if failed > 0:
                error = response["Entries"][0].get("ErrorMessage", "unknown")
                logger.error(
                    "Failed to publish %s: %s", event.type.value, error
                )
                return None

            entry_id = response["Entries"][0].get("EventId")
            logger.debug(
                "Published %s from %s (id=%s)",
                event.type.value, event.source, entry_id,
            )
            return entry_id

        except ClientError:
            logger.exception("EventBridge put_event failed for %s", event.type.value)
            return None

    def put_events(self, events: list[Event]) -> int:
        """Publish multiple events in batches of 10.

        Returns the number of successfully published events.
        """
        if not events:
            return 0

        published = 0
        for i in range(0, len(events), _MAX_BATCH_SIZE):
            batch = events[i : i + _MAX_BATCH_SIZE]
            entries = [self._to_eb_entry(e) for e in batch]

            try:
                response = self._client.put_events(Entries=entries)
                failed = response.get("FailedEntryCount", 0)
                published += len(batch) - failed

                if failed > 0:
                    for j, entry_resp in enumerate(response["Entries"]):
                        if "ErrorMessage" in entry_resp:
                            logger.error(
                                "Batch publish failed for event %s: %s",
                                batch[j].id,
                                entry_resp["ErrorMessage"],
                            )
            except ClientError:
                logger.exception("EventBridge batch put_events failed")

        logger.info("Published %d / %d events", published, len(events))
        return published

    def _to_eb_entry(self, event: Event) -> dict[str, Any]:
        """Convert our Event model to an EventBridge PutEvents entry."""
        detail = event.model_dump(mode="json")
        # Ensure timestamp is ISO string
        if isinstance(detail.get("timestamp"), str):
            time_str = detail["timestamp"]
        else:
            time_str = datetime.now(timezone.utc).isoformat()

        return {
            "Source": f"{_SOURCE_PREFIX}.{event.source}",
            "DetailType": event.type.value,
            "Detail": json.dumps(detail),
            "EventBusName": self.bus_name,
            "Time": datetime.fromisoformat(time_str.replace("Z", "+00:00")),
        }

    @staticmethod
    def from_eventbridge(raw_event: dict) -> Event:
        """Parse an incoming EventBridge event into our Event model.

        Handles both direct EventBridge invocations and EventBridge
        Scheduler payloads (which wrap the detail differently).
        """
        # EventBridge Scheduler wraps events differently
        if "detail" in raw_event:
            detail = raw_event["detail"]
        elif "Detail" in raw_event:
            detail = (
                json.loads(raw_event["Detail"])
                if isinstance(raw_event["Detail"], str)
                else raw_event["Detail"]
            )
        else:
            # Assume the raw event IS the detail (e.g., test invocations)
            detail = raw_event

        return Event.model_validate(detail)

    @staticmethod
    def is_heartbeat(raw_event: dict) -> bool:
        """Check if the incoming event is a scheduled heartbeat."""
        detail_type = raw_event.get("detail-type", raw_event.get("DetailType", ""))
        return detail_type == EventType.HEARTBEAT_TICK.value

    @staticmethod
    def heartbeat_agent(raw_event: dict) -> str:
        """Extract the target agent from a heartbeat event."""
        detail = raw_event.get("detail", raw_event.get("Detail", {}))
        if isinstance(detail, str):
            detail = json.loads(detail)
        return detail.get("payload", {}).get("agent", "unknown")
