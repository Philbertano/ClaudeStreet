"""DLQ replayer — reprocesses failed events from the dead letter queue.

Triggered by CloudWatch alarm on DLQ depth > 0.
Reads messages from the DLQ and republishes them to EventBridge.
"""

from __future__ import annotations

import json
import logging
import os

import boto3

from claudestreet.core.event_bus import EventBridgeClient
from claudestreet.models.events import Event

logger = logging.getLogger(__name__)

_MAX_MESSAGES_PER_INVOCATION = 10


def handler(event: dict, context) -> dict:
    """Lambda handler triggered by CloudWatch alarm on DLQ depth."""
    dlq_url = os.environ.get("DLQ_URL", "")
    if not dlq_url:
        logger.error("DLQ_URL environment variable not set")
        return {"statusCode": 500, "error": "DLQ_URL not configured"}

    sqs = boto3.client("sqs")
    eb = EventBridgeClient()

    replayed = 0
    failed = 0

    for _ in range(_MAX_MESSAGES_PER_INVOCATION):
        response = sqs.receive_message(
            QueueUrl=dlq_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=1,
        )

        messages = response.get("Messages", [])
        if not messages:
            break

        for msg in messages:
            try:
                body = json.loads(msg["Body"])

                # Extract the original EventBridge event from the SQS message
                # SQS wraps EventBridge events in a specific envelope
                detail = body.get("detail", body)
                parsed_event = Event.model_validate(detail)

                # Republish to EventBridge
                result = eb.put_event(parsed_event)
                if result:
                    replayed += 1
                    # Delete from DLQ on successful replay
                    sqs.delete_message(
                        QueueUrl=dlq_url,
                        ReceiptHandle=msg["ReceiptHandle"],
                    )
                    logger.info(
                        "Replayed event %s (%s) from DLQ",
                        parsed_event.id, parsed_event.type.value,
                    )
                else:
                    failed += 1
                    logger.warning("Failed to replay event from DLQ")

            except Exception:
                failed += 1
                logger.exception("Failed to process DLQ message")

    logger.info("DLQ replay complete: %d replayed, %d failed", replayed, failed)
    return {
        "statusCode": 200,
        "replayed": replayed,
        "failed": failed,
    }
