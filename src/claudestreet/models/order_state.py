"""Order lifecycle state machine.

Defines valid order states and transitions. Each transition is enforced
via conditional DynamoDB writes to prevent duplicate state changes.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


# Valid transitions: current_state -> set of allowed next states
VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING: {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.FAILED},
    OrderState.SUBMITTED: {
        OrderState.ACCEPTED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.FAILED,
    },
    OrderState.ACCEPTED: {
        OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED,
    },
    OrderState.PARTIAL_FILL: {
        OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED,
    },
    # Terminal states — no further transitions
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.FAILED: set(),
}

TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.FAILED}


def is_valid_transition(current: OrderState, target: OrderState) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())


def is_terminal(state: OrderState) -> bool:
    return state in TERMINAL_STATES


class OrderStateMachine:
    """Manages order lifecycle with conditional DynamoDB transitions."""

    ORDER_TIMEOUT_SECONDS = 300  # 5 minutes

    def __init__(self, memory) -> None:
        from claudestreet.core.memory import DynamoMemory
        self.memory: DynamoMemory = memory

    def transition(
        self,
        trade_id: str,
        current_state: OrderState,
        target_state: OrderState,
        metadata: dict | None = None,
    ) -> bool:
        """Attempt a state transition with conditional write.

        Returns True if transition succeeded, False if it was rejected
        (e.g., another process already transitioned the order).
        """
        if not is_valid_transition(current_state, target_state):
            logger.error(
                "Invalid order transition %s -> %s for trade %s",
                current_state.value, target_state.value, trade_id,
            )
            return False

        from boto3.dynamodb.conditions import Attr
        from botocore.exceptions import ClientError
        from claudestreet.core.memory import _to_decimal
        from datetime import datetime, timezone

        update_expr = "SET order_state = :new_state, order_updated_at = :now"
        expr_values: dict = {
            ":new_state": target_state.value,
            ":expected": current_state.value,
            ":now": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            update_expr += ", order_metadata = :meta"
            expr_values[":meta"] = metadata

        try:
            self.memory._trades.update_item(
                Key={"trade_id": trade_id},
                UpdateExpression=update_expr,
                ConditionExpression=Attr("order_state").eq(current_state.value),
                ExpressionAttributeValues=_to_decimal(expr_values),
            )
            logger.info(
                "Order %s: %s -> %s", trade_id, current_state.value, target_state.value
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(
                    "Order %s transition rejected (race condition): %s -> %s",
                    trade_id, current_state.value, target_state.value,
                )
                return False
            raise

    def check_timeout(self, trade_id: str, submitted_at_iso: str) -> bool:
        """Check if an order in SUBMITTED state has timed out (>5 min)."""
        from datetime import datetime, timezone

        submitted_at = datetime.fromisoformat(submitted_at_iso.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - submitted_at).total_seconds()
        return elapsed > self.ORDER_TIMEOUT_SECONDS
