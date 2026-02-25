"""DynamoDB memory — durable state for the swarm.

Replaces SQLite with DynamoDB for true serverless operation.
Each table is designed for specific access patterns with GSIs
for the queries agents need.

Tables:
  claudestreet-trades      — open/closed trade records
  claudestreet-strategies  — evolved strategy population
  claudestreet-snapshots   — portfolio performance over time
  claudestreet-events      — event audit log with TTL

All writes are idempotent (conditional puts / upserts) to handle
Lambda retries safely.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _to_decimal(obj: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal(i) for i in obj]
    return obj


def _from_decimal(obj: Any) -> Any:
    """Recursively convert Decimals back to float."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(i) for i in obj]
    return obj


class DynamoMemory:
    """DynamoDB-backed state store for the trading swarm.

    All methods are synchronous (boto3 DynamoDB resource is sync).
    Lambda handlers call these directly — no async needed.
    """

    def __init__(
        self,
        trades_table: str | None = None,
        strategies_table: str | None = None,
        snapshots_table: str | None = None,
        events_table: str | None = None,
        region: str | None = None,
    ) -> None:
        region = region or os.environ.get("AWS_REGION", "us-east-1")
        dynamodb = boto3.resource("dynamodb", region_name=region)

        self._trades = dynamodb.Table(
            trades_table or os.environ.get("TRADES_TABLE", "claudestreet-trades")
        )
        self._strategies = dynamodb.Table(
            strategies_table or os.environ.get("STRATEGIES_TABLE", "claudestreet-strategies")
        )
        self._snapshots = dynamodb.Table(
            snapshots_table or os.environ.get("SNAPSHOTS_TABLE", "claudestreet-snapshots")
        )
        self._events = dynamodb.Table(
            events_table or os.environ.get("EVENTS_TABLE", "claudestreet-events")
        )

    # ──────────────────────────────────────────────
    # Trades
    # ──────────────────────────────────────────────

    def record_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_id: str,
    ) -> None:
        self._trades.put_item(
            Item=_to_decimal({
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_id": strategy_id,
                "status": "open",
                "pnl": 0.0,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }),
            ConditionExpression=Attr("trade_id").not_exists(),  # idempotent
        )

    def record_trade_close(
        self, trade_id: str, exit_price: float, pnl: float
    ) -> None:
        self._trades.update_item(
            Key={"trade_id": trade_id},
            UpdateExpression=(
                "SET #s = :status, exit_price = :exit_price, "
                "pnl = :pnl, closed_at = :closed_at"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=_to_decimal({
                ":status": "closed",
                ":exit_price": exit_price,
                ":pnl": pnl,
                ":closed_at": datetime.now(timezone.utc).isoformat(),
            }),
        )

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            response = self._trades.query(
                IndexName="symbol-status-index",
                KeyConditionExpression=(
                    Key("symbol").eq(symbol) & Key("status").eq("open")
                ),
            )
        else:
            response = self._trades.scan(
                FilterExpression=Attr("status").eq("open"),
            )
        return [_from_decimal(item) for item in response.get("Items", [])]

    def get_strategy_trades(self, strategy_id: str) -> list[dict]:
        response = self._trades.query(
            IndexName="strategy-index",
            KeyConditionExpression=Key("strategy_id").eq(strategy_id),
        )
        return [_from_decimal(item) for item in response.get("Items", [])]

    # ──────────────────────────────────────────────
    # Strategies
    # ──────────────────────────────────────────────

    def save_strategy(self, strategy_data: dict) -> None:
        item = {
            "strategy_id": strategy_data["id"],
            "name": strategy_data["name"],
            "strategy_type": strategy_data["strategy_type"],
            "genome_json": json.dumps(strategy_data["genome"]),
            "fitness_json": json.dumps(strategy_data["fitness"]),
            "generation": strategy_data.get("generation", 0),
            "is_active": 1 if strategy_data.get("is_active", True) else 0,
            "created_at": strategy_data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "total_trades": strategy_data.get("total_trades", 0),
            "total_pnl": strategy_data.get("total_pnl", 0.0),
        }
        self._strategies.put_item(Item=_to_decimal(item))

    def get_active_strategies(self) -> list[dict]:
        response = self._strategies.query(
            IndexName="active-index",
            KeyConditionExpression=Key("is_active").eq(1),
            ScanIndexForward=False,  # sort by total_pnl descending
        )
        rows = []
        for item in response.get("Items", []):
            item = _from_decimal(item)
            item["id"] = item.pop("strategy_id")
            item["genome"] = json.loads(item.pop("genome_json", "{}"))
            item["fitness"] = json.loads(item.pop("fitness_json", "{}"))
            rows.append(item)
        return rows

    def retire_strategy(self, strategy_id: str) -> None:
        self._strategies.update_item(
            Key={"strategy_id": strategy_id},
            UpdateExpression="SET is_active = :inactive",
            ExpressionAttributeValues={":inactive": 0},
        )

    # ──────────────────────────────────────────────
    # Portfolio snapshots
    # ──────────────────────────────────────────────

    def record_snapshot(
        self,
        cash: float,
        positions_value: float,
        total_value: float,
        daily_pnl: float,
        open_positions: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._snapshots.put_item(
            Item=_to_decimal({
                "date": now.strftime("%Y-%m-%d"),
                "timestamp": now.isoformat(),
                "cash": cash,
                "positions_value": positions_value,
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "open_positions": open_positions,
            })
        )

    def get_portfolio_history(self, days: int = 30) -> list[dict]:
        from datetime import timedelta

        results = []
        today = datetime.now(timezone.utc).date()
        for i in range(days):
            date_str = (today - timedelta(days=i)).isoformat()
            response = self._snapshots.query(
                KeyConditionExpression=Key("date").eq(date_str),
                ScanIndexForward=False,
                Limit=1,
            )
            items = response.get("Items", [])
            if items:
                results.append(_from_decimal(items[0]))
        return results

    # ──────────────────────────────────────────────
    # Event audit log
    # ──────────────────────────────────────────────

    def log_event(
        self,
        event_id: str,
        event_type: str,
        source: str,
        timestamp: str,
        payload: dict,
        correlation_id: str | None,
    ) -> None:
        now = datetime.now(timezone.utc)
        ttl = int(now.timestamp()) + (30 * 86400)  # 30 day TTL
        item: dict[str, Any] = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "source": source,
            "payload_json": json.dumps(payload),
            "ttl": ttl,
        }
        if correlation_id:
            item["correlation_id"] = correlation_id
        self._events.put_item(Item=_to_decimal(item))

    def get_event_chain(self, correlation_id: str) -> list[dict]:
        response = self._events.query(
            IndexName="correlation-index",
            KeyConditionExpression=Key("correlation_id").eq(correlation_id),
            ScanIndexForward=True,
        )
        return [_from_decimal(item) for item in response.get("Items", [])]
