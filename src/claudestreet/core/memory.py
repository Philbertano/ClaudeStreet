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
        decisions_table: str | None = None,
        patterns_table: str | None = None,
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
        self._decisions = dynamodb.Table(
            decisions_table or os.environ.get("DECISIONS_TABLE", "claudestreet-decisions")
        )
        self._patterns = dynamodb.Table(
            patterns_table or os.environ.get("PATTERNS_TABLE", "claudestreet-patterns")
        )

    # ──────────────────────────────────────────────
    # Trades
    # ──────────────────────────────────────────────

    def record_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_id: str,
    ) -> None:
        try:
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
                    "order_state": "pending",
                    "pnl": 0.0,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                }),
                ConditionExpression=Attr("trade_id").not_exists(),  # idempotent
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug("Duplicate trade_open %s, skipping", trade_id)
            else:
                raise

    def record_trade_close(
        self, trade_id: str, exit_price: float, pnl: float
    ) -> None:
        try:
            self._trades.update_item(
                Key={"trade_id": trade_id},
                UpdateExpression=(
                    "SET #s = :status, exit_price = :exit_price, "
                    "pnl = :pnl, closed_at = :closed_at"
                ),
                ConditionExpression=Attr("status").eq("open"),
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues=_to_decimal({
                    ":status": "closed",
                    ":exit_price": exit_price,
                    ":pnl": pnl,
                    ":closed_at": datetime.now(timezone.utc).isoformat(),
                }),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug("Duplicate trade_close %s, skipping", trade_id)
            else:
                raise

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            items: list[dict] = []
            query_kwargs = {
                "IndexName": "symbol-status-index",
                "KeyConditionExpression": (
                    Key("symbol").eq(symbol) & Key("status").eq("open")
                ),
            }
            while True:
                response = self._trades.query(**query_kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                query_kwargs["ExclusiveStartKey"] = last_key
            return [_from_decimal(item) for item in items]

        # Full scan with pagination to handle >1 MB of data
        items: list[dict] = []
        scan_kwargs = {"FilterExpression": Attr("status").eq("open")}
        while True:
            response = self._trades.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
        return [_from_decimal(item) for item in items]

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
            "wins": strategy_data.get("wins", 0),
            "losses": strategy_data.get("losses", 0),
            "kelly_fraction_mult": strategy_data.get("kelly_fraction_mult", 0.5),
            "regime_preference": strategy_data.get("regime_preference", ""),
            "version": strategy_data.get("version", 1),
        }
        expected_version = strategy_data.get("version", 1)
        try:
            self._strategies.put_item(
                Item=_to_decimal(item),
                ConditionExpression=(
                    Attr("strategy_id").not_exists()
                    | Attr("version").lt(expected_version)
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning("Strategy %s version conflict, skipping", strategy_data["id"])
            else:
                raise

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
        snapshot_id: str | None = None,
        market_context: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        item: dict[str, Any] = {
            "date": now.strftime("%Y-%m-%d"),
            "timestamp": now.isoformat(),
            "snapshot_id": snapshot_id or f"snap-{now.strftime('%Y%m%d%H%M%S')}",
            "cash": cash,
            "positions_value": positions_value,
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "open_positions": open_positions,
        }
        if market_context:
            item["market_context"] = market_context
        self._snapshots.put_item(Item=_to_decimal(item))

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
        try:
            self._events.put_item(
                Item=_to_decimal(item),
                ConditionExpression=Attr("event_id").not_exists(),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug("Duplicate event %s, skipping", event_id)
            else:
                raise

    def get_event_chain(self, correlation_id: str) -> list[dict]:
        response = self._events.query(
            IndexName="correlation-index",
            KeyConditionExpression=Key("correlation_id").eq(correlation_id),
            ScanIndexForward=True,
        )
        return [_from_decimal(item) for item in response.get("Items", [])]

    # ──────────────────────────────────────────────
    # System state (regime, etc.)
    # ──────────────────────────────────────────────

    def set_current_regime(self, regime: str) -> None:
        self._strategies.put_item(
            Item={
                "strategy_id": "__system_regime__",
                "regime": regime,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "is_active": 0,
            }
        )

    def get_current_regime(self) -> str:
        response = self._strategies.get_item(
            Key={"strategy_id": "__system_regime__"},
        )
        item = response.get("Item")
        if item:
            return item.get("regime", "")
        return ""

    # ──────────────────────────────────────────────
    # EPIC cache (symbol → IG EPIC mapping)
    # ──────────────────────────────────────────────

    def get_epic_cache(self, symbol: str) -> str | None:
        """Look up cached EPIC for a ticker symbol."""
        response = self._strategies.get_item(
            Key={"strategy_id": f"__epic_cache__{symbol}"},
        )
        item = response.get("Item")
        if item:
            return item.get("epic", None)
        return None

    def put_epic_cache(self, symbol: str, epic: str) -> None:
        """Cache a symbol → EPIC mapping."""
        self._strategies.put_item(
            Item={
                "strategy_id": f"__epic_cache__{symbol}",
                "epic": epic,
                "symbol": symbol,
                "is_active": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_all_epic_cache(self) -> dict[str, str]:
        """Get all cached symbol → EPIC mappings."""
        items: list[dict] = []
        scan_kwargs = {
            "FilterExpression": Attr("strategy_id").begins_with("__epic_cache__"),
        }
        while True:
            response = self._strategies.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
        return {item["symbol"]: item["epic"] for item in items if "symbol" in item and "epic" in item}

    # ──────────────────────────────────────────────
    # Decision Ledger
    # ──────────────────────────────────────────────

    def record_decision_step(
        self,
        correlation_id: str | None,
        step_key: str,
        agent: str,
        symbol: str,
        reasoning: dict,
        strategy_id: str | None = None,
    ) -> None:
        """Record a single agent's reasoning step in the decision ledger.

        Idempotent: skips duplicate writes for the same correlation_id + step_key.
        """
        if not correlation_id:
            return

        now = datetime.now(timezone.utc)
        ttl = int(now.timestamp()) + (90 * 86400)  # 90 day TTL

        item: dict[str, Any] = {
            "correlation_id": correlation_id,
            "step_key": step_key,
            "agent": agent,
            "symbol": symbol,
            "timestamp": now.isoformat(),
            "reasoning": reasoning,
            "ttl": ttl,
        }
        if strategy_id:
            item["strategy_id"] = strategy_id

        try:
            self._decisions.put_item(
                Item=_to_decimal(item),
                ConditionExpression=Attr("correlation_id").not_exists(),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug("Duplicate decision step %s/%s, skipping", correlation_id, step_key)
            else:
                raise

    def get_decision_chain(self, correlation_id: str) -> list[dict]:
        """Get all decision steps for a correlation_id in chronological order."""
        response = self._decisions.query(
            KeyConditionExpression=Key("correlation_id").eq(correlation_id),
            ScanIndexForward=True,
        )
        return [_from_decimal(item) for item in response.get("Items", [])]

    def get_decisions_by_symbol(self, symbol: str, limit: int = 50) -> list[dict]:
        """Get recent decisions for a symbol (most recent first)."""
        response = self._decisions.query(
            IndexName="symbol-index",
            KeyConditionExpression=Key("symbol").eq(symbol),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_from_decimal(item) for item in response.get("Items", [])]

    def get_decisions_by_strategy(self, strategy_id: str, limit: int = 50) -> list[dict]:
        """Get recent decisions for a strategy (most recent first)."""
        response = self._decisions.query(
            IndexName="strategy-index",
            KeyConditionExpression=Key("strategy_id").eq(strategy_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_from_decimal(item) for item in response.get("Items", [])]

    # ──────────────────────────────────────────────
    # Pattern Library
    # ──────────────────────────────────────────────

    def update_pattern(
        self,
        symbol: str,
        fingerprint: str,
        pnl: float,
        confidence: float,
    ) -> None:
        """Atomically update pattern win/loss/pnl counters."""
        pattern_key = f"{symbol}:{fingerprint}"
        now = datetime.now(timezone.utc)
        is_win = 1 if pnl > 0 else 0
        is_loss = 1 if pnl <= 0 else 0

        self._patterns.update_item(
            Key={"pattern_key": pattern_key},
            UpdateExpression=(
                "ADD occurrences :one, wins :win, losses :loss, total_pnl :pnl "
                "SET symbol = :sym, fingerprint = :fp, last_seen = :ts, "
                "avg_confidence = if_not_exists(avg_confidence, :zero) + :conf_delta"
            ),
            ExpressionAttributeValues=_to_decimal({
                ":one": 1,
                ":win": is_win,
                ":loss": is_loss,
                ":pnl": pnl,
                ":sym": symbol,
                ":fp": fingerprint,
                ":ts": now.isoformat(),
                ":zero": 0,
                ":conf_delta": confidence,
            }),
        )

    def get_pattern(self, symbol: str, fingerprint: str) -> dict | None:
        """Get a specific pattern by symbol and fingerprint."""
        pattern_key = f"{symbol}:{fingerprint}"
        response = self._patterns.get_item(Key={"pattern_key": pattern_key})
        item = response.get("Item")
        return _from_decimal(item) if item else None

    def get_patterns_for_symbol(
        self, symbol: str, min_occurrences: int = 5
    ) -> list[dict]:
        """Get patterns for a symbol with minimum occurrence threshold."""
        response = self._patterns.query(
            IndexName="symbol-index",
            KeyConditionExpression=Key("symbol").eq(symbol),
            ScanIndexForward=False,
        )
        results = []
        for item in response.get("Items", []):
            item = _from_decimal(item)
            if item.get("occurrences", 0) >= min_occurrences:
                results.append(item)
        return results
