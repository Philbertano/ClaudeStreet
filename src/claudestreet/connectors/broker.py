"""Broker connector — IG Markets CFD trading via REST API.

Supports live and demo IG accounts. Credentials loaded from
AWS Secrets Manager via the config system. Uses the trading_ig
library for session management and order submission.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

logger = logging.getLogger(__name__)

# Session refresh interval (5 minutes)
_SESSION_REFRESH_SECS = 300


class BrokerConnector:
    """Synchronous IG Markets broker interface for order management."""

    def __init__(
        self,
        api_key: str = "",
        username: str = "",
        password: str = "",
        acc_number: str = "",
        acc_type: str = "LIVE",
        memory: Any = None,
    ) -> None:
        self.api_key = api_key
        self.username = username
        self.password = password
        self.acc_number = acc_number
        self.acc_type = acc_type
        self.memory = memory
        self._ig: Any = None
        self._session_created_at: float = 0.0
        # In-memory EPIC cache for the lifetime of this connector
        self._epic_cache: dict[str, str] = {}

    def _get_session(self):
        """Create or refresh the IG API session lazily."""
        now = time.time()
        if self._ig is not None and (now - self._session_created_at) < _SESSION_REFRESH_SECS:
            return self._ig

        try:
            from trading_ig import IGService
        except ImportError:
            raise ImportError(
                "trading-ig not installed. "
                "Install with: pip install 'claudestreet[broker]'"
            )

        acc_type_upper = self.acc_type.upper()
        if acc_type_upper not in ("LIVE", "DEMO"):
            acc_type_upper = "LIVE"

        self._ig = IGService(
            username=self.username,
            password=self.password,
            api_key=self.api_key,
            acc_type=acc_type_upper,
            acc_number=self.acc_number,
            use_rate_limiter=True,
        )
        self._ig.create_session()
        self._session_created_at = time.time()
        logger.info("IG session created (acc_type=%s)", acc_type_upper)
        return self._ig

    def _ensure_session(self):
        """Get session, retrying auth on failure."""
        try:
            return self._get_session()
        except Exception:
            # Force a fresh session on auth failure
            self._ig = None
            self._session_created_at = 0.0
            return self._get_session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying submit_order (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float | None = None,
        order_type: str = "market",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        currency_code: str = "GBP",
        epic: str | None = None,
    ) -> dict[str, Any]:
        """Submit an order to IG Markets.

        For market orders: creates an open position immediately.
        For limit orders: creates a working order.
        """
        ig = self._ensure_session()
        resolved_epic = epic or self.resolve_epic(symbol)
        ig_direction = "BUY" if side.lower() == "buy" else "SELL"

        if order_type == "market":
            response = ig.create_open_position(
                epic=resolved_epic,
                direction=ig_direction,
                size=quantity,
                currency_code=currency_code,
                order_type="MARKET",
                force_open=True,
                guaranteed_stop=False,
                stop_level=stop_loss,
                limit_level=take_profit,
            )
        else:
            # Limit / working order
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            response = ig.create_working_order(
                epic=resolved_epic,
                direction=ig_direction,
                size=quantity,
                currency_code=currency_code,
                order_type="LIMIT",
                level=limit_price,
                force_open=True,
                guaranteed_stop=False,
                stop_level=stop_loss,
                limit_level=take_profit,
                time_in_force="GOOD_TILL_CANCELLED",
            )

        deal_ref = response.get("dealReference", "")
        # Confirm the deal to get the deal_id
        confirmation = ig.fetch_deal_by_deal_reference(deal_ref)

        deal_id = confirmation.get("dealId", deal_ref)
        status = confirmation.get("dealStatus", "UNKNOWN").lower()
        level = confirmation.get("level", limit_price or 0)

        logger.info(
            "[broker] IG %s %s %.2f %s @ %.2f (deal=%s, status=%s)",
            ig_direction, resolved_epic, quantity, symbol, float(level or 0),
            deal_id, status,
        )

        return {
            "order_id": deal_id,
            "status": "filled" if status == "accepted" else status,
            "fill_price": float(level or 0),
            "filled_qty": quantity if status == "accepted" else 0,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying get_positions (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions from IG."""
        ig = self._ensure_session()
        response = ig.fetch_open_positions()

        positions = []
        if response is not None and hasattr(response, "iterrows"):
            for _, row in response.iterrows():
                positions.append({
                    "symbol": row.get("instrumentName", ""),
                    "quantity": float(row.get("size", 0)),
                    "entry_price": float(row.get("level", 0)),
                    "current_price": float(row.get("bid", 0)),
                    "unrealized_pnl": float(row.get("profit", 0)),
                    "deal_id": row.get("dealId", ""),
                    "epic": row.get("epic", ""),
                    "direction": row.get("direction", ""),
                })
        return positions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying close_position (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
    def close_position(
        self,
        deal_id: str,
        direction: str,
        size: float,
        epic: str | None = None,
    ) -> dict[str, Any]:
        """Close an existing position by deal_id."""
        ig = self._ensure_session()

        # To close, we send the opposite direction
        close_direction = "SELL" if direction.upper() == "BUY" else "BUY"

        response = ig.close_open_position(
            deal_id=deal_id,
            direction=close_direction,
            size=size,
            order_type="MARKET",
        )

        deal_ref = response.get("dealReference", "")
        confirmation = ig.fetch_deal_by_deal_reference(deal_ref)

        return {
            "order_id": confirmation.get("dealId", deal_ref),
            "status": confirmation.get("dealStatus", "UNKNOWN").lower(),
            "fill_price": float(confirmation.get("level", 0)),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying get_account_info (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
    def get_account_info(self) -> dict[str, Any]:
        """Get account balance and margin information."""
        ig = self._ensure_session()
        accounts = ig.fetch_accounts()

        if accounts is not None and hasattr(accounts, "iterrows"):
            for _, row in accounts.iterrows():
                if row.get("accountId") == self.acc_number or self.acc_number == "":
                    return {
                        "balance": float(row.get("balance", 0)),
                        "deposit": float(row.get("deposit", 0)),
                        "profit_loss": float(row.get("profitLoss", 0)),
                        "available": float(row.get("available", 0)),
                    }

        return {"balance": 0, "deposit": 0, "profit_loss": 0, "available": 0}

    def resolve_epic(self, symbol: str) -> str:
        """Resolve a ticker symbol to an IG EPIC code.

        Resolution order:
        1. In-memory cache (fast)
        2. DynamoDB EPIC cache via memory (persistent)
        3. IG market search API (slow, cached back to DynamoDB)
        """
        # 1. In-memory cache
        if symbol in self._epic_cache:
            return self._epic_cache[symbol]

        # 2. DynamoDB cache
        if self.memory is not None:
            cached = self.memory.get_epic_cache(symbol)
            if cached:
                self._epic_cache[symbol] = cached
                return cached

        # 3. IG market search
        epic = self._search_epic(symbol)
        self._epic_cache[symbol] = epic

        # Persist to DynamoDB
        if self.memory is not None:
            self.memory.put_epic_cache(symbol, epic)

        return epic

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying _search_epic (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
    def _search_epic(self, symbol: str) -> str:
        """Search IG markets API for a symbol's EPIC code."""
        ig = self._ensure_session()
        results = ig.search_markets(symbol)

        if results is not None and hasattr(results, "iterrows"):
            for _, row in results.iterrows():
                epic = row.get("epic", "")
                if epic:
                    logger.info("[broker] Resolved %s -> %s", symbol, epic)
                    return epic

        raise ValueError(f"Could not resolve EPIC for symbol: {symbol}")
