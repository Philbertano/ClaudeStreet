"""Broker connector — synchronous order submission for Lambda.

Supports Alpaca (paper + live). Secrets loaded from
AWS Secrets Manager via the config system.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BrokerConnector:
    """Synchronous broker interface for order management."""

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import alpaca_trade_api as tradeapi
                self._client = tradeapi.REST(
                    self.api_key, self.secret_key, self.base_url
                )
            except ImportError:
                raise ImportError(
                    "alpaca-trade-api not installed. "
                    "Install with: pip install 'claudestreet[broker]'"
                )
        return self._client

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float | None = None,
        order_type: str = "limit",
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "qty": quantity,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if limit_price and order_type == "limit":
                kwargs["limit_price"] = str(limit_price)

            order = client.submit_order(**kwargs)
            return {
                "order_id": order.id,
                "status": order.status,
                "fill_price": float(order.filled_avg_price or limit_price or 0),
                "filled_qty": int(order.filled_qty or 0),
            }
        except Exception as e:
            logger.exception("Order failed for %s", symbol)
            return {"order_id": None, "status": "error", "error": str(e)}

    def get_positions(self) -> list[dict[str, Any]]:
        try:
            client = self._get_client()
            return [
                {
                    "symbol": p.symbol,
                    "quantity": int(p.qty),
                    "entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                }
                for p in client.list_positions()
            ]
        except Exception:
            logger.exception("Failed to get positions")
            return []
