"""Market data connector — fetches live and historical prices.

Synchronous interface for Lambda compatibility.
Uses yfinance for market data.
"""

from __future__ import annotations

import logging
from functools import partial

import pandas as pd

from claudestreet.models.events import MarketTickPayload

logger = logging.getLogger(__name__)


class MarketDataConnector:
    """Synchronous market data provider backed by yfinance."""

    def get_latest_tick(self, symbol: str) -> MarketTickPayload | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            price = getattr(info, "last_price", None) or 0.0
            prev_close = getattr(info, "previous_close", None) or price
            change_pct = (price - prev_close) / prev_close if prev_close > 0 else 0.0

            hist = ticker.history(period="1d")
            volume = int(hist["Volume"].iloc[-1]) if not hist.empty else 0

            return MarketTickPayload(
                symbol=symbol,
                price=round(price, 2),
                volume=volume,
                change_pct=round(change_pct, 4),
            )
        except Exception:
            logger.exception("Failed to fetch tick for %s", symbol)
            return None

    def get_historical(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            df.columns = [c.title() for c in df.columns]
            return df
        except Exception:
            logger.exception("Failed to fetch history for %s", symbol)
            return None

    def get_batch_ticks(self, symbols: list[str]) -> dict[str, MarketTickPayload]:
        ticks = {}
        for sym in symbols:
            tick = self.get_latest_tick(sym)
            if tick:
                ticks[sym] = tick
        return ticks
