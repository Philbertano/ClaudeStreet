"""Market data connector — fetches live and historical prices.

Synchronous interface for Lambda compatibility.
Uses yfinance for market data.
"""

from __future__ import annotations

import logging

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from claudestreet.models.events import MarketTickPayload

logger = logging.getLogger(__name__)


class MarketDataConnector:
    """Synchronous market data provider backed by yfinance."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying get_latest_tick (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.warning(
            "Retrying get_historical (attempt %d)", rs.attempt_number
        ),
        reraise=True,
    )
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

    def get_correlation_matrix(
        self, symbols: list[str], period: str = "3mo"
    ) -> pd.DataFrame | None:
        """Compute rolling correlation matrix between symbols using returns."""
        try:

            data = {}
            for sym in symbols:
                df = self.get_historical(sym, period=period)
                if df is not None and not df.empty:
                    data[sym] = df["Close"].pct_change().dropna()

            if len(data) < 2:
                return None

            returns_df = pd.DataFrame(data)
            return returns_df.corr()
        except Exception:
            logger.exception("Failed to compute correlation matrix")
            return None

    def get_cross_asset_signals(self) -> dict[str, float]:
        """Fetch cross-asset signals: VIX, VIX3M term structure, TLT, UUP, GLD."""
        signals: dict[str, float] = {}
        cross_assets = {
            "^VIX": "vix",
            "^VIX3M": "vix3m",
            "TLT": "tlt",
            "UUP": "uup",
            "GLD": "gld",
        }

        for ticker, key in cross_assets.items():
            tick = self.get_latest_tick(ticker)
            if tick:
                signals[key] = tick.price
                signals[f"{key}_change"] = tick.change_pct

        # VIX term structure (contango/backwardation)
        if "vix" in signals and "vix3m" in signals and signals["vix3m"] > 0:
            signals["vix_term_structure"] = signals["vix"] / signals["vix3m"]

        return signals
