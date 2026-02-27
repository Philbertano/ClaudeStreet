"""Market regime detection — classifies current market conditions.

Uses rolling volatility, trend strength, and optionally VIX level
to classify the market into regimes. Strategies should match their
behavior to the current regime.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """Classify market regime from OHLCV data."""

    def __init__(
        self,
        vol_window: int = 20,
        trend_window: int = 60,
        vol_high_threshold: float = 0.25,
        trend_threshold: float = 0.001,
    ) -> None:
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_high_threshold = vol_high_threshold
        self.trend_threshold = trend_threshold

    def detect(
        self,
        df: pd.DataFrame,
        vix_level: float | None = None,
    ) -> MarketRegime:
        """Classify current market regime.

        Args:
            df: OHLCV DataFrame (needs at least trend_window+10 bars).
            vix_level: Optional VIX level for volatility confirmation.

        Returns:
            Current MarketRegime classification.
        """
        if len(df) < self.trend_window + 10:
            return MarketRegime.MEAN_REVERTING

        close = df["Close"].values.astype(float)

        # 1. Realized volatility (20-day annualized)
        returns = np.diff(np.log(close[-self.vol_window - 1:]))
        realized_vol = float(np.std(returns) * np.sqrt(252))

        # 2. Trend strength (linear regression slope over 60 days)
        trend_data = close[-self.trend_window:]
        x = np.arange(len(trend_data))
        slope = float(np.polyfit(x, trend_data, 1)[0])
        # Normalize slope by price level
        normalized_slope = slope / float(np.mean(trend_data))

        # 3. VIX override — very high VIX always = HIGH_VOLATILITY
        if vix_level is not None and vix_level > 30:
            return MarketRegime.HIGH_VOLATILITY

        # 4. Classification logic
        if realized_vol > self.vol_high_threshold:
            return MarketRegime.HIGH_VOLATILITY

        if normalized_slope > self.trend_threshold:
            return MarketRegime.TRENDING_BULL
        elif normalized_slope < -self.trend_threshold:
            return MarketRegime.TRENDING_BEAR

        return MarketRegime.MEAN_REVERTING

    def detect_for_symbols(
        self,
        symbol_data: dict[str, pd.DataFrame],
        vix_level: float | None = None,
    ) -> dict[str, MarketRegime]:
        """Detect regime for multiple symbols."""
        return {
            symbol: self.detect(df, vix_level)
            for symbol, df in symbol_data.items()
            if df is not None and not df.empty
        }

    def get_regime_summary(self, df: pd.DataFrame) -> dict:
        """Get detailed regime metrics for diagnostics."""
        if len(df) < self.trend_window + 10:
            return {"regime": MarketRegime.MEAN_REVERTING.value, "metrics": {}}

        close = df["Close"].values.astype(float)

        returns = np.diff(np.log(close[-self.vol_window - 1:]))
        realized_vol = float(np.std(returns) * np.sqrt(252))

        trend_data = close[-self.trend_window:]
        x = np.arange(len(trend_data))
        slope, intercept = np.polyfit(x, trend_data, 1)
        normalized_slope = slope / float(np.mean(trend_data))

        regime = self.detect(df)

        return {
            "regime": regime.value,
            "metrics": {
                "realized_vol_20d": round(realized_vol, 4),
                "trend_slope_60d": round(float(slope), 4),
                "normalized_slope": round(normalized_slope, 6),
                "current_price": round(float(close[-1]), 2),
                "price_20d_ago": round(float(close[-self.vol_window]), 2),
            },
        }
