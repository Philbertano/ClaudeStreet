"""Technical analysis skill — indicator computation and signal evaluation.

Adapts OpenClaw's Skill pattern: a self-contained module that provides
domain-specific capabilities to agents. This skill computes standard
technical indicators and produces buy/sell recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalysisSkill:
    """Compute technical indicators and generate trading signals."""

    def compute_indicators(
        self,
        df: pd.DataFrame,
        params: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute all technical indicators on OHLCV data.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume].
            params: Strategy genome parameters to override defaults.

        Returns:
            Dict of indicator name → current value.
        """
        p = params or {}
        indicators: dict[str, float] = {}

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        if len(close) < 2:
            return indicators

        current_price = float(close[-1])
        indicators["close"] = current_price
        indicators["prev_close"] = float(close[-2])

        # --- RSI ---
        rsi_period = int(p.get("rsi_period", 14))
        indicators["rsi"] = self._rsi(close, rsi_period)

        # --- MACD ---
        macd_fast = int(p.get("macd_fast", 12))
        macd_slow = int(p.get("macd_slow", 26))
        macd_signal_period = int(p.get("macd_signal", 9))
        macd_line, signal_line, histogram = self._macd(
            close, macd_fast, macd_slow, macd_signal_period
        )
        indicators["macd"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = histogram

        # --- Bollinger Bands ---
        bb_period = int(p.get("bb_period", 20))
        bb_std = p.get("bb_std", 2.0)
        upper, middle, lower = self._bollinger(close, bb_period, bb_std)
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower
        indicators["bb_pct_b"] = (
            (current_price - lower) / (upper - lower)
            if (upper - lower) > 0 else 0.5
        )

        # --- EMAs ---
        ema_fast = int(p.get("ema_fast", 9))
        ema_slow = int(p.get("ema_slow", 21))
        indicators["ema_fast"] = self._ema(close, ema_fast)
        indicators["ema_slow"] = self._ema(close, ema_slow)
        indicators["ema_crossover"] = (
            1.0 if indicators["ema_fast"] > indicators["ema_slow"] else -1.0
        )

        # --- Volume ---
        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        indicators["volume"] = float(volume[-1])
        indicators["avg_volume_20"] = avg_vol
        indicators["volume_ratio"] = float(volume[-1]) / avg_vol if avg_vol > 0 else 1.0

        # --- ATR (Average True Range) ---
        indicators["atr_14"] = self._atr(high, low, close, 14)

        # --- SMA 50 / 200 ---
        if len(close) >= 50:
            indicators["sma_50"] = float(np.mean(close[-50:]))
        if len(close) >= 200:
            indicators["sma_200"] = float(np.mean(close[-200:]))

        return indicators

    def evaluate(
        self, indicators: dict[str, float], params: dict[str, float] | None = None
    ) -> tuple[str, float]:
        """Produce a recommendation from indicators.

        Returns:
            (recommendation, confidence) where recommendation is one of
            "strong_buy", "buy", "hold", "sell", "strong_sell".
        """
        p = params or {}
        score = 0.0
        signals = 0

        # RSI signals
        rsi = indicators.get("rsi", 50)
        oversold = p.get("rsi_oversold", 30)
        overbought = p.get("rsi_overbought", 70)
        if rsi < oversold:
            score += 1.0
            signals += 1
        elif rsi > overbought:
            score -= 1.0
            signals += 1
        else:
            signals += 1

        # MACD crossover
        macd_hist = indicators.get("macd_histogram", 0)
        if macd_hist > 0:
            score += 0.8
        elif macd_hist < 0:
            score -= 0.8
        signals += 1

        # Bollinger Band position
        bb_pct = indicators.get("bb_pct_b", 0.5)
        if bb_pct < 0.1:
            score += 0.7   # near lower band — oversold
        elif bb_pct > 0.9:
            score -= 0.7   # near upper band — overbought
        signals += 1

        # EMA crossover
        ema_cross = indicators.get("ema_crossover", 0)
        score += ema_cross * 0.6
        signals += 1

        # Volume confirmation
        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            score *= 1.2   # amplify signal on high volume

        # SMA 50/200 trend
        sma_50 = indicators.get("sma_50")
        sma_200 = indicators.get("sma_200")
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                score += 0.4  # golden cross territory
            else:
                score -= 0.4  # death cross territory
            signals += 1

        # Normalize to recommendation
        if signals == 0:
            return "hold", 0.0

        avg_score = score / signals
        confidence = min(abs(avg_score), 1.0)

        if avg_score > 0.6:
            return "strong_buy", confidence
        elif avg_score > 0.2:
            return "buy", confidence
        elif avg_score < -0.6:
            return "strong_sell", confidence
        elif avg_score < -0.2:
            return "sell", confidence
        return "hold", confidence

    def summarize(
        self, symbol: str, indicators: dict[str, float], recommendation: str
    ) -> str:
        rsi = indicators.get("rsi", 0)
        macd_h = indicators.get("macd_histogram", 0)
        bb_pct = indicators.get("bb_pct_b", 0.5)
        ema_cross = "bullish" if indicators.get("ema_crossover", 0) > 0 else "bearish"

        return (
            f"{symbol}: {recommendation.upper()} | "
            f"RSI={rsi:.1f} MACD_H={macd_h:.4f} "
            f"BB%B={bb_pct:.2f} EMA={ema_cross}"
        )

    # --- Internal indicator calculations ---

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = float(data[0])
        for val in data[1:]:
            ema = (float(val) - ema) * multiplier + ema
        return ema

    @staticmethod
    def _macd(
        close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float, float, float]:
        if len(close) < slow:
            return 0.0, 0.0, 0.0

        def ema_series(data, period):
            result = [float(data[0])]
            mult = 2 / (period + 1)
            for val in data[1:]:
                result.append((float(val) - result[-1]) * mult + result[-1])
            return result

        ema_fast = ema_series(close, fast)
        ema_slow = ema_series(close, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = ema_series(macd_line, signal)

        return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]

    @staticmethod
    def _bollinger(
        close: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> tuple[float, float, float]:
        if len(close) < period:
            p = float(close[-1])
            return p, p, p
        window = close[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        return middle + num_std * std, middle, middle - num_std * std

    @staticmethod
    def _atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        if len(close) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            tr = max(
                float(high[i]) - float(low[i]),
                abs(float(high[i]) - float(close[i - 1])),
                abs(float(low[i]) - float(close[i - 1])),
            )
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0
