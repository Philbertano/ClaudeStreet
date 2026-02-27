"""Technical analysis skill — indicator computation and signal evaluation.

Computes standard and advanced technical indicators and produces
buy/sell recommendations. Includes Stochastic RSI, MFI,
Accumulation/Distribution, relative strength vs SPY,
volume-price divergence, and multi-timeframe confirmation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalysisSkill:
    """Compute technical indicators and generate trading signals."""

    def compute_indicators(
        self,
        df: pd.DataFrame,
        params: dict[str, float] | None = None,
        spy_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute all technical indicators on OHLCV data.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume].
            params: Strategy genome parameters to override defaults.
            spy_df: Optional SPY DataFrame for relative strength calculation.

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

        # --- Stochastic RSI ---
        indicators["stoch_rsi_k"], indicators["stoch_rsi_d"] = self._stochastic_rsi(
            close, rsi_period
        )

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

        # --- Money Flow Index (MFI) ---
        indicators["mfi"] = self._mfi(high, low, close, volume, 14)

        # --- Accumulation/Distribution Line ---
        indicators["ad_line"] = self._accumulation_distribution(high, low, close, volume)

        # --- Relative Strength vs SPY ---
        if spy_df is not None and len(spy_df) >= 20:
            spy_close = spy_df["Close"].values
            indicators["relative_strength_spy"] = self._relative_strength(
                close, spy_close, 20
            )

        # --- Volume-Price Divergence ---
        indicators["volume_price_divergence"] = self._volume_price_divergence(
            close, volume, 20
        )

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

        # Stochastic RSI
        stoch_k = indicators.get("stoch_rsi_k")
        if stoch_k is not None:
            if stoch_k < 0.2:
                score += 0.5  # oversold
            elif stoch_k > 0.8:
                score -= 0.5  # overbought
            signals += 1

        # MFI
        mfi = indicators.get("mfi")
        if mfi is not None:
            if mfi < 20:
                score += 0.4  # money flow oversold
            elif mfi > 80:
                score -= 0.4  # money flow overbought
            signals += 1

        # Volume-price divergence (bearish signal)
        vpd = indicators.get("volume_price_divergence", 0)
        if vpd < -0.5:
            score -= 0.3  # new highs on declining volume
            signals += 1

        # Relative strength vs SPY
        rs = indicators.get("relative_strength_spy")
        if rs is not None:
            score += rs * 0.3  # favor stocks outperforming market
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

    def evaluate_detailed(
        self, indicators: dict[str, float], params: dict[str, float] | None = None
    ) -> dict:
        """Full signal breakdown with condition fingerprint.

        Returns dict with recommendation, confidence, score, per-signal
        details, and a deterministic fingerprint string.
        """
        p = params or {}
        score = 0.0
        signals = 0
        signal_details: dict[str, dict] = {}
        fingerprint_parts: list[str] = []

        # RSI
        rsi = indicators.get("rsi", 50)
        oversold = p.get("rsi_oversold", 30)
        overbought = p.get("rsi_overbought", 70)
        if rsi < oversold:
            rsi_score = 1.0
            rsi_state = "oversold"
        elif rsi > overbought:
            rsi_score = -1.0
            rsi_state = "overbought"
        else:
            rsi_score = 0.0
            rsi_state = "neutral"
        score += rsi_score
        signals += 1
        signal_details["rsi"] = {"value": rsi, "score": rsi_score, "state": rsi_state}
        fingerprint_parts.append(f"rsi={rsi_state}")

        # MACD
        macd_hist = indicators.get("macd_histogram", 0)
        if macd_hist > 0:
            macd_score = 0.8
            macd_state = "bullish"
        elif macd_hist < 0:
            macd_score = -0.8
            macd_state = "bearish"
        else:
            macd_score = 0.0
            macd_state = "neutral"
        score += macd_score
        signals += 1
        signal_details["macd"] = {"value": macd_hist, "score": macd_score, "state": macd_state}
        fingerprint_parts.append(f"macd={macd_state}")

        # Bollinger Band
        bb_pct = indicators.get("bb_pct_b", 0.5)
        if bb_pct < 0.1:
            bb_score = 0.7
            bb_state = "lower"
        elif bb_pct > 0.9:
            bb_score = -0.7
            bb_state = "upper"
        else:
            bb_score = 0.0
            bb_state = "mid"
        score += bb_score
        signals += 1
        signal_details["bb"] = {"value": bb_pct, "score": bb_score, "state": bb_state}
        fingerprint_parts.append(f"bb={bb_state}")

        # EMA crossover
        ema_cross = indicators.get("ema_crossover", 0)
        ema_score = ema_cross * 0.6
        if ema_cross > 0:
            ema_state = "bullish"
        elif ema_cross < 0:
            ema_state = "bearish"
        else:
            ema_state = "neutral"
        score += ema_score
        signals += 1
        signal_details["ema"] = {"value": ema_cross, "score": ema_score, "state": ema_state}
        fingerprint_parts.append(f"ema={ema_state}")

        # Volume confirmation (amplifier, not a signal count)
        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 2.0:
            score *= 1.2

        # SMA 50/200
        sma_50 = indicators.get("sma_50")
        sma_200 = indicators.get("sma_200")
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                sma_score = 0.4
                sma_state = "golden"
            else:
                sma_score = -0.4
                sma_state = "death"
            score += sma_score
            signals += 1
            signal_details["sma"] = {"value": sma_50 - sma_200, "score": sma_score, "state": sma_state}
            fingerprint_parts.append(f"sma={sma_state}")

        # Stochastic RSI
        stoch_k = indicators.get("stoch_rsi_k")
        if stoch_k is not None:
            if stoch_k < 0.2:
                stoch_score = 0.5
                stoch_state = "oversold"
            elif stoch_k > 0.8:
                stoch_score = -0.5
                stoch_state = "overbought"
            else:
                stoch_score = 0.0
                stoch_state = "neutral"
            score += stoch_score
            signals += 1
            signal_details["stoch_rsi"] = {"value": stoch_k, "score": stoch_score, "state": stoch_state}
            fingerprint_parts.append(f"stoch_rsi={stoch_state}")

        # MFI
        mfi = indicators.get("mfi")
        if mfi is not None:
            if mfi < 20:
                mfi_score = 0.4
                mfi_state = "oversold"
            elif mfi > 80:
                mfi_score = -0.4
                mfi_state = "overbought"
            else:
                mfi_score = 0.0
                mfi_state = "neutral"
            score += mfi_score
            signals += 1
            signal_details["mfi"] = {"value": mfi, "score": mfi_score, "state": mfi_state}
            fingerprint_parts.append(f"mfi={mfi_state}")

        # Volume-price divergence
        vpd = indicators.get("volume_price_divergence", 0)
        if vpd < -0.5:
            score -= 0.3
            signals += 1

        # Relative strength vs SPY
        rs = indicators.get("relative_strength_spy")
        if rs is not None:
            score += rs * 0.3
            signals += 1

        # Normalize
        if signals == 0:
            return {
                "recommendation": "hold",
                "confidence": 0.0,
                "score": 0.0,
                "signals": signal_details,
                "fingerprint": "|".join(fingerprint_parts),
            }

        avg_score = score / signals
        confidence = min(abs(avg_score), 1.0)

        if avg_score > 0.6:
            recommendation = "strong_buy"
        elif avg_score > 0.2:
            recommendation = "buy"
        elif avg_score < -0.6:
            recommendation = "strong_sell"
        elif avg_score < -0.2:
            recommendation = "sell"
        else:
            recommendation = "hold"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "score": round(avg_score, 6),
            "signals": signal_details,
            "fingerprint": "|".join(fingerprint_parts),
        }

    def summarize(
        self, symbol: str, indicators: dict[str, float], recommendation: str
    ) -> str:
        rsi = indicators.get("rsi", 0)
        macd_h = indicators.get("macd_histogram", 0)
        bb_pct = indicators.get("bb_pct_b", 0.5)
        ema_cross = "bullish" if indicators.get("ema_crossover", 0) > 0 else "bearish"
        mfi = indicators.get("mfi", 50)
        stoch = indicators.get("stoch_rsi_k", 0.5)

        return (
            f"{symbol}: {recommendation.upper()} | "
            f"RSI={rsi:.1f} StochRSI={stoch:.2f} MFI={mfi:.1f} "
            f"MACD_H={macd_h:.4f} BB%B={bb_pct:.2f} EMA={ema_cross}"
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
    def _stochastic_rsi(
        close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14
    ) -> tuple[float, float]:
        """Compute Stochastic RSI (%K and %D)."""
        if len(close) < rsi_period + stoch_period + 1:
            return 0.5, 0.5

        # Compute RSI series
        rsi_values = []
        for i in range(stoch_period + 3, 0, -1):
            end = len(close) - i if i > 0 else len(close)
            if end < rsi_period + 1:
                continue
            segment = close[:end]
            deltas = np.diff(segment[-(rsi_period + 1):])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        if len(rsi_values) < stoch_period:
            return 0.5, 0.5

        recent = rsi_values[-stoch_period:]
        rsi_min = min(recent)
        rsi_max = max(recent)
        rsi_range = rsi_max - rsi_min

        if rsi_range == 0:
            stoch_k = 0.5
        else:
            stoch_k = (rsi_values[-1] - rsi_min) / rsi_range

        # %D is 3-period SMA of %K — compute from recent %K values
        if len(rsi_values) >= stoch_period + 2:
            # Compute %K for last 3 points to get %D
            k_values = []
            for offset in range(3):
                idx = len(rsi_values) - offset
                window = rsi_values[idx - stoch_period:idx]
                w_min = min(window)
                w_max = max(window)
                w_range = w_max - w_min
                if w_range == 0:
                    k_values.append(0.5)
                else:
                    k_values.append((rsi_values[idx - 1] - w_min) / w_range)
            stoch_d = sum(k_values) / len(k_values)
        else:
            stoch_d = stoch_k
        return float(stoch_k), float(stoch_d)

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

    @staticmethod
    def _mfi(
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        volume: np.ndarray, period: int = 14,
    ) -> float:
        """Money Flow Index — volume-weighted RSI."""
        if len(close) < period + 1:
            return 50.0

        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        positive_flow = 0.0
        negative_flow = 0.0

        for i in range(-period, 0):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow += float(raw_money_flow[i])
            elif typical_price[i] < typical_price[i - 1]:
                negative_flow += float(raw_money_flow[i])

        if negative_flow == 0:
            return 100.0
        money_ratio = positive_flow / negative_flow
        return float(100 - (100 / (1 + money_ratio)))

    @staticmethod
    def _accumulation_distribution(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    ) -> float:
        """Accumulation/Distribution Line (current value)."""
        if len(close) < 2:
            return 0.0

        ad = 0.0
        for i in range(len(close)):
            hl_range = float(high[i]) - float(low[i])
            if hl_range > 0:
                clv = ((float(close[i]) - float(low[i])) - (float(high[i]) - float(close[i]))) / hl_range
                ad += clv * float(volume[i])
        return ad

    @staticmethod
    def _relative_strength(
        stock_close: np.ndarray, benchmark_close: np.ndarray, period: int = 20
    ) -> float:
        """Relative strength: stock return - benchmark return over period."""
        if len(stock_close) < period + 1 or len(benchmark_close) < period + 1:
            return 0.0

        stock_return = (float(stock_close[-1]) - float(stock_close[-period - 1])) / float(stock_close[-period - 1])
        bench_return = (float(benchmark_close[-1]) - float(benchmark_close[-period - 1])) / float(benchmark_close[-period - 1])

        return stock_return - bench_return

    @staticmethod
    def _volume_price_divergence(
        close: np.ndarray, volume: np.ndarray, period: int = 20
    ) -> float:
        """Detect volume-price divergence.

        Returns negative value if price making new highs on declining volume (bearish).
        Returns positive value if price making new lows on increasing volume (could be capitulation).
        """
        if len(close) < period:
            return 0.0

        recent_close = close[-period:]
        recent_volume = volume[-period:]

        # Check if near period highs
        price_pct = (float(close[-1]) - float(np.min(recent_close))) / max(
            float(np.max(recent_close)) - float(np.min(recent_close)), 0.01
        )

        # Volume trend (linear regression slope)
        x = np.arange(period)
        if np.std(recent_volume.astype(float)) > 0:
            vol_slope = float(np.polyfit(x, recent_volume.astype(float), 1)[0])
            vol_slope_normalized = vol_slope / float(np.mean(recent_volume))
        else:
            vol_slope_normalized = 0.0

        # Near highs + declining volume = bearish divergence
        if price_pct > 0.8 and vol_slope_normalized < -0.01:
            return -1.0 * abs(vol_slope_normalized) * 10

        # Near lows + increasing volume = potential reversal
        if price_pct < 0.2 and vol_slope_normalized > 0.01:
            return 1.0 * abs(vol_slope_normalized) * 10

        return 0.0
