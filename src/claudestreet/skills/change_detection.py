"""Change-point detection — CUSUM on daily strategy PnL streams.

Detects when the cumulative deviation of strategy PnL exceeds a
threshold, indicating a regime change or strategy breakdown.
Triggers REGIME_CHANGE event for emergency evolution.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CUSUMDetector:
    """Cumulative Sum (CUSUM) change-point detector for PnL streams.

    Detects both positive and negative shifts in the mean of a stream.
    When the cumulative sum exceeds a threshold, a change point is flagged.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.0,
        min_observations: int = 10,
    ) -> None:
        """
        Args:
            threshold: CUSUM threshold for triggering a change point.
                Higher = fewer false positives, slower detection.
            drift: Allowable drift before counting as change (slack parameter).
            min_observations: Minimum observations before detection starts.
        """
        self.threshold = threshold
        self.drift = drift
        self.min_observations = min_observations

    def detect(self, values: list[float]) -> dict[str, Any]:
        """Run CUSUM on a PnL stream.

        Args:
            values: List of daily PnL values.

        Returns:
            Dict with:
                - change_detected: bool
                - change_index: int or None (index of change point)
                - direction: "positive" or "negative" or None
                - cusum_pos: current positive CUSUM value
                - cusum_neg: current negative CUSUM value
        """
        if len(values) < self.min_observations:
            return {
                "change_detected": False,
                "change_index": None,
                "direction": None,
                "cusum_pos": 0.0,
                "cusum_neg": 0.0,
            }

        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr[:self.min_observations]))
        std = float(np.std(arr[:self.min_observations]))

        if std == 0:
            std = 1.0

        # Normalize
        normalized = (arr - mean) / std

        cusum_pos = 0.0
        cusum_neg = 0.0
        change_index = None
        direction = None

        for i in range(self.min_observations, len(normalized)):
            cusum_pos = max(0, cusum_pos + normalized[i] - self.drift)
            cusum_neg = max(0, cusum_neg - normalized[i] - self.drift)

            if cusum_pos > self.threshold:
                change_index = i
                direction = "positive"
                break
            if cusum_neg > self.threshold:
                change_index = i
                direction = "negative"
                break

        return {
            "change_detected": change_index is not None,
            "change_index": change_index,
            "direction": direction,
            "cusum_pos": round(cusum_pos, 4),
            "cusum_neg": round(cusum_neg, 4),
            "mean_baseline": round(mean, 4),
            "std_baseline": round(std, 4),
        }

    def detect_multi_strategy(
        self, strategy_pnls: dict[str, list[float]]
    ) -> dict[str, dict[str, Any]]:
        """Run CUSUM on multiple strategy PnL streams.

        Args:
            strategy_pnls: Dict of strategy_id → list of daily PnL values.

        Returns:
            Dict of strategy_id → CUSUM detection result.
        """
        results = {}
        for strategy_id, pnls in strategy_pnls.items():
            results[strategy_id] = self.detect(pnls)
        return results

    def detect_aggregate(self, values: list[float]) -> dict[str, Any]:
        """Detect change point on aggregate (portfolio-level) PnL stream.

        Uses a more sensitive threshold for aggregate detection since
        portfolio-level shifts are more significant.
        """
        sensitive = CUSUMDetector(
            threshold=self.threshold * 0.7,
            drift=self.drift,
            min_observations=self.min_observations,
        )
        return sensitive.detect(values)
