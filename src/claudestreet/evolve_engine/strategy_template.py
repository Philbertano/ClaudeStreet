"""Base class that all LLM-generated strategies must inherit from.

Claude writes subclasses of CustomStrategy. Each one is a self-contained
trading strategy with its own evaluate() logic. The evolution engine
dynamically loads and backtests these.

Generated code is stored in S3 as Python source files and loaded
via restricted exec() with whitelisted imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CustomStrategy(ABC):
    """Interface for LLM-generated trading strategies.

    All generated strategies must:
      1. Inherit from CustomStrategy
      2. Set name and description
      3. Implement evaluate() returning (recommendation, confidence)

    Available imports in generated code:
      numpy (as np), pandas (as pd), math

    Recommendations: "strong_buy", "buy", "hold", "sell", "strong_sell"
    Confidence: 0.0 to 1.0
    """

    name: str = "unnamed"
    description: str = ""
    version: str = "1.0"

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def evaluate(
        self, df: pd.DataFrame, params: dict[str, float]
    ) -> tuple[str, float]:
        """Evaluate the strategy on OHLCV data.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume].
                 At least 200 rows of daily data.
            params: Strategy genome parameters (tunable by evolution).

        Returns:
            (recommendation, confidence) where:
              recommendation: "strong_buy"|"buy"|"hold"|"sell"|"strong_sell"
              confidence: float 0.0 to 1.0
        """
        ...

    def get_default_params(self) -> dict[str, float]:
        """Return default parameter values for this strategy."""
        return {}


# ── Template shown to Claude when generating strategies ──

STRATEGY_TEMPLATE = '''\
import math
import numpy as np
import pandas as pd
from claudestreet.evolve_engine.strategy_template import CustomStrategy


class {class_name}(CustomStrategy):
    """
    {description}
    """

    name = "{name}"
    description = "{description}"
    version = "1.0"

    def evaluate(self, df: pd.DataFrame, params: dict[str, float]) -> tuple[str, float]:
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        # --- Your strategy logic here ---
        # Must return (recommendation, confidence)
        # recommendation: "strong_buy", "buy", "hold", "sell", "strong_sell"
        # confidence: 0.0 to 1.0

        return "hold", 0.0

    def get_default_params(self) -> dict[str, float]:
        return {{}}
'''
