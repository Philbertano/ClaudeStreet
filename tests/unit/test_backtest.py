"""Tests for the backtesting engine and sandbox."""

import numpy as np
import pandas as pd

from claudestreet.evolve_engine.backtest import BacktestEngine
from claudestreet.evolve_engine.sandbox import load_strategy_from_source, validate_strategy
from claudestreet.evolve_engine.strategy_template import CustomStrategy


VALID_STRATEGY_CODE = '''
import numpy as np
import pandas as pd
from claudestreet.evolve_engine.strategy_template import CustomStrategy

class SimpleRSIStrategy(CustomStrategy):
    name = "simple-rsi"
    description = "Buy when RSI < 30, sell when RSI > 70"
    version = "1.0"

    def evaluate(self, df: pd.DataFrame, params: dict[str, float]) -> tuple[str, float]:
        close = df["Close"].values
        if len(close) < 15:
            return "hold", 0.0

        period = int(params.get("rsi_period", 14))
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if rsi < 30:
            return "buy", min((30 - rsi) / 30, 1.0)
        elif rsi > 70:
            return "sell", min((rsi - 70) / 30, 1.0)
        return "hold", 0.0

    def get_default_params(self) -> dict[str, float]:
        return {"rsi_period": 14.0}
'''


def _make_test_df(n_bars: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV data with a trend."""
    np.random.seed(42)
    price = 100.0
    data = []
    for _ in range(n_bars):
        change = np.random.normal(0.0005, 0.02)
        open_ = price
        close = price * (1 + change)
        high = max(open_, close) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_, close) * (1 - abs(np.random.normal(0, 0.005)))
        volume = int(np.random.uniform(1e6, 5e6))
        data.append([open_, high, low, close, volume])
        price = close
    return pd.DataFrame(data, columns=["Open", "High", "Low", "Close", "Volume"])


def test_load_valid_strategy():
    cls = load_strategy_from_source(VALID_STRATEGY_CODE)
    assert cls is not None
    assert issubclass(cls, CustomStrategy)
    assert cls.name == "simple-rsi"


def test_validate_valid_strategy():
    cls = load_strategy_from_source(VALID_STRATEGY_CODE)
    errors = validate_strategy(cls)
    assert errors == [], f"Unexpected errors: {errors}"


def test_load_invalid_syntax():
    bad_code = "class Foo(CustomStrategy):\n    def evaluate(self"
    cls = load_strategy_from_source(bad_code)
    assert cls is None


def test_validate_no_name():
    code = '''
import pandas as pd
from claudestreet.evolve_engine.strategy_template import CustomStrategy

class NoNameStrategy(CustomStrategy):
    def evaluate(self, df, params):
        return "hold", 0.0
'''
    cls = load_strategy_from_source(code)
    assert cls is not None
    errors = validate_strategy(cls)
    assert any("name" in e.lower() for e in errors)


def test_backtest_runs():
    cls = load_strategy_from_source(VALID_STRATEGY_CODE)
    assert cls is not None

    df = _make_test_df(200)
    engine = BacktestEngine(initial_capital=100000.0)
    result = engine.run(cls, df, symbol="TEST")

    assert result.error is None
    assert result.total_bars == 200
    assert result.fitness.total_trades >= 0
    assert len(result.equity_curve) > 0


def test_backtest_fitness_computed():
    cls = load_strategy_from_source(VALID_STRATEGY_CODE)
    df = _make_test_df(300)
    engine = BacktestEngine(initial_capital=100000.0)
    result = engine.run(cls, df, symbol="TEST")

    # Fitness should have been computed
    f = result.fitness
    assert isinstance(f.composite, float)
    assert isinstance(f.sharpe_ratio, float)
    assert 0.0 <= f.win_rate <= 1.0
    assert 0.0 <= f.max_drawdown <= 1.0
