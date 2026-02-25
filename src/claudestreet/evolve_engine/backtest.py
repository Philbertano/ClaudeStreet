"""Backtesting engine — evaluates strategies against historical data.

Walks through OHLCV bars, calls the strategy's evaluate() on each,
simulates trades with stop-loss/take-profit, and computes fitness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import pandas as pd

from claudestreet.evolution.fitness import FitnessEvaluator
from claudestreet.evolve_engine.strategy_template import CustomStrategy
from claudestreet.models.strategy import FitnessScore

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: float
    entry_bar: int
    stop_loss: float
    take_profit: float
    exit_price: float = 0.0
    exit_bar: int = 0
    pnl: float = 0.0
    status: str = "open"


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    fitness: FitnessScore = field(default_factory=FitnessScore)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    total_bars: int = 0
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"FAILED: {self.error}"
        f = self.fitness
        return (
            f"{self.strategy_name} on {self.symbol}: "
            f"return=${f.total_return:,.2f} sharpe={f.sharpe_ratio:.2f} "
            f"dd={f.max_drawdown:.1%} wr={f.win_rate:.0%} "
            f"trades={f.total_trades} pf={f.profit_factor:.2f} "
            f"composite={f.composite:.4f}"
        )


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        lookback_bars: int = 50,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.lookback_bars = lookback_bars
        self.evaluator = FitnessEvaluator()

    def run(
        self,
        strategy_cls: Type[CustomStrategy],
        df: pd.DataFrame,
        symbol: str = "TEST",
        params: dict[str, float] | None = None,
    ) -> BacktestResult:
        """Run a backtest on historical OHLCV data.

        Args:
            strategy_cls: The strategy class to test.
            df: OHLCV DataFrame (must have Open, High, Low, Close, Volume).
            symbol: Symbol name for labeling.
            params: Strategy parameters (uses defaults if not provided).

        Returns:
            BacktestResult with fitness metrics and trade log.
        """
        result = BacktestResult(
            strategy_name=strategy_cls.name,
            symbol=symbol,
            total_bars=len(df),
        )

        try:
            strategy = strategy_cls()
            params = params or strategy.get_default_params()
        except Exception as e:
            result.error = f"Failed to instantiate strategy: {e}"
            return result

        capital = self.initial_capital
        position: BacktestTrade | None = None
        trades: list[BacktestTrade] = []
        equity: list[float] = [capital]

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        for i in range(self.lookback_bars, len(df)):
            bar_close = float(close[i])
            bar_high = float(high[i])
            bar_low = float(low[i])

            # Check open position for SL/TP
            if position and position.status == "open":
                hit_sl = False
                hit_tp = False

                if position.side == "buy":
                    hit_sl = bar_low <= position.stop_loss
                    hit_tp = bar_high >= position.take_profit
                else:
                    hit_sl = bar_high >= position.stop_loss
                    hit_tp = bar_low <= position.take_profit

                if hit_sl or hit_tp:
                    if hit_sl:
                        position.exit_price = position.stop_loss
                    else:
                        position.exit_price = position.take_profit

                    position.exit_bar = i
                    position.status = "closed"

                    if position.side == "buy":
                        position.pnl = (position.exit_price - position.entry_price) * (
                            capital * self.position_size_pct / position.entry_price
                        )
                    else:
                        position.pnl = (position.entry_price - position.exit_price) * (
                            capital * self.position_size_pct / position.entry_price
                        )

                    capital += position.pnl
                    trades.append(position)
                    position = None

            # Get strategy signal (only if no open position)
            if position is None:
                window = df.iloc[max(0, i - 200) : i + 1].copy()
                try:
                    recommendation, confidence = strategy.evaluate(window, params)
                except Exception as e:
                    result.error = f"evaluate() failed at bar {i}: {e}"
                    break

                if recommendation in ("strong_buy", "buy") and confidence > 0.5:
                    sl = bar_close * (1 - self.stop_loss_pct)
                    tp = bar_close * (1 + self.take_profit_pct)
                    position = BacktestTrade(
                        symbol=symbol, side="buy",
                        entry_price=bar_close, entry_bar=i,
                        stop_loss=sl, take_profit=tp,
                    )
                elif recommendation in ("strong_sell", "sell") and confidence > 0.5:
                    sl = bar_close * (1 + self.stop_loss_pct)
                    tp = bar_close * (1 - self.take_profit_pct)
                    position = BacktestTrade(
                        symbol=symbol, side="sell",
                        entry_price=bar_close, entry_bar=i,
                        stop_loss=sl, take_profit=tp,
                    )

            equity.append(capital)

        # Close any remaining position at last price
        if position and position.status == "open":
            position.exit_price = float(close[-1])
            position.exit_bar = len(df) - 1
            position.status = "closed"
            if position.side == "buy":
                position.pnl = (position.exit_price - position.entry_price) * (
                    capital * self.position_size_pct / position.entry_price
                )
            else:
                position.pnl = (position.entry_price - position.exit_price) * (
                    capital * self.position_size_pct / position.entry_price
                )
            capital += position.pnl
            trades.append(position)

        # Compute fitness
        closed_dicts = [{"pnl": t.pnl} for t in trades if t.status == "closed"]
        result.fitness = self.evaluator.evaluate(closed_dicts)
        result.trades = trades
        result.equity_curve = equity

        logger.info("Backtest %s: %s", strategy_cls.name, result.summary())
        return result
