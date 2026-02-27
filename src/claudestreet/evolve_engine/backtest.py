"""Backtesting engine — evaluates strategies against historical data.

Walks through OHLCV bars, calls the strategy's evaluate() on each,
simulates trades with stop-loss/take-profit, slippage, commissions,
gap risk, and computes fitness.

Supports walk-forward validation: train/validate/test splits.
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
    commission: float = 0.0
    slippage: float = 0.0
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
    split: str = "full"  # "train", "validate", "test", "full"

    def summary(self) -> str:
        if self.error:
            return f"FAILED: {self.error}"
        f = self.fitness
        return (
            f"{self.strategy_name} on {self.symbol} [{self.split}]: "
            f"return=${f.total_return:,.2f} sharpe={f.sharpe_ratio:.2f} "
            f"dd={f.max_drawdown:.1%} wr={f.win_rate:.0%} "
            f"trades={f.total_trades} pf={f.profit_factor:.2f} "
            f"composite={f.composite:.4f}"
        )


class BacktestEngine:
    """Walk-forward backtesting engine with realistic cost modeling."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        lookback_bars: int = 50,
        slippage_bps: float = 5.0,
        commission_per_trade: float = 1.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.lookback_bars = lookback_bars
        self.slippage_bps = slippage_bps  # basis points (5 bps = 0.05%)
        self.commission_per_trade = commission_per_trade
        self.evaluator = FitnessEvaluator()

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage model: fill worse than expected."""
        spread_pct = self.slippage_bps / 10000
        if side == "buy":
            return price * (1 + spread_pct)
        return price * (1 - spread_pct)

    def _apply_gap_risk(
        self, bar_open: float, stop_loss: float, side: str
    ) -> float | None:
        """Check if bar opens beyond stop — fill at open (gap risk)."""
        if side == "buy" and bar_open <= stop_loss:
            return bar_open
        if side == "sell" and bar_open >= stop_loss:
            return bar_open
        return None

    def run(
        self,
        strategy_cls: Type[CustomStrategy],
        df: pd.DataFrame,
        symbol: str = "TEST",
        params: dict[str, float] | None = None,
    ) -> BacktestResult:
        """Run a backtest on historical OHLCV data with realistic cost modeling."""
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
        open_ = df["Open"].values if "Open" in df.columns else close

        for i in range(self.lookback_bars, len(df)):
            bar_close = float(close[i])
            bar_high = float(high[i])
            bar_low = float(low[i])
            bar_open = float(open_[i])

            # Check open position for SL/TP
            if position and position.status == "open":
                # Gap risk: check if bar opens beyond stop
                gap_fill = self._apply_gap_risk(bar_open, position.stop_loss, position.side)

                hit_sl = False
                hit_tp = False

                if gap_fill is not None:
                    # Gapped through stop — fill at open price
                    position.exit_price = gap_fill
                    hit_sl = True
                else:
                    if position.side == "buy":
                        hit_sl = bar_low <= position.stop_loss
                        hit_tp = bar_high >= position.take_profit
                    else:
                        hit_sl = bar_high >= position.stop_loss
                        hit_tp = bar_low <= position.take_profit

                if hit_sl or hit_tp:
                    if gap_fill is None:
                        if hit_sl:
                            position.exit_price = position.stop_loss
                        else:
                            position.exit_price = position.take_profit

                    # Apply slippage on exit
                    exit_side = "sell" if position.side == "buy" else "buy"
                    position.exit_price = self._apply_slippage(
                        position.exit_price, exit_side
                    )
                    position.exit_bar = i
                    position.status = "closed"

                    # Dynamic equity: use current capital for position sizing
                    qty = capital * self.position_size_pct / position.entry_price
                    if position.side == "buy":
                        position.pnl = (position.exit_price - position.entry_price) * qty
                    else:
                        position.pnl = (position.entry_price - position.exit_price) * qty

                    # Deduct commission
                    position.commission = self.commission_per_trade
                    position.pnl -= position.commission

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
                    # Apply slippage on entry
                    entry_price = self._apply_slippage(bar_close, "buy")
                    sl = entry_price * (1 - self.stop_loss_pct)
                    tp = entry_price * (1 + self.take_profit_pct)
                    position = BacktestTrade(
                        symbol=symbol, side="buy",
                        entry_price=entry_price, entry_bar=i,
                        stop_loss=sl, take_profit=tp,
                        slippage=entry_price - bar_close,
                    )
                elif recommendation in ("strong_sell", "sell") and confidence > 0.5:
                    entry_price = self._apply_slippage(bar_close, "sell")
                    sl = entry_price * (1 + self.stop_loss_pct)
                    tp = entry_price * (1 - self.take_profit_pct)
                    position = BacktestTrade(
                        symbol=symbol, side="sell",
                        entry_price=entry_price, entry_bar=i,
                        stop_loss=sl, take_profit=tp,
                        slippage=bar_close - entry_price,
                    )

            equity.append(capital)

        # Close any remaining position at last price
        if position and position.status == "open":
            exit_side = "sell" if position.side == "buy" else "buy"
            position.exit_price = self._apply_slippage(float(close[-1]), exit_side)
            position.exit_bar = len(df) - 1
            position.status = "closed"
            qty = capital * self.position_size_pct / position.entry_price
            if position.side == "buy":
                position.pnl = (position.exit_price - position.entry_price) * qty
            else:
                position.pnl = (position.entry_price - position.exit_price) * qty
            position.commission = self.commission_per_trade
            position.pnl -= position.commission
            capital += position.pnl
            trades.append(position)

        # Compute fitness
        closed_dicts = [{"pnl": t.pnl} for t in trades if t.status == "closed"]
        result.fitness = self.evaluator.evaluate(closed_dicts)
        result.trades = trades
        result.equity_curve = equity

        logger.info("Backtest %s: %s", strategy_cls.name, result.summary())
        return result

    def run_walk_forward(
        self,
        strategy_cls: Type[CustomStrategy],
        df: pd.DataFrame,
        symbol: str = "TEST",
        params: dict[str, float] | None = None,
        train_pct: float = 0.70,
        validate_pct: float = 0.15,
    ) -> dict[str, BacktestResult]:
        """Run walk-forward validation: train on first 70%, validate 15%, test 15%.

        Returns dict of split_name → BacktestResult.
        """
        n = len(df)
        train_end = int(n * train_pct)
        validate_end = int(n * (train_pct + validate_pct))

        train_df = df.iloc[:train_end]
        validate_df = df.iloc[train_end:validate_end]
        test_df = df.iloc[validate_end:]

        results = {}

        for split_name, split_df in [
            ("train", train_df),
            ("validate", validate_df),
            ("test", test_df),
        ]:
            if len(split_df) < self.lookback_bars + 10:
                results[split_name] = BacktestResult(
                    strategy_name=strategy_cls.name,
                    symbol=symbol,
                    total_bars=len(split_df),
                    error=f"Insufficient bars for {split_name} split ({len(split_df)})",
                    split=split_name,
                )
                continue

            result = self.run(strategy_cls, split_df, symbol=symbol, params=params)
            result.split = split_name
            results[split_name] = result

        return results
