"""Fitness evaluator — scores strategies on historical trade performance.

Composite fitness balances returns, risk-adjusted returns (Sharpe),
drawdown control, win rate, and profit factor. This multi-objective
approach prevents evolution from optimizing for one metric at
the expense of others.
"""

from __future__ import annotations

import math

from claudestreet.models.strategy import FitnessScore


class FitnessEvaluator:
    """Evaluate strategy fitness from closed trade history."""

    def __init__(
        self,
        weight_return: float = 0.25,
        weight_sharpe: float = 0.30,
        weight_drawdown: float = 0.20,
        weight_win_rate: float = 0.15,
        weight_profit_factor: float = 0.10,
    ) -> None:
        self.weights = {
            "return": weight_return,
            "sharpe": weight_sharpe,
            "drawdown": weight_drawdown,
            "win_rate": weight_win_rate,
            "profit_factor": weight_profit_factor,
        }

    def evaluate(self, closed_trades: list[dict]) -> FitnessScore:
        if not closed_trades:
            return FitnessScore()

        pnls = [t.get("pnl", 0.0) for t in closed_trades]
        total_return = sum(pnls)
        total_trades = len(pnls)

        # Win rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        # Profit factor (gross profit / gross loss)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else
            (10.0 if gross_profit > 0 else 0.0)
        )

        # Sharpe ratio (annualized, assuming daily trades)
        mean_pnl = total_return / total_trades
        if total_trades > 1:
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (total_trades - 1)
            std_pnl = math.sqrt(variance)
            sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown from cumulative PnL curve
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Normalize max_drawdown as percentage of peak (capped at 1.0)
        max_drawdown_pct = max_dd / peak if peak > 0 else 0.0
        max_drawdown_pct = min(max_drawdown_pct, 1.0)

        # Composite score (all components normalized to ~[0, 1] range)
        normalized_return = self._sigmoid(total_return / max(1, total_trades), scale=100)
        normalized_sharpe = self._sigmoid(sharpe, scale=2)
        normalized_drawdown = 1.0 - max_drawdown_pct   # lower drawdown = better
        normalized_pf = min(profit_factor / 3.0, 1.0)   # cap at 3.0

        composite = (
            self.weights["return"] * normalized_return
            + self.weights["sharpe"] * normalized_sharpe
            + self.weights["drawdown"] * normalized_drawdown
            + self.weights["win_rate"] * win_rate
            + self.weights["profit_factor"] * normalized_pf
        )

        return FitnessScore(
            total_return=round(total_return, 2),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_drawdown_pct, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            total_trades=total_trades,
            composite=round(composite, 4),
        )

    @staticmethod
    def _sigmoid(x: float, scale: float = 1.0) -> float:
        """Squash value to [0, 1] range."""
        return 1.0 / (1.0 + math.exp(-x / scale))
