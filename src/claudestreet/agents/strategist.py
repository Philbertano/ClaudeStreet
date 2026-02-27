"""Strategist agent — strategy selection and trade proposals.

Receives analysis, matches it against the active strategy population,
and generates trade proposals. Uses Kelly Criterion for position sizing,
Thompson Sampling for strategy selection, and regime filtering.
"""

from __future__ import annotations

import logging
import random

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import (
    Event, EventType, EventPriority,
    AnalysisPayload, TradeProposalPayload,
)
from claudestreet.models.strategy import Strategy, StrategyGenome

logger = logging.getLogger(__name__)


class StrategistAgent(BaseAgent):
    agent_id = "strategist"
    description = "Strategy selection and trade proposals"

    def __init__(self, memory, config) -> None:
        super().__init__(memory, config)
        self._strategies: list[Strategy] | None = None
        self._current_regime: str | None = None

    def _load_strategies(self) -> list[Strategy]:
        """Load active strategies from DynamoDB (cached per invocation)."""
        if self._strategies is not None:
            return self._strategies

        stored = self.memory.get_active_strategies()
        strategies = []
        for s in stored:
            genome = StrategyGenome.model_validate(s["genome"])
            strategies.append(Strategy(
                id=s["id"],
                name=s["name"],
                strategy_type=s["strategy_type"],
                genome=genome,
                generation=s.get("generation", 0),
                total_trades=s.get("total_trades", 0),
                total_pnl=s.get("total_pnl", 0.0),
                regime_preference=s.get("regime_preference", ""),
                kelly_fraction_mult=s.get("kelly_fraction_mult", 0.5),
                wins=s.get("wins", 0),
                losses=s.get("losses", 0),
            ))

        if not strategies:
            strategies = self._seed_initial()

        self._strategies = strategies
        return strategies

    def _seed_initial(self) -> list[Strategy]:
        """Seed initial population if none exists."""
        from claudestreet.evolution.population import Population

        pop = Population(
            population_size=self.config.get("population_size", 20),
            survivors=self.config.get("survivors_per_generation", 5),
        )
        strategies = pop.seed()
        for s in strategies:
            self.memory.save_strategy({
                "id": s.id,
                "name": s.name,
                "strategy_type": s.strategy_type,
                "genome": s.genome.model_dump(),
                "fitness": s.fitness.model_dump(),
                "generation": s.generation,
                "is_active": True,
                "created_at": s.created_at.isoformat(),
            })
        logger.info("Seeded %d initial strategies", len(strategies))
        return strategies

    def _get_current_regime(self) -> str:
        """Load current regime from DynamoDB (cached per invocation)."""
        if self._current_regime is None:
            self._current_regime = self.memory.get_current_regime()
        return self._current_regime

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.ANALYSIS_COMPLETE:
            return self._propose_trades(event)
        if event.type == EventType.REGIME_CHANGE:
            regime = event.payload.get("regime", "")
            self.memory.set_current_regime(regime)
            self._current_regime = regime
            logger.info("[strategist] Regime updated to: %s", regime)
            return []
        return []

    def _select_strategies_thompson(
        self, strategies: list[Strategy], k: int = 3
    ) -> list[Strategy]:
        """Select top-K strategies using Thompson Sampling.

        For each strategy, sample from Beta(wins+1, losses+1).
        Select the top-K by sampled value.
        """
        if len(strategies) <= k:
            return strategies

        scored = []
        for s in strategies:
            # Beta distribution: higher wins → higher expected sample
            sample = random.betavariate(s.wins + 1, s.losses + 1)
            scored.append((sample, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def _filter_by_regime(self, strategies: list[Strategy]) -> list[Strategy]:
        """Filter strategies by current market regime."""
        current_regime = self._get_current_regime()
        if not current_regime:
            return strategies

        filtered = []
        for s in strategies:
            # Strategies with no preference run in all regimes
            if not s.regime_preference:
                filtered.append(s)
                continue
            # Match regime
            if s.regime_preference == current_regime:
                filtered.append(s)

        # Fallback: if no strategies match, use all
        return filtered if filtered else strategies

    def _kelly_position_size(
        self,
        strategy: Strategy,
        price: float,
        stop_loss: float,
        confidence: float,
        capital: float,
    ) -> int:
        """Compute position size using half-Kelly Criterion.

        kelly_fraction = (win_rate * avg_win/avg_loss - (1-win_rate)) / (avg_win/avg_loss)
        position_size = capital * kelly_fraction * kelly_mult
        Scaled by signal confidence.
        """
        total_trades = strategy.total_trades
        wins = strategy.wins

        # Need minimum trades for Kelly to be meaningful
        if total_trades < 10:
            # Fall back to fixed risk sizing
            risk_pct = self.config.get("max_portfolio_risk_pct", 0.02)
            risk_per_share = abs(price - stop_loss)
            if risk_per_share <= 0:
                return 1
            quantity = max(1, int((capital * risk_pct) / risk_per_share))
            max_shares = int((capital * self.config.get("max_position_pct", 0.15)) / price)
            return min(quantity, max_shares)

        win_rate = wins / total_trades if total_trades > 0 else 0.5

        # Estimate avg win/loss ratio from fitness or default
        avg_win_loss = strategy.fitness.profit_factor if strategy.fitness.profit_factor > 0 else 1.5

        # Kelly formula
        if avg_win_loss <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (win_rate * avg_win_loss - (1 - win_rate)) / avg_win_loss

        # Clamp: never negative, never more than 25%
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))

        # Apply strategy's Kelly multiplier (half-Kelly default)
        kelly_mult = strategy.kelly_fraction_mult
        effective_fraction = kelly_fraction * kelly_mult

        # Scale by confidence: only full size at high confidence
        confidence_scale = max(0.0, (confidence - 0.5) * 2)
        effective_fraction *= max(0.2, confidence_scale)  # minimum 20% of Kelly size

        # Calculate shares
        position_value = capital * effective_fraction
        quantity = max(1, int(position_value / price))

        # Hard cap: max position percentage
        max_shares = int((capital * self.config.get("max_position_pct", 0.15)) / price)
        return min(quantity, max_shares)

    def _propose_trades(self, event: Event) -> list[Event]:
        analysis = AnalysisPayload(**event.payload)
        strategies = self._load_strategies()
        events: list[Event] = []

        # Filter by regime
        regime_filtered = self._filter_by_regime(strategies)

        # Select top strategies via Thompson Sampling
        selected = self._select_strategies_thompson(
            [s for s in regime_filtered if s.is_active],
            k=self.config.get("thompson_k", 3),
        )

        for strategy in selected:
            params = strategy.genome.to_params()
            if analysis.confidence < params.get("confidence_threshold", 0.6):
                continue

            side = None
            if analysis.recommendation in ("strong_buy", "buy"):
                side = "buy"
            elif analysis.recommendation in ("strong_sell", "sell"):
                side = "sell"
            if not side:
                continue

            price = analysis.technical.get("close", 0.0)
            if price <= 0:
                continue

            # ATR-based stop placement when ATR is available
            atr = analysis.technical.get("atr_14", 0.0)
            sl_pct = params.get("stop_loss_pct", 0.05)
            tp_pct = params.get("take_profit_pct", 0.10)

            if atr > 0 and price > 0:
                # ATR-based: stop at 2x ATR, take profit at 3x ATR
                atr_stop = atr * 2.0
                atr_tp = atr * 3.0
                if side == "buy":
                    stop_loss = price - atr_stop
                    take_profit = price + atr_tp
                else:
                    stop_loss = price + atr_stop
                    take_profit = price - atr_tp
            else:
                if side == "buy":
                    stop_loss = price * (1 - sl_pct)
                    take_profit = price * (1 + tp_pct)
                else:
                    stop_loss = price * (1 + sl_pct)
                    take_profit = price * (1 - tp_pct)

            # Kelly Criterion position sizing
            capital = self.config.get("initial_capital", 100000.0)
            quantity = self._kelly_position_size(
                strategy, price, stop_loss, analysis.confidence, capital,
            )

            proposal = TradeProposalPayload(
                symbol=analysis.symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                strategy_id=strategy.id,
                confidence=analysis.confidence,
                rationale=f"{analysis.summary} | {strategy.name} gen{strategy.generation}",
                signal_id=event.payload.get("signal_id", ""),
            )

            events.append(self.emit(
                EventType.TRADE_PROPOSED,
                payload=proposal.model_dump(),
                parent=event,
                priority=EventPriority.HIGH,
            ))

        return events
