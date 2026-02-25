"""Strategist agent — strategy selection and trade proposals.

Receives analysis, matches it against the active strategy population,
and generates trade proposals. The genetic evolution cycle runs
separately in the Fargate evolution engine.
"""

from __future__ import annotations

import logging

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

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.ANALYSIS_COMPLETE:
            return self._propose_trades(event)
        return []

    def _propose_trades(self, event: Event) -> list[Event]:
        analysis = AnalysisPayload(**event.payload)
        strategies = self._load_strategies()
        events: list[Event] = []

        for strategy in strategies:
            if not strategy.is_active:
                continue

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

            sl_pct = params.get("stop_loss_pct", 0.05)
            tp_pct = params.get("take_profit_pct", 0.10)
            if side == "buy":
                stop_loss = price * (1 - sl_pct)
                take_profit = price * (1 + tp_pct)
            else:
                stop_loss = price * (1 + sl_pct)
                take_profit = price * (1 - tp_pct)

            # Position sizing from risk
            capital = self.config.get("initial_capital", 100000.0)
            risk_pct = self.config.get("max_portfolio_risk_pct", 0.02)
            risk_per_share = abs(price - stop_loss)
            quantity = max(1, int((capital * risk_pct) / risk_per_share)) if risk_per_share > 0 else 1
            max_shares = int((capital * self.config.get("max_position_pct", 0.15)) / price)
            quantity = min(quantity, max_shares)

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
            )

            events.append(self.emit(
                EventType.TRADE_PROPOSED,
                payload=proposal.model_dump(),
                parent=event,
                priority=EventPriority.HIGH,
            ))

        return events
