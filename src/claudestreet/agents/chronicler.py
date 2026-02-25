"""Chronicler agent — the swarm's memory keeper.

Records every trade, tracks performance, generates snapshots,
and feeds data to the evolution engine.
"""

from __future__ import annotations

import logging

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import Event, EventType
from claudestreet.models.trade import PortfolioSnapshot

logger = logging.getLogger(__name__)


class ChroniclerAgent(BaseAgent):
    agent_id = "chronicler"
    description = "Performance tracking and portfolio snapshots"

    def process(self, event: Event) -> list[Event]:
        # Log every event to DynamoDB audit trail
        self.memory.log_event(
            event_id=event.id,
            event_type=event.type.value,
            source=event.source,
            timestamp=event.timestamp.isoformat(),
            payload=event.payload,
            correlation_id=event.correlation_id,
        )

        if event.type == EventType.TRADE_EXECUTED:
            self._record_execution(event)
        elif event.type == EventType.POSITION_CLOSED:
            self._record_close(event)
        elif event.type == EventType.STRATEGY_EVOLVED:
            logger.info("[chronicler] Evolution gen %d", event.payload.get("generation", 0))

        return []

    def heartbeat(self) -> list[Event]:
        """Generate portfolio snapshot."""
        open_trades = self.memory.get_open_trades()
        capital = self.config.get("initial_capital", 100000.0)

        positions_value = sum(
            t.get("entry_price", 0) * t.get("quantity", 0) for t in open_trades
        )

        # Estimate cash from capital minus invested
        invested = positions_value
        cash = capital - invested
        total_value = cash + positions_value

        self.memory.record_snapshot(
            cash=round(cash, 2),
            positions_value=round(positions_value, 2),
            total_value=round(total_value, 2),
            daily_pnl=0.0,
            open_positions=len(open_trades),
        )

        logger.info("[chronicler] Snapshot: value=%.2f positions=%d",
                     total_value, len(open_trades))

        return [self.emit(
            EventType.PORTFOLIO_SNAPSHOT,
            payload=PortfolioSnapshot(
                cash=round(cash, 2),
                positions_value=round(positions_value, 2),
                total_value=round(total_value, 2),
                open_positions=len(open_trades),
            ).model_dump(mode="json"),
        )]

    def _record_execution(self, event: Event) -> None:
        payload = event.payload
        logger.info("[chronicler] Trade: %s %d %s @ %.2f",
                     payload.get("side"), payload.get("quantity"),
                     payload.get("symbol"), payload.get("fill_price"))

    def _record_close(self, event: Event) -> None:
        trade_id = event.payload.get("trade_id", "")
        exit_price = event.payload.get("exit_price", 0.0)
        pnl = event.payload.get("pnl", 0.0)
        if trade_id:
            self.memory.record_trade_close(trade_id, exit_price, pnl)
        logger.info("[chronicler] Closed: pnl=%.2f", pnl)
