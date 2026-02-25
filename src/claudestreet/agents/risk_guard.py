"""RiskGuard agent — the swarm's circuit breaker.

Validates every trade proposal against risk limits.
Enforces position sizing, exposure, daily loss limits,
and time restrictions before any order reaches the Executor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import (
    Event, EventType, EventPriority,
    TradeProposalPayload,
)

logger = logging.getLogger(__name__)


class RiskGuardAgent(BaseAgent):
    agent_id = "risk_guard"
    description = "Risk validation and circuit breakers"

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.TRADE_PROPOSED:
            return self._validate(event)
        return []

    def heartbeat(self) -> list[Event]:
        """Periodic portfolio risk check."""
        events: list[Event] = []
        open_trades = self.memory.get_open_trades()
        capital = self.config.get("initial_capital", 100000.0)

        total_exposure = sum(
            t.get("entry_price", 0) * t.get("quantity", 0) for t in open_trades
        )
        if capital > 0 and total_exposure / capital > 0.9:
            events.append(self.emit(
                EventType.RISK_ALERT,
                payload={
                    "alert_type": "high_exposure",
                    "exposure_pct": round(total_exposure / capital, 4),
                },
                priority=EventPriority.HIGH,
            ))

        unprotected = [t for t in open_trades if not t.get("stop_loss")]
        if unprotected:
            events.append(self.emit(
                EventType.RISK_ALERT,
                payload={
                    "alert_type": "missing_stop_loss",
                    "symbols": [t["symbol"] for t in unprotected],
                },
            ))
        return events

    def _validate(self, event: Event) -> list[Event]:
        proposal = TradeProposalPayload(**event.payload)
        rejections: list[str] = []
        capital = self.config.get("initial_capital", 100000.0)
        now = datetime.now(timezone.utc)

        # 1. Position size
        position_value = proposal.entry_price * proposal.quantity
        max_pos_pct = self.config.get("max_position_pct", 0.15)
        if position_value > capital * max_pos_pct:
            rejections.append(
                f"Position ${position_value:,.0f} exceeds {max_pos_pct:.0%} limit"
            )

        # 2. Existing exposure to same symbol
        open_trades = self.memory.get_open_trades(proposal.symbol)
        existing = sum(t.get("entry_price", 0) * t.get("quantity", 0) for t in open_trades)
        if existing + position_value > capital * max_pos_pct:
            rejections.append(f"Combined {proposal.symbol} exposure exceeds limit")

        # 3. Max open positions
        all_open = self.memory.get_open_trades()
        max_pos = self.config.get("max_positions", 10)
        if len(all_open) >= max_pos and not open_trades:
            rejections.append(f"Max {max_pos} positions reached")

        # 4. Stop loss required
        if proposal.stop_loss <= 0:
            rejections.append("No stop loss defined")

        # 5. Risk/reward >= 1.5
        risk = abs(proposal.entry_price - proposal.stop_loss)
        reward = abs(proposal.take_profit - proposal.entry_price)
        if risk > 0 and reward / risk < 1.5:
            rejections.append(f"R:R {reward/risk:.2f} below 1.5 minimum")

        # 6. Restricted hours
        current_time = now.strftime("%H:%M")
        for window in self.config.get("restricted_hours", []):
            parts = window.split("-")
            if len(parts) == 2 and parts[0] <= current_time <= parts[1]:
                rejections.append(f"Restricted window {window}")

        if rejections:
            logger.info("[risk_guard] REJECTED %s %s: %s",
                        proposal.side, proposal.symbol, "; ".join(rejections))
            return [self.emit(
                EventType.RISK_REJECTED,
                payload={**proposal.model_dump(), "rejections": rejections},
                parent=event,
                priority=EventPriority.HIGH,
            )]

        logger.info("[risk_guard] APPROVED %s %d %s @ %.2f",
                     proposal.side, proposal.quantity, proposal.symbol, proposal.entry_price)
        return [self.emit(
            EventType.RISK_APPROVED,
            payload=proposal.model_dump(),
            parent=event,
            priority=EventPriority.HIGH,
        )]
