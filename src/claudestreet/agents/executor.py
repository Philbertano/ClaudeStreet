"""Executor agent — trade execution and order management.

Only acts on RiskGuard-approved proposals. Supports paper
trading (default) and live Alpaca API execution.
"""

from __future__ import annotations

import logging
import uuid

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import (
    Event, EventType, EventPriority,
    TradeProposalPayload, TradeExecutedPayload,
)

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    agent_id = "executor"
    description = "Trade execution via broker API"

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.RISK_APPROVED:
            return self._execute(event)
        return []

    def _execute(self, event: Event) -> list[Event]:
        proposal = TradeProposalPayload(**event.payload)
        paper = self.config.get("paper_trading", True)

        if paper:
            return self._paper_execute(proposal, event)
        return self._live_execute(proposal, event)

    def _paper_execute(self, proposal: TradeProposalPayload, parent: Event) -> list[Event]:
        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        trade_id = f"trade-{uuid.uuid4().hex[:8]}"

        self.memory.record_trade_open(
            trade_id=trade_id,
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=proposal.quantity,
            entry_price=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            strategy_id=proposal.strategy_id,
        )

        logger.info("[executor] PAPER %s %d %s @ %.2f",
                     proposal.side, proposal.quantity, proposal.symbol, proposal.entry_price)

        return [self.emit(
            EventType.TRADE_EXECUTED,
            payload={
                **TradeExecutedPayload(
                    symbol=proposal.symbol,
                    side=proposal.side,
                    quantity=proposal.quantity,
                    fill_price=proposal.entry_price,
                    order_id=order_id,
                    strategy_id=proposal.strategy_id,
                ).model_dump(mode="json"),
                "trade_id": trade_id,
                "paper": True,
            },
            parent=parent,
            priority=EventPriority.HIGH,
        )]

    def _live_execute(self, proposal: TradeProposalPayload, parent: Event) -> list[Event]:
        from claudestreet.connectors.broker import BrokerConnector

        broker = BrokerConnector(
            api_key=self.config.get("alpaca_api_key", ""),
            secret_key=self.config.get("alpaca_secret_key", ""),
            base_url=self.config.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
        )

        try:
            result = broker.submit_order(
                symbol=proposal.symbol,
                side=proposal.side,
                quantity=proposal.quantity,
                limit_price=proposal.entry_price,
            )

            if result.get("status") in ("filled", "new", "accepted"):
                trade_id = f"trade-{uuid.uuid4().hex[:8]}"
                fill_price = result.get("fill_price", proposal.entry_price)

                self.memory.record_trade_open(
                    trade_id=trade_id,
                    symbol=proposal.symbol,
                    side=proposal.side,
                    quantity=proposal.quantity,
                    entry_price=fill_price,
                    stop_loss=proposal.stop_loss,
                    take_profit=proposal.take_profit,
                    strategy_id=proposal.strategy_id,
                )

                return [self.emit(
                    EventType.TRADE_EXECUTED,
                    payload={
                        **TradeExecutedPayload(
                            symbol=proposal.symbol,
                            side=proposal.side,
                            quantity=proposal.quantity,
                            fill_price=fill_price,
                            order_id=result.get("order_id", ""),
                            strategy_id=proposal.strategy_id,
                        ).model_dump(mode="json"),
                        "trade_id": trade_id,
                        "paper": False,
                    },
                    parent=parent,
                    priority=EventPriority.HIGH,
                )]
            else:
                return [self.emit(
                    EventType.TRADE_FAILED,
                    payload={"symbol": proposal.symbol, "reason": result.get("status", "unknown")},
                    parent=parent,
                )]
        except Exception as e:
            logger.exception("[executor] Live execution failed")
            return [self.emit(
                EventType.TRADE_FAILED,
                payload={"symbol": proposal.symbol, "reason": str(e)},
                parent=parent,
            )]
