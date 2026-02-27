"""Executor agent — trade execution and order management.

Only acts on RiskGuard-approved proposals. Supports paper
trading (default) and live IG Markets API execution.
Tracks order lifecycle via OrderStateMachine.
"""

from __future__ import annotations

import logging
import uuid

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import (
    Event, EventType, EventPriority,
    TradeProposalPayload, TradeExecutedPayload,
)
from claudestreet.models.order_state import OrderState, OrderStateMachine

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    agent_id = "executor"
    description = "Trade execution via broker API"

    def __init__(self, memory, config) -> None:
        super().__init__(memory, config)
        self._osm = OrderStateMachine(memory)

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

        # Track order lifecycle: PENDING -> SUBMITTED -> FILLED (instant for paper)
        self._osm.transition(trade_id, OrderState.PENDING, OrderState.SUBMITTED)
        self._osm.transition(trade_id, OrderState.SUBMITTED, OrderState.ACCEPTED)
        self._osm.transition(trade_id, OrderState.ACCEPTED, OrderState.FILLED, metadata={
            "order_id": order_id,
            "fill_price": proposal.entry_price,
            "paper": True,
        })

        logger.info("[executor] PAPER %s %.2f %s @ %.2f",
                     proposal.side, proposal.quantity, proposal.symbol, proposal.entry_price)

        try:
            self.memory.record_decision_step(
                correlation_id=parent.correlation_id,
                step_key="executor",
                agent=self.agent_id,
                symbol=proposal.symbol,
                strategy_id=proposal.strategy_id,
                reasoning={
                    "order_id": order_id,
                    "fill_price": proposal.entry_price,
                    "expected_price": proposal.entry_price,
                    "slippage_bps": 0.0,
                    "status": "filled",
                    "paper_trading": True,
                },
            )
        except Exception:
            logger.debug("Failed to record executor decision step", exc_info=True)

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
            api_key=self.config.get("ig_api_key", ""),
            username=self.config.get("ig_username", ""),
            password=self.config.get("ig_password", ""),
            acc_number=self.config.get("ig_acc_number", ""),
            acc_type=self.config.get("ig_acc_type", "LIVE"),
            memory=self.memory,
        )

        trade_id = f"trade-{uuid.uuid4().hex[:8]}"

        # Track whether the trade record was created successfully
        trade_recorded = False

        try:
            # Record trade open with PENDING state
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
            trade_recorded = True

            # Transition to SUBMITTED
            self._osm.transition(trade_id, OrderState.PENDING, OrderState.SUBMITTED)

            result = broker.submit_order(
                symbol=proposal.symbol,
                side=proposal.side,
                quantity=proposal.quantity,
                order_type="market",
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                currency_code=self.config.get("ig_default_currency", "GBP"),
            )

            if result.get("status") in ("filled", "new", "accepted"):
                fill_price = result.get("fill_price", proposal.entry_price)

                # Transition through states
                self._osm.transition(trade_id, OrderState.SUBMITTED, OrderState.ACCEPTED)
                self._osm.transition(trade_id, OrderState.ACCEPTED, OrderState.FILLED, metadata={
                    "order_id": result.get("order_id", ""),
                    "fill_price": fill_price,
                    "filled_qty": result.get("filled_qty", 0),
                })

                try:
                    slippage_bps = (
                        round((fill_price - proposal.entry_price) / proposal.entry_price * 10000, 2)
                        if proposal.entry_price > 0 else 0.0
                    )
                    self.memory.record_decision_step(
                        correlation_id=parent.correlation_id,
                        step_key="executor",
                        agent=self.agent_id,
                        symbol=proposal.symbol,
                        strategy_id=proposal.strategy_id,
                        reasoning={
                            "order_id": result.get("order_id", ""),
                            "fill_price": fill_price,
                            "expected_price": proposal.entry_price,
                            "slippage_bps": slippage_bps,
                            "status": result.get("status", ""),
                            "paper_trading": False,
                        },
                    )
                except Exception:
                    logger.debug("Failed to record executor decision step", exc_info=True)

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
                # Order rejected by broker
                self._osm.transition(
                    trade_id, OrderState.SUBMITTED, OrderState.REJECTED,
                    metadata={"reason": result.get("status", "unknown")},
                )
                return [self.emit(
                    EventType.TRADE_FAILED,
                    payload={
                        "symbol": proposal.symbol,
                        "trade_id": trade_id,
                        "reason": result.get("status", "unknown"),
                    },
                    parent=parent,
                )]
        except Exception as e:
            logger.exception("[executor] Live execution failed")
            # Only attempt state transition if the trade record exists in DynamoDB
            if trade_recorded:
                try:
                    self._osm.transition(
                        trade_id, OrderState.SUBMITTED, OrderState.FAILED,
                        metadata={"error": str(e)},
                    )
                except Exception:
                    logger.debug("Could not transition trade %s to FAILED", trade_id)
            return [self.emit(
                EventType.TRADE_FAILED,
                payload={
                    "symbol": proposal.symbol,
                    "trade_id": trade_id,
                    "reason": str(e),
                },
                parent=parent,
            )]
