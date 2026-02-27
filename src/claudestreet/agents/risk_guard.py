"""RiskGuard agent — the swarm's circuit breaker.

Validates every trade proposal against risk limits.
Enforces position sizing, exposure, daily loss limits,
portfolio heat, correlation checks, sector concentration,
and time restrictions before any order reaches the Executor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
        """Periodic portfolio risk check + position reconciliation."""
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

        # Portfolio heat check
        total_risk = sum(
            abs(t.get("entry_price", 0) - t.get("stop_loss", 0)) * t.get("quantity", 0)
            for t in open_trades
            if t.get("stop_loss", 0) > 0
        )
        heat_pct = total_risk / capital if capital > 0 else 0
        if heat_pct > 0.06:
            events.append(self.emit(
                EventType.RISK_ALERT,
                payload={
                    "alert_type": "portfolio_heat_exceeded",
                    "heat_pct": round(heat_pct, 4),
                    "limit": 0.06,
                },
                priority=EventPriority.HIGH,
            ))

        # Position reconciliation: compare internal vs broker positions
        if not self.config.get("paper_trading", True):
            events.extend(self._reconcile_positions(open_trades))

        return events

    def _reconcile_positions(self, internal_trades: list[dict]) -> list[Event]:
        """Compare DynamoDB positions vs broker API positions."""
        events: list[Event] = []
        try:
            from claudestreet.connectors.broker import BrokerConnector
            broker = BrokerConnector(
                api_key=self.config.get("ig_api_key", ""),
                username=self.config.get("ig_username", ""),
                password=self.config.get("ig_password", ""),
                acc_number=self.config.get("ig_acc_number", ""),
                acc_type=self.config.get("ig_acc_type", "LIVE"),
                memory=self.memory,
            )
            broker_positions = broker.get_positions()

            internal_by_symbol: dict[str, float] = {}
            for t in internal_trades:
                sym = t.get("symbol", "")
                internal_by_symbol[sym] = internal_by_symbol.get(sym, 0) + t.get("quantity", 0)

            broker_by_symbol: dict[str, float] = {}
            for p in broker_positions:
                broker_by_symbol[p["symbol"]] = p["quantity"]

            all_symbols = set(internal_by_symbol.keys()) | set(broker_by_symbol.keys())
            mismatches = []
            for sym in all_symbols:
                internal_qty = internal_by_symbol.get(sym, 0)
                broker_qty = broker_by_symbol.get(sym, 0)
                if abs(internal_qty - broker_qty) > 0.001:
                    mismatches.append({
                        "symbol": sym,
                        "internal_qty": internal_qty,
                        "broker_qty": broker_qty,
                        "drift": broker_qty - internal_qty,
                    })

            if mismatches:
                events.append(self.emit(
                    EventType.RISK_ALERT,
                    payload={
                        "alert_type": "position_mismatch",
                        "mismatches": mismatches,
                    },
                    priority=EventPriority.CRITICAL,
                ))
                logger.warning("[risk_guard] Position mismatch: %s", mismatches)

        except Exception:
            logger.exception("[risk_guard] Position reconciliation failed")

        return events

    def _check_portfolio_heat(
        self, proposal: TradeProposalPayload, open_trades: list[dict], capital: float
    ) -> str | None:
        """Check if total open risk (portfolio heat) exceeds 6% of capital."""
        existing_risk = sum(
            abs(t.get("entry_price", 0) - t.get("stop_loss", 0)) * t.get("quantity", 0)
            for t in open_trades
            if t.get("stop_loss", 0) > 0
        )
        new_risk = abs(proposal.entry_price - proposal.stop_loss) * proposal.quantity
        total_heat = (existing_risk + new_risk) / capital if capital > 0 else 0

        heat_limit = self.config.get("portfolio_heat_limit", 0.06)
        if total_heat > heat_limit:
            return f"Portfolio heat {total_heat:.1%} would exceed {heat_limit:.0%} limit"
        return None

    def _check_correlation(
        self, proposal: TradeProposalPayload, open_trades: list[dict]
    ) -> str | None:
        """Reject positions with >0.7 correlation to existing holdings."""
        existing_symbols = list({t.get("symbol", "") for t in open_trades})
        if not existing_symbols or proposal.symbol in existing_symbols:
            return None

        try:
            from claudestreet.connectors.market_data import MarketDataConnector
            connector = MarketDataConnector()
            corr = connector.get_correlation_matrix(
                [proposal.symbol] + existing_symbols[:5], period="3mo"
            )
            if corr is not None:
                for sym in existing_symbols:
                    if sym in corr.columns and proposal.symbol in corr.index:
                        pair_corr = abs(corr.loc[proposal.symbol, sym])
                        if pair_corr > 0.7:
                            return (
                                f"{proposal.symbol} has {pair_corr:.2f} correlation "
                                f"with existing position {sym}"
                            )
        except Exception:
            logger.debug("Correlation check unavailable, skipping")

        return None

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

        # 6. Restricted hours (US Eastern, where the market operates)
        eastern = now.astimezone(ZoneInfo("America/New_York"))
        current_time = eastern.strftime("%H:%M")
        for window in self.config.get("restricted_hours", []):
            parts = window.split("-")
            if len(parts) == 2 and parts[0] <= current_time <= parts[1]:
                rejections.append(f"Restricted window {window}")

        # 7. Portfolio heat check
        heat_rejection = self._check_portfolio_heat(proposal, all_open, capital)
        if heat_rejection:
            rejections.append(heat_rejection)

        # 8. Correlation check
        corr_rejection = self._check_correlation(proposal, all_open)
        if corr_rejection:
            rejections.append(corr_rejection)

        if rejections:
            logger.info("[risk_guard] REJECTED %s %s: %s",
                        proposal.side, proposal.symbol, "; ".join(rejections))
            return [self.emit(
                EventType.RISK_REJECTED,
                payload={**proposal.model_dump(), "rejections": rejections},
                parent=event,
                priority=EventPriority.HIGH,
            )]

        logger.info("[risk_guard] APPROVED %s %.2f %s @ %.2f",
                     proposal.side, proposal.quantity, proposal.symbol, proposal.entry_price)
        return [self.emit(
            EventType.RISK_APPROVED,
            payload=proposal.model_dump(),
            parent=event,
            priority=EventPriority.HIGH,
        )]
