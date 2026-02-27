"""Chronicler agent — the swarm's memory keeper.

Records every trade, tracks performance, generates snapshots,
computes signal-to-outcome attribution, maintains online fitness
with exponential decay, and auto-retires/promotes strategies.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from claudestreet.agents.base import BaseAgent
from claudestreet.models.events import Event, EventType, EventPriority
from claudestreet.models.trade import PortfolioSnapshot

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 50  # last N trades for fitness calculation
_DECAY_HALFLIFE_DAYS = 30  # exponential decay half-life
_AUTO_RETIRE_THRESHOLD = 0.30
_AUTO_PROMOTE_THRESHOLD = 0.70


class ChroniclerAgent(BaseAgent):
    agent_id = "chronicler"
    description = "Performance tracking, attribution, and portfolio snapshots"

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
            return self._record_close(event)
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

    def _record_close(self, event: Event) -> list[Event]:
        """Record trade close, compute attribution, and update strategy fitness."""
        trade_id = event.payload.get("trade_id", "")
        exit_price = event.payload.get("exit_price", 0.0)
        pnl = event.payload.get("pnl", 0.0)
        strategy_id = event.payload.get("strategy_id", "")
        signal_id = event.payload.get("signal_id", "")

        if trade_id:
            self.memory.record_trade_close(trade_id, exit_price, pnl)

        logger.info("[chronicler] Closed: pnl=%.2f strategy=%s", pnl, strategy_id)

        events: list[Event] = []

        # Signal-to-outcome attribution
        if strategy_id and event.correlation_id:
            self._record_attribution(event, pnl, strategy_id, signal_id)

        # Online fitness update
        if strategy_id:
            events.extend(self._update_strategy_fitness(strategy_id, pnl))

        return events

    def _record_attribution(
        self,
        event: Event,
        pnl: float,
        strategy_id: str,
        signal_id: str,
    ) -> None:
        """Store signal-to-outcome attribution record."""
        confidence = event.payload.get("confidence", 0.5)
        position_size = event.payload.get("quantity", 1) * event.payload.get("entry_price", 0)

        # Normalize PnL by conviction
        signal_quality = pnl / (confidence * position_size) if confidence * position_size > 0 else 0

        attribution = {
            "correlation_id": event.correlation_id,
            "strategy_id": strategy_id,
            "signal_id": signal_id,
            "pnl": pnl,
            "confidence": confidence,
            "position_size": position_size,
            "signal_quality": round(signal_quality, 6),
            "symbol": event.payload.get("symbol", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Store in events table with a special event type
        self.memory.log_event(
            event_id=f"attr-{event.correlation_id}",
            event_type="attribution.signal_outcome",
            source=self.agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=attribution,
            correlation_id=event.correlation_id,
        )

    def _update_strategy_fitness(
        self, strategy_id: str, pnl: float
    ) -> list[Event]:
        """Recalculate strategy fitness from recent trades with exponential decay."""
        events: list[Event] = []

        trades = self.memory.get_strategy_trades(strategy_id)
        closed = [t for t in trades if t.get("status") == "closed"]

        if not closed:
            return events

        # Take last N trades
        recent = closed[-_ROLLING_WINDOW:]

        # Apply exponential decay: recent trades weighted more
        now = datetime.now(timezone.utc)
        weighted_pnls = []
        wins = 0
        losses = 0

        for t in recent:
            t_pnl = t.get("pnl", 0)
            closed_at = t.get("closed_at", "")

            # Compute decay weight
            weight = 1.0
            if closed_at:
                try:
                    trade_time = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
                    days_ago = (now - trade_time).days
                    weight = math.exp(-0.693 * days_ago / _DECAY_HALFLIFE_DAYS)  # ln(2)/halflife
                except (ValueError, TypeError):
                    weight = 0.5

            weighted_pnls.append(t_pnl * weight)

            if t_pnl > 0:
                wins += 1
            elif t_pnl < 0:
                losses += 1

        # Compute fitness metrics
        total_trades = len(recent)
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_pnl = sum(weighted_pnls)
        gross_profit = sum(p for p in weighted_pnls if p > 0)
        gross_loss = abs(sum(p for p in weighted_pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 3.0

        # Simple composite fitness
        composite = (
            0.25 * min(total_pnl / 10000, 1.0)  # normalized return
            + 0.30 * min(profit_factor / 3.0, 1.0)  # profit factor
            + 0.15 * win_rate
            + 0.30 * (1.0 - min(gross_loss / 10000, 1.0))  # drawdown proxy
        )
        composite = max(0.0, min(1.0, composite))

        # Update strategy in DynamoDB
        fitness_data = {
            "total_return": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(min(profit_factor, 10.0), 4),
            "total_trades": total_trades,
            "composite": round(composite, 4),
        }

        try:
            strategies = self.memory.get_active_strategies()
            for s in strategies:
                if s["id"] == strategy_id:
                    self.memory.save_strategy({
                        "id": strategy_id,
                        "name": s.get("name", "unnamed"),
                        "strategy_type": s.get("strategy_type", "custom"),
                        "genome": s.get("genome", {}),
                        "fitness": fitness_data,
                        "generation": s.get("generation", 0),
                        "is_active": True,
                        "total_trades": total_trades,
                        "total_pnl": round(sum(t.get("pnl", 0) for t in closed), 2),
                        "wins": wins,
                        "losses": losses,
                        "version": s.get("version", 0) + 1,
                    })
                    break
        except Exception:
            logger.exception("Failed to update strategy fitness for %s", strategy_id)

        # Auto-retire if fitness is too low
        if composite < _AUTO_RETIRE_THRESHOLD and total_trades >= 20:
            self.memory.retire_strategy(strategy_id)
            events.append(self.emit(
                EventType.STRATEGY_RETIRED,
                payload={
                    "strategy_id": strategy_id,
                    "reason": f"Auto-retired: rolling fitness {composite:.4f} < {_AUTO_RETIRE_THRESHOLD}",
                    "composite": composite,
                    "total_trades": total_trades,
                },
                priority=EventPriority.HIGH,
            ))
            logger.info("[chronicler] Auto-retired strategy %s (fitness=%.4f)", strategy_id, composite)

        # Auto-promote: log if doing well (sizing weight increase handled by strategist)
        elif composite > _AUTO_PROMOTE_THRESHOLD and total_trades >= 10:
            logger.info("[chronicler] Strategy %s performing well (fitness=%.4f)", strategy_id, composite)

        return events
