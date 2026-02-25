"""Event definitions for the ClaudeStreet event bus.

Every inter-agent communication flows through typed events.
Events are immutable, timestamped, and carry a correlation_id
so the full causal chain of any trade can be reconstructed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # --- Market data ---
    MARKET_TICK = "market.tick"
    MARKET_DAILY = "market.daily"
    VOLUME_SPIKE = "market.volume_spike"
    PRICE_ANOMALY = "market.price_anomaly"

    # --- Signals ---
    SIGNAL_DETECTED = "signal.detected"
    SIGNAL_EXPIRED = "signal.expired"

    # --- Analysis ---
    ANALYSIS_COMPLETE = "analysis.complete"
    SENTIMENT_UPDATE = "analysis.sentiment"

    # --- Strategy ---
    TRADE_PROPOSED = "strategy.trade_proposed"
    STRATEGY_EVOLVED = "strategy.evolved"
    STRATEGY_RETIRED = "strategy.retired"

    # --- Risk ---
    RISK_APPROVED = "risk.approved"
    RISK_REJECTED = "risk.rejected"
    RISK_ALERT = "risk.alert"
    CIRCUIT_BREAKER = "risk.circuit_breaker"

    # --- Execution ---
    TRADE_EXECUTED = "execution.trade_executed"
    TRADE_FAILED = "execution.trade_failed"
    ORDER_FILLED = "execution.order_filled"
    POSITION_CLOSED = "execution.position_closed"

    # --- Performance ---
    PERFORMANCE_UPDATE = "performance.update"
    PORTFOLIO_SNAPSHOT = "performance.snapshot"

    # --- System ---
    AGENT_STARTED = "system.agent_started"
    AGENT_STOPPED = "system.agent_stopped"
    HEARTBEAT_TICK = "system.heartbeat"
    SWARM_SHUTDOWN = "system.shutdown"


class EventPriority(int, Enum):
    CRITICAL = 0   # circuit breakers, stop losses
    HIGH = 1       # trade execution, risk alerts
    NORMAL = 2     # analysis, signals
    LOW = 3        # performance updates, logging


class Event(BaseModel):
    """Immutable event flowing through the event bus."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    type: EventType
    source: str                          # agent_id that produced this event
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: str | None = None    # links events in a causal chain
    parent_id: str | None = None         # the event that triggered this one

    def spawn(
        self,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
        priority: EventPriority | None = None,
    ) -> Event:
        """Create a child event inheriting this event's correlation chain."""
        return Event(
            type=event_type,
            source=source,
            payload=payload or {},
            priority=priority or self.priority,
            correlation_id=self.correlation_id or self.id,
            parent_id=self.id,
        )


class MarketTickPayload(BaseModel):
    symbol: str
    price: float
    volume: int
    bid: float | None = None
    ask: float | None = None
    change_pct: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SignalPayload(BaseModel):
    symbol: str
    signal_type: str          # "bullish_crossover", "oversold_bounce", etc.
    strength: float           # 0.0 to 1.0
    indicators: dict[str, float] = Field(default_factory=dict)
    timeframe: str = "1d"


class TradeProposalPayload(BaseModel):
    symbol: str
    side: str                 # "buy" or "sell"
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_id: str
    confidence: float         # 0.0 to 1.0
    rationale: str = ""


class TradeExecutedPayload(BaseModel):
    symbol: str
    side: str
    quantity: int
    fill_price: float
    order_id: str
    strategy_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalysisPayload(BaseModel):
    symbol: str
    technical: dict[str, float] = Field(default_factory=dict)
    sentiment_score: float | None = None
    recommendation: str = "hold"   # "strong_buy", "buy", "hold", "sell", "strong_sell"
    confidence: float = 0.0
    summary: str = ""
