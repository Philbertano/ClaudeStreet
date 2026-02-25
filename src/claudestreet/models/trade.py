"""Trade and position models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class Order(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    side: OrderSide
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    filled_at: datetime | None = None
    strategy_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Position(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    side: OrderSide
    quantity: int
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_id: str = ""
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.status == PositionStatus.CLOSED:
            return 0.0
        direction = 1 if self.side == OrderSide.BUY else -1
        return direction * (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        direction = 1 if self.side == OrderSide.BUY else -1
        return direction * (self.current_price - self.entry_price) / self.entry_price

    def should_stop_loss(self) -> bool:
        if self.stop_loss <= 0:
            return False
        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss
        return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        if self.take_profit <= 0:
            return False
        if self.side == OrderSide.BUY:
            return self.current_price >= self.take_profit
        return self.current_price <= self.take_profit


class PortfolioSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cash: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    open_positions: int = 0
    trades_today: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
