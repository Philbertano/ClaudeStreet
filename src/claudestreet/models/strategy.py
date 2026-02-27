"""Strategy models — the DNA of trading behavior."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class StrategyGene(BaseModel):
    """A single tunable parameter in a strategy's genome."""

    name: str
    value: float
    min_val: float
    max_val: float
    step: float = 0.01       # minimum mutation increment

    def clamp(self) -> None:
        self.value = max(self.min_val, min(self.max_val, self.value))


class StrategyGenome(BaseModel):
    """Complete genetic encoding of a trading strategy."""

    genes: dict[str, StrategyGene] = Field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.genes[name].value

    def set(self, name: str, value: float) -> None:
        self.genes[name].value = value
        self.genes[name].clamp()

    def to_params(self) -> dict[str, float]:
        return {name: gene.value for name, gene in self.genes.items()}

    @classmethod
    def default_momentum(cls) -> StrategyGenome:
        """Default momentum strategy genome."""
        return cls(genes={
            "rsi_period": StrategyGene(name="rsi_period", value=14, min_val=5, max_val=50, step=1),
            "rsi_oversold": StrategyGene(name="rsi_oversold", value=30, min_val=15, max_val=45, step=1),
            "rsi_overbought": StrategyGene(name="rsi_overbought", value=70, min_val=55, max_val=85, step=1),
            "macd_fast": StrategyGene(name="macd_fast", value=12, min_val=5, max_val=20, step=1),
            "macd_slow": StrategyGene(name="macd_slow", value=26, min_val=15, max_val=50, step=1),
            "macd_signal": StrategyGene(name="macd_signal", value=9, min_val=5, max_val=20, step=1),
            "bb_period": StrategyGene(name="bb_period", value=20, min_val=10, max_val=50, step=1),
            "bb_std": StrategyGene(name="bb_std", value=2.0, min_val=1.0, max_val=3.5, step=0.1),
            "ema_fast": StrategyGene(name="ema_fast", value=9, min_val=3, max_val=20, step=1),
            "ema_slow": StrategyGene(name="ema_slow", value=21, min_val=15, max_val=60, step=1),
            "volume_spike_mult": StrategyGene(name="volume_spike_mult", value=2.5, min_val=1.5, max_val=5.0, step=0.1),
            "stop_loss_pct": StrategyGene(name="stop_loss_pct", value=0.05, min_val=0.01, max_val=0.15, step=0.005),
            "take_profit_pct": StrategyGene(name="take_profit_pct", value=0.10, min_val=0.02, max_val=0.30, step=0.005),
            "confidence_threshold": StrategyGene(name="confidence_threshold", value=0.6, min_val=0.3, max_val=0.95, step=0.05),
        })

    @classmethod
    def default_mean_reversion(cls) -> StrategyGenome:
        """Default mean-reversion strategy genome."""
        return cls(genes={
            "bb_period": StrategyGene(name="bb_period", value=20, min_val=10, max_val=50, step=1),
            "bb_std": StrategyGene(name="bb_std", value=2.0, min_val=1.0, max_val=3.5, step=0.1),
            "rsi_period": StrategyGene(name="rsi_period", value=14, min_val=5, max_val=50, step=1),
            "rsi_oversold": StrategyGene(name="rsi_oversold", value=25, min_val=10, max_val=40, step=1),
            "rsi_overbought": StrategyGene(name="rsi_overbought", value=75, min_val=60, max_val=90, step=1),
            "mean_lookback": StrategyGene(name="mean_lookback", value=50, min_val=20, max_val=200, step=5),
            "entry_z_score": StrategyGene(name="entry_z_score", value=2.0, min_val=1.0, max_val=3.5, step=0.1),
            "exit_z_score": StrategyGene(name="exit_z_score", value=0.5, min_val=0.0, max_val=1.5, step=0.1),
            "stop_loss_pct": StrategyGene(name="stop_loss_pct", value=0.04, min_val=0.01, max_val=0.10, step=0.005),
            "take_profit_pct": StrategyGene(name="take_profit_pct", value=0.06, min_val=0.02, max_val=0.15, step=0.005),
            "confidence_threshold": StrategyGene(name="confidence_threshold", value=0.55, min_val=0.3, max_val=0.95, step=0.05),
        })


class FitnessScore(BaseModel):
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    composite: float = 0.0         # weighted combination of the above


class Strategy(BaseModel):
    id: str = Field(default_factory=lambda: f"strat-{uuid.uuid4().hex[:8]}")
    name: str = "unnamed"
    strategy_type: str = "momentum"    # "momentum", "mean_reversion", "hybrid"
    genome: StrategyGenome = Field(default_factory=StrategyGenome.default_momentum)
    fitness: FitnessScore = Field(default_factory=FitnessScore)
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_evaluated: datetime | None = None
    total_trades: int = 0
    total_pnl: float = 0.0
    regime_preference: str = ""  # "trending_bull", "trending_bear", "mean_reverting", "high_volatility", "" = all
    kelly_fraction_mult: float = 0.5  # half-Kelly default, range 0.25-0.75
    # Thompson Sampling stats
    wins: int = 0
    losses: int = 0
