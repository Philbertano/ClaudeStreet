"""Tests for agent business logic (no AWS dependencies)."""

from unittest.mock import MagicMock

from claudestreet.agents.risk_guard import RiskGuardAgent
from claudestreet.models.events import Event, EventType, TradeProposalPayload


def _make_memory():
    """Create a mock memory for testing."""
    memory = MagicMock()
    memory.get_open_trades.return_value = []
    return memory


def _make_config():
    return {
        "initial_capital": 100000.0,
        "max_positions": 10,
        "max_position_pct": 0.15,
        "restricted_hours": [],
    }


def test_risk_guard_approves_valid_trade():
    memory = _make_memory()
    agent = RiskGuardAgent(memory=memory, config=_make_config())

    proposal = TradeProposalPayload(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=150.0,
        stop_loss=142.5,
        take_profit=165.0,
        strategy_id="test-strat",
        confidence=0.8,
    )

    event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.RISK_APPROVED


def test_risk_guard_rejects_no_stop_loss():
    memory = _make_memory()
    agent = RiskGuardAgent(memory=memory, config=_make_config())

    proposal = TradeProposalPayload(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=150.0,
        stop_loss=0,        # no stop loss
        take_profit=165.0,
        strategy_id="test-strat",
        confidence=0.8,
    )

    event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.RISK_REJECTED
    assert "stop loss" in result[0].payload["rejections"][0].lower()


def test_risk_guard_rejects_oversized_position():
    memory = _make_memory()
    agent = RiskGuardAgent(memory=memory, config=_make_config())

    proposal = TradeProposalPayload(
        symbol="AAPL",
        side="buy",
        quantity=200,
        entry_price=150.0,   # 200 * 150 = 30k, > 15% of 100k
        stop_loss=142.5,
        take_profit=165.0,
        strategy_id="test-strat",
        confidence=0.8,
    )

    event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.RISK_REJECTED


def test_risk_guard_rejects_bad_risk_reward():
    memory = _make_memory()
    agent = RiskGuardAgent(memory=memory, config=_make_config())

    proposal = TradeProposalPayload(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=150.0,
        stop_loss=140.0,      # risk = 10
        take_profit=155.0,    # reward = 5, R:R = 0.5
        strategy_id="test-strat",
        confidence=0.8,
    )

    event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.RISK_REJECTED
