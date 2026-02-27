"""Regression tests for critical bug fixes.

Bug 1: Step Functions approval always passes — handler now returns 'approved' field
Bug 2: Max positions check bypass — removed 'or bool(open_trades)' clause
Bug 3: Stop/take-profit hardcodes "sell" — now uses signal side for close direction
Bug 4: get_strategy_trades missing pagination — now loops through all pages
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from claudestreet.agents.analyst import AnalystAgent
from claudestreet.agents.risk_guard import RiskGuardAgent
from claudestreet.models.events import (
    Event,
    EventType,
    SignalPayload,
    TradeProposalPayload,
)


# ── Helpers ──


def _make_memory():
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


def _valid_proposal(**overrides):
    defaults = dict(
        symbol="AAPL",
        side="buy",
        quantity=10,
        entry_price=150.0,
        stop_loss=142.5,
        take_profit=165.0,
        strategy_id="test-strat",
        confidence=0.8,
    )
    defaults.update(overrides)
    return TradeProposalPayload(**defaults)


# ── Bug 1: Handler returns 'approved' field ──


def _run_handler_with_fake_agent(output_events):
    """Run the handler factory with a fake agent that returns given events."""
    from claudestreet.core.event_bus import EventBridgeClient
    from claudestreet.handlers.base import create_handler

    class FakeAgent:
        agent_id = "fake_agent"

        def __init__(self, **kwargs):
            pass

        def process(self, event):
            return output_events

        def heartbeat(self):
            return []

    mock_eb = MagicMock()
    mock_eb.put_events.return_value = len(output_events)

    parsed_event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload={},
    )

    with patch("claudestreet.handlers.base._get_infra") as mock_infra, \
         patch.object(EventBridgeClient, "is_heartbeat", return_value=False), \
         patch.object(EventBridgeClient, "from_eventbridge", return_value=parsed_event):
        mock_infra.return_value = (_make_memory(), mock_eb, _make_config())
        handler = create_handler(FakeAgent)
        return handler({"detail-type": "trade"}, None)


def test_handler_returns_approved_true_for_risk_approved():
    """Handler must set approved=True when RiskGuard emits RISK_APPROVED."""
    result = _run_handler_with_fake_agent([
        Event(type=EventType.RISK_APPROVED, source="risk_guard", payload={"symbol": "AAPL"}),
    ])
    assert result["statusCode"] == 200
    assert result["approved"] is True


def test_handler_returns_approved_false_for_risk_rejected():
    """Handler must set approved=False when RiskGuard emits RISK_REJECTED."""
    result = _run_handler_with_fake_agent([
        Event(type=EventType.RISK_REJECTED, source="risk_guard", payload={"rejections": ["x"]}),
    ])
    assert result["statusCode"] == 200
    assert result["approved"] is False


# ── Bug 2: Max positions check bypass ──


def test_max_positions_enforced_with_existing_trades():
    """Max positions must be enforced even when the symbol already has open trades."""
    memory = _make_memory()
    config = _make_config()
    config["max_positions"] = 3

    # 3 open positions total (at the limit)
    all_open = [
        {"symbol": "AAPL", "entry_price": 150, "quantity": 10, "stop_loss": 140},
        {"symbol": "GOOG", "entry_price": 100, "quantity": 5, "stop_loss": 90},
        {"symbol": "MSFT", "entry_price": 200, "quantity": 8, "stop_loss": 190},
    ]

    # Symbol-specific open trades (AAPL already has a position)
    symbol_open = [all_open[0]]

    # First call (with symbol) returns symbol trades, second call (all) returns all
    memory.get_open_trades.side_effect = [symbol_open, all_open]

    agent = RiskGuardAgent(memory=memory, config=config)
    proposal = _valid_proposal(symbol="AAPL", quantity=5, entry_price=150.0)

    event = Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.RISK_REJECTED
    rejections = result[0].payload.get("rejections", [])
    assert any("max" in r.lower() or "positions" in r.lower() for r in rejections)


# ── Bug 3: Stop/take-profit close direction ──


def test_stop_loss_close_direction_for_short():
    """SL on a short (side='sell') should recommend 'buy' to close."""
    memory = _make_memory()
    config = _make_config()
    agent = AnalystAgent(memory=memory, config=config)

    signal = SignalPayload(
        symbol="AAPL",
        signal_type="stop_loss",
        strength=1.0,
        indicators={"trigger_price": 160.0},
        timeframe="tick",
        side="sell",
    )
    event = Event(
        type=EventType.SIGNAL_DETECTED,
        source="sentinel",
        payload=signal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.ANALYSIS_COMPLETE
    assert result[0].payload["recommendation"] == "buy"


def test_stop_loss_close_direction_for_long():
    """SL on a long (side='buy') should recommend 'sell' to close."""
    memory = _make_memory()
    config = _make_config()
    agent = AnalystAgent(memory=memory, config=config)

    signal = SignalPayload(
        symbol="AAPL",
        signal_type="stop_loss",
        strength=1.0,
        indicators={"trigger_price": 140.0},
        timeframe="tick",
        side="buy",
    )
    event = Event(
        type=EventType.SIGNAL_DETECTED,
        source="sentinel",
        payload=signal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.ANALYSIS_COMPLETE
    assert result[0].payload["recommendation"] == "sell"


def test_stop_loss_legacy_no_side():
    """SL with no side (legacy) should fallback to recommendation='sell'."""
    memory = _make_memory()
    config = _make_config()
    agent = AnalystAgent(memory=memory, config=config)

    signal = SignalPayload(
        symbol="AAPL",
        signal_type="take_profit",
        strength=1.0,
        indicators={"trigger_price": 170.0},
        timeframe="tick",
        # side defaults to ""
    )
    event = Event(
        type=EventType.SIGNAL_DETECTED,
        source="sentinel",
        payload=signal.model_dump(),
    )

    result = agent.process(event)
    assert len(result) == 1
    assert result[0].type == EventType.ANALYSIS_COMPLETE
    assert result[0].payload["recommendation"] == "sell"


# ── Bug 4: get_strategy_trades pagination ──


def test_get_strategy_trades_pagination():
    """get_strategy_trades must paginate through all DynamoDB pages."""
    from claudestreet.core.memory import DynamoMemory

    # Mock the DynamoDB table
    mock_table = MagicMock()

    # Simulate two pages of results
    page1_items = [
        {"trade_id": f"t{i}", "strategy_id": "strat-1", "pnl": Decimal("10.5")}
        for i in range(3)
    ]
    page2_items = [
        {"trade_id": f"t{i}", "strategy_id": "strat-1", "pnl": Decimal("20.0")}
        for i in range(3, 5)
    ]

    mock_table.query.side_effect = [
        {"Items": page1_items, "LastEvaluatedKey": {"trade_id": "t2"}},
        {"Items": page2_items},  # no LastEvaluatedKey = last page
    ]

    with patch("boto3.resource"):
        mem = DynamoMemory.__new__(DynamoMemory)
        mem._trades = mock_table

    result = mem.get_strategy_trades("strat-1")

    assert len(result) == 5
    assert mock_table.query.call_count == 2
    # Verify Decimals were converted to float
    assert all(isinstance(item["pnl"], float) for item in result)
