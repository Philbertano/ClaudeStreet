"""Tests for decision ledger, pattern library, and evaluate_detailed."""

import os

import boto3
import pytest
from moto import mock_aws

from claudestreet.core.memory import DynamoMemory
from claudestreet.skills.technical_analysis import TechnicalAnalysisSkill


# ── DynamoDB fixtures ──


def _create_tables(dynamodb):
    """Create all DynamoDB tables needed by DynamoMemory."""
    # Trades table
    dynamodb.create_table(
        TableName="test-trades",
        KeySchema=[{"AttributeName": "trade_id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "trade_id", "AttributeType": "S"},
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
            {"AttributeName": "strategy_id", "AttributeType": "S"},
            {"AttributeName": "opened_at", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "symbol-status-index",
                "KeySchema": [
                    {"AttributeName": "symbol", "KeyType": "HASH"},
                    {"AttributeName": "status", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "strategy-index",
                "KeySchema": [
                    {"AttributeName": "strategy_id", "KeyType": "HASH"},
                    {"AttributeName": "opened_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )
    # Strategies table
    dynamodb.create_table(
        TableName="test-strategies",
        KeySchema=[{"AttributeName": "strategy_id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "strategy_id", "AttributeType": "S"},
            {"AttributeName": "is_active", "AttributeType": "N"},
            {"AttributeName": "total_pnl", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "active-index",
                "KeySchema": [
                    {"AttributeName": "is_active", "KeyType": "HASH"},
                    {"AttributeName": "total_pnl", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )
    # Snapshots table
    dynamodb.create_table(
        TableName="test-snapshots",
        KeySchema=[
            {"AttributeName": "date", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "date", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    # Events table
    dynamodb.create_table(
        TableName="test-events",
        KeySchema=[
            {"AttributeName": "event_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "event_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
            {"AttributeName": "correlation_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "correlation-index",
                "KeySchema": [
                    {"AttributeName": "correlation_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )
    # Decisions table
    dynamodb.create_table(
        TableName="test-decisions",
        KeySchema=[
            {"AttributeName": "correlation_id", "KeyType": "HASH"},
            {"AttributeName": "step_key", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "correlation_id", "AttributeType": "S"},
            {"AttributeName": "step_key", "AttributeType": "S"},
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"},
            {"AttributeName": "strategy_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "symbol-index",
                "KeySchema": [
                    {"AttributeName": "symbol", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "strategy-index",
                "KeySchema": [
                    {"AttributeName": "strategy_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )
    # Patterns table
    dynamodb.create_table(
        TableName="test-patterns",
        KeySchema=[{"AttributeName": "pattern_key", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "pattern_key", "AttributeType": "S"},
            {"AttributeName": "symbol", "AttributeType": "S"},
            {"AttributeName": "occurrences", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "symbol-index",
                "KeySchema": [
                    {"AttributeName": "symbol", "KeyType": "HASH"},
                    {"AttributeName": "occurrences", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    )


@pytest.fixture
def memory():
    with mock_aws():
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        _create_tables(dynamodb)
        yield DynamoMemory(
            trades_table="test-trades",
            strategies_table="test-strategies",
            snapshots_table="test-snapshots",
            events_table="test-events",
            decisions_table="test-decisions",
            patterns_table="test-patterns",
            region="us-east-1",
        )


# ── Decision Ledger tests ──


def test_record_decision_step_and_get_chain(memory):
    """Write 3 steps, read chain, verify order and content."""
    corr_id = "corr-001"

    memory.record_decision_step(
        correlation_id=corr_id,
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"score": 0.8, "fingerprint": "rsi=oversold|macd=bullish"},
    )
    memory.record_decision_step(
        correlation_id=corr_id,
        step_key="strategist:STRAT-001",
        agent="strategist",
        symbol="AAPL",
        strategy_id="STRAT-001",
        reasoning={"thompson_score": 0.75, "quantity": 10},
    )
    memory.record_decision_step(
        correlation_id=corr_id,
        step_key="risk_guard",
        agent="risk_guard",
        symbol="AAPL",
        strategy_id="STRAT-001",
        reasoning={"approved": True, "checks": {}},
    )

    chain = memory.get_decision_chain(corr_id)
    assert len(chain) == 3
    assert chain[0]["step_key"] == "analyst"
    assert chain[1]["step_key"] == "risk_guard"  # sorted by step_key (sort key)
    assert chain[2]["step_key"] == "strategist:STRAT-001"
    assert chain[0]["reasoning"]["score"] == 0.8


def test_decision_step_idempotent(memory):
    """Write same step twice, no error on duplicate."""
    corr_id = "corr-002"

    memory.record_decision_step(
        correlation_id=corr_id,
        step_key="analyst",
        agent="analyst",
        symbol="MSFT",
        reasoning={"score": 0.5},
    )
    # Second write should be silently skipped
    memory.record_decision_step(
        correlation_id=corr_id,
        step_key="analyst",
        agent="analyst",
        symbol="MSFT",
        reasoning={"score": 0.9},  # different data
    )

    chain = memory.get_decision_chain(corr_id)
    assert len(chain) == 1
    assert chain[0]["reasoning"]["score"] == 0.5  # original preserved


def test_decision_step_skips_none_correlation(memory):
    """Should silently skip when correlation_id is None."""
    memory.record_decision_step(
        correlation_id=None,
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"test": True},
    )
    # No error, nothing written


def test_get_decisions_by_symbol(memory):
    """Verify GSI query by symbol returns matching decisions."""
    memory.record_decision_step(
        correlation_id="corr-a",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"score": 0.8},
    )
    memory.record_decision_step(
        correlation_id="corr-b",
        step_key="analyst",
        agent="analyst",
        symbol="MSFT",
        reasoning={"score": 0.3},
    )
    memory.record_decision_step(
        correlation_id="corr-c",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"score": 0.6},
    )

    results = memory.get_decisions_by_symbol("AAPL")
    assert len(results) == 2
    symbols = {r["symbol"] for r in results}
    assert symbols == {"AAPL"}


def test_get_decisions_by_strategy(memory):
    """Verify GSI query by strategy returns matching decisions."""
    memory.record_decision_step(
        correlation_id="corr-x",
        step_key="strategist:S1",
        agent="strategist",
        symbol="AAPL",
        strategy_id="S1",
        reasoning={"thompson_score": 0.7},
    )
    memory.record_decision_step(
        correlation_id="corr-y",
        step_key="strategist:S2",
        agent="strategist",
        symbol="AAPL",
        strategy_id="S2",
        reasoning={"thompson_score": 0.4},
    )

    results = memory.get_decisions_by_strategy("S1")
    assert len(results) == 1
    assert results[0]["strategy_id"] == "S1"


# ── Pattern Library tests ──


def test_update_pattern_creates_new(memory):
    """First update creates pattern with occurrences=1."""
    memory.update_pattern("AAPL", "rsi=oversold|macd=bullish", 150.0, 0.8)

    pattern = memory.get_pattern("AAPL", "rsi=oversold|macd=bullish")
    assert pattern is not None
    assert pattern["occurrences"] == 1
    assert pattern["wins"] == 1
    assert pattern["losses"] == 0
    assert pattern["total_pnl"] == 150.0
    assert pattern["symbol"] == "AAPL"
    assert pattern["fingerprint"] == "rsi=oversold|macd=bullish"


def test_update_pattern_increments(memory):
    """Second update increments counters atomically."""
    fp = "rsi=neutral|macd=bearish"
    memory.update_pattern("TSLA", fp, 200.0, 0.7)
    memory.update_pattern("TSLA", fp, -50.0, 0.6)
    memory.update_pattern("TSLA", fp, 100.0, 0.9)

    pattern = memory.get_pattern("TSLA", fp)
    assert pattern is not None
    assert pattern["occurrences"] == 3
    assert pattern["wins"] == 2
    assert pattern["losses"] == 1
    assert abs(pattern["total_pnl"] - 250.0) < 0.01


def test_get_patterns_for_symbol(memory):
    """Verify GSI query with min_occurrences filter."""
    fp1 = "rsi=oversold"
    fp2 = "rsi=neutral"
    # fp1 gets 6 occurrences
    for i in range(6):
        memory.update_pattern("SPY", fp1, 10.0, 0.5)
    # fp2 gets 3 occurrences (below threshold)
    for i in range(3):
        memory.update_pattern("SPY", fp2, -5.0, 0.4)

    results = memory.get_patterns_for_symbol("SPY", min_occurrences=5)
    assert len(results) == 1
    assert results[0]["fingerprint"] == fp1


# ── Snapshot extension test ──


def test_record_snapshot_with_market_context(memory):
    """Verify market_context is stored in snapshot."""
    ctx = {"regime": "trending_bull", "long_count": 3, "short_count": 1}
    memory.record_snapshot(
        cash=50000.0,
        positions_value=50000.0,
        total_value=100000.0,
        daily_pnl=500.0,
        open_positions=4,
        market_context=ctx,
    )

    history = memory.get_portfolio_history(days=1)
    assert len(history) >= 1
    latest = history[0]
    assert "market_context" in latest
    assert latest["market_context"]["regime"] == "trending_bull"


# ── evaluate_detailed tests ──


def test_evaluate_detailed_returns_signals_and_fingerprint():
    """Verify structure of evaluate_detailed output."""
    ta = TechnicalAnalysisSkill()
    indicators = {
        "rsi": 25.0,
        "macd_histogram": 0.5,
        "bb_pct_b": 0.05,
        "ema_crossover": 1.0,
        "volume_ratio": 1.5,
        "stoch_rsi_k": 0.15,
        "mfi": 18.0,
    }

    result = ta.evaluate_detailed(indicators)

    assert "recommendation" in result
    assert "confidence" in result
    assert "score" in result
    assert "signals" in result
    assert "fingerprint" in result

    # Check signal detail structure
    assert "rsi" in result["signals"]
    assert result["signals"]["rsi"]["state"] == "oversold"
    assert "macd" in result["signals"]
    assert result["signals"]["macd"]["state"] == "bullish"
    assert "bb" in result["signals"]
    assert result["signals"]["bb"]["state"] == "lower"

    # Fingerprint should be deterministic
    fp = result["fingerprint"]
    assert "rsi=oversold" in fp
    assert "macd=bullish" in fp
    assert "bb=lower" in fp


def test_evaluate_detailed_matches_evaluate():
    """recommendation and confidence should match evaluate()."""
    ta = TechnicalAnalysisSkill()
    indicators = {
        "rsi": 45.0,
        "macd_histogram": -0.3,
        "bb_pct_b": 0.5,
        "ema_crossover": -1.0,
        "volume_ratio": 1.0,
    }

    rec_simple, conf_simple = ta.evaluate(indicators)
    detailed = ta.evaluate_detailed(indicators)

    assert detailed["recommendation"] == rec_simple
    assert abs(detailed["confidence"] - conf_simple) < 1e-6
