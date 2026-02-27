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


# ══════════════════════════════════════════════════
# EDGE CASE TESTS — Decision Ledger
# ══════════════════════════════════════════════════


def test_decision_step_empty_reasoning(memory):
    """Empty reasoning dict should be stored without error."""
    memory.record_decision_step(
        correlation_id="corr-empty",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={},
    )

    chain = memory.get_decision_chain("corr-empty")
    assert len(chain) == 1
    assert chain[0]["reasoning"] == {}


def test_decision_step_nested_reasoning(memory):
    """Complex nested reasoning dict should survive DynamoDB round-trip."""
    nested = {
        "signals": {
            "rsi": {"value": 25.3, "score": 1.0, "state": "oversold"},
            "macd": {"value": 0.5, "score": 0.8, "state": "bullish"},
        },
        "fingerprint": "rsi=oversold|macd=bullish",
        "list_data": [1.0, 2.5, 3.7],
    }
    memory.record_decision_step(
        correlation_id="corr-nested",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning=nested,
    )

    chain = memory.get_decision_chain("corr-nested")
    assert len(chain) == 1
    r = chain[0]["reasoning"]
    assert r["signals"]["rsi"]["state"] == "oversold"
    assert r["list_data"] == [1.0, 2.5, 3.7]


def test_decision_step_has_ttl(memory):
    """Decision steps should have TTL attribute set."""
    memory.record_decision_step(
        correlation_id="corr-ttl",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"test": True},
    )

    chain = memory.get_decision_chain("corr-ttl")
    assert len(chain) == 1
    assert "ttl" in chain[0]
    assert chain[0]["ttl"] > 0


def test_decision_step_without_strategy_id(memory):
    """Steps without strategy_id should not have the attribute."""
    memory.record_decision_step(
        correlation_id="corr-nostrat",
        step_key="analyst",
        agent="analyst",
        symbol="AAPL",
        reasoning={"test": True},
    )

    chain = memory.get_decision_chain("corr-nostrat")
    assert len(chain) == 1
    # strategy_id should not be present (not in GSI query results either)
    assert "strategy_id" not in chain[0]


def test_get_decision_chain_nonexistent(memory):
    """Querying non-existent correlation_id returns empty list."""
    chain = memory.get_decision_chain("does-not-exist")
    assert chain == []


def test_get_decisions_by_symbol_empty(memory):
    """Querying symbol with no decisions returns empty list."""
    results = memory.get_decisions_by_symbol("ZZZZ")
    assert results == []


def test_get_decisions_by_strategy_empty(memory):
    """Querying strategy with no decisions returns empty list."""
    results = memory.get_decisions_by_strategy("no-such-strategy")
    assert results == []


def test_get_decisions_by_symbol_respects_limit(memory):
    """Limit parameter should cap the number of results."""
    for i in range(10):
        memory.record_decision_step(
            correlation_id=f"corr-limit-{i}",
            step_key="analyst",
            agent="analyst",
            symbol="AAPL",
            reasoning={"i": i},
        )

    results = memory.get_decisions_by_symbol("AAPL", limit=3)
    assert len(results) == 3


# ══════════════════════════════════════════════════
# EDGE CASE TESTS — Pattern Library
# ══════════════════════════════════════════════════


def test_update_pattern_zero_pnl(memory):
    """Zero PnL should count as a loss (pnl <= 0)."""
    memory.update_pattern("AAPL", "rsi=neutral", 0.0, 0.5)

    pattern = memory.get_pattern("AAPL", "rsi=neutral")
    assert pattern is not None
    assert pattern["occurrences"] == 1
    assert pattern["wins"] == 0
    assert pattern["losses"] == 1
    assert pattern["total_pnl"] == 0.0


def test_update_pattern_negative_pnl(memory):
    """Negative PnL should count as a loss."""
    memory.update_pattern("AAPL", "rsi=overbought", -500.0, 0.7)

    pattern = memory.get_pattern("AAPL", "rsi=overbought")
    assert pattern is not None
    assert pattern["wins"] == 0
    assert pattern["losses"] == 1
    assert pattern["total_pnl"] == -500.0


def test_update_pattern_small_positive_pnl(memory):
    """Even a tiny positive PnL should count as a win."""
    memory.update_pattern("AAPL", "rsi=neutral", 0.01, 0.5)

    pattern = memory.get_pattern("AAPL", "rsi=neutral")
    assert pattern["wins"] == 1
    assert pattern["losses"] == 0


def test_get_pattern_nonexistent(memory):
    """Querying non-existent pattern returns None."""
    result = memory.get_pattern("AAPL", "does-not-exist")
    assert result is None


def test_get_patterns_for_symbol_empty(memory):
    """Querying symbol with no patterns returns empty list."""
    results = memory.get_patterns_for_symbol("ZZZZ")
    assert results == []


def test_get_patterns_min_occurrences_zero(memory):
    """min_occurrences=0 should return all patterns."""
    memory.update_pattern("GOOG", "rsi=neutral", 10.0, 0.5)

    results = memory.get_patterns_for_symbol("GOOG", min_occurrences=0)
    assert len(results) == 1


def test_pattern_confidence_accumulates(memory):
    """avg_confidence should accumulate across updates."""
    memory.update_pattern("AAPL", "rsi=oversold", 50.0, 0.8)
    memory.update_pattern("AAPL", "rsi=oversold", 30.0, 0.6)

    pattern = memory.get_pattern("AAPL", "rsi=oversold")
    assert pattern is not None
    # avg_confidence accumulates: 0 + 0.8 + 0.6 = 1.4
    assert abs(pattern["avg_confidence"] - 1.4) < 0.01


def test_pattern_last_seen_updates(memory):
    """last_seen should reflect the most recent update."""
    memory.update_pattern("AAPL", "rsi=oversold", 50.0, 0.8)
    first_pattern = memory.get_pattern("AAPL", "rsi=oversold")
    first_ts = first_pattern["last_seen"]

    memory.update_pattern("AAPL", "rsi=oversold", 30.0, 0.6)
    second_pattern = memory.get_pattern("AAPL", "rsi=oversold")
    second_ts = second_pattern["last_seen"]

    assert second_ts >= first_ts


# ══════════════════════════════════════════════════
# EDGE CASE TESTS — Snapshot with market_context
# ══════════════════════════════════════════════════


def test_record_snapshot_without_market_context(memory):
    """Snapshot without market_context should work (backwards compatible)."""
    memory.record_snapshot(
        cash=100000.0,
        positions_value=0.0,
        total_value=100000.0,
        daily_pnl=0.0,
        open_positions=0,
    )

    history = memory.get_portfolio_history(days=1)
    assert len(history) >= 1
    assert "market_context" not in history[0]


def test_record_snapshot_empty_market_context(memory):
    """Empty market_context dict should not be stored (falsy)."""
    memory.record_snapshot(
        cash=100000.0,
        positions_value=0.0,
        total_value=100000.0,
        daily_pnl=0.0,
        open_positions=0,
        market_context={},
    )

    history = memory.get_portfolio_history(days=1)
    assert len(history) >= 1
    # Empty dict is falsy, so market_context should not be stored
    assert "market_context" not in history[0]


# ══════════════════════════════════════════════════
# EDGE CASE TESTS — evaluate_detailed
# ══════════════════════════════════════════════════


def test_evaluate_detailed_empty_indicators():
    """Empty indicators should return hold with 0 confidence."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({})

    assert result["recommendation"] == "hold"
    assert result["confidence"] == 0.0
    # Should still have fingerprint parts for the 4 hardcoded signals
    assert "rsi=neutral" in result["fingerprint"]
    assert "macd=neutral" in result["fingerprint"]


def test_evaluate_detailed_only_rsi():
    """Only RSI provided — should still produce valid result."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({"rsi": 20.0})

    assert result["recommendation"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert "rsi=oversold" in result["fingerprint"]
    assert "rsi" in result["signals"]


def test_evaluate_detailed_sma_missing():
    """Missing SMA 50/200 should not appear in signals or fingerprint."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({
        "rsi": 50.0,
        "macd_histogram": 0.0,
        "bb_pct_b": 0.5,
        "ema_crossover": 0.0,
    })

    assert "sma" not in result["signals"]
    assert "sma=" not in result["fingerprint"]


def test_evaluate_detailed_sma_golden_cross():
    """SMA golden cross should appear in signals and fingerprint."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({
        "rsi": 50.0,
        "macd_histogram": 0.0,
        "bb_pct_b": 0.5,
        "ema_crossover": 0.0,
        "sma_50": 200.0,
        "sma_200": 180.0,
    })

    assert "sma" in result["signals"]
    assert result["signals"]["sma"]["state"] == "golden"
    assert "sma=golden" in result["fingerprint"]


def test_evaluate_detailed_sma_death_cross():
    """SMA death cross should appear in signals and fingerprint."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({
        "rsi": 50.0,
        "macd_histogram": 0.0,
        "bb_pct_b": 0.5,
        "ema_crossover": 0.0,
        "sma_50": 180.0,
        "sma_200": 200.0,
    })

    assert result["signals"]["sma"]["state"] == "death"
    assert "sma=death" in result["fingerprint"]


def test_evaluate_detailed_stoch_rsi_none():
    """stoch_rsi_k=None should be excluded from signals."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({"rsi": 50.0})

    assert "stoch_rsi" not in result["signals"]
    assert "stoch_rsi=" not in result["fingerprint"]


def test_evaluate_detailed_stoch_rsi_zero():
    """stoch_rsi_k=0.0 is a value (not None) — should be 'oversold'."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({"stoch_rsi_k": 0.0})

    assert "stoch_rsi" in result["signals"]
    assert result["signals"]["stoch_rsi"]["state"] == "oversold"


def test_evaluate_detailed_mfi_none():
    """mfi=None should be excluded from signals."""
    ta = TechnicalAnalysisSkill()
    result = ta.evaluate_detailed({"rsi": 50.0})

    assert "mfi" not in result["signals"]


def test_evaluate_detailed_mfi_boundaries():
    """MFI at exact boundaries."""
    ta = TechnicalAnalysisSkill()

    # At boundary: mfi=20 should be neutral (not < 20)
    result = ta.evaluate_detailed({"mfi": 20.0})
    assert result["signals"]["mfi"]["state"] == "neutral"

    # Just below: mfi=19 should be oversold
    result = ta.evaluate_detailed({"mfi": 19.0})
    assert result["signals"]["mfi"]["state"] == "oversold"

    # At boundary: mfi=80 should be neutral (not > 80)
    result = ta.evaluate_detailed({"mfi": 80.0})
    assert result["signals"]["mfi"]["state"] == "neutral"

    # Just above: mfi=81 should be overbought
    result = ta.evaluate_detailed({"mfi": 81.0})
    assert result["signals"]["mfi"]["state"] == "overbought"


def test_evaluate_detailed_extreme_rsi():
    """RSI at extreme values (0 and 100)."""
    ta = TechnicalAnalysisSkill()

    result = ta.evaluate_detailed({"rsi": 0.0})
    assert result["signals"]["rsi"]["state"] == "oversold"

    result = ta.evaluate_detailed({"rsi": 100.0})
    assert result["signals"]["rsi"]["state"] == "overbought"


def test_evaluate_detailed_high_volume_amplifies():
    """Volume ratio > 2.0 should amplify score."""
    ta = TechnicalAnalysisSkill()
    indicators_low_vol = {"rsi": 25.0, "volume_ratio": 1.0}
    indicators_high_vol = {"rsi": 25.0, "volume_ratio": 3.0}

    result_low = ta.evaluate_detailed(indicators_low_vol)
    result_high = ta.evaluate_detailed(indicators_high_vol)

    # High volume should produce higher absolute score
    assert abs(result_high["score"]) >= abs(result_low["score"])


def test_evaluate_detailed_fingerprint_deterministic():
    """Same indicators should always produce same fingerprint."""
    ta = TechnicalAnalysisSkill()
    indicators = {
        "rsi": 25.0,
        "macd_histogram": 0.5,
        "bb_pct_b": 0.05,
        "ema_crossover": 1.0,
        "stoch_rsi_k": 0.15,
        "mfi": 18.0,
    }

    fp1 = ta.evaluate_detailed(indicators)["fingerprint"]
    fp2 = ta.evaluate_detailed(indicators)["fingerprint"]

    assert fp1 == fp2


def test_evaluate_detailed_all_bearish():
    """All bearish signals should produce strong_sell or sell."""
    ta = TechnicalAnalysisSkill()
    indicators = {
        "rsi": 80.0,
        "macd_histogram": -1.0,
        "bb_pct_b": 0.95,
        "ema_crossover": -1.0,
        "sma_50": 180.0,
        "sma_200": 200.0,
        "stoch_rsi_k": 0.9,
        "mfi": 85.0,
    }

    result = ta.evaluate_detailed(indicators)
    assert result["recommendation"] in ("sell", "strong_sell")
    assert result["signals"]["rsi"]["state"] == "overbought"
    assert result["signals"]["macd"]["state"] == "bearish"
