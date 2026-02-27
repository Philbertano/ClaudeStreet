"""Tests for agent decision step recording integration.

Verifies that each agent correctly calls record_decision_step
when processing events, and that failures don't block the pipeline.
"""

from unittest.mock import MagicMock

from claudestreet.agents.risk_guard import RiskGuardAgent
from claudestreet.agents.executor import ExecutorAgent
from claudestreet.agents.chronicler import ChroniclerAgent
from claudestreet.agents.strategist import StrategistAgent
from claudestreet.models.events import (
    Event, EventType,
    TradeProposalPayload, AnalysisPayload,
)
from claudestreet.models.strategy import Strategy, StrategyGenome, FitnessScore


# ── Helpers ──


def _make_memory():
    memory = MagicMock()
    memory.get_open_trades.return_value = []
    memory.get_active_strategies.return_value = []
    memory.get_current_regime.return_value = ""
    memory.get_decision_chain.return_value = []
    memory.get_strategy_trades.return_value = []
    return memory


def _make_config():
    return {
        "initial_capital": 100000.0,
        "max_positions": 10,
        "max_position_pct": 0.15,
        "restricted_hours": [],
        "paper_trading": True,
        "thompson_k": 3,
    }


def _make_proposal_event(correlation_id="corr-001"):
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
    return Event(
        type=EventType.TRADE_PROPOSED,
        source="strategist",
        payload=proposal.model_dump(),
        correlation_id=correlation_id,
    )


# ── RiskGuard decision step tests ──


class TestRiskGuardDecisionSteps:
    """Verify risk_guard records decision steps on approve and reject."""

    def test_records_step_on_approval(self):
        memory = _make_memory()
        agent = RiskGuardAgent(memory=memory, config=_make_config())
        event = _make_proposal_event()

        result = agent.process(event)

        assert result[0].type == EventType.RISK_APPROVED
        memory.record_decision_step.assert_called_once()
        call_kwargs = memory.record_decision_step.call_args[1]
        assert call_kwargs["correlation_id"] == "corr-001"
        assert call_kwargs["step_key"] == "risk_guard"
        assert call_kwargs["agent"] == "risk_guard"
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["strategy_id"] == "test-strat"
        reasoning = call_kwargs["reasoning"]
        assert reasoning["approved"] is True
        assert "checks" in reasoning
        assert reasoning["checks"]["stop_loss"]["passed"] is True

    def test_records_step_on_rejection(self):
        memory = _make_memory()
        agent = RiskGuardAgent(memory=memory, config=_make_config())

        proposal = TradeProposalPayload(
            symbol="AAPL",
            side="buy",
            quantity=10,
            entry_price=150.0,
            stop_loss=0,  # will be rejected
            take_profit=165.0,
            strategy_id="test-strat",
            confidence=0.8,
        )
        event = Event(
            type=EventType.TRADE_PROPOSED,
            source="strategist",
            payload=proposal.model_dump(),
            correlation_id="corr-002",
        )

        result = agent.process(event)

        assert result[0].type == EventType.RISK_REJECTED
        memory.record_decision_step.assert_called_once()
        reasoning = memory.record_decision_step.call_args[1]["reasoning"]
        assert reasoning["approved"] is False
        assert len(reasoning["rejections"]) > 0
        assert reasoning["checks"]["stop_loss"]["passed"] is False

    def test_records_all_check_results(self):
        """Verify all 8 checks are tracked in reasoning."""
        memory = _make_memory()
        agent = RiskGuardAgent(memory=memory, config=_make_config())
        event = _make_proposal_event()

        agent.process(event)

        checks = memory.record_decision_step.call_args[1]["reasoning"]["checks"]
        expected_checks = {
            "position_size", "symbol_exposure", "max_positions",
            "stop_loss", "risk_reward", "restricted_hours",
            "portfolio_heat", "correlation",
        }
        assert set(checks.keys()) == expected_checks
        # All checks should have passed for a valid trade
        for check_name, check_data in checks.items():
            assert "passed" in check_data, f"Missing 'passed' in check {check_name}"

    def test_pipeline_continues_if_decision_step_fails(self):
        """Decision step failure must not block the pipeline."""
        memory = _make_memory()
        memory.record_decision_step.side_effect = Exception("DynamoDB error")
        agent = RiskGuardAgent(memory=memory, config=_make_config())
        event = _make_proposal_event()

        result = agent.process(event)

        # Pipeline should still produce a result
        assert len(result) == 1
        assert result[0].type == EventType.RISK_APPROVED

    def test_records_step_with_none_correlation_id(self):
        """Should still call record_decision_step even if correlation_id is None."""
        memory = _make_memory()
        agent = RiskGuardAgent(memory=memory, config=_make_config())

        proposal = TradeProposalPayload(
            symbol="AAPL", side="buy", quantity=10,
            entry_price=150.0, stop_loss=142.5, take_profit=165.0,
            strategy_id="test-strat", confidence=0.8,
        )
        event = Event(
            type=EventType.TRADE_PROPOSED,
            source="strategist",
            payload=proposal.model_dump(),
            correlation_id=None,
        )

        result = agent.process(event)
        assert len(result) == 1
        # record_decision_step should be called with None correlation_id
        # (memory.py will skip writing internally)
        memory.record_decision_step.assert_called_once()


# ── Executor decision step tests ──


class TestExecutorDecisionSteps:
    """Verify executor records decision steps after execution."""

    def test_paper_execute_records_step(self):
        memory = _make_memory()
        agent = ExecutorAgent(memory=memory, config=_make_config())

        proposal = TradeProposalPayload(
            symbol="AAPL", side="buy", quantity=10,
            entry_price=150.0, stop_loss=142.5, take_profit=165.0,
            strategy_id="test-strat", confidence=0.8,
        )
        event = Event(
            type=EventType.RISK_APPROVED,
            source="risk_guard",
            payload=proposal.model_dump(),
            correlation_id="corr-exec-001",
        )

        result = agent.process(event)

        assert len(result) == 1
        assert result[0].type == EventType.TRADE_EXECUTED
        memory.record_decision_step.assert_called_once()

        call_kwargs = memory.record_decision_step.call_args[1]
        assert call_kwargs["step_key"] == "executor"
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["strategy_id"] == "test-strat"
        reasoning = call_kwargs["reasoning"]
        assert reasoning["paper_trading"] is True
        assert reasoning["slippage_bps"] == 0.0
        assert reasoning["fill_price"] == 150.0
        assert reasoning["status"] == "filled"

    def test_paper_execute_continues_if_step_fails(self):
        memory = _make_memory()
        memory.record_decision_step.side_effect = Exception("DynamoDB error")
        agent = ExecutorAgent(memory=memory, config=_make_config())

        proposal = TradeProposalPayload(
            symbol="AAPL", side="buy", quantity=10,
            entry_price=150.0, stop_loss=142.5, take_profit=165.0,
            strategy_id="test-strat", confidence=0.8,
        )
        event = Event(
            type=EventType.RISK_APPROVED,
            source="risk_guard",
            payload=proposal.model_dump(),
            correlation_id="corr-exec-002",
        )

        result = agent.process(event)

        # Pipeline must continue despite decision step failure
        assert len(result) == 1
        assert result[0].type == EventType.TRADE_EXECUTED


# ── Chronicler decision step tests ──


class TestChroniclerDecisionSteps:
    """Verify chronicler records outcome steps and updates patterns."""

    def test_record_close_records_outcome_step(self):
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-001",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "entry_price": 150.0,
                "pnl": 100.0,
                "strategy_id": "test-strat",
            },
            correlation_id="corr-chron-001",
        )

        agent.process(event)

        # Should have called record_decision_step for the outcome
        step_calls = [
            c for c in memory.record_decision_step.call_args_list
            if c[1].get("step_key") == "chronicler:outcome"
        ]
        assert len(step_calls) == 1
        reasoning = step_calls[0][1]["reasoning"]
        assert reasoning["pnl"] == 100.0
        assert reasoning["exit_price"] == 160.0
        assert reasoning["trade_id"] == "trade-001"

    def test_record_close_updates_pattern_library(self):
        """If analyst step has fingerprint, pattern should be updated."""
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        memory.get_decision_chain.return_value = [
            {
                "step_key": "analyst",
                "reasoning": {
                    "fingerprint": "rsi=oversold|macd=bullish",
                    "confidence": 0.8,
                },
            }
        ]
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-002",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "pnl": 100.0,
                "strategy_id": "test-strat",
            },
            correlation_id="corr-chron-002",
        )

        agent.process(event)

        memory.update_pattern.assert_called_once_with(
            "AAPL", "rsi=oversold|macd=bullish", 100.0, 0.8
        )

    def test_record_close_no_pattern_update_without_fingerprint(self):
        """If analyst step has no fingerprint, no pattern update."""
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        memory.get_decision_chain.return_value = [
            {
                "step_key": "analyst",
                "reasoning": {"score": 0.5},  # no fingerprint key
            }
        ]
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-003",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "pnl": 50.0,
                "strategy_id": "test-strat",
            },
            correlation_id="corr-chron-003",
        )

        agent.process(event)

        memory.update_pattern.assert_not_called()

    def test_record_close_no_pattern_update_empty_chain(self):
        """If no analyst step in chain, no pattern update."""
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        memory.get_decision_chain.return_value = []  # empty chain
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-004",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "pnl": -20.0,
                "strategy_id": "test-strat",
            },
            correlation_id="corr-chron-004",
        )

        agent.process(event)

        memory.update_pattern.assert_not_called()

    def test_record_close_pattern_update_failure_does_not_block(self):
        """Pattern update failure must not block the pipeline."""
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        memory.get_decision_chain.side_effect = Exception("DynamoDB error")
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-005",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "pnl": 50.0,
                "strategy_id": "test-strat",
            },
            correlation_id="corr-chron-005",
        )

        # Should not raise
        agent.process(event)

    def test_record_close_without_correlation_id(self):
        """No pattern update when correlation_id is None."""
        memory = _make_memory()
        memory.get_strategy_trades.return_value = []
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        event = Event(
            type=EventType.POSITION_CLOSED,
            source="executor",
            payload={
                "trade_id": "trade-006",
                "symbol": "AAPL",
                "exit_price": 160.0,
                "pnl": 50.0,
                "strategy_id": "test-strat",
            },
            correlation_id=None,
        )

        agent.process(event)

        # get_decision_chain should NOT be called when correlation_id is None
        memory.get_decision_chain.assert_not_called()

    def test_heartbeat_includes_market_context(self):
        """Heartbeat snapshot should include market context."""
        memory = _make_memory()
        memory.get_open_trades.return_value = [
            {"entry_price": 150.0, "quantity": 10, "side": "buy", "symbol": "AAPL"},
            {"entry_price": 200.0, "quantity": 5, "side": "sell", "symbol": "MSFT"},
        ]
        memory.get_current_regime.return_value = "trending_bull"
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        agent.heartbeat()

        memory.record_snapshot.assert_called_once()
        call_kwargs = memory.record_snapshot.call_args[1]
        ctx = call_kwargs["market_context"]
        assert ctx["regime"] == "trending_bull"
        assert ctx["long_count"] == 1
        assert ctx["short_count"] == 1
        assert ctx["total_exposure_pct"] > 0

    def test_heartbeat_regime_failure_graceful(self):
        """If get_current_regime fails, market context still works."""
        memory = _make_memory()
        memory.get_open_trades.return_value = []
        memory.get_current_regime.side_effect = Exception("DynamoDB error")
        agent = ChroniclerAgent(memory=memory, config=_make_config())

        agent.heartbeat()

        # Should still produce snapshot
        memory.record_snapshot.assert_called_once()
        ctx = memory.record_snapshot.call_args[1]["market_context"]
        assert ctx["regime"] == ""  # default on failure


# ── Strategist decision step tests ──


class TestStrategistDecisionSteps:
    """Verify strategist records decision steps with Thompson scores and Kelly data."""

    def _make_strategies(self, n=2):
        strategies = []
        for i in range(n):
            s = Strategy(
                id=f"strat-{i}",
                name=f"Strategy {i}",
                strategy_type="momentum",
                genome=StrategyGenome.default_momentum(),
                fitness=FitnessScore(profit_factor=1.5, win_rate=0.6, composite=0.7),
                wins=10,
                losses=5,
                total_trades=15,
                total_pnl=500.0,
            )
            strategies.append(s)
        return strategies

    def _make_analysis_event(self, symbol="AAPL"):
        analysis = AnalysisPayload(
            symbol=symbol,
            technical={"close": 150.0, "atr_14": 3.0},
            recommendation="strong_buy",
            confidence=0.9,
            summary="Test analysis",
        )
        return Event(
            type=EventType.ANALYSIS_COMPLETE,
            source="analyst",
            payload=analysis.model_dump(),
            correlation_id="corr-strat-001",
        )

    def test_records_step_per_strategy(self):
        memory = _make_memory()
        strategies = self._make_strategies(2)
        stored = []
        for s in strategies:
            stored.append({
                "id": s.id,
                "name": s.name,
                "strategy_type": s.strategy_type,
                "genome": s.genome.model_dump(),
                "fitness": s.fitness.model_dump(),
                "wins": s.wins,
                "losses": s.losses,
                "total_trades": s.total_trades,
                "total_pnl": s.total_pnl,
                "generation": 0,
                "regime_preference": "",
                "kelly_fraction_mult": 0.5,
            })
        memory.get_active_strategies.return_value = stored

        agent = StrategistAgent(memory=memory, config=_make_config())
        event = self._make_analysis_event()

        result = agent.process(event)

        # Should have proposals and matching decision steps
        assert len(result) > 0
        for r in result:
            assert r.type == EventType.TRADE_PROPOSED

        # Each proposal should have a matching decision step call
        step_calls = memory.record_decision_step.call_args_list
        assert len(step_calls) == len(result)

        for step_call in step_calls:
            kwargs = step_call[1]
            assert kwargs["correlation_id"] == "corr-strat-001"
            assert kwargs["step_key"].startswith("strategist:")
            reasoning = kwargs["reasoning"]
            assert "thompson_score" in reasoning
            assert "all_thompson_scores" in reasoning
            assert "kelly" in reasoning
            assert "quantity" in reasoning
            assert "signal_confidence" in reasoning

    def test_pipeline_continues_if_step_fails(self):
        memory = _make_memory()
        strategies = self._make_strategies(1)
        stored = [{
            "id": strategies[0].id,
            "name": strategies[0].name,
            "strategy_type": strategies[0].strategy_type,
            "genome": strategies[0].genome.model_dump(),
            "fitness": strategies[0].fitness.model_dump(),
            "wins": strategies[0].wins,
            "losses": strategies[0].losses,
            "total_trades": strategies[0].total_trades,
            "total_pnl": strategies[0].total_pnl,
            "generation": 0,
            "regime_preference": "",
            "kelly_fraction_mult": 0.5,
        }]
        memory.get_active_strategies.return_value = stored
        memory.record_decision_step.side_effect = Exception("DynamoDB error")

        agent = StrategistAgent(memory=memory, config=_make_config())
        event = self._make_analysis_event()

        result = agent.process(event)

        # Pipeline must produce proposals despite decision step failure
        assert len(result) > 0
        assert result[0].type == EventType.TRADE_PROPOSED
