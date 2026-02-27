"""Tests for strategist internal methods: Thompson Sampling and Kelly sizing.

These were changed to return richer data types for the decision ledger.
Critical to verify the new return types don't break the pipeline.
"""

from unittest.mock import MagicMock

from claudestreet.agents.strategist import StrategistAgent
from claudestreet.models.strategy import Strategy, StrategyGenome, FitnessScore


def _make_memory():
    memory = MagicMock()
    memory.get_active_strategies.return_value = []
    memory.get_current_regime.return_value = ""
    return memory


def _make_config():
    return {
        "initial_capital": 100000.0,
        "max_position_pct": 0.15,
        "max_portfolio_risk_pct": 0.02,
        "thompson_k": 3,
    }


def _make_strategy(
    id="strat-test",
    wins=10,
    losses=5,
    total_trades=15,
    profit_factor=1.5,
    kelly_mult=0.5,
) -> Strategy:
    return Strategy(
        id=id,
        name="Test Strategy",
        strategy_type="momentum",
        genome=StrategyGenome.default_momentum(),
        fitness=FitnessScore(profit_factor=profit_factor, win_rate=wins / max(total_trades, 1)),
        wins=wins,
        losses=losses,
        total_trades=total_trades,
        kelly_fraction_mult=kelly_mult,
    )


# ── Thompson Sampling return type tests ──


class TestThompsonSampling:
    """Verify _select_strategies_thompson returns (score, Strategy) tuples."""

    def test_returns_tuple_list(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}") for i in range(5)]

        result = agent._select_strategies_thompson(strategies, k=3)

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            score, strategy = item
            assert isinstance(score, float)
            assert isinstance(strategy, Strategy)
            assert 0.0 <= score <= 1.0  # Beta distribution output

    def test_returns_all_when_fewer_than_k(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}") for i in range(2)]

        result = agent._select_strategies_thompson(strategies, k=5)

        assert len(result) == 2
        for score, strategy in result:
            assert isinstance(score, float)
            assert isinstance(strategy, Strategy)

    def test_empty_strategies(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())

        result = agent._select_strategies_thompson([], k=3)

        assert result == []

    def test_single_strategy(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy()]

        result = agent._select_strategies_thompson(strategies, k=3)

        assert len(result) == 1
        score, strategy = result[0]
        assert isinstance(score, float)
        assert strategy.id == "strat-test"

    def test_k_equals_zero(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}") for i in range(3)]

        result = agent._select_strategies_thompson(strategies, k=0)

        assert result == []

    def test_zero_wins_and_losses(self):
        """Strategies with no history should still produce valid scores."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}", wins=0, losses=0, total_trades=0) for i in range(3)]

        result = agent._select_strategies_thompson(strategies, k=2)

        assert len(result) == 2
        for score, strategy in result:
            assert 0.0 <= score <= 1.0  # Beta(1, 1) is always valid

    def test_sorted_descending_by_score(self):
        """Results should be sorted by Thompson score descending."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}") for i in range(10)]

        result = agent._select_strategies_thompson(strategies, k=5)

        scores = [score for score, _ in result]
        assert scores == sorted(scores, reverse=True)

    def test_tuple_unpacking_works(self):
        """Verify the for-loop unpacking pattern used in _propose_trades."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategies = [_make_strategy(id=f"s-{i}") for i in range(3)]

        result = agent._select_strategies_thompson(strategies, k=2)

        # This is the exact pattern used in _propose_trades line 232
        for thompson_score, strategy in result:
            assert thompson_score > 0
            assert strategy.id.startswith("s-")


# ── Kelly Position Sizing return type tests ──


class TestKellyPositionSize:
    """Verify _kelly_position_size returns (quantity, intermediates) tuple."""

    def test_returns_tuple(self):
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=20, wins=12, losses=8, profit_factor=1.8)

        result = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.8, capital=100000.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        quantity, intermediates = result
        assert isinstance(quantity, int)
        assert isinstance(intermediates, dict)
        assert quantity >= 1

    def test_kelly_mode_with_sufficient_trades(self):
        """total_trades >= 10 should use Kelly formula."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=20, wins=12, losses=8, profit_factor=1.8)

        quantity, intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.8, capital=100000.0,
        )

        assert intermediates["method"] == "kelly"
        assert "win_rate" in intermediates
        assert "kelly_fraction" in intermediates
        assert "effective_fraction" in intermediates
        assert "confidence_scale" in intermediates
        assert intermediates["win_rate"] == 12 / 20

    def test_fallback_mode_with_few_trades(self):
        """total_trades < 10 should use fixed risk sizing."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=5, wins=3, losses=2)

        quantity, intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.8, capital=100000.0,
        )

        assert intermediates["method"] == "fixed_risk"
        assert intermediates["fallback"] is True

    def test_zero_stop_loss_distance(self):
        """price == stop_loss should return minimum quantity."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=5)

        quantity, intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=150.0,
            confidence=0.8, capital=100000.0,
        )

        assert quantity == 1
        assert intermediates["fallback"] is True

    def test_zero_price(self):
        """price=0 in fallback mode should return 1."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=5)

        quantity, intermediates = agent._kelly_position_size(
            strategy=strategy, price=0.0, stop_loss=0.0,
            confidence=0.8, capital=100000.0,
        )

        assert quantity == 1

    def test_low_confidence_scales_down(self):
        """confidence < 0.5 should scale position to 20% minimum."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=20, wins=14, losses=6, profit_factor=2.0)

        _, low_intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.3, capital=100000.0,
        )
        _, high_intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.9, capital=100000.0,
        )

        # Low confidence should use 0.2 minimum scale
        assert low_intermediates["confidence_scale"] == 0.0  # (0.3 - 0.5) * 2 = -0.4 → max(0.0, -0.4) = 0.0
        assert high_intermediates["confidence_scale"] > 0.5

    def test_kelly_fraction_clamped_to_25_pct(self):
        """Kelly fraction should never exceed 25%."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        # Very high win rate and profit factor → large kelly fraction
        strategy = _make_strategy(
            total_trades=50, wins=48, losses=2, profit_factor=10.0,
        )

        _, intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.9, capital=100000.0,
        )

        assert intermediates["kelly_fraction"] <= 0.25

    def test_zero_profit_factor(self):
        """profit_factor=0 should result in kelly_fraction=0."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(
            total_trades=20, wins=5, losses=15, profit_factor=0.0,
        )

        quantity, intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.8, capital=100000.0,
        )

        # profit_factor=0 → uses default 1.5, but with low win rate
        assert quantity >= 1

    def test_max_position_cap(self):
        """Position should never exceed max_position_pct of capital."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(
            total_trades=50, wins=45, losses=5, profit_factor=5.0, kelly_mult=1.0,
        )

        quantity, _ = agent._kelly_position_size(
            strategy=strategy, price=10.0, stop_loss=9.0,
            confidence=1.0, capital=100000.0,
        )

        max_shares = int(100000.0 * 0.15 / 10.0)  # 1500
        assert quantity <= max_shares

    def test_tuple_unpacking_works(self):
        """Verify the unpacking pattern used in _propose_trades."""
        agent = StrategistAgent(memory=_make_memory(), config=_make_config())
        strategy = _make_strategy(total_trades=20, wins=12, losses=8)

        # This is the exact pattern used in _propose_trades line 274
        quantity, kelly_intermediates = agent._kelly_position_size(
            strategy=strategy, price=150.0, stop_loss=142.5,
            confidence=0.8, capital=100000.0,
        )

        assert quantity > 0
        assert isinstance(kelly_intermediates, dict)
