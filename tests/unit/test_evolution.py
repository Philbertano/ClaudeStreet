"""Tests for the genetic evolution system."""

from claudestreet.evolution.dna import DNAEncoder
from claudestreet.evolution.fitness import FitnessEvaluator
from claudestreet.evolution.mutation import Mutator
from claudestreet.evolution.population import Population
from claudestreet.models.strategy import Strategy, StrategyGenome


def test_default_genome_has_expected_genes():
    genome = StrategyGenome.default_momentum()
    params = genome.to_params()
    assert "rsi_period" in params
    assert "macd_fast" in params
    assert "stop_loss_pct" in params
    assert params["rsi_period"] == 14


def test_gene_clamp_respects_bounds():
    genome = StrategyGenome.default_momentum()
    genome.set("rsi_period", 999)
    assert genome.get("rsi_period") == 50  # max_val


def test_crossover_produces_valid_genome():
    parent_a = StrategyGenome.default_momentum()
    parent_b = StrategyGenome.default_momentum()
    parent_b.set("rsi_period", 20)

    child = DNAEncoder.crossover(parent_a, parent_b)
    assert "rsi_period" in child.genes
    assert child.genes["rsi_period"].min_val <= child.get("rsi_period") <= child.genes["rsi_period"].max_val


def test_blend_crossover_averages():
    parent_a = StrategyGenome.default_momentum()
    parent_b = StrategyGenome.default_momentum()
    parent_a.set("rsi_period", 10)
    parent_b.set("rsi_period", 30)

    child = DNAEncoder.blend_crossover(parent_a, parent_b, alpha=0.5)
    assert child.get("rsi_period") == 20


def test_mutation_stays_in_bounds():
    mutator = Mutator(mutation_rate=1.0)  # mutate every gene
    genome = StrategyGenome.default_momentum()

    for _ in range(100):
        mutated = mutator.mutate(genome)
        for name, gene in mutated.genes.items():
            assert gene.min_val <= gene.value <= gene.max_val, (
                f"{name}: {gene.value} not in [{gene.min_val}, {gene.max_val}]"
            )


def test_fitness_evaluator_positive_trades():
    evaluator = FitnessEvaluator()
    trades = [{"pnl": 100.0}, {"pnl": 50.0}, {"pnl": -30.0}, {"pnl": 80.0}]
    score = evaluator.evaluate(trades)

    assert score.total_return == 200.0
    assert score.total_trades == 4
    assert score.win_rate == 0.75
    assert score.profit_factor > 1.0
    assert score.composite > 0.0


def test_fitness_evaluator_empty():
    evaluator = FitnessEvaluator()
    score = evaluator.evaluate([])
    assert score.total_trades == 0
    assert score.composite == 0.0


def test_population_seed():
    pop = Population(population_size=10)
    strategies = pop.seed()
    assert len(strategies) == 10
    assert all(isinstance(s, Strategy) for s in strategies)
    assert any(s.strategy_type == "momentum" for s in strategies)
    assert any(s.strategy_type == "mean_reversion" for s in strategies)


def test_population_evolve():
    pop = Population(population_size=10, survivors=3)
    strategies = pop.seed()

    # Give some strategies fitness
    for i, s in enumerate(strategies):
        s.fitness.composite = i * 0.1
        s.fitness.total_trades = 15

    new_gen = pop.evolve(strategies, generation_number=1)
    assert len(new_gen) == 10
    assert all(s.generation == 1 for s in new_gen)


def test_dna_distance():
    a = StrategyGenome.default_momentum()
    b = StrategyGenome.default_momentum()
    assert DNAEncoder.distance(a, b) == 0.0

    b.set("rsi_period", 50)  # max
    assert DNAEncoder.distance(a, b) > 0.0


def test_population_diversity():
    genomes = [StrategyGenome.default_momentum() for _ in range(5)]
    # Identical genomes → zero diversity
    assert Mutator.population_diversity(genomes) == 0.0

    # Mutate them → positive diversity
    mutator = Mutator(mutation_rate=1.0)
    diverse = [mutator.mutate(g) for g in genomes]
    assert Mutator.population_diversity(diverse) > 0.0
