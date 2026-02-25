"""Population manager — the evolution engine.

Manages generations of trading strategies using a genetic algorithm:
1. Evaluate fitness of current generation
2. Select survivors (tournament selection)
3. Breed offspring via crossover
4. Mutate offspring
5. Form new generation from survivors + offspring
"""

from __future__ import annotations

import random

from claudestreet.models.strategy import Strategy, StrategyGenome, FitnessScore
from claudestreet.evolution.dna import DNAEncoder
from claudestreet.evolution.mutation import Mutator


class Population:
    """Genetic algorithm population manager for trading strategies."""

    def __init__(
        self,
        population_size: int = 20,
        survivors: int = 5,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.6,
        tournament_size: int = 3,
    ) -> None:
        self.population_size = population_size
        self.survivors = survivors
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.mutator = Mutator(mutation_rate=mutation_rate)

    def seed(self) -> list[Strategy]:
        """Create an initial random population."""
        strategies: list[Strategy] = []

        for i in range(self.population_size):
            if i % 2 == 0:
                genome = StrategyGenome.default_momentum()
                stype = "momentum"
            else:
                genome = StrategyGenome.default_mean_reversion()
                stype = "mean_reversion"

            # Randomize the genome slightly for diversity
            genome = self.mutator.mutate(genome)

            strategy = Strategy(
                name=f"{stype}-gen0-{i}",
                strategy_type=stype,
                genome=genome,
                generation=0,
            )
            strategies.append(strategy)

        return strategies

    def evolve(
        self, current_gen: list[Strategy], generation_number: int
    ) -> list[Strategy]:
        """Run one evolution cycle. Returns the new generation."""
        if not current_gen:
            return self.seed()

        # Sort by composite fitness (descending)
        ranked = sorted(
            current_gen,
            key=lambda s: s.fitness.composite,
            reverse=True,
        )

        # Select survivors (elitism — top N pass through unchanged)
        elite = ranked[: self.survivors]
        new_gen: list[Strategy] = []
        for s in elite:
            survivor = s.model_copy(deep=True)
            survivor.generation = generation_number
            new_gen.append(survivor)

        # Calculate diversity for adaptive mutation
        genomes = [s.genome for s in ranked if s.fitness.total_trades > 0]
        diversity = Mutator.population_diversity(genomes) if genomes else 1.0

        # Breed offspring to fill remaining slots
        offspring_needed = self.population_size - len(new_gen)
        for i in range(offspring_needed):
            parent_a = self._tournament_select(ranked)
            parent_b = self._tournament_select(ranked)

            # Crossover
            if random.random() < self.crossover_rate:
                child_genome = DNAEncoder.crossover(parent_a.genome, parent_b.genome)
                parent_ids = [parent_a.id, parent_b.id]
            else:
                child_genome = DNAEncoder.clone(parent_a.genome)
                parent_ids = [parent_a.id]

            # Mutation (adaptive)
            child_genome = self.mutator.adaptive_mutate(child_genome, diversity)

            child = Strategy(
                name=f"{parent_a.strategy_type}-gen{generation_number}-{i}",
                strategy_type=parent_a.strategy_type,
                genome=child_genome,
                generation=generation_number,
                parent_ids=parent_ids,
                fitness=FitnessScore(),
            )
            new_gen.append(child)

        return new_gen

    def _tournament_select(self, ranked: list[Strategy]) -> Strategy:
        """Tournament selection: pick best from random subset."""
        tournament = random.sample(
            ranked, min(self.tournament_size, len(ranked))
        )
        return max(tournament, key=lambda s: s.fitness.composite)
