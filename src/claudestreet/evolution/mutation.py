"""Mutator — genetic operators for strategy evolution.

Provides several mutation strategies:
- Gaussian: small perturbations around current value
- Uniform: random jump within gene bounds
- Creep: gradual directional drift
- Reset: occasional full randomization of a gene

Adaptive mutation rate increases when population diversity drops,
preventing premature convergence.
"""

from __future__ import annotations

import random

from claudestreet.models.strategy import StrategyGenome, StrategyGene


class Mutator:
    """Genetic mutation operators for strategy genomes."""

    def __init__(
        self,
        mutation_rate: float = 0.15,
        gaussian_sigma: float = 0.1,
        reset_probability: float = 0.02,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.gaussian_sigma = gaussian_sigma
        self.reset_probability = reset_probability

    def mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Apply mutation to a genome. Returns a new mutated copy."""
        mutated = genome.model_copy(deep=True)

        for name, gene in mutated.genes.items():
            if random.random() > self.mutation_rate:
                continue

            # Choose mutation type
            roll = random.random()
            if roll < self.reset_probability:
                # Full reset — random value within bounds
                gene.value = random.uniform(gene.min_val, gene.max_val)
            elif roll < 0.5:
                # Gaussian mutation — small perturbation
                gene_range = gene.max_val - gene.min_val
                sigma = gene_range * self.gaussian_sigma
                gene.value += random.gauss(0, sigma)
            else:
                # Creep mutation — step in gene's native resolution
                direction = random.choice([-1, 1])
                steps = random.randint(1, 3)
                gene.value += direction * gene.step * steps

            # Snap to step resolution and clamp
            if gene.step >= 1.0:
                gene.value = round(gene.value)
            gene.clamp()

        return mutated

    def adaptive_mutate(
        self,
        genome: StrategyGenome,
        population_diversity: float,
        diversity_threshold: float = 0.3,
    ) -> StrategyGenome:
        """Mutation with adaptive rate based on population diversity.

        When diversity drops below threshold, mutation rate increases
        to inject more variation and prevent convergence.
        """
        if population_diversity < diversity_threshold:
            boost = (diversity_threshold - population_diversity) / diversity_threshold
            effective_rate = min(self.mutation_rate * (1 + boost * 2), 0.8)
        else:
            effective_rate = self.mutation_rate

        original_rate = self.mutation_rate
        self.mutation_rate = effective_rate
        result = self.mutate(genome)
        self.mutation_rate = original_rate
        return result

    @staticmethod
    def population_diversity(genomes: list[StrategyGenome]) -> float:
        """Measure average pairwise distance between genomes (0..1)."""
        if len(genomes) < 2:
            return 1.0

        from claudestreet.evolution.dna import DNAEncoder

        total_dist = 0.0
        pairs = 0
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                total_dist += DNAEncoder.distance(genomes[i], genomes[j])
                pairs += 1

        return total_dist / pairs if pairs > 0 else 0.0
