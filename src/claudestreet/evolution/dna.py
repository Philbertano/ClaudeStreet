"""DNA encoder — serializes strategy genomes for crossover and mutation.

Each strategy's genome is a dict of named genes with bounded numeric
values. The DNA encoder provides utilities for genome manipulation
that the Mutator and Population classes depend on.
"""

from __future__ import annotations

import random

from claudestreet.models.strategy import StrategyGenome, StrategyGene


class DNAEncoder:
    """Utilities for manipulating strategy genomes."""

    @staticmethod
    def crossover(
        parent_a: StrategyGenome,
        parent_b: StrategyGenome,
    ) -> StrategyGenome:
        """Uniform crossover: each gene randomly chosen from either parent.

        Only crosses genes that exist in both parents. Genes unique to
        one parent are inherited from that parent.
        """
        child_genes: dict[str, StrategyGene] = {}

        all_gene_names = set(parent_a.genes.keys()) | set(parent_b.genes.keys())

        for name in all_gene_names:
            gene_a = parent_a.genes.get(name)
            gene_b = parent_b.genes.get(name)

            if gene_a and gene_b:
                # Both parents have this gene — pick randomly
                source = random.choice([gene_a, gene_b])
                child_genes[name] = source.model_copy(deep=True)
            elif gene_a:
                child_genes[name] = gene_a.model_copy(deep=True)
            elif gene_b:
                child_genes[name] = gene_b.model_copy(deep=True)

        return StrategyGenome(genes=child_genes)

    @staticmethod
    def blend_crossover(
        parent_a: StrategyGenome,
        parent_b: StrategyGenome,
        alpha: float = 0.5,
    ) -> StrategyGenome:
        """Blend crossover: child gene values are weighted averages."""
        child_genes: dict[str, StrategyGene] = {}
        all_gene_names = set(parent_a.genes.keys()) | set(parent_b.genes.keys())

        for name in all_gene_names:
            gene_a = parent_a.genes.get(name)
            gene_b = parent_b.genes.get(name)

            if gene_a and gene_b:
                blended_value = alpha * gene_a.value + (1 - alpha) * gene_b.value
                child_gene = gene_a.model_copy(deep=True)
                child_gene.value = blended_value
                child_gene.clamp()
                child_genes[name] = child_gene
            elif gene_a:
                child_genes[name] = gene_a.model_copy(deep=True)
            elif gene_b:
                child_genes[name] = gene_b.model_copy(deep=True)

        return StrategyGenome(genes=child_genes)

    @staticmethod
    def distance(a: StrategyGenome, b: StrategyGenome) -> float:
        """Normalized Euclidean distance between two genomes.

        Used to maintain genetic diversity in the population.
        """
        common_genes = set(a.genes.keys()) & set(b.genes.keys())
        if not common_genes:
            return 1.0

        sum_sq = 0.0
        for name in common_genes:
            gene_a = a.genes[name]
            gene_b = b.genes[name]
            range_val = gene_a.max_val - gene_a.min_val
            if range_val > 0:
                norm_diff = (gene_a.value - gene_b.value) / range_val
                sum_sq += norm_diff ** 2

        return (sum_sq / len(common_genes)) ** 0.5

    @staticmethod
    def clone(genome: StrategyGenome) -> StrategyGenome:
        return genome.model_copy(deep=True)
