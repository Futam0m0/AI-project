"""Phase 5.4 — Genetic Algorithm over 10 behaviour weights.

From-scratch implementation (no external libraries) that maps one-to-one onto
the course notebook GA pseudocode:

    1. Initialise a POPULATION of n random chromosomes in [0,1]^10
    2. Evaluate FITNESS = objective(candidate).
    3. ELITISM (10% of best parents copied verbatim into next generation)
    4. TOURNAMENT SELECTION (size 3) — pick two parents from fittest 1/3
       of 3 random individuals each.
    5. SINGLE-POINT CROSSOVER (75% chance) at a random column.  If no
       crossover, children are verbatim parents.
    6. UNIFORM MUTATION (1% per gene) — add a Gaussian-ish small delta
       re-clamped into [0,1].
    7. Repeat until n_generations.  Return best chromosome ever seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from algorithms.baseline_decision import BehaviourWeights
from learning.sa import default_objective_avg_party_hp


@dataclass
class GAResult:
    best_chromosome: list[float]
    best_score: float
    best_history: list[float] = field(default_factory=list)
    mean_history: list[float] = field(default_factory=list)
    n_generations: int = 0
    pop_size: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_lcg(seed: int) -> Callable[[], float]:
    state = int(seed) | 1

    def _rand() -> float:
        nonlocal state
        state = (state * 1_664_525 + 1_013_904_223) & 0x7FFFFFFF
        return state / float(0x7FFFFFFF)
    return _rand


def _random_chromosome(rand: Callable[[], float]) -> list[float]:
    return [rand() for _ in range(10)]


def _tournament_select(
    population: list[list[float]],
    fitnesses: list[float],
    k: int,
    rand: Callable[[], float],
) -> list[float]:
    n = len(population)
    best_idx = int(rand() * n) % n
    best_fit = fitnesses[best_idx]
    for _ in range(k - 1):
        idx = int(rand() * n) % n
        if fitnesses[idx] > best_fit:
            best_idx, best_fit = idx, fitnesses[idx]
    return list(population[best_idx])


def _crossover(
    mother: list[float],
    father: list[float],
    rand: Callable[[], float],
    *,
    prob: float = 0.75,
) -> tuple[list[float], list[float]]:
    if rand() > prob:
        return list(mother), list(father)
    n = len(mother)
    point = max(1, int(rand() * (n - 1))) if n > 1 else 0
    girl = mother[:point] + father[point:]
    boy = father[:point] + mother[point:]
    return girl, boy


def _mutate(
    chromo: list[float],
    rand: Callable[[], float],
    *,
    per_gene_prob: float = 0.01,
    step_std: float = 0.12,
) -> list[float]:
    out = []
    for v in chromo:
        if rand() < per_gene_prob:
            delta = (rand() * 2.0 - 1.0) * step_std
            v = v + delta
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def genetic_algorithm(
    objective_fn: Callable[[Sequence[float]], float] | None = None,
    *,
    pop_size: int = 20,
    n_generations: int = 12,
    elitism_frac: float = 0.1,
    tournament_k: int = 3,
    crossover_prob: float = 0.75,
    mutation_prob: float = 0.01,
    start_chromosome: Sequence[float] | None = None,
    seed: int = 2026,
) -> GAResult:
    objective_fn = objective_fn or default_objective_avg_party_hp
    rand = _make_lcg(seed)

    # Initial population ------------------------------------------------
    population: list[list[float]] = [_random_chromosome(rand) for _ in range(pop_size)]
    if start_chromosome is not None:
        if len(start_chromosome) != 10:
            raise ValueError("start_chromosome must have 10 elements")
        # Seed the population with one expert chromosome (the baseline default)
        population[0] = [min(1.0, max(0.0, v)) for v in start_chromosome]

    fitnesses = [float(objective_fn(p)) for p in population]
    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    best_chromosome = list(population[best_idx])
    best_score = fitnesses[best_idx]
    best_history = [best_score]
    mean_history = [sum(fitnesses) / float(len(fitnesses))]

    n_elite = max(0, min(pop_size - 2, int(round(pop_size * elitism_frac))))

    for gen in range(n_generations):
        # 1. Elitism
        sorted_idxs = sorted(range(pop_size), key=lambda i: -fitnesses[i])
        elites = [list(population[i]) for i in sorted_idxs[:n_elite]]

        # 2. Build children until we hit pop_size (minus elites)
        children: list[list[float]] = []
        while len(children) + n_elite < pop_size:
            mom = _tournament_select(population, fitnesses, tournament_k, rand)
            dad = _tournament_select(population, fitnesses, tournament_k, rand)
            g1, g2 = _crossover(mom, dad, rand, prob=crossover_prob)
            g1 = _mutate(g1, rand, per_gene_prob=mutation_prob)
            g2 = _mutate(g2, rand, per_gene_prob=mutation_prob)
            children.append(g1)
            if len(children) + n_elite < pop_size:
                children.append(g2)
        # Trim if we overshot (shouldn't happen, safety)
        children = children[: max(0, pop_size - n_elite)]
        population = elites + children
        # 3. Evaluate
        fitnesses = [float(objective_fn(p)) for p in population]
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[best_idx] > best_score:
            best_score = fitnesses[best_idx]
            best_chromosome = list(population[best_idx])
        best_history.append(best_score)
        mean_history.append(sum(fitnesses) / float(len(fitnesses)))

    return GAResult(
        best_chromosome=best_chromosome,
        best_score=best_score,
        best_history=best_history,
        mean_history=mean_history,
        n_generations=n_generations,
        pop_size=pop_size,
    )
