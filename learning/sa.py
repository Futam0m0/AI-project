"""Phase 5.3 — Simulated Annealing over a 10-weight behaviour chromosome.

Annealing is a meta-heuristic: we start at a candidate chromosome
``w ∈ [0,1]^10`` (a :class:`BehaviourWeights` flattened to a list), define a
neighbourhood ``neighbour(w)`` that tweaks one weight by a small random delta
clamped into [0,1], and define an objective score ``f(w)``.  At each step we
accept uphill moves with probability

    exp(-Δ / T)

where ``Δ = f(candidate) - f(current) ≥ 0`` and ``T`` decays every step by a
geometric schedule ``T_{t+1} = 0.99 * T_t``.

The objective ``f(w)`` here is the *average party HP remaining after a short
battle* — because higher party HP is better.  SA thus maximises the score.
(If a caller wants to *minimise* something, simply negate the objective passed
in via ``objective_fn``.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from algorithms.baseline_decision import BehaviourWeights
from algorithms.baseline_decision import weighted_decision_fn
from game.game_state import GameState


# ---------------------------------------------------------------------------
# Default objective evaluated by SA / GA if caller provides none.
# ---------------------------------------------------------------------------

def default_objective_avg_party_hp(
    chromosome: Sequence[float],
    seed: int = 2026,
    rounds: int = 15,
    seeds_per_eval: int = 2,
) -> float:
    """Objective: maximise average party HP remaining across *seeds_per_eval*
    short battles (higher is better).
    """
    weights = BehaviourWeights.from_chromosome(list(chromosome))
    dfn = weighted_decision_fn(weights)
    total_score = 0.0
    for i in range(max(1, seeds_per_eval)):
        s = seed ^ ((i + 1) * 0x9E3779B1)
        gs = GameState.new_default(seed=s)
        for _ in range(rounds):
            if gs.winner is not None:
                break
            gs.advance_one_round(decision_fn=dfn)
        hp_remaining = sum(max(0, p.hp) for p in gs.party)
        total_score += hp_remaining
    # Add a tiny bonus for winning (+10 if party won, -5 if lost) to speed
    # convergence:
    s2 = seed ^ 0x55555555
    g2 = GameState.new_default(seed=s2)
    for _ in range(rounds * 2):
        if g2.winner is not None:
            break
        g2.advance_one_round(decision_fn=dfn)
    if g2.winner == "party":
        total_score += 20.0
    elif g2.winner == "enemies":
        total_score -= 5.0
    return total_score / float(max(1, seeds_per_eval))


@dataclass
class SAResult:
    best_chromosome: list[float]
    best_score: float
    final_score: float
    history: list[float] = field(default_factory=list)
    temp_history: list[float] = field(default_factory=list)
    n_steps: int = 0


def _neighbour(
    w: list[float],
    step_std: float,
    prng: Callable[[], float],
) -> list[float]:
    """Produce a neighbouring chromosome by nudging one random weight."""
    n = len(w)
    idx = int(prng() * n) % n
    delta = (prng() * 2.0 - 1.0) * step_std
    new = list(w)
    v = new[idx] + delta
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    new[idx] = v
    return new


def simulated_annealing(
    objective_fn: Callable[[Sequence[float]], float] | None = None,
    *,
    start_chromosome: Sequence[float] | None = None,
    n_steps: int = 80,
    start_temp: float = 4.0,
    cooling: float = 0.96,
    step_std: float = 0.15,
    seed: int = 2026,
) -> SAResult:
    """Run Simulated Annealing over a 10-element chromosome in [0, 1]^10.

    The function returns the best chromosome visited (not the final one),
    so annealing can "drift back downhill" after finding a good spot and
    still report its peak candidate.
    """
    objective_fn = objective_fn or default_objective_avg_party_hp

    # Small deterministic PRNG (LCG) — so SA runs byte-for-byte reproducible
    # on any runtime, independent of random module state:
    _state = seed | 1

    def prng() -> float:
        nonlocal _state
        _state = (_state * 1_664_525 + 1_013_904_223) & 0x7FFFFFFF
        return _state / float(0x7FFFFFFF)

    if start_chromosome is None:
        start = BehaviourWeights.default().to_chromosome()
    else:
        if len(start_chromosome) != 10:
            raise ValueError("start_chromosome must have 10 elements")
        start = [min(1.0, max(0.0, v)) for v in start_chromosome]

    current = start
    current_score = float(objective_fn(start))
    best = list(current)
    best_score = current_score
    history = [current_score]
    temp_history = [start_temp]

    temp = float(start_temp)
    step = 0
    for step in range(n_steps):
        cand = _neighbour(current, step_std, prng)
        cand_score = float(objective_fn(cand))
        delta = cand_score - current_score
        if delta > 0:
            accept = True
        else:
            # acceptance prob for uphill (here cand < current → uphill in -score
            # terms is "worse"; use standard Boltzmann.)
            if temp <= 1e-9:
                accept = False
            else:
                accept = prng() < math.exp(delta / temp)
        if accept:
            current, current_score = cand, cand_score
        if current_score > best_score:
            best, best_score = list(current), current_score
        temp *= cooling
        history.append(current_score)
        temp_history.append(temp)

    return SAResult(
        best_chromosome=best,
        best_score=best_score,
        final_score=current_score,
        history=history,
        temp_history=temp_history,
        n_steps=step + 1,
    )
