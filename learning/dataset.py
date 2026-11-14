"""Phase 5.1 — combat dataset generator.

The goal of this module is purely to expose a deterministic, reproducible
generator that produces a *design matrix* ``X`` (behaviour weights per row)
and a *response vector* ``y`` (expected damage per round, a single float
per row) that the gradient-descent linear regressor in :mod:`learning.gd`
can fit.

The generator samples 10-weight chromosomes (see
:class:`algorithms.baseline_decision.BehaviourWeights`) and evaluates them
by running a short, deterministic simulated combat using the game engine.
The output per sample row is ``(w_0, w_1, ..., w_9, expected_damage)``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable

from algorithms.baseline_decision import BehaviourWeights, weighted_decision_fn
from game.game_state import GameState


@dataclass(frozen=True)
class CombatDataset:
    """A small, deterministic regression dataset for behaviour-weight tuning."""

    feature_names: list[str]
    X: list[list[float]]      # n × 10  (each row is a chromosome)
    y: list[float]            # n × 1   (response: expected damage/round)
    seed: int
    rows_per_observation: int = 5
    rounds_per_battle: int = 12

    def __len__(self) -> int:
        return len(self.y)

    def save_csv(self, csv_path: str) -> None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.feature_names + ["expected_damage_per_round"])
            for row_x, row_y in zip(self.X, self.y):
                writer.writerow([f"{v:.6f}" for v in row_x] + [f"{row_y:.4f}"])

    @classmethod
    def load_csv(cls, csv_path: str) -> "CombatDataset":
        X: list[list[float]] = []
        y: list[float] = []
        feature_names: list[str] = []
        with open(csv_path, "r", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            feature_names = [c for c in header if c != "expected_damage_per_round"]
            for row in reader:
                if not row:
                    continue
                X.append([float(v) for v in row[:len(feature_names)]])
                y.append(float(row[-1]))
        return cls(feature_names=feature_names, X=X, y=y, seed=0)


# 10 standard feature names matching the BehaviourWeights chromosome layout:
FEATURE_NAMES = [
    "w0_warrior_melee_range",
    "w1_warrior_charge_willingness",
    "w2_healer_heal_threshold",
    "w3_healer_heal_range",
    "w4_healer_stickiness",
    "w5_scout_melee_range",
    "w6_scout_exit_priority",
    "w7_enemy_charge_willingness",
    "w8_defend_when_stuck_prob",
    "w9_panic_at_low_hp",
]


def _evaluate_once(weights: BehaviourWeights, seed: int, rounds: int) -> float:
    """Run a single battle and return (damage dealt to enemies) / rounds."""
    gs = GameState.new_default(seed=seed)
    dfn = weighted_decision_fn(weights)
    dmg_before = sum(getattr(e, "max_hp", 0) - e.hp for e in gs.enemies if getattr(e, "max_hp", None))
    total_dealt_accumulator = 0.0
    # Estimate by (max_hp - current_hp) delta summed across enemies per round.
    for _ in range(rounds):
        prev = sum(max(0, getattr(e, "max_hp", 0) - e.hp) for e in gs.enemies
                   if getattr(e, "max_hp", None))
        if gs.winner is not None:
            break
        gs.advance_one_round(decision_fn=dfn)
        after = sum(max(0, getattr(e, "max_hp", 0) - e.hp) for e in gs.enemies
                    if getattr(e, "max_hp", None))
        total_dealt_accumulator += max(0.0, after - prev)
    return total_dealt_accumulator / float(max(1, rounds))


def generate_dataset(
    n_rows: int = 40,
    seed: int = 2026,
    rounds_per_battle: int = 12,
    rows_per_observation: int = 3,
    evaluator: Callable[[BehaviourWeights, int, int], float] | None = None,
) -> CombatDataset:
    """Generate a small regression dataset.

    Pseudo-random chromosomes are produced with a tiny deterministic LCG so
    results are 100% reproducible across any Python runtime (do not depend
    on :mod:`random` global seeding state — the generator owns its own).
    """
    import math
    state = seed * 1_664_525 + 1_013_904_223

    def rand() -> float:
        nonlocal state
        state = (state * 1_103_515_245 + 12_345) & 0x7FFFFFFF
        return state / float(0x7FFFFFFF)

    evaluator = evaluator or _evaluate_once

    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n_rows):
        chromo = [rand() for _ in range(10)]
        # Clamp to [0,1] — LCG is already in range; sanity guard.
        chromo = [min(1.0, max(0.0, c)) for c in chromo]
        weights = BehaviourWeights.from_chromosome(chromo)
        # Take a tiny average over a couple of offset seeds to reduce
        # single-sample noise without making generation slow.
        agg = 0.0
        for j in range(rows_per_observation):
            agg += evaluator(weights, seed ^ (j * 0x5555), rounds_per_battle)
        response = agg / float(max(1, rows_per_observation))
        X.append(chromo)
        y.append(response)

    return CombatDataset(feature_names=FEATURE_NAMES, X=X, y=y, seed=seed,
                         rows_per_observation=rows_per_observation,
                         rounds_per_battle=rounds_per_battle)
