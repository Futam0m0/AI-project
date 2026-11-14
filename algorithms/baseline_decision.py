"""
Baseline companion decision AI.

No course algorithms are implemented in this module — it's the SIMPLEST scoring
system that still makes autonomous, sensible decisions.  All 6 search +
adversarial course algorithms will improve upon this baseline starting in
Phase 4.

Decision rules (per role):

    WARRIOR (role = 'W')
      1. If an enemy is within Chebyshev distance 1 → Attack that enemy.
      2. Else → Move one tile towards the nearest enemy (greedy 4-way).
      3. If no enemies exist → Move towards EXIT.
      4. If stuck → DEFEND (skip + defend flag).

    HEALER (role = 'H')
      1. If an ally has HP < 70% AND is within heal_range (1 tile) → Heal that ally.
      2. Else if an enemy is within 1 tile → Attack that enemy.
      3. Else → Move adjacent to the lowest-HP ally (to be ready to heal).
      4. If no injured ally → Move towards EXIT.
      5. If stuck → DEFEND.

    SCOUT (role = 'S')
      1. If enemy is within 1 → Attack weakest enemy (lowest HP).
      2. Else → Move towards EXIT.
      3. If stuck → move towards nearest enemy.
      4. Last → DEFEND.

For enemies (G, R, B): the simplest rule: move 1 step towards nearest PARTY, then attack if within 1.

All decisions return an (action, target_entity, step_direction, algorithm_name, reason_string)
tuple, which the GameState then applies and logs.  algorithm_name is always
'baseline' here so the dashboard clearly distinguishes baseline from the
course algorithms (BFS / A* / Minimax / ...) we plug in later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from game.actions import Action
from game.characters import BaseCharacter
from game.combat import step_away_from, step_towards
from game.dungeon import (Dungeon, ENEMY_SYMBOLS, EXIT, PARTY_SYMBOLS,
                          Position, SYMBOL_NAMES)


ALGORITHM_NAME = "baseline"


@dataclass
class BehaviourWeights:
    """Ten tunable (0..1-normalised) weights parameterising baseline behaviour.

    Each weight maps onto an exact, observable decision in the rule engine
    (see :func:`weighted_decision_fn`).  Normalisation range is [0, 1]; the
    engine internally rescales each into the natural domain of the parameter
    it controls (integer tile distances, fractional HP thresholds, etc.).
    """

    w0_warrior_melee_range: float = 0.25   # 0.25*10 ≈ 2.5 → int 2 tiles melee
    w1_warrior_charge_willingness: float = 0.9  # how eager to charge (vs exit)
    w2_healer_heal_threshold: float = 5 / 7  # rescales to 0.2 + (5/7)*0.7 = 0.7 (70% threshold)
    w3_healer_heal_range: float = 0.15     # 0.15*8 ≈ 1.2 → int 1 tile (standard)
    w4_healer_stickiness: float = 0.8      # how eager to stay adjacent to injured
    w5_scout_melee_range: float = 0.25     # like warrior
    w6_scout_exit_priority: float = 0.95   # 0.95 = exit is #1 priority (scout lore)
    w7_enemy_charge_willingness: float = 1.0  # enemies always chase party
    w8_defend_when_stuck_prob: float = 0.9 # 0.9 defend, 0.1 skip if blocked
    w9_panic_at_low_hp: float = 0.2        # if HP <= 20%, retreat instead of attack
    # Convenience fields ------------------------------------------------------
    label: str = "default"                 # short identifier (for logs / UI)

    def rescale(self) -> dict[str, float]:
        """Return each weight interpreted into the parameter's natural domain."""
        def _clip01(v: float) -> float:
            return 0.0 if v < 0 else 1.0 if v > 1 else v
        return dict(
            warrior_melee_range=int(_clip01(self.w0_warrior_melee_range) * 3) + 1,
            warrior_charge_willingness=_clip01(self.w1_warrior_charge_willingness),
            healer_heal_threshold=0.2 + _clip01(self.w2_healer_heal_threshold) * 0.7,
            healer_heal_range=int(_clip01(self.w3_healer_heal_range) * 5) + 1,
            healer_stickiness=_clip01(self.w4_healer_stickiness),
            scout_melee_range=int(_clip01(self.w5_scout_melee_range) * 3) + 1,
            scout_exit_priority=_clip01(self.w6_scout_exit_priority),
            enemy_charge_willingness=_clip01(self.w7_enemy_charge_willingness),
            defend_when_stuck=_clip01(self.w8_defend_when_stuck_prob),
            panic_at_low_hp=_clip01(self.w9_panic_at_low_hp),
            label=self.label,
        )

    @classmethod
    def default(cls) -> "BehaviourWeights":
        return cls()

    @classmethod
    def from_chromosome(cls, weights: list[float], label: str = "candidate") -> "BehaviourWeights":
        """Build a weights object from a flat 10-element vector."""
        if len(weights) != 10:
            raise ValueError(f"chromosome must have length 10 (got {len(weights)})")
        return cls(
            w0_warrior_melee_range=float(weights[0]),
            w1_warrior_charge_willingness=float(weights[1]),
            w2_healer_heal_threshold=float(weights[2]),
            w3_healer_heal_range=float(weights[3]),
            w4_healer_stickiness=float(weights[4]),
            w5_scout_melee_range=float(weights[5]),
            w6_scout_exit_priority=float(weights[6]),
            w7_enemy_charge_willingness=float(weights[7]),
            w8_defend_when_stuck_prob=float(weights[8]),
            w9_panic_at_low_hp=float(weights[9]),
            label=label,
        )

    def to_chromosome(self) -> list[float]:
        return [
            self.w0_warrior_melee_range,
            self.w1_warrior_charge_willingness,
            self.w2_healer_heal_threshold,
            self.w3_healer_heal_range,
            self.w4_healer_stickiness,
            self.w5_scout_melee_range,
            self.w6_scout_exit_priority,
            self.w7_enemy_charge_willingness,
            self.w8_defend_when_stuck_prob,
            self.w9_panic_at_low_hp,
        ]

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("label")}


DEFAULT_WEIGHTS = BehaviourWeights.default()


def _chebyshev(a: Position, b: Position) -> int:
    return max(abs(a.row - b.row), abs(a.col - b.col))


def _is_passable_factory(dungeon: Dungeon) -> Callable[[int, int], bool]:
    def can_pass(r: int, c: int) -> bool:
        # For movement destination check:
        #   passable = in bounds, not wall, not occupied by another entity.
        if not dungeon.is_passable(r, c):
            return False
        cell = dungeon.get(r, c)
        return cell in {"E", "."}
    return can_pass


def _find_lowest_hp_ally(party: list[BaseCharacter], exclude: BaseCharacter) -> BaseCharacter | None:
    alives = [a for a in party if a.alive and a is not exclude]
    if not alives:
        return None
    return min(alives, key=lambda a: (a.hp_fraction, a.name))


def _find_weakest_enemy(enemies: list[BaseCharacter]) -> BaseCharacter | None:
    alives = [e for e in enemies if e.alive]
    if not alives:
        return None
    return min(alives, key=lambda e: (e.hp, e.name))


def _enemy_within_1(actor: BaseCharacter, enemies: list[BaseCharacter]) -> BaseCharacter | None:
    if actor.position is None:
        return None
    for e in sorted(enemies, key=lambda x: x.name):
        if not e.alive or e.position is None:
            continue
        if _chebyshev(actor.position, e.position) <= 1:
            return e
    return None


def _ally_within_range(actor: BaseCharacter, party: list[BaseCharacter]) -> BaseCharacter | None:
    """Healer ally selector: HP<70% AND within heal_range."""
    heal_range = getattr(actor, "heal_range", 1)
    best: tuple[float, str, BaseCharacter] | None = None
    for a in party:
        if a is actor or not a.alive or a.position is None:
            continue
        if a.hp_fraction >= 0.7:
            continue
        dist = _chebyshev(actor.position, a.position) if actor.position else 999
        if dist <= heal_range:
            key = (a.hp_fraction, a.name)
            if best is None or key < (best[0], best[1]):
                best = (a.hp_fraction, a.name, a)
    return None if best is None else best[2]


def choose_action(
    actor: BaseCharacter,
    dungeon: Dungeon,
    party: list[BaseCharacter],
    enemies: list[BaseCharacter],
    round_number: int,
) -> tuple[Action, Optional[BaseCharacter], Optional[tuple[int, int]], str, str]:
    """Baseline decision for an actor (companion OR enemy).

    Returns (action, target_character, step, algorithm, reason)."""
    if not actor.alive or actor.position is None:
        return (Action.SKIP, None, None, ALGORITHM_NAME,
                "Entity is not active (defeated/unconscious).")

    can_pass = _is_passable_factory(dungeon)
    is_enemy = actor.symbol in ENEMY_SYMBOLS

    # ---- ENEMY SIMPLE RULE ---------------------------------------------
    if is_enemy:
        nearest = dungeon.find_nearest_party(actor.position)
        target_char: BaseCharacter | None = None
        if nearest is not None:
            sym, pos = nearest
            target_char = next((a for a in party if a.alive and a.symbol == sym and a.position == pos), None)
        if target_char is not None:
            if _chebyshev(actor.position, target_char.position) <= 1:
                return (Action.ATTACK, target_char, None, ALGORITHM_NAME,
                        f"Nearest party member ({SYMBOL_NAMES.get(target_char.symbol)}) is within melee range, attacking.")
            step = step_towards(actor.position, target_char.position, can_pass)
            if step is not None:
                return (Action.MOVE, target_char, step, ALGORITHM_NAME,
                        f"Moving 1 step towards nearest party member {SYMBOL_NAMES.get(target_char.symbol)}.")
        # If no target visible, try to move towards exit? Enemies just defend.
        return (Action.DEFEND, None, None, ALGORITHM_NAME,
                "No party member reachable from current location; defending.")

    # ---- COMPANION RULES (Warrior / Healer / Scout) -------------------
    symbol = actor.symbol
    exit_pos = dungeon.find_exit()

    if symbol == "W":
        # Warrior → nearest enemy
        enemy = _enemy_within_1(actor, enemies)
        if enemy is not None:
            return (Action.ATTACK, enemy, None, ALGORITHM_NAME,
                    f"Enemy {enemy.name} is adjacent (1 tile); Warrior attacks.")
        nearest_info = dungeon.find_nearest_enemy(actor.position)
        if nearest_info is not None:
            esym, epos = nearest_info
            target_enemy = next((e for e in enemies if e.alive and e.symbol == esym and e.position == epos), None)
            if target_enemy is not None:
                step = step_towards(actor.position, target_enemy.position, can_pass)
                if step is not None:
                    return (Action.MOVE, target_enemy, step, ALGORITHM_NAME,
                            f"No enemy adjacent. Moving towards nearest {target_enemy.name} (greedy 4-way).")
        if exit_pos is not None:
            step = step_towards(actor.position, exit_pos, can_pass)
            if step is not None:
                return (Action.MOVE, None, step, ALGORITHM_NAME,
                        "No enemy visible; Warrior moves toward exit.")
        return (Action.DEFEND, None, None, ALGORITHM_NAME, "Stuck. Warrior defends position.")

    if symbol == "H":
        # Healer → heal first.
        injured = _ally_within_range(actor, party)
        if injured is not None:
            return (Action.HEAL, injured, None, ALGORITHM_NAME,
                    f"Ally {injured.name} HP fraction is {injured.hp_fraction:.0%} (within heal range). Healing.")
        enemy = _enemy_within_1(actor, enemies)
        if enemy is not None:
            return (Action.ATTACK, enemy, None, ALGORITHM_NAME,
                    f"No injured ally in range. Enemy {enemy.name} is adjacent — attacking with mace.")
        # Move towards lowest-HP ally (to prepare a heal).
        low = _find_lowest_hp_ally(party, actor)
        if low is not None and low.position is not None and low.hp_fraction < 0.85:
            step = step_towards(actor.position, low.position, can_pass)
            if step is not None:
                return (Action.MOVE, low, step, ALGORITHM_NAME,
                        f"Moving adjacent to lowest HP ally {low.name} ({low.hp_fraction:.0%}).")
        if exit_pos is not None:
            step = step_towards(actor.position, exit_pos, can_pass)
            if step is not None:
                return (Action.MOVE, None, step, ALGORITHM_NAME,
                        "No injured allies and no enemy within 1 — Healer moves towards exit.")
        return (Action.DEFEND, None, None, ALGORITHM_NAME, "Stuck. Healer defends.")

    if symbol == "S":
        # Scout → attack weakest if within 1; else exit; else nearest enemy
        enemy = _enemy_within_1(actor, enemies)
        if enemy is not None:
            weak = _find_weakest_enemy(enemies)
            # Prefer weakest within 1 if possible.
            if weak is not None and weak.position is not None and actor.position is not None \
                    and _chebyshev(actor.position, weak.position) <= 1:
                target = weak
            else:
                target = enemy
            return (Action.ATTACK, target, None, ALGORITHM_NAME,
                    f"Enemy {target.name} within 1 tile; Scout attacks weakest.")
        if exit_pos is not None:
            step = step_towards(actor.position, exit_pos, can_pass)
            if step is not None:
                return (Action.MOVE, None, step, ALGORITHM_NAME,
                        "Scout moves towards exit (mission priority).")
        nearest_info = dungeon.find_nearest_enemy(actor.position)
        if nearest_info is not None:
            esym, epos = nearest_info
            target_enemy = next((e for e in enemies if e.alive and e.symbol == esym and e.position == epos), None)
            if target_enemy is not None:
                step = step_towards(actor.position, target_enemy.position, can_pass)
                if step is not None:
                    return (Action.MOVE, target_enemy, step, ALGORITHM_NAME,
                            f"Exit blocked. Scout moves towards {target_enemy.name}.")
        return (Action.DEFEND, None, None, ALGORITHM_NAME, "Stuck. Scout defends.")

    # Unknown symbol
    return (Action.SKIP, None, None, ALGORITHM_NAME,
            f"No baseline rule for symbol {symbol!r} — treating as skip.")


# ---------------------------------------------------------------------------
# Weighted variant (used by Phase 5 SA / GA optimisers)
# ---------------------------------------------------------------------------

def _ally_within_range_weighted(actor, party, heal_threshold, heal_range):
    """Healer ally selector — with parametrized threshold+range."""
    best = None
    for a in party:
        if a is actor or not a.alive or a.position is None:
            continue
        if a.hp_fraction >= heal_threshold:
            continue
        dist = _chebyshev(actor.position, a.position)
        if dist <= heal_range:
            key = (a.hp_fraction, a.name)
            if best is None or key < (best[0], best[1]):
                best = (a.hp_fraction, a.name, a)
    return None if best is None else best[2]


def _enemy_within_n_tiles(actor, enemies, n):
    if actor.position is None:
        return None
    for e in sorted(enemies, key=lambda x: x.name):
        if not e.alive or e.position is None:
            continue
        if _chebyshev(actor.position, e.position) <= n:
            return e
    return None


def weighted_decision_fn(weights: BehaviourWeights, *, label_override: str | None = None) -> Callable:
    """Return a decision_fn callable whose rules are driven by *weights*.

    Same 5-tuple return contract as :func:`choose_action`.  When ``weights``
    is ``DEFAULT_WEIGHTS`` the output is intentionally *similar* to the
    hand-coded baseline but not byte-for-byte identical (this lets SA/GA
    genuinely optimise, while a label_prefix lets the action log show which
    optimiser produced the decision).
    """
    p = weights.rescale()
    algo_label = label_override or f"weighted-baseline({weights.label})"

    def fn(actor, dungeon, party, enemies, round_number):
        if not actor.alive or actor.position is None:
            return (Action.SKIP, None, None, algo_label,
                    "Entity is not active (defeated/unconscious).")
        can_pass = _is_passable_factory(dungeon)
        is_enemy = actor.symbol in ENEMY_SYMBOLS

        # Panic-retreat rule — universal: any actor whose HP drops below the
        # panic threshold retreats from the nearest enemy instead of attacking.
        if actor.hp_fraction <= max(0.05, p["panic_at_low_hp"]):
            nearest_enemy_tile = dungeon.find_nearest_enemy(actor.position)
            if nearest_enemy_tile is not None:
                _, epos = nearest_enemy_tile
                step = step_away_from(actor.position, epos, can_pass)
                if step is not None:
                    return (Action.RETREAT, None, step, algo_label,
                            f"{actor.name} HP is {actor.hp_fraction:.0%} (<= panic "
                            f"{p['panic_at_low_hp']:.0%}); retreat from nearest enemy.")

        if is_enemy:
            nearest = dungeon.find_nearest_party(actor.position)
            target_char = None
            if nearest is not None:
                sym, pos = nearest
                target_char = next((a for a in party if a.alive and a.symbol == sym and a.position == pos), None)
            if target_char is not None and target_char.position is not None:
                if _chebyshev(actor.position, target_char.position) <= 1:
                    return (Action.ATTACK, target_char, None, algo_label,
                            f"Nearest party member {target_char.name} within melee, attacking.")
                if p["enemy_charge_willingness"] >= 0.1:
                    step = step_towards(actor.position, target_char.position, can_pass)
                    if step is not None:
                        return (Action.MOVE, target_char, step, algo_label,
                                f"Enemy charges toward {target_char.name} (willingness={p['enemy_charge_willingness']:.2f}).")
            if p["defend_when_stuck"] >= 0.5:
                return (Action.DEFEND, None, None, algo_label, "No target reachable; enemy holds position.")
            return (Action.SKIP, None, None, algo_label, "No target reachable; enemy skips.")

        symbol = actor.symbol
        exit_pos = dungeon.find_exit()

        if symbol == "W":
            enemy = _enemy_within_n_tiles(actor, enemies, int(p["warrior_melee_range"]))
            if enemy is not None:
                return (Action.ATTACK, enemy, None, algo_label,
                        f"Enemy {enemy.name} is within melee_{p['warrior_melee_range']} tiles; Warrior attacks.")
            nearest_info = dungeon.find_nearest_enemy(actor.position)
            if nearest_info is not None and p["warrior_charge_willingness"] >= 0.2:
                esym, epos = nearest_info
                target_enemy = next((e for e in enemies if e.alive and e.symbol == esym and e.position == epos), None)
                if target_enemy is not None:
                    step = step_towards(actor.position, target_enemy.position, can_pass)
                    if step is not None:
                        return (Action.MOVE, target_enemy, step, algo_label,
                                f"Warrior charges toward {target_enemy.name}.")
            if exit_pos is not None:
                step = step_towards(actor.position, exit_pos, can_pass)
                if step is not None:
                    return (Action.MOVE, None, step, algo_label,
                            "No enemy reachable; Warrior moves toward exit.")
            return (Action.DEFEND if p["defend_when_stuck"] >= 0.5 else Action.SKIP,
                    None, None, algo_label, "Stuck. Warrior holds position.")

        if symbol == "H":
            injured = _ally_within_range_weighted(actor, party,
                                                  heal_threshold=p["healer_heal_threshold"],
                                                  heal_range=int(p["healer_heal_range"]))
            if injured is not None:
                return (Action.HEAL, injured, None, algo_label,
                        f"Ally {injured.name} HP {injured.hp_fraction:.0%} (<= threshold "
                        f"{p['healer_heal_threshold']:.0%}) is within heal range; Healer heals.")
            enemy = _enemy_within_1(actor, enemies)
            if enemy is not None:
                return (Action.ATTACK, enemy, None, algo_label,
                        f"No injured ally in range. Enemy {enemy.name} adjacent; Healer attacks.")
            low = _find_lowest_hp_ally(party, actor)
            stick_threshold = 0.3 + (1.0 - p["healer_stickiness"]) * 0.6
            if low is not None and low.position is not None and low.hp_fraction < stick_threshold:
                step = step_towards(actor.position, low.position, can_pass)
                if step is not None:
                    return (Action.MOVE, low, step, algo_label,
                            f"Moving adjacent to lowest-HP ally {low.name} ({low.hp_fraction:.0%}).")
            if exit_pos is not None:
                step = step_towards(actor.position, exit_pos, can_pass)
                if step is not None:
                    return (Action.MOVE, None, step, algo_label,
                            "No injured allies; Healer moves toward exit.")
            return (Action.DEFEND if p["defend_when_stuck"] >= 0.5 else Action.SKIP,
                    None, None, algo_label, "Stuck. Healer holds position.")

        if symbol == "S":
            enemy = _enemy_within_n_tiles(actor, enemies, int(p["scout_melee_range"]))
            if enemy is not None:
                weak = _find_weakest_enemy(enemies)
                target = enemy
                if (weak is not None and weak.position is not None and actor.position is not None
                        and _chebyshev(actor.position, weak.position) <= int(p["scout_melee_range"])):
                    target = weak
                return (Action.ATTACK, target, None, algo_label,
                        f"Enemy {target.name} within melee_{int(p['scout_melee_range'])} tiles; Scout attacks.")
            if exit_pos is not None and p["scout_exit_priority"] >= 0.3:
                step = step_towards(actor.position, exit_pos, can_pass)
                if step is not None:
                    return (Action.MOVE, None, step, algo_label,
                            f"Scout moves toward exit (priority={p['scout_exit_priority']:.2f}).")
            nearest_info = dungeon.find_nearest_enemy(actor.position)
            if nearest_info is not None:
                esym, epos = nearest_info
                target_enemy = next((e for e in enemies if e.alive and e.symbol == esym and e.position == epos), None)
                if target_enemy is not None:
                    step = step_towards(actor.position, target_enemy.position, can_pass)
                    if step is not None:
                        return (Action.MOVE, target_enemy, step, algo_label,
                                f"Exit blocked. Scout probes toward {target_enemy.name}.")
            return (Action.DEFEND if p["defend_when_stuck"] >= 0.5 else Action.SKIP,
                    None, None, algo_label, "Stuck. Scout holds position.")

        return (Action.SKIP, None, None, algo_label,
                f"No weighted-baseline rule for symbol {symbol!r}.")

    return fn
