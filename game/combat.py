"""
Simplified turn-based D&D combat mechanics.

Only the rules needed for simulation are implemented — NO full SRD rulebook.
Rules used:

    1. Roll d20, add attack_bonus. If result >= target AC → HIT.
       Else → MISS (0 damage).
    2. On HIT: roll damage dice (e.g. 1d8) + damage_bonus. Cap min damage 1.
    3. Advantage/disadvantage optional (simplified OFF by default).
    4. Defending: next incoming attack vs you gets disadvantage (2 d20 → pick lower).
    5. Healer heal: roll heal_dice + heal_bonus. Ally HP clamped to max.
    6. Retreat: step AWAY from nearest enemy (3-tile radius), next attack on retreater has advantage? No — simplified, retreat just moves + adds DEFEND flag.
    7. Skip: do nothing.

All functions in this module are PURE (no side effects on dungeon/state).
They return a dict of {key: delta} which the caller (game_state) applies.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from game.characters import BaseCharacter
from game.dungeon import Position, MOVEMENT_STEPS


_DICE_RE = re.compile(r"^(\d+)d(\d+)$")


def roll_d20(seed: Optional[int] = None, advantage: bool = False, disadvantage: bool = False) -> int:
    """Roll a single d20.  Pure helper.

    Advantage/disadvantage rules per SRD simplified.
    """
    rng = random.Random(seed) if seed is not None else random
    rolls = [rng.randint(1, 20)]
    if advantage or disadvantage:
        rolls.append(rng.randint(1, 20))
    if advantage:
        return max(rolls)
    if disadvantage:
        return min(rolls)
    return rolls[0]


def roll_damage(dice_expression: str, bonus: int = 0, seed: Optional[int] = None) -> int:
    """Roll damage dice like '1d8' + bonus.  Returns total, minimum 1.

    Example: roll_damage('2d6', 3) → 5..15 inclusive.
    """
    m = _DICE_RE.match(dice_expression.strip())
    if not m:
        raise ValueError(f"Unknown dice expression: {dice_expression!r}")
    num, sides = int(m.group(1)), int(m.group(2))
    rng = random.Random(seed) if seed is not None else random
    total = sum(rng.randint(1, sides) for _ in range(num)) + bonus
    return max(1, total)


def perform_attack(
    attacker: BaseCharacter,
    defender: BaseCharacter,
    *,
    attacker_advantage: bool = False,
    defender_has_defend_flag: bool = False,
    seed: Optional[int] = None,
) -> tuple[int, int, bool]:
    """Compute (attack_roll, damage, hit) and apply damage if hit.

    Returns (d20_roll_after_modifiers_applied, damage_total_applied, hit_bool).
    Damage actually applied (clamped to HP remaining on defender) is returned.
    """
    disadvantage = bool(defender_has_defend_flag)
    raw = roll_d20(seed=seed, advantage=attacker_advantage, disadvantage=disadvantage)
    attack_total = raw + attacker.attack_bonus
    hit = attack_total >= defender.ac
    rolled_dmg = roll_damage(attacker.damage_dice, attacker.damage_bonus, seed=seed) if hit else 0
    applied = defender.apply_damage(rolled_dmg) if hit else 0
    return attack_total, applied, hit


def perform_heal(
    healer: BaseCharacter,
    target: BaseCharacter,
    seed: Optional[int] = None,
) -> int:
    """Returns HP actually restored (capped at target max)."""
    dice = getattr(healer, "heal_dice", "1d8")
    bonus = getattr(healer, "heal_bonus", 0)
    rolled = roll_damage(dice, bonus, seed=seed)
    restored = target.apply_healing(rolled)
    return restored


def step_towards(
    from_pos: Position,
    to_pos: Position,
    is_passable,
) -> Optional[tuple[int, int]]:
    """Greedy 4-directional step towards to_pos.  Return step tuple (dr, dc)
    or None if no legal step exists."""
    # Generate candidates sorted by manhattan improvement (steepest descent first).
    candidates = []
    for step in MOVEMENT_STEPS:
        new_pos = from_pos + step
        if not is_passable(new_pos.row, new_pos.col):
            continue
        new_dist = new_pos.manhattan(to_pos)
        candidates.append((new_dist, step))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    # Pick the step giving smallest new distance (ties broken by MOVEMENT_STEPS order).
    return candidates[0][1]


def step_away_from(
    from_pos: Position,
    threat_pos: Position,
    is_passable,
) -> Optional[tuple[int, int]]:
    """Greedy step that increases distance from threat.  Used for retreat."""
    current = from_pos.manhattan(threat_pos)
    best = None
    best_dist = current
    for step in MOVEMENT_STEPS:
        new_pos = from_pos + step
        if not is_passable(new_pos.row, new_pos.col):
            continue
        d = new_pos.manhattan(threat_pos)
        if d > best_dist:
            best_dist = d
            best = step
    return best
