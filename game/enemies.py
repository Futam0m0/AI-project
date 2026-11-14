"""
Enemy presets: Goblin, Redbrand Ruffian, Bugbear.

All stat blocks are the standard SRD 5.1 OGL values.  These are simplified for
the simulator (one weapon attack only, no special actions, no legendary).

For reference (Open Game License v1.0a content):
  - Goblin:      HP 10 (tuned up from SRD 7 for longer fights), AC 15, Scimitar +5 to hit, 1d6+2 slashing
  - Bugbear:     HP 27 (5d8+5), AC 16 (hide + shield), Morningstar +4 to hit, 2d8+2 piercing + extra
  - Redbrand Ruffian is LMoP-specific and not in SRD: we approximate with
    HP=22 (tuned up from 16 for longer fights), AC=14 (studded leather), Shortsword +4, 1d6+1 piercing.
    Code comments explicitly mark this as "LMoP simplified".
"""

from __future__ import annotations

from game.characters import BaseCharacter, STATUS_LITERAL
from game.dungeon import Position


class Goblin(BaseCharacter):
    """SRD OGL Goblin stat-block (level 1 grunt)."""

    def __init__(self, name: str = "Goblin", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="G",
            hp_max=10,
            hp=10,
            ac=15,
            attack_bonus=5,
            damage_dice="1d6",
            damage_bonus=2,
            position=position,
            objective="Attack nearest party member with ranged/mob tactics.",
        )


class RedbrandRuffian(BaseCharacter):
    """LMoP homebrew simplified → equivalent to a bandit captain +3.

    NOTE: LMoP content.  NOT in SRD."""

    def __init__(self, name: str = "Redbrand Ruffian", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="R",
            hp_max=22,
            hp=22,
            ac=14,
            attack_bonus=4,
            damage_dice="1d6",
            damage_bonus=1,
            position=position,
            objective="Engage any ally that is visible (simplified).",
        )


class Bugbear(BaseCharacter):
    """SRD OGL Bugbear stat block (CR 1 threat)."""

    def __init__(self, name: str = "Bugbear", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="B",
            hp_max=27,
            hp=27,
            ac=16,
            attack_bonus=4,
            damage_dice="2d8",
            damage_bonus=2,
            position=position,
            objective="Reach nearest party member; prefer Warrior (threat).",
        )


def make_default_encounter_three_enemies() -> list[BaseCharacter]:
    """Returns a simple first-encounter size: 2 Goblins + 1 Redbrand Ruffian.

    Rough balance: ~50% win rate for level-3 party of 3.  Good baseline for GA
    / SA optimization later."""
    return [Goblin("Goblin 1"), Goblin("Goblin 2"), RedbrandRuffian("Ruffian 1")]
