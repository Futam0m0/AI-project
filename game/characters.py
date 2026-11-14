"""
Party characters: Warrior, Healer, Scout.

Stats are sourced from SRD 5.1 OGL Class entries for Fighter, Cleric, and Rogue
respectively, simplified for a beginner-friendly demo (no multiclass, no feats,
no subclass).  The exact HP totals are: Warrior=d10×3 → 28; Healer=d8×2 → 20;
Scout (Rogue)=d8×2+d6 → 22 (simplified).  Attack bonuses ≈ level 3 character
(+4 proficiency + ability modifier).  Damage dice = typical weapon dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from game.dungeon import Position, SYMBOL_NAMES

ROLE_LITERAL = Literal["Warrior", "Healer", "Scout"]
STATUS_LITERAL = Literal["alive", "unconscious", "defeated"]


@dataclass
class BaseCharacter:
    """Base class shared by both party members and enemies."""
    name: str
    symbol: str
    hp_max: int
    hp: int
    ac: int
    attack_bonus: int
    damage_dice: str          # e.g. "1d8+2"
    damage_bonus: int = 0
    position: Position | None = None
    status: STATUS_LITERAL = "alive"
    objective: str = "Follow baseline decision AI"

    def __post_init__(self) -> None:
        # Clamp initial HP.
        self.hp = max(0, min(self.hp, self.hp_max))

    @property
    def alive(self) -> bool:
        return self.status == "alive" and self.hp > 0

    @property
    def hp_fraction(self) -> float:
        if self.hp_max <= 0:
            return 0.0
        return self.hp / self.hp_max

    def apply_damage(self, damage: int) -> int:
        """Deal damage, update status, return actual HP delta (never negative, clamped to HP)."""
        dmg = max(0, damage)
        before = self.hp
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.status = "unconscious"
        return before - self.hp

    def apply_healing(self, amount: int) -> int:
        heal = max(0, amount)
        before = self.hp
        self.hp = min(self.hp_max, self.hp + heal)
        if self.hp > 0 and self.status in {"unconscious", "defeated"}:
            # Simple rule: any healing that restores to positive → alive.
            self.status = "alive"
        return self.hp - before

    def display_name(self) -> str:
        return f"{self.name} ({SYMBOL_NAMES.get(self.symbol, self.symbol)})"


# ---- 3 party companion presets (simplified SRD OGL values, level 3 chars) ----


class Warrior(BaseCharacter):
    """Tank / melee DPS.  High HP, high AC, best attack bonus.

    OGL Fighter (d10 hit die)."""

    def __init__(self, name: str = "Brann the Bold", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="W",
            hp_max=28,
            hp=28,
            ac=18,
            attack_bonus=5,
            damage_dice="1d8",
            damage_bonus=3,
            position=position,
            objective="Melee nearest enemy; hold the front line.",
        )


class Healer(BaseCharacter):
    """Support / backline healer.  Low HP, has a heal ability (1d8+2).

    OGL Cleric (d8 hit die), Life domain equivalent (simplified)."""

    def __init__(self, name: str = "Mara the Merciful", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="H",
            hp_max=20,
            hp=20,
            ac=14,
            attack_bonus=3,
            damage_dice="1d6",
            damage_bonus=1,
            position=position,
            objective="Heal the lowest HP ally; attack if no one is hurt.",
        )
        # Healer has its own heal parameters separate from attack/damage.
        # 1d8+2 = 3-10 per heal (Cure Wounds level 2).
        self.heal_dice: str = "1d8"
        self.heal_bonus: int = 2
        self.heal_range: int = 1  # Chebyshev distance (adjacent + diagonal)


class Scout(BaseCharacter):
    """Mobile striker.  Mid HP, good AC, focus-fire weakest enemy.

    OGL Rogue (d8 hit die, sneak attack simplified)."""

    def __init__(self, name: str = "Ryn the Quick", position: Position | None = None):
        super().__init__(
            name=name,
            symbol="S",
            hp_max=22,
            hp=22,
            ac=15,
            attack_bonus=4,
            damage_dice="1d6",
            damage_bonus=2,
            position=position,
            objective="Move toward the exit; attack weak enemies en route.",
        )
        # Extra scout: "sneak attack" bonus when adjacent to an ally.
        self.sneak_attack_dice: str = "1d6"


def make_default_party() -> list[BaseCharacter]:
    """Return the 3 default companions in order."""
    return [Warrior(), Healer(), Scout()]
