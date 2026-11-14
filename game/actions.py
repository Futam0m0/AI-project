"""
Action enum + ActionResult dataclass.

Every decision a companion or enemy makes becomes one ActionResult object.
These objects are written to the game action log and rendered in the Streamlit
Action dashboard (Section C).  The __str__ format mirrors the brief's example:

    Round 3 — Warrior
    Action: Move towards Goblin
    Algorithm: BFS
    Reason: Found a valid path around the wall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from game.characters import BaseCharacter
from game.dungeon import Position


class Action(Enum):
    """Every possible action an entity can take in one turn."""
    MOVE = "Move"
    ATTACK = "Attack"
    HEAL = "Heal"
    DEFEND = "Defend"
    RETREAT = "Retreat"
    SKIP = "Skip"


@dataclass
class ActionResult:
    """Structured result of one turn's decision (action log entry)."""

    round_number: int
    actor: BaseCharacter
    action: Action
    target: Optional[BaseCharacter] = None
    target_position: Optional[Position] = None
    algorithm: str = "baseline"
    reason: str = ""
    delta_hp_actor: int = 0   # heal: positive; damage taken: negative
    delta_hp_target: int = 0  # damage dealt: negative; healed: positive
    # Optional human-readable extra (e.g. dice rolls, path length, etc.)
    context: dict = field(default_factory=dict)

    # ---- Display helpers (brief §3.C format) ----------------------------
    def header_line(self) -> str:
        return f"Round {self.round_number} — {self.actor.display_name()}"

    def action_line(self) -> str:
        parts = [f"Action: {self.action.value}"]
        if self.target is not None:
            parts[-1] += f" {self.target.display_name()}"
        elif self.target_position is not None:
            parts[-1] += f" to ({self.target_position.row},{self.target_position.col})"
        return "    ".join(parts)

    def algorithm_line(self) -> str:
        return f"Algorithm: {self.algorithm}"

    def reason_line(self) -> str:
        return f"Reason: {self.reason or 'No explanation recorded.'}"

    def __str__(self) -> str:
        return "\n".join([
            self.header_line(),
            "  " + self.action_line(),
            "  " + self.algorithm_line(),
            "  " + self.reason_line(),
        ])
