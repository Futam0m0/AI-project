"""
Overall game state + one-round advance orchestrator.

Turn order (simplified, no initiative rolls):
    Warrior → Healer → Scout → each alive Enemy (in insertion order).

Win conditions:
    1. Any party member (W/H/S) steps onto the EXIT tile → WIN.
    2. All enemies defeated → WIN.
Lose conditions:
    1. All 3 party members unconscious/defeated → LOSE.

Statistics tracked (for dashboard Section E):
    battles_won / battles_lost
    enemies_defeated
    party_survival_rate  (fraction of 3 alive when game ends)
    damage_dealt / damage_received / healing_performed
    turns_taken
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Optional

from algorithms import baseline_decision
from game.actions import Action, ActionResult
from game.characters import BaseCharacter, Healer, Warrior, Scout, make_default_party
from game.combat import perform_attack, perform_heal, step_towards, step_away_from
from game.dungeon import (Dungeon, ENEMY_SYMBOLS, EXIT, PARTY_SYMBOLS,
                          Position)
from game.enemies import Bugbear, Goblin, RedbrandRuffian, make_default_encounter_three_enemies

# Maximum rounds before a battle is declared a draw (stalemate safety net).
# Matches the experiment runner's MAX_ROUNDS_PER_BATTLE so all callers agree.
MAX_ROUNDS = 60


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

@dataclass
class GameStats:
    """Collects Section-E statistics.  Reset on game.reset()."""
    battles_won: int = 0
    battles_lost: int = 0
    enemies_defeated: int = 0
    damage_dealt: int = 0          # damage party → enemies
    damage_received: int = 0       # damage enemies → party
    healing_performed: int = 0
    turns_taken: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "battles_won": self.battles_won,
            "battles_lost": self.battles_lost,
            "enemies_defeated": self.enemies_defeated,
            "party_survival_rate": self.party_survival_rate(3),
            "damage_dealt": self.damage_dealt,
            "damage_received": self.damage_received,
            "healing_performed": self.healing_performed,
            "turns_taken": self.turns_taken,
            "extra": self.extra,
        }

    def party_survival_rate(self, total_party_size: int = 3) -> float:
        if total_party_size <= 0:
            return 0.0
        dead = self.battles_lost * total_party_size   # simplification
        alive_after = max(total_party_size - dead, 0)
        return alive_after / total_party_size


# ---------------------------------------------------------------------------
# Game state object
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    dungeon: Dungeon
    party: list[BaseCharacter]
    enemies: list[BaseCharacter]
    action_log: list[ActionResult] = field(default_factory=list)
    stats: GameStats = field(default_factory=GameStats)
    round_number: int = 0
    winner: Optional[str] = None   # "party" | "enemies" | None
    defend_flags: dict[str, bool] = field(default_factory=dict)  # actor_id → True
    seed: Optional[int] = None

    # ----- initialization helpers ----------------------------------------
    @classmethod
    def new_default(cls, seed: Optional[int] = None) -> "GameState":
        """Build a game: 8x10 default dungeon, party at (1,1)-(1,3), 3 enemies scattered."""
        dungeon = Dungeon(rows=8, cols=10)
        party = make_default_party()
        enemies = make_default_encounter_three_enemies()

        # Placement — 100% deterministic with seed if passed.
        rng = random.Random(seed)
        placements_party = [(1, 1), (1, 2), (1, 3)]  # Warrior, Healer, Scout from left col
        for (char, (r, c)) in zip(party, placements_party):
            char.position = Position(r, c)
            assert dungeon.place(char.symbol, r, c), f"Placement failure {char.symbol}@({r},{c})"

        # Enemy placements: safe interior positions not too close to exit.
        candidate_slots = [
            (4, 7), (5, 7), (6, 4), (3, 5), (2, 7),
        ]
        # Shuffle by seed for variety.
        rng.shuffle(candidate_slots)
        placed = 0
        for enemy in enemies:
            while candidate_slots and placed < len(enemies):
                r, c = candidate_slots.pop(0)
                if dungeon.place(enemy.symbol, r, c):
                    enemy.position = Position(r, c)
                    placed += 1
                    break
            else:
                # Fallback — find a floor cell.
                for r in range(1, dungeon.rows - 1):
                    for c in range(1, dungeon.cols - 1):
                        if dungeon.place(enemy.symbol, r, c):
                            enemy.position = Position(r, c)
                            placed += 1
                            break
                    if enemy.position is not None:
                        break

        state = cls(dungeon=dungeon, party=party, enemies=enemies, seed=seed)
        return state

    # ----- helpers --------------------------------------------------------
    def party_alive(self) -> list[BaseCharacter]:
        return [a for a in self.party if a.alive]

    def enemies_alive(self) -> list[BaseCharacter]:
        return [e for e in self.enemies if e.alive]

    def game_over(self) -> tuple[bool, Optional[str]]:
        if not self.enemies_alive():
            return True, "party"
        exit_pos = self.dungeon.find_exit()
        if exit_pos is not None:
            exit_r, exit_c = exit_pos
            if self.dungeon.grid[exit_r][exit_c] in PARTY_SYMBOLS:
                return True, "party"
        if not self.party_alive():
            return True, "enemies"
        # Stalemate safety net: if no side can win within MAX_ROUNDS, declare a
        # draw instead of looping forever (e.g. both sides trapped behind walls).
        if self.round_number >= MAX_ROUNDS:
            return True, "draw"
        return False, None

    def reset(self, new_seed: Optional[int] = None) -> None:
        """Rebuild the game to its default starting state.

        Replaces self contents in place so external references to this
        GameState object stay valid (Streamlit st.session_state-friendly).
        """
        if new_seed is not None:
            self.seed = new_seed
        fresh = GameState.new_default(seed=self.seed)
        self.dungeon = fresh.dungeon
        self.party = fresh.party
        self.enemies = fresh.enemies
        self.action_log = []
        self.stats = GameStats()
        self.round_number = 0
        self.winner = None
        self.defend_flags = {}

    # ----- apply one action --------------------------------------------
    def apply_action(self, actor: BaseCharacter, decision: tuple) -> ActionResult:
        action, target_char, step, algorithm, reason = decision
        result = ActionResult(
            round_number=self.round_number,
            actor=actor,
            action=action,
            target=target_char,
            target_position=(
                actor.position + step if (step is not None and actor.position is not None) else None
            ),
            algorithm=algorithm,
            reason=reason,
        )
        # --- Execute.
        if action == Action.SKIP:
            return result
        if action == Action.DEFEND:
            self.defend_flags[id(actor)] = True
            return result
        if action == Action.RETREAT:
            # Step away from nearest threat.
            if actor.position is None:
                return result
            nearest = self.dungeon.find_nearest(actor.position, ENEMY_SYMBOLS)
            if nearest is not None and actor.position is not None:
                sym, pos = nearest
                step2 = step_away_from(
                    actor.position, pos,
                    lambda r, c: self.dungeon.is_passable(r, c) and
                                 self.dungeon.get(r, c) in {".", EXIT},
                )
                if step2 is not None:
                    self._do_move(actor, step2)
                    self.defend_flags[id(actor)] = True
            return result
        if action == Action.MOVE and step is not None and actor.position is not None:
            self._do_move(actor, step)
            # If move lands on exit → check game over immediately.
            over, winner = self.game_over()
            if over:
                self.winner = winner
                if winner == "party":
                    self.stats.battles_won += 1
            return result
        if action == Action.ATTACK and target_char is not None:
            if not actor.alive or not target_char.alive:
                return result
            defender_flag = self.defend_flags.get(id(target_char), False)
            attack_total, damage, hit = perform_attack(
                actor, target_char,
                defender_has_defend_flag=defender_flag,
                seed=self._seed_for(actor, "atk"),
            )
            result.context = {"attack_roll": attack_total, "target_ac": target_char.ac, "hit": hit}
            result.delta_hp_target = -damage
            if hit:
                dealt = damage
                if actor.symbol in PARTY_SYMBOLS:
                    self.stats.damage_dealt += dealt
                else:
                    self.stats.damage_received += dealt
                if not target_char.alive:
                    target_char.status = "defeated"
                    if target_char.symbol in ENEMY_SYMBOLS:
                        self.stats.enemies_defeated += 1
                    # Clear the defeated character's symbol from the grid so the
                    # corpse does not block movement for either side (would
                    # otherwise cause permanent stalemates in narrow corridors).
                    if target_char.position is not None:
                        self.dungeon.remove_entity(*target_char.position)
            return result
        if action == Action.HEAL and target_char is not None:
            if not actor.alive:
                return result
            heal = perform_heal(actor, target_char, seed=self._seed_for(actor, "heal"))
            result.delta_hp_target = heal
            result.context = {"heal_rolled": heal}
            self.stats.healing_performed += heal
            return result
        return result

    def _do_move(self, actor: BaseCharacter, step: tuple[int, int]) -> None:
        assert actor.position is not None
        before = actor.position
        ok = self.dungeon.move(actor.symbol, before, step)
        if ok:
            actor.position = before + step

    def _seed_for(self, actor: BaseCharacter, tag: str) -> int | None:
        """Deterministic per-actor per-round seed derived from master seed if set."""
        if self.seed is None:
            return None
        # Build a deterministic integer hash from (seed, round, actor identity, tag).
        # Actor identity uses a stable CRC32 of the actor's unique name rather than
        # id(actor) (memory address), so the same seed reproduces the same battle
        # across process runs.
        parts = (int(self.seed) & 0xFFFFFFFF,
                 self.round_number & 0xFFFFFFFF,
                 zlib.crc32(actor.name.encode()) & 0xFFFFFFFF,
                 zlib.crc32(tag.encode()) & 0xFFFFFFFF)
        rng = random.Random()
        rng.seed(sum((p << shift) for p, shift in zip(parts, (0, 32, 64, 96))))
        return rng.randint(0, (1 << 30) - 1)

    # ----- main round ---------------------------------------------------
    def _end_game(self, winner: str) -> None:
        """Set the winner and record battles_won / battles_lost exactly once.

        Centralised so every game_over() check records the outcome
        consistently, whether the game ends at the start, middle, or end of
        a round.  ``winner`` is one of "party" | "enemies" | "draw".
        """
        self.winner = winner
        if winner == "party" and self.stats.battles_won == 0:
            self.stats.battles_won = 1
        elif winner == "enemies" and self.stats.battles_lost == 0:
            self.stats.battles_lost = 1
        # "draw" → neither battles_won nor battles_lost incremented.

    def advance_one_round(
        self,
        decision_fn=baseline_decision.choose_action,
    ) -> None:
        """Run exactly one round of the game in W → H → S → enemies order.

        Parameters
        ----------
        decision_fn : callable
            ``(actor, dungeon, party, enemies, round_num) -> (Action, target, step, str_algo, str_reason)``
            matching the ``baseline_decision.choose_action`` signature.  Later
            phases (4+) swap this out for BFS / A* / Minimax etc. by passing a
            wrapper from ``algorithms.algorithm_selection.dispatch``.
        """
        over, winner = self.game_over()
        if over:
            self._end_game(winner)
            return
        self.round_number += 1
        self.stats.turns_taken += 1
        # Party first.
        for actor in self.party:
            over, winner = self.game_over()
            if over:
                self._end_game(winner)
                return
            if not actor.alive:
                continue
            decision = decision_fn(actor, self.dungeon, self.party, self.enemies, self.round_number)
            res = self.apply_action(actor, decision)
            self.action_log.append(res)
        # Enemies second.
        for enemy in self.enemies:
            over, winner = self.game_over()
            if over:
                self._end_game(winner)
                return
            if not enemy.alive:
                continue
            decision = decision_fn(enemy, self.dungeon, self.party, self.enemies, self.round_number)
            res = self.apply_action(enemy, decision)
            self.action_log.append(res)
        # End-of-round win/loss bookkeeping.
        over, winner = self.game_over()
        if over:
            self._end_game(winner)

    # ----- utilities ----------------------------------------------------
    def to_json_dict(self) -> dict:
        """Serialise for Streamlit dashboard or experiment CSV later."""
        return {
            "round_number": self.round_number,
            "winner": self.winner,
            "stats": self.stats.to_dict(),
            "grid": [list(r) for r in self.dungeon.grid],
            "party": [
                {
                    "name": c.name, "symbol": c.symbol, "hp_max": c.hp_max, "hp": c.hp,
                    "ac": c.ac, "status": c.status, "objective": c.objective,
                    "position": None if c.position is None else (c.position.row, c.position.col),
                }
                for c in self.party
            ],
            "enemies": [
                {
                    "name": e.name, "symbol": e.symbol, "hp_max": e.hp_max, "hp": e.hp,
                    "ac": e.ac, "status": e.status,
                    "position": None if e.position is None else (e.position.row, e.position.col),
                }
                for e in self.enemies
            ],
            "action_log_last_30": [
                {
                    "round": r.round_number, "actor": r.actor.name,
                    "action": r.action.value,
                    "target": r.target.name if r.target else None,
                    "algorithm": r.algorithm, "reason": r.reason,
                    "dHP_target": r.delta_hp_target, "context": r.context,
                }
                for r in self.action_log[-30:]
            ],
        }
