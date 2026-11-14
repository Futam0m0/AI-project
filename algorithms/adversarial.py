"""
Adversarial search algorithms — matches notebook Chapter 4 (Adversarial & Stochastic):

  1. ``minimax``           — classic minimax on a deterministic zero-sum combat state.
                              MAX = our actor; MIN = the opposing side.  Returns action at root.
  2. ``cutoff_search``     — 3-LEVEL SIMPLE CUTOFF: run minimax up to a fixed ``max_depth``
                              and on non-terminal leaves call ``eval_fn(state)``.  Demonstrates
                              the same cut-off technique shown in the course notebook when
                              the tree is too deep to search fully.
  3. ``alpha_beta``        — Minimax + Alpha-Beta pruning (reduces states without changing
                              final value, so returned action == minimax action).
  4. ``expectimax``        — Stochastic (chance nodes) variant: opponent response is drawn
                              from a 3-way distribution [crit 0.1 / hit 0.7 / miss 0.2].
                              Values are EXPECTED VALUES over chance outcomes (= weighted sum),
                              not worst-case as in pure Minimax.

The game tree is built over a tiny, intentionally simple deterministic 2-agent combat
state so the lecturer can step through by hand and verify. State shape:

    CombatState(our_hp, their_hp, our_turn: bool)   — BOTH sides have actions from {Attack, Heal, Defend}.
    - Attack   → deals 5 damage (deterministic for pure Minimax/Alpha; Expectimax applies crit/miss envelope).
    - Heal     → restores 4 HP up to our max.
    - Defend   → halves (floor) any damage we take on the opponent's next turn.

Terminal states: our_hp <= 0 (MIN wins → value -1.0) OR their_hp <= 0 (MAX wins → value +1.0).

All functions expose:
  * a low-level pair ``(best_value, best_action, expansions)`` called by tests for verification
  * a high-level ``*_decision(actor, dungeon, party, enemies, round_num)`` 5-tuple wrapper
    compatible with ``GameState.advance_one_round(decision_fn=...)`` (used by Streamlit dropdown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Callable

from algorithms import baseline_decision
from game.actions import Action
from game.characters import BaseCharacter
from game.dungeon import Dungeon, ENEMY_SYMBOLS, chebyshev

ADV_ACTIONS = ("Attack", "Heal", "Defend")
AdvAction = Literal["Attack", "Heal", "Defend"]


@dataclass
class CombatState:
    """2-player deterministic combat subtree state.

    MAX acts when ``our_turn`` is True, MIN acts when False.
    ``max_hp`` is the starting HP cap for Heal clamping.
    ``defend_buff``: when true, the next Attack damage received by the defender is halved.
    """
    our_hp: int
    their_hp: int
    our_turn: bool
    max_hp: int = 20
    our_defend_next: bool = False
    their_defend_next: bool = False

    def terminal(self) -> bool:
        return self.our_hp <= 0 or self.their_hp <= 0

    def terminal_value(self) -> float:
        """Range [-1.0, +1.0].  Positive = MAX wins."""
        if self.their_hp <= 0 and self.our_hp <= 0:
            return 0.0
        if self.their_hp <= 0:
            return +1.0
        if self.our_hp <= 0:
            return -1.0
        raise RuntimeError("Not a terminal state, cannot score.")

    # -------- apply one action -----------------------------------------------------------
    def apply(self, actor_side: Literal["our", "their"], action: AdvAction) -> "CombatState":
        s = CombatState(
            self.our_hp, self.their_hp, not self.our_turn,
            self.max_hp, self.our_defend_next, self.their_defend_next,
        )
        dmg = 5
        heal_amt = 4
        if action == "Attack":
            if actor_side == "our":
                final_dmg = dmg // 2 if s.their_defend_next else dmg
                s.their_hp -= final_dmg
                s.their_defend_next = False
            else:
                final_dmg = dmg // 2 if s.our_defend_next else dmg
                s.our_hp -= final_dmg
                s.our_defend_next = False
        elif action == "Heal":
            if actor_side == "our":
                s.our_hp = min(s.max_hp, s.our_hp + heal_amt)
            else:
                # MIN's heal caps at same max_hp. Reuse for symmetric state.
                s.their_hp = min(s.max_hp, s.their_hp + heal_amt)
        elif action == "Defend":
            if actor_side == "our":
                s.our_defend_next = True
            else:
                s.their_defend_next = True
        return s


def heuristic_eval(state: CombatState) -> float:
    """Heuristic used by CUTOFF search (simple 3-level cutoff) on non-terminal leaves.

    Range roughly [-1.0, +1.0]: score = (our_hp - their_hp) / max_hp.
    """
    diff = state.our_hp - state.their_hp
    cap = max(1, state.max_hp)
    return max(-1.0, min(1.0, diff / cap))


# ===========================================================================
# 1. Pure Minimax (bounded by max_depth; beyond depth → heuristic_eval)
# ===========================================================================
def minimax_search(
    state: CombatState,
    *,
    max_depth: int = 5,
) -> tuple[float, AdvAction, int]:
    """Returns (best_root_value, best_root_action, expansions).

    The tree is bounded by ``max_depth`` plies of alternating turns.  Beyond
    the bound we call ``heuristic_eval(state)`` instead of fully recursing —
    this prevents infinite recursion when both players can Heal forever,
    and keeps the search fast for the dashboard (hundreds of states per
    call, not millions).  Default = 5 plies = 3^5 = 243 leaf nodes.
    """
    expansions = 0

    def recurse(s: CombatState, depth: int) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return heuristic_eval(s)
        if s.our_turn:
            best = -1e9
            for a in ADV_ACTIONS:
                best = max(best, recurse(s.apply("our", a), depth - 1))
            return best
        else:
            best = +1e9
            for a in ADV_ACTIONS:
                best = min(best, recurse(s.apply("their", a), depth - 1))
            return best

    best_val = -1e9
    best_act: AdvAction = "Attack"
    for a in ADV_ACTIONS:
        v = recurse(state.apply("our", a), max_depth - 1)
        if v > best_val:
            best_val = v
            best_act = a
    expansions += 1
    return best_val, best_act, expansions


# ===========================================================================
# 2. Minimax with 3-LEVEL SIMPLE CUTOFF
# ===========================================================================
def cutoff_search(
    state: CombatState,
    max_depth: int = 3,
    eval_fn: Callable[[CombatState], float] = heuristic_eval,
) -> tuple[float, AdvAction, int]:
    expansions = 0

    def recurse(s: CombatState, depth: int) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return eval_fn(s)
        if s.our_turn:
            best = -1e9
            for a in ADV_ACTIONS:
                best = max(best, recurse(s.apply("our", a), depth - 1))
            return best
        else:
            best = +1e9
            for a in ADV_ACTIONS:
                best = min(best, recurse(s.apply("their", a), depth - 1))
            return best

    best_val = -1e9
    best_act: AdvAction = "Attack"
    for a in ADV_ACTIONS:
        v = recurse(state.apply("our", a), max_depth - 1)
        if v > best_val:
            best_val = v
            best_act = a
    expansions += 1
    return best_val, best_act, expansions


# ===========================================================================
# 3. Minimax + ALPHA-BETA pruning (bounded by max_depth)
# ===========================================================================
def alphabeta_search(
    state: CombatState,
    *,
    max_depth: int = 5,
) -> tuple[float, AdvAction, int]:
    expansions = 0

    def max_value(s: CombatState, depth: int, alpha: float, beta: float) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return heuristic_eval(s)
        v = -1e9
        for a in ADV_ACTIONS:
            v = max(v, min_value(s.apply("our", a), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(s: CombatState, depth: int, alpha: float, beta: float) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return heuristic_eval(s)
        v = +1e9
        for a in ADV_ACTIONS:
            v = min(v, max_value(s.apply("their", a), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    best_val = -1e9
    best_act: AdvAction = "Attack"
    alpha = -1e9
    beta = +1e9
    for a in ADV_ACTIONS:
        v = min_value(state.apply("our", a), max_depth - 1, alpha, beta)
        if v > best_val:
            best_val = v
            best_act = a
        alpha = max(alpha, v)
    expansions += 1
    return best_val, best_act, expansions


# ===========================================================================
# 4. EXPECTIMAX (stochastic — chance nodes on opponent's attack outcome)
# ===========================================================================

# 3-way opponent attack outcome distribution (sums exactly to 1):
#   miss 0.2 → damage × 0
#   hit  0.7 → damage × 1
#   crit 0.1 → damage × 2
_CHANCE_OUTCOMES: tuple[tuple[float, float], ...] = (
    (0.2, 0.0),
    (0.7, 1.0),
    (0.1, 2.0),
)


def _expectimax_apply_chance(
    state: CombatState,
    attacker: Literal["our", "their"],
    defender_defends: bool,
) -> list[tuple[float, CombatState]]:
    """Produce 3 weighted successor states, one per chance outcome.

    Weights sum to exactly 1.0, used for Expectimax expected-value calculation.
    """
    base = 5
    results: list[tuple[float, CombatState]] = []
    for prob, mult in _CHANCE_OUTCOMES:
        raw_dmg = int(base * mult)
        dmg = (raw_dmg // 2) if defender_defends else raw_dmg
        ns = CombatState(state.our_hp, state.their_hp, not state.our_turn,
                         state.max_hp, state.our_defend_next, state.their_defend_next)
        if attacker == "our":
            ns.their_hp -= dmg
            ns.their_defend_next = False
        else:
            ns.our_hp -= dmg
            ns.our_defend_next = False
        results.append((prob, ns))
    return results


def expectimax_search(
    state: CombatState,
    *,
    max_depth: int = 4,
) -> tuple[float, AdvAction, int]:
    expansions = 0

    def max_val(s: CombatState, depth: int) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return heuristic_eval(s)
        best = -1e9
        for a in ADV_ACTIONS:
            if a == "Attack":
                exp_v = 0.0
                for prob, ns in _expectimax_apply_chance(s, "our", s.their_defend_next):
                    exp_v += prob * chance_val(ns, depth - 1)
                best = max(best, exp_v)
            else:
                ns = s.apply("our", a)
                best = max(best, chance_val(ns, depth - 1))
        return best

    def min_val(s: CombatState, depth: int) -> float:
        nonlocal expansions
        expansions += 1
        if s.terminal():
            return s.terminal_value()
        if depth <= 0:
            return heuristic_eval(s)
        worst = +1e9
        for a in ADV_ACTIONS:
            if a == "Attack":
                exp_v = 0.0
                for prob, ns in _expectimax_apply_chance(s, "their", s.our_defend_next):
                    exp_v += prob * max_val(ns, depth - 1)
                worst = min(worst, exp_v)
            else:
                ns = s.apply("their", a)
                worst = min(worst, max_val(ns, depth - 1))
        return worst

    def chance_val(s: CombatState, depth: int) -> float:
        return min_val(s, depth)

    best_val = -1e9
    best_act: AdvAction = "Attack"
    for a in ADV_ACTIONS:
        if a == "Attack":
            v = 0.0
            for prob, ns in _expectimax_apply_chance(state, "our", state.their_defend_next):
                v += prob * min_val(ns, max_depth - 1)
        else:
            v = min_val(state.apply("our", a), max_depth - 1)
        if v > best_val:
            best_val = v
            best_act = a
    expansions += 1
    return best_val, best_act, expansions


# ===========================================================================
# High-level wrappers → unified 5-tuple decision_fn shape
#
#   When to engage adversarial mode:
#     * If actor is not alive OR position is None → fall back to baseline.
#     * If an enemy is within Chebyshev ≤ 2 (adjacent melee) we're in combat
#       → build a small "us vs nearest enemy" CombatState from the actual current
#       HP, run the search, map the returned AdvAction → (Action enum, target,
#       step_tup, algo_name, reason) 5-tuple.
#     * Otherwise (not in combat melee range): fall back to baseline so characters
#       still walk to the dungeon exit when no enemy is near.
# ===========================================================================

# String labels used in the Action Log ("Algorithm:" field — exact per §3.C contract)
MINIMAX_LABEL      = "Minimax (full-game-tree, adversarial)"
CUTOFF_LABEL       = "Cutoff Search (Minimax depth-3, HP-diff heuristic leaves)"
ALPHABETA_LABEL    = "Minimax + Alpha-Beta Pruning"
EXPECTIMAX_LABEL   = "Expectimax (0.2 miss / 0.7 hit / 0.1 crit chance nodes)"


def _build_adversarial_decision(
    actor: BaseCharacter,
    dungeon: Dungeon,
    party: list[BaseCharacter],
    enemies: list[BaseCharacter],
    round_num: int,
    search_name: Literal["minimax", "alphabeta", "expectimax", "cutoff"],
) -> tuple[Action, Optional[BaseCharacter], Optional[tuple[int, int]], str, str]:
    if not actor.alive or actor.position is None:
        return baseline_decision.choose_action(actor, dungeon, party, enemies, round_num)

    # 1. Find nearest OPPONENT within melee range. The opponent list depends on
    #    which side the actor is on: a party member fights enemies, an enemy
    #    fights the party.  This avoids an enemy selecting itself (Chebyshev 0)
    #    as its own target.
    is_enemy_actor = actor.symbol in ENEMY_SYMBOLS
    opponents = party if is_enemy_actor else enemies

    target: Optional[BaseCharacter] = None
    best_d: Optional[int] = None
    for e in opponents:
        if not e.alive or e.position is None:
            continue
        if e is actor:
            continue
        d = chebyshev(actor.position, e.position)
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_d = d
            target = e
    if target is None or best_d is None or best_d > 2:
        return baseline_decision.choose_action(actor, dungeon, party, enemies, round_num)

    # 2. Build a tiny 1v1 CombatState. Max HP is capped for clarity.
    cap = max(10, actor.hp_max, target.hp_max)
    our_hp = max(0, actor.hp); their_hp = max(0, target.hp)
    state = CombatState(our_hp=our_hp, their_hp=their_hp, our_turn=True, max_hp=cap)
    label = MINIMAX_LABEL
    if search_name == "minimax":
        value, adv_action, expanded = minimax_search(state)
        label = MINIMAX_LABEL
    elif search_name == "alphabeta":
        value, adv_action, expanded = alphabeta_search(state)
        label = ALPHABETA_LABEL
    elif search_name == "expectimax":
        value, adv_action, expanded = expectimax_search(state)
        label = EXPECTIMAX_LABEL
    elif search_name == "cutoff":
        value, adv_action, expanded = cutoff_search(state, max_depth=3)
        label = CUTOFF_LABEL
    else:
        return baseline_decision.choose_action(actor, dungeon, party, enemies, round_num)

    # 3. Map string AdvAction → game.actions.Action enum + optional target + step
    if adv_action == "Attack":
        reason = (
            f"One-v-one subtree against nearest {target.name} searched {expanded} "
            f"states, returned action Attack (root value {value:+.2f}).  Enemy in "
            f"melee range (Chebyshev {best_d})."
        )
        return (Action.ATTACK, target, None, label, reason)
    if adv_action == "Heal":
        # Heal target = ourselves (we only have self-heal in simple rules).
        reason = (
            f"Healing self — adversarial subtree ({expanded} states) recommends recovery "
            f"(root value {value:+.2f}). Our HP {our_hp}/{cap} is low relative to threat."
        )
        return (Action.HEAL, actor, None, label, reason)
    # Defend
    reason = (
        f"Defensive stance — adversarial subtree ({expanded} states) recommends Defend "
        f"(root value {value:+.2f}) to halve next hit from {target.name}."
    )
    return (Action.DEFEND, None, None, label, reason)


def minimax_decision(actor, dungeon, party, enemies, round_num):
    return _build_adversarial_decision(actor, dungeon, party, enemies, round_num, "minimax")

def alphabeta_decision(actor, dungeon, party, enemies, round_num):
    return _build_adversarial_decision(actor, dungeon, party, enemies, round_num, "alphabeta")

def expectimax_decision(actor, dungeon, party, enemies, round_num):
    return _build_adversarial_decision(actor, dungeon, party, enemies, round_num, "expectimax")

def cutoff_decision(actor, dungeon, party, enemies, round_num):
    return _build_adversarial_decision(actor, dungeon, party, enemies, round_num, "cutoff")
