"""Modular algorithm selector — combines three independent selectors.

The 13 existing algorithms are organised into three layers so the user can
mix-and-match approaches instead of picking a single monolithic strategy:

  * **Decision-making** — the "brain": role rules + behaviour weights.
        baseline, weighted-baseline, sa-opt, ga-opt

  * **Navigation** — movement/pathfinding when NO enemy is in melee range.
        baseline-nav (defer to decision-making), bfs, dfs, ids, ucs,
        greedy, astar

  * **Combat** — tactical decisions when an enemy IS within Chebyshev ≤ 2.
        baseline-combat (defer to decision-making), minimax, alphabeta,
        expectimax

Routing rule
------------
For each actor on each turn:

  1. Determine whether the actor is "in combat" (any living opponent within
     Chebyshev distance ≤ 2).
  2. **In combat** → call the *Combat* selector's function.
     If the combat selector is ``baseline-combat``, defer to the
     *Decision-making* function (which handles combat with its own weights).
  3. **Not in combat** → call the *Navigation* selector's function.
     If the navigation selector is ``baseline-nav``, defer to the
     *Decision-making* function (which handles movement with its own weights).

The game engine remains responsible for validating and applying every action;
this module only *routes* to the right existing algorithm.  No algorithm is
rewritten — we reuse :func:`algorithms.algorithm_selection.dispatch` for every
option.

Backward compatibility
----------------------
The original :func:`dispatch` and :data:`AVAILABLE_ALGORITHMS` are untouched,
so the experiment runner and any single-strategy callers keep working.
"""
from __future__ import annotations

from typing import Callable

from algorithms.algorithm_selection import dispatch
from game.characters import BaseCharacter
from game.dungeon import Dungeon, ENEMY_SYMBOLS, chebyshev

# DecisionFn signature: (actor, dungeon, party, enemies, round_num) -> 5-tuple
DecisionFn = Callable[
    [BaseCharacter, Dungeon, list[BaseCharacter], list[BaseCharacter], int],
    tuple,
]

# ---------------------------------------------------------------------------
# Option lists + display labels (shown in the Streamlit sidebar selectboxes)
# ---------------------------------------------------------------------------

DECISION_OPTIONS: list[str] = [
    "baseline",
    "weighted-baseline",
    "sa-opt",
    "ga-opt",
]

NAVIGATION_OPTIONS: list[str] = [
    "baseline-nav",
    "bfs",
    "dfs",
    "ids",
    "ucs",
    "greedy",
    "astar",
]

COMBAT_OPTIONS: list[str] = [
    "baseline-combat",
    "minimax",
    "alphabeta",
    "expectimax",
]

DECISION_LABELS: dict[str, str] = {
    "baseline":          "Baseline role rules",
    "weighted-baseline": "Weighted baseline (10 tunable weights)",
    "sa-opt":            "SA-optimized weights (avg party HP)",
    "ga-opt":            "GA-optimized weights (10-gene chromosome)",
}

NAVIGATION_LABELS: dict[str, str] = {
    "baseline-nav": "Baseline navigation (defer to decision-making)",
    "bfs":          "BFS — Breadth-First Search",
    "dfs":          "DFS — Depth-First Search",
    "ids":          "IDS — Iterative Deepening DFS",
    "ucs":          "UCS — Uniform-Cost Search",
    "greedy":       "Greedy Best-First Search",
    "astar":        "A* Search (f = g + h)",
}

COMBAT_LABELS: dict[str, str] = {
    "baseline-combat": "Baseline combat (defer to decision-making)",
    "minimax":         "Minimax (2-player adversarial)",
    "alphabeta":       "Alpha-Beta Pruning",
    "expectimax":      "Expectimax (stochastic adversary)",
}


# ---------------------------------------------------------------------------
# Combat-range helper
# ---------------------------------------------------------------------------

def _is_in_combat(
    actor: BaseCharacter,
    party: list[BaseCharacter],
    enemies: list[BaseCharacter],
) -> bool:
    """Return True if any living opponent is within Chebyshev distance ≤ 2.

    For a party member the opponents are the enemies; for an enemy the
    opponents are the party.  This mirrors the threshold used by the search
    wrappers and the adversarial module.
    """
    if not actor.alive or actor.position is None:
        return False
    is_enemy_actor = actor.symbol in ENEMY_SYMBOLS
    opponents = party if is_enemy_actor else enemies
    for opp in opponents:
        if not opp.alive or opp.position is None:
            continue
        if opp is actor:
            continue
        if chebyshev(actor.position, opp.position) <= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Public builder — returns a single DecisionFn that routes across layers
# ---------------------------------------------------------------------------

def build_decision_fn(
    decision_name: str,
    navigation_name: str,
    combat_name: str,
) -> DecisionFn:
    """Combine three selectors into one decision function.

    Parameters
    ----------
    decision_name : str
        One of :data:`DECISION_OPTIONS`.  Provides the base role rules and
        weights; used directly when the matching layer is set to ``baseline``.
    navigation_name : str
        One of :data:`NAVIGATION_OPTIONS`.  Controls movement when not in
        combat.  ``baseline-nav`` defers to *decision_name*.
    combat_name : str
        One of :data:`COMBAT_OPTIONS`.  Controls combat when an opponent is
        within Chebyshev ≤ 2.  ``baseline-combat`` defers to *decision_name*.

    Returns
    -------
    DecisionFn
        A callable compatible with ``GameState.advance_one_round``.
    """
    base_fn = dispatch(decision_name)

    # Resolve the navigation and combat functions once (dispatch is cheap and
    # cache-friendly; adversarial imports happen lazily inside dispatch).
    nav_is_baseline = navigation_name == "baseline-nav"
    combat_is_baseline = combat_name == "baseline-combat"

    nav_fn = None if nav_is_baseline else dispatch(navigation_name)
    combat_fn = None if combat_is_baseline else dispatch(combat_name)

    def _combined(
        actor: BaseCharacter,
        dungeon: Dungeon,
        party: list[BaseCharacter],
        enemies: list[BaseCharacter],
        round_num: int,
    ):
        if _is_in_combat(actor, party, enemies):
            if combat_is_baseline:
                return base_fn(actor, dungeon, party, enemies, round_num)
            return combat_fn(actor, dungeon, party, enemies, round_num)
        # Not in combat → navigation layer.
        if nav_is_baseline:
            return base_fn(actor, dungeon, party, enemies, round_num)
        return nav_fn(actor, dungeon, party, enemies, round_num)

    return _combined


def combination_label(
    decision_name: str,
    navigation_name: str,
    combat_name: str,
) -> str:
    """Short human-readable label for the current 3-selector combination."""
    d = DECISION_LABELS.get(decision_name, decision_name)
    n = NAVIGATION_LABELS.get(navigation_name, navigation_name)
    c = COMBAT_LABELS.get(combat_name, combat_name)
    return f"Decision: {d}  |  Nav: {n}  |  Combat: {c}"
