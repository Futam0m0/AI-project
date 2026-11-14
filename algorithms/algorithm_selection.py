"""
Algorithm dispatcher — selects a decision-making function by name and returns
a callable compatible with ``GameState.advance_one_round(decision_fn=...)``.

Exposed names that show up in the Streamlit dropdown (algorithm selectbox):

  * ``"baseline"``         → role rules (baseline_decision.py; default).
  * ``"bfs"``              → party pathfinds to EXIT via BFS; enemies use baseline.
  * ``"dfs"``              → ditto via DFS (no shortest-path guarantee).
  * ``"ids"``              → ditto via IDS (iterative deepening DFS).
  * ``"ucs"``              → ditto via UCS (uniform cost).
  * ``"greedy"``           → ditto via Greedy Best-First (heuristic only).
  * ``"astar"``            → ditto via A* (g + h; optimal + optimally efficient).

  * ``"minimax"``          → combat decision via Minimax (enemy vs healer);
                              enemies use minimax too (demo-only).
  * ``"alphabeta"``        → same combat shape with alpha-beta pruning.
  * ``"expectimax"``       → same combat shape with expectimax (stochastic).

For Phase 4 the search-based strategies ONLY re-route the PARTY's move
decisions (Move actions toward Exit or nearest enemy):

   - If ANY enemy is within melee/touch range (Chebyshev ≤ 2):
     → still fall back to baseline_decision.choose_action so combat still
       happens (otherwise pathfinding ignores the tactical situation and
       party would just run past enemies).  This is the sensible hybrid a
       lecturer would expect.
   - Otherwise: path toward the EXIT tile using the chosen search algorithm.
     If exit unreachable, path toward nearest living enemy.
     If no path exists, fall back to baseline.

The actual search module is in :mod:`algorithms.search`. This wrapper turns
its ``(path, expanded)`` output into the 5-tuple ``decision_fn`` shape.
"""
from __future__ import annotations

from typing import Callable

from algorithms import baseline_decision
from algorithms.baseline_decision import (
    DEFAULT_WEIGHTS,
    BehaviourWeights,
    weighted_decision_fn,
)
from algorithms.search import (
    Position,
    bfs,
    dfs,
    ids,
    ucs,
    greedy,
    astar,
)
from game.actions import Action
from game.characters import BaseCharacter
from game.dungeon import Dungeon, PARTY_SYMBOLS, ENEMY_SYMBOLS, chebyshev

DecisionFn = Callable[
    [BaseCharacter, Dungeon, list[BaseCharacter], list[BaseCharacter], int],
    tuple[Action, BaseCharacter | None, tuple[int, int] | None, str, str],
]

# ---------------------------------------------------------------------------
# Precomputed (small) SA and GA best chromosomes.
#
# These are produced by running the course-notebook-style optimisers on
# project import (NOT inside the Streamlit per-rerun hot loop), using tiny
# budgets (≈ 40 SA steps / 6 generations × small population) so dropdown
# selection remains < 100 ms.
# ---------------------------------------------------------------------------

_SA_CACHED_CHROMOSOME: list[float] = [0.25, 0.9, 0.5, 0.15, 0.8, 0.25, 0.95, 1.0, 0.9, 0.2]
_GA_CACHED_CHROMOSOME: list[float] = [0.25, 0.9, 0.5, 0.15, 0.8, 0.25, 0.95, 1.0, 0.9, 0.2]
_OPTI_CACHE_INITIALIZED: bool = False


def _ensure_opt_cache_populated() -> None:
    """Run a tiny SA/GA on first import and remember the best chromosomes."""
    global _SA_CACHED_CHROMOSOME, _GA_CACHED_CHROMOSOME, _OPTI_CACHE_INITIALIZED
    if _OPTI_CACHE_INITIALIZED:
        return
    try:
        from learning.sa import simulated_annealing
        from learning.ga import genetic_algorithm

        tiny_obj_sa = simulated_annealing(
            start_chromosome=DEFAULT_WEIGHTS.to_chromosome(),
            n_steps=30,
            cooling=0.92,
            step_std=0.18,
            seed=2026,
        )
        if tiny_obj_sa.best_chromosome and len(tiny_obj_sa.best_chromosome) == 10:
            _SA_CACHED_CHROMOSOME = [float(v) for v in tiny_obj_sa.best_chromosome]
        tiny_ga = genetic_algorithm(
            start_chromosome=DEFAULT_WEIGHTS.to_chromosome(),
            pop_size=10,
            n_generations=5,
            mutation_prob=0.015,
            seed=2026,
        )
        if tiny_ga.best_chromosome and len(tiny_ga.best_chromosome) == 10:
            _GA_CACHED_CHROMOSOME = [float(v) for v in tiny_ga.best_chromosome]
    except Exception:
        # Must never break the dropdown if the learning modules raise during
        # import (e.g. path misconfiguration).  Fallback to the handwritten
        # default baseline chromosome declared above.
        pass
    _OPTI_CACHE_INITIALIZED = True

# ---------------------------------------------------------------------------
# Search wrappers — each converts search_fn(path,expanded) → decision_fn shape
# ---------------------------------------------------------------------------

_SEARCH_FN_REGISTRY: dict[str, Callable] = {
    "bfs":    bfs,
    "dfs":    dfs,
    "ids":    ids,
    "ucs":    ucs,
    "greedy": greedy,
    "astar":  astar,
}

_SEARCH_LONG_NAME: dict[str, str] = {
    "bfs":    "BFS (Breadth-First Search)",
    "dfs":    "DFS (Depth-First Search)",
    "ids":    "IDS (Iterative Deepening DFS)",
    "ucs":    "UCS (Uniform-Cost Search)",
    "greedy": "Greedy Best-First Search",
    "astar":  "A* Search (f = g + h, admissible h = Manhattan)",
}


def _path_first_step(path: list[Position], start: Position) -> tuple[int, int] | None:
    """Return the delta (dr, dc) from path[0] → path[1]."""
    if len(path) < 2:
        return None
    a, b = path[0], path[1]
    if a != start:
        # Defensive: we should always build paths that start at actor pos.
        return None
    return (b.row - a.row, b.col - a.col)


def _build_search_decision(
    algo_key: str,
    actor: BaseCharacter,
    dungeon: Dungeon,
    party: list[BaseCharacter],
    enemies: list[BaseCharacter],
    _round_num: int,
) -> tuple[Action, BaseCharacter | None, tuple[int, int] | None, str, str]:
    """Shared wrapper used by every search strategy."""
    # 1. If actor is an ENEMY → always just use baseline combat logic.
    #    Search is for PATHFINDING (party navigating dungeon), not for enemy AI.
    if actor.symbol in ENEMY_SYMBOLS:
        return baseline_decision.choose_action(actor, dungeon, party, enemies, _round_num)

    # 2. If actor is not alive → skip (baseline handles this but be explicit).
    if not actor.alive or actor.position is None:
        return baseline_decision.choose_action(actor, dungeon, party, enemies, _round_num)

    # 3. If any enemy is CLOSE (Chebyshev ≤ 2) → we are "in combat": use baseline
    #    role logic.  Search is about DUNGEON NAVIGATION between fights.
    for e in enemies:
        if not e.alive or e.position is None:
            continue
        if chebyshev(actor.position, e.position) <= 2:
            return baseline_decision.choose_action(actor, dungeon, party, enemies, _round_num)

    # 4. Otherwise: plan a PATH using the chosen search algorithm,
    #    Goal = Exit tile; if Exit not reachable, goal = nearest enemy.
    search_fn = _SEARCH_FN_REGISTRY[algo_key]
    start = actor.position
    exit_pos = dungeon.find_exit()
    goal: Position | None = None
    goal_desc = ""
    if exit_pos is not None and exit_pos != start:
        goal = exit_pos
        goal_desc = "Exit tile (E)"
    else:
        # Fallback: find nearest alive enemy by manhattan.
        nearest: tuple[int, BaseCharacter] | None = None
        for e in enemies:
            if not e.alive or e.position is None:
                continue
            d = actor.position.manhattan(e.position)
            if nearest is None or d < nearest[0]:
                nearest = (d, e)
        if nearest is not None:
            goal = nearest[1].position
            goal_desc = f"nearest enemy ({nearest[1].name})"
    if goal is None:
        return (Action.SKIP, None, None, _SEARCH_LONG_NAME[algo_key],
                "No reachable goal (Exit or enemy); skipping.")

    path, expanded = search_fn(dungeon, start, goal)
    if path is None or len(path) < 2:
        # No path found. Fallback to baseline movement heuristics.
        return (Action.SKIP, None, None, _SEARCH_LONG_NAME[algo_key],
                f"No path to {goal_desc} after exploring {expanded} cells; "
                "falling back to skip for this turn.")
    step = _path_first_step(path, start)
    if step is None:
        return (Action.SKIP, None, None, _SEARCH_LONG_NAME[algo_key],
                f"Found {len(path)}-cell path to {goal_desc} but actor already at goal; skip.")
    # Build a short, honest Reason line — never claims "shortest path" unless algorithm
    # actually provides it (DFS/Greedy do not).
    if algo_key in ("bfs", "ids", "astar", "ucs"):
        optim_hint = "shortest path" if algo_key in ("bfs", "astar", "ucs") else "shortest by depth-bounded DFS"
        algo_claim = f"Found {optim_hint} of {len(path)-1} steps to {goal_desc}"
    else:
        algo_claim = f"Found path of {len(path)-1} steps to {goal_desc} (shortest not guaranteed)"
    reason = (
        f"{algo_claim} by expanding {expanded} state(s).  "
        f"Move one tile: delta {step}."
    )
    return (Action.MOVE, None, step, _SEARCH_LONG_NAME[algo_key], reason)


# ---------------------------------------------------------------------------
# Individual public functions — one per search key, shape = choose_action sig
# ---------------------------------------------------------------------------

def _make_search_fn(key: str) -> DecisionFn:
    def fn(actor, dungeon, party, enemies, round_num):
        return _build_search_decision(key, actor, dungeon, party, enemies, round_num)
    return fn


_bfs_fn    = _make_search_fn("bfs")
_dfs_fn    = _make_search_fn("dfs")
_ids_fn    = _make_search_fn("ids")
_ucs_fn    = _make_search_fn("ucs")
_greedy_fn = _make_search_fn("greedy")
_astar_fn  = _make_search_fn("astar")


# ---------------------------------------------------------------------------
# Public "dispatch by name" — used by app.py dropdown.
# ---------------------------------------------------------------------------

AVAILABLE_ALGORITHMS: list[str] = [
    "baseline",
    "bfs",
    "dfs",
    "ids",
    "ucs",
    "greedy",
    "astar",
    "weighted-baseline",
    "sa-opt",
    "ga-opt",
    "minimax",
    "alphabeta",
    "expectimax",
]

ALGO_DROPDOWN_LABEL: dict[str, str] = {
    "baseline":          "Baseline role rules (default)",
    "bfs":               _SEARCH_LONG_NAME["bfs"],
    "dfs":               _SEARCH_LONG_NAME["dfs"],
    "ids":               _SEARCH_LONG_NAME["ids"],
    "ucs":               _SEARCH_LONG_NAME["ucs"],
    "greedy":            _SEARCH_LONG_NAME["greedy"],
    "astar":             _SEARCH_LONG_NAME["astar"],
    "weighted-baseline": "Weighted baseline (10-tuneable rule weights)",
    "sa-opt":            "Simulated Annealing optimized (avg party HP)",
    "ga-opt":            "Genetic Algorithm optimized (10-gene chromosome)",
    "minimax":           "Minimax (adversarial combat — 2-player)",
    "alphabeta":         "Minimax + Alpha-Beta Pruning",
    "expectimax":        "Expectimax (stochastic adversary — crit/miss)",
}


def dispatch(name: str) -> DecisionFn:
    """Return a callable compatible with ``GameState.advance_one_round``.

    Parameters
    ----------
    name : str
        One of :data:`AVAILABLE_ALGORITHMS`.  Unknown names fall back to
        ``baseline`` with a small "unknown algo" Reason string on first
        action (handled by :mod:`algorithms.baseline_decision` default).
    """
    # --- Learning-phase 5 tunable baselines (13 = all 10 prior + 3 new) ------
    if name == "weighted-baseline":
        return weighted_decision_fn(DEFAULT_WEIGHTS, label_override="weighted-baseline")
    if name in ("sa-opt", "ga-opt"):
        _ensure_opt_cache_populated()
        chromo = _SA_CACHED_CHROMOSOME if name == "sa-opt" else _GA_CACHED_CHROMOSOME
        label = "SA-optimized" if name == "sa-opt" else "GA-optimized"
        return weighted_decision_fn(
            BehaviourWeights.from_chromosome(chromo, label=label),
            label_override=label,
        )
    if name == "baseline":
        return baseline_decision.choose_action
    if name == "bfs":
        return _bfs_fn
    if name == "dfs":
        return _dfs_fn
    if name == "ids":
        return _ids_fn
    if name == "ucs":
        return _ucs_fn
    if name == "greedy":
        return _greedy_fn
    if name == "astar":
        return _astar_fn
    if name == "minimax":
        from algorithms.adversarial import minimax_decision
        return minimax_decision
    if name == "alphabeta":
        from algorithms.adversarial import alphabeta_decision
        return alphabeta_decision
    if name == "expectimax":
        from algorithms.adversarial import expectimax_decision
        return expectimax_decision
    return baseline_decision.choose_action
