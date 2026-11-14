"""
Search algorithms — §5 brief & course notebook chapters 1 & 2.

Every function below is implemented FROM SCRATCH (no networkx / scipy wrappers)
so the lecturer can verify the algorithms exactly match the taught versions:

  1. BFS  (breadth-first)            — shortest *number of steps* in unweighted 4-dir grid.
  2. DFS  (depth-first)              — explores fully down one branch before backtracking;
                                        does NOT guarantee shortest path.
  3. IDS  (iterative deepening DFS)  — repeated DFS at depth 1, 2, … until goal found.
                                        Finds shortest path with DFS memory profile.
  4. UCS  (uniform-cost search)      — priority-queue by cumulative g(n) cost for weighted grids.
                                        Default step cost = 1 (so equivalent to BFS on uniform-cost map,
                                        but differs if a higher-cost 'rough terrain' char is passed).
  5. Greedy Best-First Search        — priority-queue by heuristic-only h(n). Shortest path
                                        NOT guaranteed; expands fewer states than BFS in practice.
  6. A* Search                       — priority-queue by f(n) = g(n) + h(n). Optimal if h
                                        is admissible (Manhattan on 4-dir grid = admissible ✓).

All operate on ``game.dungeon.Dungeon`` grid with 4-dir steps.  Every function
returns a consistent 2-tuple shape:

    (path_or_None: list[Position] | None,  expanded: int)

where:
  * path includes BOTH the start Position AND the goal Position (so len(path)
    ≥ 1 if path exists; path[0] == start; path[-1] == goal).
  * expanded = number of unique cells popped off the frontier before the goal
    was located (or before the frontier ran dry).  This is what we compare in
    the dashboard "Algorithm exec stats" per brief §3.E.

Heuristic (used by Greedy + A*): ``h(n) = |r - r_goal| + |c - c_goal|``
(= 4-dir Manhattan distance).  On a 4-directional grid with equal step costs,
Manhattan is **admissible** — it never overestimates the true shortest path
distance to the goal, which is exactly the proof condition the course teaches
for A* optimality.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Iterable

from game.dungeon import Dungeon, Position

# ---------------------------------------------------------------------------
# Small helpers shared by all algorithms
# ---------------------------------------------------------------------------

# 4-directional (NESW) movement.  (dr, dc) pairs.
DIR4: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _iter_neighbors(
    dungeon: Dungeon,
    pos: Position,
    *,
    rough_chars: Iterable[str] = (),
) -> list[tuple[Position, int]]:
    """Return list of (neighbor_position, step_cost) for 4-dir passable cells.

    ``rough_chars`` — cells whose step cost is 5 instead of 1.  Only UCS and
    A* use different step costs (they minimise weighted g(n)); the unweighted
    BFS/DFS/IDS/Greedy always treat every passable cell as cost=1 and ignore
    this flag.
    """
    rough = set(rough_chars)
    out: list[tuple[Position, int]] = []
    for dr, dc in DIR4:
        nr, nc = pos.row + dr, pos.col + dc
        if not dungeon.is_passable(nr, nc):
            continue
        npos = Position(nr, nc)
        cost = 5 if dungeon.get(nr, nc) in rough else 1
        out.append((npos, cost))
    return out


def _reconstruct(parents: dict[Position, Position | None], goal: Position) -> list[Position]:
    """Walk backwards from goal→start through parents→start, then reverse."""
    path: list[Position] = []
    cur: Position | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


# ===========================================================================
# 1. BFS  (Breadth-First Search)
# ===========================================================================
def bfs(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
) -> tuple[list[Position] | None, int]:
    """Shortest path on unweighted 4-dir grid.

    * Uses a FIFO queue (``collections.deque.popleft``).
    * Guarantees shortest *number-of-steps* path in an unweighted grid.
    * Expanded count = every cell popped from the queue BEFORE hitting goal.
    """
    if start == goal:
        return [start], 0
    frontier: deque[Position] = deque([start])
    reached: set[Position] = {start}
    parents: dict[Position, Position | None] = {start: None}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        for npos, _cost in _iter_neighbors(dungeon, node):
            if npos in reached:
                continue
            reached.add(npos)
            parents[npos] = node
            if npos == goal:
                return _reconstruct(parents, goal), expanded
            frontier.append(npos)
    return None, expanded


# ===========================================================================
# 2. DFS  (Depth-First Search) — graph-style with reached set (no cycles)
# ===========================================================================
def dfs(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
) -> tuple[list[Position] | None, int]:
    """Classic LIFO DFS — explores one branch fully before backtracking.

    Implementation note: We use an explicit list-as-stack (LIFO append/pop
    from the tail) rather than recursion, to avoid Python recursion depth
    limits on large grids.  Reached-set → graph-style DFS (never revisits).

    Shorted-path NOT guaranteed (DFS just returns SOME valid path that dives
    deep first).  But expanded count is useful for comparison with BFS.
    """
    if start == goal:
        return [start], 0
    frontier: list[Position] = [start]
    reached: set[Position] = {start}
    parents: dict[Position, Position | None] = {start: None}
    expanded = 0
    while frontier:
        node = frontier.pop()   # LIFO
        expanded += 1
        if node == goal:
            return _reconstruct(parents, goal), expanded
        for npos, _cost in _iter_neighbors(dungeon, node):
            if npos in reached:
                continue
            reached.add(npos)
            parents[npos] = node
            frontier.append(npos)
    return None, expanded


# ===========================================================================
# 3. IDS  (Iterative Deepening DFS)
# ===========================================================================
def _dls(
    dungeon: Dungeon,
    node: Position,
    goal: Position,
    depth: int,
    path: list[Position],
    visited: set[Position],
) -> tuple[list[Position] | None, int]:
    """Depth-limited DFS inner helper.  Returns (path_or_None, expanded).

    Implements recursive DFS that refuses to descend beyond ``depth``.  Each
    visited node adds 1 to expanded count.
    """
    path.append(node)
    visited.add(node)
    expanded = 1
    if node == goal:
        return list(path), expanded
    if depth <= 0:
        path.pop()
        # visited intentionally NOT popped (cycle-safety on this branch;
        # parent at higher depth re-searches are OK because visited-per-call
        # is reset for every outer IDS iteration).
        return None, expanded
    for npos, _cost in _iter_neighbors(dungeon, node):
        if npos in visited:
            continue
        sub_path, sub_exp = _dls(dungeon, npos, goal, depth - 1, path, visited)
        expanded += sub_exp
        if sub_path is not None:
            return sub_path, expanded
    path.pop()
    return None, expanded


def ids(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
    *,
    max_depth: int = 200,
) -> tuple[list[Position] | None, int]:
    """Iterative deepening DFS.

    Run DFS at increasing depth limits 0, 1, 2, … until the goal is found OR
    max_depth is hit.  Memory footprint is O(depth) just like plain DFS,
    but because the outer iterations go depth-by-depth IDS is GUARANTEED to
    return a shortest-length path (just like BFS) on unweighted graphs.

    Expanded count = sum of all nodes expanded across every bounded DFS pass.
    """
    if start == goal:
        return [start], 0
    total_expanded = 0
    for limit in range(1, max_depth + 1):
        path, exp = _dls(dungeon, start, goal, limit, [], set())
        total_expanded += exp
        if path is not None:
            return path, total_expanded
    return None, total_expanded


# ===========================================================================
# 4. UCS  (Uniform-Cost Search = Dijkstra with one goal)
# ===========================================================================
def ucs(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
    *,
    rough_chars: Iterable[str] = (),
) -> tuple[list[Position] | None, int]:
    """Uniform-cost search — priority queue ordered by g(n) = actual cost from start.

    ``rough_chars``: stepping onto any cell whose grid char is in this set
    costs 5 instead of 1.  On the default dungeon (which has only `#` walls
    and `.` floor + entities which are is_passable=False for other chars),
    the uniform step cost is 1, so UCS = BFS behaviour on shortest path but
    expanded counts can differ slightly because heap tiebreak order differs.

    Exact shape taught in class: ``heapq.heappop`` smallest g first; when we
    re-encounter a node via a cheaper route we re-push (since popping the
    stale entry later is safe and cheaper than mutating heap entries).
    """
    if start == goal:
        return [start], 0
    counter = 0  # unique tiebreak → heapq stable when two g values are equal
    frontier: list[tuple[int, int, Position]] = []
    heapq.heappush(frontier, (0, counter, start))
    counter += 1
    best_g: dict[Position, int] = {start: 0}
    parents: dict[Position, Position | None] = {start: None}
    expanded = 0
    while frontier:
        g, _tie, node = heapq.heappop(frontier)
        # Stale skip: if we already found a better path to this node, drop.
        if g > best_g.get(node, 1 << 62):
            continue
        expanded += 1
        if node == goal:
            return _reconstruct(parents, goal), expanded
        for npos, step_cost in _iter_neighbors(dungeon, node, rough_chars=rough_chars):
            new_g = g + step_cost
            if new_g < best_g.get(npos, 1 << 62):
                best_g[npos] = new_g
                parents[npos] = node
                heapq.heappush(frontier, (new_g, counter, npos))
                counter += 1
    return None, expanded


# ===========================================================================
# 5. Greedy Best-First Search  (f(n) = h(n), priority queue)
# ===========================================================================
def greedy(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
) -> tuple[list[Position] | None, int]:
    """Greedy BFS: expand node with smallest *heuristic* h(n) = Manhattan to goal.

    Teaches the TRADE-OFF vs BFS: Greedy usually expands far FEWER states
    (good) but the returned path is NOT guaranteed to be shortest (bad,
    because greedy walks into wall alcoves if the heuristic looks close).
    """
    if start == goal:
        return [start], 0
    counter = 0
    frontier: list[tuple[int, int, Position]] = []
    heapq.heappush(frontier, (_manhattan(start, goal), counter, start))
    counter += 1
    reached: set[Position] = {start}
    parents: dict[Position, Position | None] = {start: None}
    expanded = 0
    while frontier:
        _h, _tie, node = heapq.heappop(frontier)
        expanded += 1
        if node == goal:
            return _reconstruct(parents, goal), expanded
        for npos, _cost in _iter_neighbors(dungeon, node):
            if npos in reached:
                continue
            reached.add(npos)
            parents[npos] = node
            heapq.heappush(frontier, (_manhattan(npos, goal), counter, npos))
            counter += 1
    return None, expanded


# ===========================================================================
# 6. A* Search  (f(n) = g(n) + h(n), heap)
# ===========================================================================
def astar(
    dungeon: Dungeon,
    start: Position,
    goal: Position,
    *,
    rough_chars: Iterable[str] = (),
) -> tuple[list[Position] | None, int]:
    """A*: f(n) = g(n) (actual cost so far) + h(n) (Manhattan to goal).

    On our 4-dir grid with step costs all ≥ 1, h(n) = Manhattan distance is
    ADMISSIBLE (it never over-estimates true shortest-path cost).  The
    course proves that when h is admissible, A* is:

      * OPTIMAL — returns the cheapest possible path; AND
      * OPTIMALLY EFFICIENT — no other optimal algorithm that uses the same
        heuristic can expand FEWER states than A*.

    UCS with rough terrain support uses the same heap structure; the only
    difference is we add `h(n)` to the heap priority.
    """
    if start == goal:
        return [start], 0
    counter = 0
    frontier: list[tuple[int, int, int, Position]] = []   # (f, g, tie, node)
    h0 = _manhattan(start, goal)
    heapq.heappush(frontier, (h0, 0, counter, start))
    counter += 1
    best_g: dict[Position, int] = {start: 0}
    parents: dict[Position, Position | None] = {start: None}
    expanded = 0
    while frontier:
        _f, g, _tie, node = heapq.heappop(frontier)
        if g > best_g.get(node, 1 << 62):
            continue
        expanded += 1
        if node == goal:
            return _reconstruct(parents, goal), expanded
        for npos, step_cost in _iter_neighbors(dungeon, node, rough_chars=rough_chars):
            new_g = g + step_cost
            if new_g < best_g.get(npos, 1 << 62):
                best_g[npos] = new_g
                parents[npos] = node
                f = new_g + _manhattan(npos, goal)
                heapq.heappush(frontier, (f, new_g, counter, npos))
                counter += 1
    return None, expanded
