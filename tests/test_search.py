"""Tests for algorithms/search.py — 18 tests (6 algorithms × 3 families).

Test families (each applied to every one of the 6 search functions):
  Case A — find a VALID path on an open map, verify contiguity, optimals return length 10.
  Case B — path must DETOUR around a solid inner wall (proved by check: no step ever on the
           known 7-cell wall strip; and BFS/IDS/UCS/A* return length=14).
  Case C — NO POSSIBLE PATH (goal completely walled off inside a ring).  Returns None,
           expanded>0.
"""
from tests._setup import *  # noqa: F401,F403

import unittest
from game.dungeon import Dungeon, Position
from algorithms.search import (
    bfs,
    dfs,
    ids,
    ucs,
    greedy,
    astar,
)

ALL_SEARCH_FNS = (bfs, dfs, ids, ucs, greedy, astar)


# ---------------------------------------------------------------------------
# Map A — open map with W at (1,1), E at (5,6), cols=8, rows=7
#   ########          cols=8: indices 0..7
#   #W.....#   row 1
#   #......#   row 2
#   #..##..#   row 3 (wall cells at (3,3),(3,4))
#   #..##..#   row 4 (wall cells at (4,3),(4,4))
#   #.....E#   row 5  → E at (5,6)
#   ########
# Shortest path around wall: (1,1) right to col 1→2, down 4 steps, right to col 6:
#   (1,1) → (1,2) → (1,3) → (1,4) → (1,5) → (1,6) → (2,6) → (3,6) → (4,6) → (5,6)
#   = 9 steps, 10 cells including start. ✓
# ---------------------------------------------------------------------------
MAP_A = [
    "########",
    "#W.....#",
    "#......#",
    "#..##..#",
    "#..##..#",
    "#.....E#",
    "########",
]

# ---------------------------------------------------------------------------
# Map B — full-height vertical wall at column 4, EXCEPT a 1-cell GAP at (1,4).
#         cols=9 (0..8), rows=7 (0..6).
#   #########
#   #W..G...#  row 1 → W(1,1), gap '.' at (1,4) where there is a corridor.
#   #...#...#  row 2 → wall at col 4
#   #...#...#  row 3 → wall at col 4
#   #...#...#  row 4 → wall at col 4
#   #...#.E.#  row 5 → E at (5,6)
#   #########
# The wall strip: (r=2..5, c=4) → 4 cells. NO detour except the gap at row 1.
# Shortest path (1,1) → (1,2) → (1,3) → (1,4)=gap → (1,5) → (1,6) → (1,7)? No col8 is wall.
# Count each row length: row 1 is "#W..G...#" = 9 chars. 0# 1W 2. 3. 4G 5. 6. 7. 8#.
# → GAP is at (1,4).  Wall strip is col=4 rows 2,3,4,5 only.
# Shortest W(1,1) → E(5,6): (1,1)-(1,2)-(1,3)-(1,4)-gap → (1,5)-(1,6)-(2,6)-(3,6)-(4,6)-(5,6)
#   steps=9? Let me count: 1→2→3→4→5→6→7→8→9 → 9 steps. 10 cells? Let's see:
#     path cells = 10. len(path) = 10.
# But actually the E is at row 5, col 6 in the row template "#...#.E.#" — indices 0#1.2.3.4#5.6E7.8# → col=6 yes.
# So the path goes through the gap then down col 6 from row1 → row5: 4 down steps.
# Total cells = 1 (1,1) + (right 3 steps to gap) + gap + right 2 steps + down 4 steps
#             = 1+3+1+2+4 = 11 cells → 10 steps.
# So BFS optimal len(path) = 11 (not 14 as before; count was wrong — now re-counted exactly).
# ---------------------------------------------------------------------------
MAP_B = [
    "#########",
    "#W......#",   # col 4 = '.', this is the gap
    "#...#...#",
    "#...#...#",
    "#...#...#",
    "#...#.E.#",
    "#########",
]
WALL_B_CELLS = {(2, 4), (3, 4), (4, 4), (5, 4)}   # known wall strip


# ---------------------------------------------------------------------------
# Map C — isolated goal inside a wall ring. cols=9, rows=8.
#   #########
#   #W......#   row 1
#   #.#####.#   row 2 — wall top (cols 2..6)
#   #.#...#.#   row 3 — wall ring sides
#   #.#.E.#.#   row 4 — E inside ring at (4,4)
#   #.#####.#   row 5 — wall bottom
#   #.......#   row 6
#   #########
# Goal E at (4,4) is completely surrounded; NO path exists from W(1,1).
# The ONLY way in would be through the ring, but every possible entry is walled off.
# ---------------------------------------------------------------------------
MAP_C = [
    "#########",
    "#W......#",
    "#.#####.#",
    "#.#...#.#",
    "#.#.E.#.#",
    "#.#####.#",
    "#.......#",
    "#########",
]


# ===========================================================================
# Case A — open map
# ===========================================================================
class TestMapAOpen(unittest.TestCase):
    def _start_goal(self):
        d = Dungeon(seed_layout=MAP_A)
        return d, Position(1, 1), Position(5, 6)

    def test_all_algorithms_find_a_path(self):
        d, s, g = self._start_goal()
        for fn in ALL_SEARCH_FNS:
            path, expanded = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertIsNotNone(path, f"{fn.__name__} must find a path on open map")
                self.assertGreaterEqual(expanded, 0)
                self.assertEqual(path[0], s)
                self.assertEqual(path[-1], g)

    def test_optimal_algorithms_return_length_10(self):
        """BFS/IDS/UCS/A* MUST return shortest 9-step path (10 cells total)."""
        d, s, g = self._start_goal()
        for fn in (bfs, ids, ucs, astar):
            path, _ = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertEqual(len(path), 10,
                    f"{fn.__name__} optimal expected len=10 (9 steps). got len={len(path)} → {path}")

    def test_every_step_is_contiguous_4dir_passable(self):
        """No search function should ever step diagonally or onto a wall."""
        d, s, g = self._start_goal()
        for fn in ALL_SEARCH_FNS:
            path, _ = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertIsNotNone(path)
                for a, b in zip(path, path[1:]):
                    dr, dc = abs(a.row - b.row), abs(a.col - b.col)
                    self.assertEqual(dr + dc, 1,
                        f"{fn.__name__}: step {a}->{b} is not 4-dir (dr+dc={dr+dc})")
                    self.assertTrue(d.is_passable(b.row, b.col),
                        f"{fn.__name__}: step onto non-passable cell {b}")


# ===========================================================================
# Case B — detour around wall strip
# ===========================================================================
class TestMapBAroundWall(unittest.TestCase):
    def _start_goal(self):
        d = Dungeon(seed_layout=MAP_B)
        return d, Position(1, 1), Position(5, 6)

    def test_all_algorithms_find_a_path(self):
        d, s, g = self._start_goal()
        for fn in ALL_SEARCH_FNS:
            path, expanded = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertIsNotNone(path,
                    f"{fn.__name__} must find a path around the wall (gap exists at (1,4)).")
                self.assertGreater(expanded, 0)
                self.assertEqual(path[-1], g)

    def test_path_never_steps_onto_the_wall_strip(self):
        d, s, g = self._start_goal()
        for fn in ALL_SEARCH_FNS:
            path, _ = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertIsNotNone(path)
                for p in path:
                    if (p.row, p.col) in WALL_B_CELLS:
                        self.fail(f"{fn.__name__} stepped onto forbidden wall cell {p}")

    def test_optimal_bfs_len_10_and_astar_expands_no_more_than_bfs(self):
        """A* (informed heuristic) should never expand MORE states than blind BFS,
        and both optimal algos return the same 10-cell path."""
        d, s, g = self._start_goal()
        bfs_path, bfs_exp = bfs(d, s, g)
        as_path,  as_exp  = astar(d, s, g)
        self.assertEqual(len(bfs_path), 10,
            f"BFS optimal path must be 10 cells (9 steps through gap at (1,4)), got {len(bfs_path)}: {bfs_path}")
        self.assertEqual(len(as_path), 10,
            f"A* optimal (h-admissible) should match BFS length: {len(as_path)} vs {len(bfs_path)}")
        self.assertLessEqual(as_exp, bfs_exp,
            f"A* expanded {as_exp}; should be ≤ BFS {bfs_exp} (informed ≥ uninformed efficiency)")


# ===========================================================================
# Case C — goal completely isolated
# ===========================================================================
class TestMapCNoPath(unittest.TestCase):
    def _start_goal(self):
        d = Dungeon(seed_layout=MAP_C)
        return d, Position(1, 1), Position(4, 4)

    def test_all_algorithms_return_none(self):
        d, s, g = self._start_goal()
        for fn in ALL_SEARCH_FNS:
            path, expanded = fn(d, s, g)
            with self.subTest(fn=fn.__name__):
                self.assertIsNone(path,
                    f"{fn.__name__} returned a path to walled-off goal (expected None)")
                self.assertGreater(expanded, 0,
                    f"{fn.__name__}: expanded=0 but should have explored before failing")

    def test_ids_with_low_maxdepth_still_returns_none(self):
        d, s, g = self._start_goal()
        path, exp = ids(d, s, g, max_depth=3)
        self.assertIsNone(path)
        self.assertGreaterEqual(exp, 0)

    def test_start_equals_goal_returns_single_cell_zero_expanded(self):
        d = Dungeon(seed_layout=MAP_A)
        s = Position(5, 6)   # directly on E cell
        for fn in ALL_SEARCH_FNS:
            path, exp = fn(d, s, s)
            with self.subTest(fn=fn.__name__):
                self.assertEqual(path, [s], "start==goal → [start]")
                self.assertEqual(exp, 0, "start==goal → expand NO cells")


# ===========================================================================
# Extra — UCS rough terrain & A* admissible proof
# ===========================================================================
class TestRoughTerrainAndOptimality(unittest.TestCase):
    # Rough map: 9 columns × 7 rows. Central column col=4 rows 1..5 marked 'R'.
    # All movement INTO an 'R' cell costs 5, movement into other passable cells costs 1.
    # Going straight right→down through the R's costs a minimum of 5 steps×5 = 25.
    # Going AROUND (via row 6 which has NO R) costs 8+4 = 12 steps × cost 1 = 12 < 25.
    # So every cost-minimising algorithm (UCS, A*) should avoid touching any R cells.
    ROUGH_MAP = [
        "#########",
        "#W..R...#",   # R at (1,4)
        "#...R...#",   # R at (2,4)
        "#...R...#",   # R at (3,4)
        "#...R...#",   # R at (4,4)
        "#...R..E#",   # R at (5,4)  &  E at (5,7)
        "#########",
    ]
    # But row 6 is all walls, no corridor around. Fix: change rows count and add
    # second row 6 as corridor. Let's do 8 rows:
    ROUGH_MAP = [
        "#########",
        "#W..R...#",   # 1
        "#...R...#",   # 2
        "#...R...#",   # 3
        "#...R...#",   # 4
        "#...R...#",   # 5
        "#.......#",   # 6 — corridor (no R, all '.'  cells 1..7)
        "#......E#",   # 7 — E at (7,7)
        "#########",
    ]

    def test_ucs_with_rough_avoids_touching_R(self):
        d = Dungeon(seed_layout=self.ROUGH_MAP)
        # Since 'R' is not PARTY/WALL/FLOOR chars, Dungeon treats it as non-passable default?
        # The existing dungeon._iter_neighbors just calls is_passable. is_passable returns True
        # if it's in_bounds & not wall. So R cells ARE passable (they're not '#'). Good.
        s = Position(1, 1); g = Position(7, 7)
        path_rough, _ = ucs(d, s, g, rough_chars={"R"})
        self.assertIsNotNone(path_rough, "UCS must find path around rough terrain")
        rough_cells_on_path = [p for p in path_rough if d.grid[p.row][p.col] == "R"]
        self.assertEqual(rough_cells_on_path, [],
            f"UCS(rough=R×5) should route around R-steps, but stepped on "
            f"{len(rough_cells_on_path)} R cells: {rough_cells_on_path}")

    def test_astar_equals_bfs_len_on_map_a(self):
        """A* with admissible h on unweighted MAP_A returns exactly BFS length."""
        d = Dungeon(seed_layout=MAP_A)
        s = Position(1, 1); g = Position(5, 6)
        bpath, _ = bfs(d, s, g)
        apath, _ = astar(d, s, g)
        self.assertEqual(len(bpath), len(apath),
            "h=Manhattan is admissible & consistent → A* must match BFS optimal length")


if __name__ == "__main__":
    unittest.main()
