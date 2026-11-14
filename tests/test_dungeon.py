"""Tests for game/dungeon.py — 10 tests as per plan (Step 2.2)."""
from tests._setup import *  # noqa: F401,F403  (imports sys.path setup)

import unittest
from game.dungeon import Dungeon, Position, PARTY_SYMBOLS, ENEMY_SYMBOLS, EXIT


def mini_map() -> list[str]:
    """Fully controlled 5x6 dungeon for unit tests:
       ######
       #W..G#
       #.##.#
       #..S.#
       #...HE
       ######
    """
    return [
        "######",
        "#W..G#",
        "#.##.#",
        "#..S.#",
        "#...HE",
        "######",
    ]


class TestDungeonBasics(unittest.TestCase):
    def test_seed_layout_dimensions(self):
        d = Dungeon(seed_layout=mini_map())
        self.assertEqual(d.rows, 6)
        self.assertEqual(d.cols, 6)

    def test_walls_block(self):
        d = Dungeon(seed_layout=mini_map())
        self.assertTrue(d.is_wall(0, 0))
        self.assertTrue(d.is_wall(2, 2))
        self.assertFalse(d.is_wall(1, 1))
        self.assertFalse(d.in_bounds(100, 100))
        self.assertTrue(d.is_wall(100, 100))  # OOB counts as wall

    def test_place_returns_false_on_wall(self):
        d = Dungeon(seed_layout=mini_map())
        self.assertFalse(d.place("G", 0, 0))   # wall
        self.assertFalse(d.place("X", 1, 1))   # W already there
        # Can we place on an empty floor next to W?  Yes: (1,2) was '.'
        self.assertTrue(d.place("R", 1, 2))
        self.assertEqual(d.get(1, 2), "R")

    def test_move_cannot_step_onto_entity(self):
        layout = [
            "####",
            "#WG#",
            "####",
        ]
        d = Dungeon(seed_layout=layout)
        # Step right: W → (1,1) → right would be (1,2)=G (enemy occupied) → False
        self.assertFalse(d.move("W", Position(1, 1), (0, 1)))
        self.assertEqual(d.get(1, 1), "W")
        self.assertEqual(d.get(1, 2), "G")

    def test_move_updates_both_cells(self):
        layout = ["#####", "#W..#", "#####"]
        d = Dungeon(seed_layout=layout)
        self.assertTrue(d.move("W", Position(1, 1), (0, 1)))
        self.assertEqual(d.get(1, 1), ".")
        self.assertEqual(d.get(1, 2), "W")

    def test_find_exit(self):
        d = Dungeon(seed_layout=mini_map())
        self.assertEqual(d.find_exit(), Position(4, 5))

    def test_find_nearest_enemy(self):
        d = Dungeon(seed_layout=mini_map())
        near = d.find_nearest_enemy(Position(3, 3))
        self.assertIsNotNone(near)
        sym, pos = near
        self.assertEqual(sym, "G")
        # Goblin is at (1,4); manhattan from (3,3) = 2+1 = 3. Only enemy, so correct.
        self.assertEqual(pos, Position(1, 4))

    def test_find_nearest_party_for_enemy(self):
        d = Dungeon(seed_layout=mini_map())
        near = d.find_nearest_party(Position(1, 4))  # Goblin looking for party
        self.assertIsNotNone(near)
        sym, pos = near
        # Warrior at (1,1) — distance 3, Scout at (3,3) distance 2+1=3, Healer (4,4) dist 3+0=3 → Warrior is tie-break smaller row,col
        self.assertIn(sym, PARTY_SYMBOLS)

    def test_party_and_enemy_symbols_distinct(self):
        self.assertEqual(PARTY_SYMBOLS & ENEMY_SYMBOLS, set(),
                         "Entity symbols must never overlap so find_nearest_* works.")

    def test_to_string_roundtrip(self):
        m = mini_map()
        d = Dungeon(seed_layout=m)
        out_lines = d.to_string().split("\n")
        self.assertEqual(out_lines, m, "Seed layout must be identical after render.")


class TestDefaultDungeonObstacle(unittest.TestCase):
    """The default 8x10 dungeon ships with one extra internal obstacle.  These
    tests guarantee the obstacle never blocks actor placement, the exit, or
    reachable routes from the party start area."""

    def test_default_dungeon_has_obstacle_at_4_5(self):
        d = Dungeon(rows=8, cols=10)
        self.assertTrue(d.is_wall(4, 5),
                        "Expected the added obstacle (wall) at (4,5) in the default dungeon.")

    def test_exit_reachable_from_party_start(self):
        from collections import deque
        d = Dungeon(rows=8, cols=10)
        start = (1, 1)  # Warrior's default placement
        exit_pos = d.find_exit()
        self.assertIsNotNone(exit_pos)
        seen = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in seen:
                    continue
                if not d.in_bounds(nr, nc) or d.is_wall(nr, nc):
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        self.assertIn((exit_pos.row, exit_pos.col), seen,
                      "Exit must be reachable from the party start area despite the obstacle.")

    def test_default_placement_slots_not_blocked(self):
        from game.game_state import GameState
        g = GameState.new_default(seed=7)
        for a in g.party:
            self.assertIsNotNone(a.position, f"{a.name} was not placed (obstacle blocking?).")
        for e in g.enemies:
            self.assertIsNotNone(e.position, f"{e.name} was not placed (obstacle blocking?).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
