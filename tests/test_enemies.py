"""Tests for game/enemies.py (6 tests)."""
from tests._setup import *  # noqa: F401,F403
import unittest
from game.enemies import Goblin, RedbrandRuffian, Bugbear, make_default_encounter_three_enemies


class TestEnemies(unittest.TestCase):
    def test_goblin_stats_from_srd(self):
        g = Goblin()
        self.assertEqual(g.hp_max, 10)
        self.assertEqual(g.ac, 15)
        self.assertEqual(g.attack_bonus, 5)
        self.assertEqual(g.damage_dice, "1d6")
        self.assertEqual(g.damage_bonus, 2)
        self.assertEqual(g.symbol, "G")

    def test_bugbear_stats_from_srd(self):
        b = Bugbear()
        self.assertEqual(b.hp_max, 27)
        self.assertEqual(b.ac, 16)
        self.assertEqual(b.damage_dice, "2d8")

    def test_redbrand_ruffian_symbol_R(self):
        r = RedbrandRuffian()
        self.assertEqual(r.symbol, "R")
        self.assertLessEqual(r.hp_max, 30)
        self.assertGreaterEqual(r.hp_max, 10)
        self.assertEqual(r.attack_bonus, 4)

    def test_encounter_size_three(self):
        enc = make_default_encounter_three_enemies()
        self.assertEqual(len(enc), 3)

    def test_encounter_symbols_are_enemy_set(self):
        from game.dungeon import ENEMY_SYMBOLS
        enc = make_default_encounter_three_enemies()
        for e in enc:
            self.assertIn(e.symbol, ENEMY_SYMBOLS)

    def test_defeated_enemy_status_after_zero_hp(self):
        g = Goblin()
        g.apply_damage(9999)
        self.assertEqual(g.hp, 0)
        self.assertEqual(g.status, "unconscious")


if __name__ == "__main__":
    unittest.main(verbosity=2)
