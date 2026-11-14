"""Tests for game/characters.py (8 tests)."""
from tests._setup import *  # noqa: F401,F403

import unittest
from game.characters import Warrior, Healer, Scout, make_default_party


class TestCharacters(unittest.TestCase):
    def test_default_party_size_3(self):
        self.assertEqual(len(make_default_party()), 3)

    def test_warrior_hp_28_ac18(self):
        w = Warrior()
        self.assertEqual(w.hp_max, 28)
        self.assertEqual(w.hp, 28)
        self.assertEqual(w.ac, 18)

    def test_healer_has_heal_parameters(self):
        h = Healer()
        self.assertEqual(h.heal_dice, "1d8")
        self.assertEqual(h.heal_bonus, 2)
        self.assertEqual(h.heal_range, 1)

    def test_scout_symbol_s(self):
        s = Scout()
        self.assertEqual(s.symbol, "S")

    def test_apply_damage_sets_hp_zero_not_negative(self):
        w = Warrior()
        dealt = w.apply_damage(1000)
        self.assertEqual(dealt, 28)
        self.assertEqual(w.hp, 0)
        self.assertEqual(w.status, "unconscious")

    def test_apply_healing_capped_at_hp_max(self):
        w = Warrior()
        w.apply_damage(10)
        self.assertEqual(w.hp, 18)
        healed = w.apply_healing(9999)
        self.assertEqual(healed, 10)  # capped to 28 (max)
        self.assertEqual(w.hp, 28)

    def test_healing_restores_status_alive(self):
        w = Warrior()
        w.apply_damage(1000)
        self.assertEqual(w.status, "unconscious")
        w.apply_healing(1)
        self.assertEqual(w.status, "alive")

    def test_unique_positions_default_none(self):
        party = make_default_party()
        # All start with None position until placed.
        self.assertTrue(all(a.position is None for a in party),
                        "Default party positions are None (game.place() sets them later).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
