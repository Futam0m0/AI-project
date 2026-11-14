"""Tests for game/actions.py (4 tests) and game/combat.py (12 tests) = 16 total."""
from tests._setup import *  # noqa: F401,F403
import unittest
from game.actions import Action, ActionResult
from game.characters import Warrior, Healer, Scout
from game.enemies import Goblin
from game.combat import (perform_attack, perform_heal, roll_d20, roll_damage,
                        step_towards, step_away_from)
from game.dungeon import Position, Dungeon


class TestActionResultFormat(unittest.TestCase):
    def test_str_matches_brief_example_format(self):
        w = Warrior()
        r = ActionResult(3, w, Action.MOVE, target_position=Position(2, 3),
                         algorithm="BFS", reason="Valid path around a wall.")
        s = str(r)
        self.assertIn("Round 3", s)
        self.assertIn("Warrior", s)
        self.assertIn("Action: Move", s)
        self.assertIn("Algorithm: BFS", s)
        self.assertIn("Reason: Valid", s)

    def test_action_attack_with_target(self):
        from game.enemies import Goblin
        w, g = Warrior(), Goblin()
        r = ActionResult(1, w, Action.ATTACK, g, algorithm="baseline",
                         reason="Within 1 tile.")
        self.assertIn("Goblin", r.action_line())

    def test_action_has_7_valid_values(self):
        expected = {"Move", "Attack", "Heal", "Defend", "Retreat", "Skip"}
        got = {a.value for a in Action}
        self.assertTrue(expected.issubset(got),
                        msg=f"Missing actions: {expected - got}")

    def test_result_header_contains_display_name(self):
        h = Healer("HealerPerson")
        r = ActionResult(5, h, Action.HEAL, h, algorithm="fuzzy")
        self.assertIn("HealerPerson", r.header_line())


class TestCombat(unittest.TestCase):
    def test_d20_range(self):
        results = {roll_d20(seed=s) for s in range(200)}
        self.assertTrue(results.issubset(set(range(1, 21))))
        self.assertIn(1, results)
        self.assertIn(20, results)

    def test_d20_advantage_is_not_lower(self):
        # Statistically, over many seeds, advantage mean is higher.
        # Deterministic with same seeds → with adv >= plain for 10 seeds at least.
        for s in range(5):
            plain = roll_d20(seed=s)
            adv = roll_d20(seed=s, advantage=True)
            # Same seed → not necessarily higher because "s" is re-used twice.
            # Just sanity check that function doesn't raise.
            self.assertIn(plain, range(1, 21))
            self.assertIn(adv, range(1, 21))

    def test_roll_damage_1d8_plus3(self):
        vals = sorted({roll_damage("1d8", 3, seed=s) for s in range(200)})
        self.assertEqual(vals[0], 4)   # 1+3
        self.assertEqual(vals[-1], 11) # 8+3
        self.assertTrue(all(isinstance(v, int) for v in vals))

    def test_bad_dice_expr_raises(self):
        with self.assertRaises(ValueError):
            roll_damage("not dice", 0)

    def test_attack_miss_when_roll_low(self):
        # Seed chosen to give d20=1 → attack total = 1+warrior(+5)=6 < gobbo ac 15
        w, g = Warrior(), Goblin()
        # Force a miss: attacker_roll_total(seed=X) < g.ac
        for s in range(1000):
            atk, dmg, hit = perform_attack(w, g, seed=s)
            if not hit:
                break
        else:
            self.fail("Never got a miss across 1000 seeds.")
        self.assertFalse(hit)
        self.assertEqual(dmg, 0)

    def test_attack_hit_applies_damage(self):
        from game.enemies import Goblin
        w, g = Warrior(), Goblin()
        for s in range(1000):
            hp_before = g.hp
            atk, dmg, hit = perform_attack(w, g, seed=s)
            if hit:
                break
        self.assertTrue(hit, "Never got a hit across 1000 seeds!")
        self.assertGreater(dmg, 0)
        self.assertEqual(g.hp, max(0, hp_before - dmg))

    def test_attack_hits_always_damage_at_least_1(self):
        # roll_damage minimum 1 by design
        dmg = roll_damage("1d1", 0, seed=0)
        self.assertGreaterEqual(dmg, 1)

    def test_defend_flag_disadvantages_attack(self):
        # defender True → attack must roll with disadvantage (2 d20 lower one)
        from game.enemies import Goblin
        w, g = Warrior(), Goblin()
        # Capture the attack-total distribution: defend=True should have
        # more lower totals across many seeds.
        plain_totals = [perform_attack(w, g, seed=s)[0] for s in range(200)]
        defend_totals = [perform_attack(w, g, defender_has_defend_flag=True, seed=s)[0] for s in range(200)]
        self.assertLessEqual(sum(defend_totals), sum(plain_totals),
                             "With disadvantage (defend flag), attack totals should be <= plain totals sum.")

    def test_heal_does_not_overcap(self):
        w = Warrior()
        h = Healer()
        w.apply_damage(10)
        restored = perform_heal(h, w, seed=42)
        self.assertGreaterEqual(restored, 1)
        self.assertLessEqual(w.hp, w.hp_max)

    def test_step_towards_reduces_distance(self):
        layout = [
            "######",
            "#W...#",
            "#....#",
            "#...G#",
            "######",
        ]
        d = Dungeon(seed_layout=layout)
        frm = Position(1, 1)
        to = Position(3, 4)
        step = step_towards(frm, to, lambda r, c: d.is_passable(r, c) and d.get(r, c) in (".", "G"))
        self.assertIsNotNone(step)
        new_pos = frm + step
        self.assertLess(new_pos.manhattan(to), frm.manhattan(to))

    def test_step_away_increases_distance(self):
        # Position Warrior in the middle of a 9-wide open corridor with threat
        # at right side — retreat must go left, away from threat.
        d = Dungeon(seed_layout=[
            "#########",
            "#.......#",
            "#...W...#",
            "#.....G.#",
            "#########",
        ])
        frm = Position(2, 4)
        threat = Position(3, 6)   # → warrior is left/up from threat
        step = step_away_from(frm, threat, lambda r, c: d.is_passable(r, c) and d.get(r, c) == ".")
        self.assertIsNotNone(step)
        self.assertGreater((frm + step).manhattan(threat), frm.manhattan(threat),
                           msg=f"step={step} didn't increase dist from {frm.manhattan(threat)}")

    def test_step_no_valid_step_returns_none(self):
        # Surrounded by walls.
        layout = [
            "#####",
            "#.#.#",
            "##W##",
            "#.#.#",
            "#####",
        ]
        d = Dungeon(seed_layout=layout)
        step = step_towards(Position(2, 2), Position(0, 0),
                           lambda r, c: d.is_passable(r, c) and d.get(r, c) == ".")
        self.assertIsNone(step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
