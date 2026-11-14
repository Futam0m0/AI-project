"""Tests for algorithms/baseline_decision.py — 10 tests (Step 2.8 plan)."""
from tests._setup import *  # noqa: F401,F403
import unittest
from algorithms import baseline_decision
from game.characters import Warrior, Healer, Scout, make_default_party
from game.enemies import Goblin, make_default_encounter_three_enemies
from game.dungeon import Dungeon, Position


def small_fight_map() -> list[str]:
    return [
        "##########",
        "#WHS..G..#",
        "#........#",
        "#..G.....#",
        "#......R.#",
        "#.......E#",
        "##########",
    ]


class TestBaselineDecision(unittest.TestCase):
    def _setup(self):
        d = Dungeon(seed_layout=small_fight_map())
        party = make_default_party()
        party[0].position = Position(1, 1); d.place("W", 1, 1)
        party[1].position = Position(1, 2); d.place("H", 1, 2)
        party[2].position = Position(1, 3); d.place("S", 1, 3)
        enemies = make_default_encounter_three_enemies()
        # Place goblin1 = G at (1,6), goblin2 = G at (3,3), ruffian = R at (4,7)
        g1, g2, r = enemies
        g1.position = Position(1, 6); d.place("G", 1, 6)
        g2.position = Position(3, 3); d.place("G", 3, 3)
        r.position = Position(4, 7); d.place("R", 4, 7)
        return d, party, enemies

    def test_warrior_attacks_adjacent_goblin(self):
        # Place a goblin adjacent to warrior to trigger attack rule.
        layout = ["#####", "#WG.#", "#####"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior()]
        enemies = [Goblin()]
        party[0].position = Position(1, 1)
        enemies[0].position = Position(1, 2)
        action, target, step, algo, reason = baseline_decision.choose_action(
            party[0], d, party, enemies, 1)
        self.assertEqual(action.value, "Attack")
        self.assertIs(target, enemies[0])

    def test_warrior_moves_towards_nearest_enemy(self):
        d, party, enemies = self._setup()
        action, target, step, algo, _ = baseline_decision.choose_action(
            party[0], d, party, enemies, 1)
        self.assertEqual(action.value, "Move")
        self.assertIsNotNone(step)
        # Should not attack because nearest goblin at (1,6) is 5 away.
        # Step should be towards right/down direction.
        self.assertIn(step, [(0, 1), (1, 0)])
        self.assertEqual(algo, "baseline")

    def test_healer_heals_injured_ally_in_range(self):
        layout = ["#####", "#WH.#", "#####"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("Warr", Position(1, 1)), Healer("Mara", Position(1, 2))]
        # Injure warrior to hp=5 (<50%).
        party[0].hp = 5
        enemies = []
        action, target, step, algo, _ = baseline_decision.choose_action(
            party[1], d, party, enemies, 1)
        self.assertEqual(action.value, "Heal")
        self.assertIs(target, party[0])

    def test_healer_heals_ally_between_50_and_70_percent(self):
        """Healing threshold is 70%: an ally at ~64% HP must be healed even
        though it is above the old 50% threshold."""
        layout = ["#####", "#WH.#", "#####"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("Warr", Position(1, 1)), Healer("Mara", Position(1, 2))]
        # Warrior hp_max = 28; hp=18 -> 64.3% (between 50% and 70%).
        party[0].hp = 18
        self.assertGreater(party[0].hp_fraction, 0.5)
        self.assertLess(party[0].hp_fraction, 0.7)
        enemies = []
        action, target, *_ = baseline_decision.choose_action(
            party[1], d, party, enemies, 1)
        self.assertEqual(action.value, "Heal",
                         f"Warrior at {party[0].hp_fraction:.0%} HP is below the 70% threshold and must be healed.")
        self.assertIs(target, party[0])

    def test_healer_does_not_heal_ally_above_70_percent(self):
        """Above the 70% threshold the healer must not heal (no injured ally)."""
        layout = ["#####", "#WH.#", "#####"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("Warr", Position(1, 1)), Healer("Mara", Position(1, 2))]
        # Warrior hp=20 -> 71.4% (above 70%).
        party[0].hp = 20
        self.assertGreaterEqual(party[0].hp_fraction, 0.7)
        enemies = []
        action, *_ = baseline_decision.choose_action(
            party[1], d, party, enemies, 1)
        self.assertNotEqual(action.value, "Heal",
                            f"Warrior at {party[0].hp_fraction:.0%} HP is above the 70% threshold.")

    def test_healer_attacks_enemy_adjacent_when_no_injury(self):
        layout = ["#######", "#WHG..#", "#######"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("W", Position(1, 1)),
                 Healer("H", Position(1, 2))]
        enemies = [Goblin("G", Position(1, 3))]
        action, target, *_ = baseline_decision.choose_action(
            party[1], d, party, enemies, 1)
        self.assertEqual(action.value, "Attack")
        self.assertIs(target, enemies[0])

    def test_scout_attacks_weakest_within_1(self):
        # S at (1,1).  Place Weak Goblin adj at (1,2), Strong at (2,1) also adj.
        # Both within 1 — scout MUST pick the weakest (weak goblin).
        layout = ["######",
                  "#S..##",
                  "#...##",
                  "######"]
        d = Dungeon(seed_layout=layout)
        party = [Scout("Scout", Position(1, 1))]
        g_strong = Goblin("Strong", Position(2, 1)); g_strong.hp = 7
        g_weak   = Goblin("Weak",   Position(1, 2)); g_weak.hp   = 2
        enemies = [g_strong, g_weak]
        # Place them on dungeon grid too.
        d.place("G", *g_strong.position)  # grid slot 2,1
        d.place("G", *g_weak.position)
        action, target, *_ = baseline_decision.choose_action(
            party[0], d, party, enemies, 1)
        self.assertEqual(action.value, "Attack")
        self.assertIs(target, g_weak, "Scout must prefer weakest enemy within 1.")

    def test_scout_moves_towards_exit_not_enemy_when_far(self):
        d, party, enemies = self._setup()
        # Scout at (1,3) — no enemy within 1 → exit at (5,8) is priority.
        action, target, step, *_ = baseline_decision.choose_action(
            party[2], d, party, enemies, 1)
        self.assertEqual(action.value, "Move")
        # Step should head towards exit row 5, col 8.
        self.assertIn(step, [(1, 0), (0, 1)])

    def test_enemy_moves_towards_nearest_party(self):
        layout = ["#######", "#W..G.#", "#######"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("W", Position(1, 1))]
        g = Goblin("G", Position(1, 4))
        enemies = [g]
        action, target, step, *_ = baseline_decision.choose_action(
            g, d, party, enemies, 1)
        self.assertEqual(action.value, "Move")
        self.assertIs(target, party[0])
        self.assertIn(step, [(0, -1)])  # step left towards W.

    def test_enemy_attacks_when_adjacent(self):
        layout = ["#####", "#WG.#", "#####"]
        d = Dungeon(seed_layout=layout)
        party = [Warrior("W", Position(1, 1))]
        g = Goblin("G", Position(1, 2))
        enemies = [g]
        action, target, *_ = baseline_decision.choose_action(
            g, d, party, enemies, 1)
        self.assertEqual(action.value, "Attack")
        self.assertIs(target, party[0])

    def test_all_algorithms_field_is_baseline(self):
        d, party, enemies = self._setup()
        # Make a list of every actor's decision — algorithm must be "baseline".
        for actor in list(party) + list(enemies):
            _, _, _, algo, _ = baseline_decision.choose_action(
                actor, d, party, enemies, 1)
            self.assertEqual(algo, "baseline",
                             msg=f"{actor.name} reported non-baseline algorithm {algo!r}.")

    def test_unconscious_entity_returns_skip(self):
        w = Warrior(); w.hp = 0; w.status = "unconscious"
        w.position = Position(1, 1)
        d = Dungeon(seed_layout=["#####", "#W..#", "#####"])
        action, *_ = baseline_decision.choose_action(
            w, d, [w], [], 1)
        self.assertEqual(action.value, "Skip")


if __name__ == "__main__":
    unittest.main(verbosity=2)
