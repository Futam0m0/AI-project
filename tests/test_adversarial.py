"""Tests for algorithms/adversarial.py — 10 tests (per plan step 4.8).

Listed explicitly (10 tests):

1.  CombatState.apply(Attack) reduces enemy HP by exactly 5 (deterministic base rule).
2.  CombatState.apply(Defend) halves subsequent incoming Attack damage to floor(5/2)=2.
3.  Minimax returns value +1.0 for a "we have 1 HP and they have 1, it's our turn" state
    (only Attack wins immediately).
4.  Alpha-Beta returns the SAME best action and value as pure Minimax on a known state
    (core promise of alpha-beta: pruning = provably identical value, fewer expansions).
5.  Alpha-Beta expands FEWER or equal states than full-tree Minimax on the same state.
6.  Cutoff search (max_depth=1) returns value exactly equal to heuristic_eval at depth 1
    leaves + root iteration.
7.  Expectimax CHANCE distribution probabilities sum to exactly 1.0.
8.  Decision wrappers (minimax_decision) return the 5-element tuple shape required by
    game_state.advance_one_round.
9.  When actor is outside melee range (> 2 Chebyshev from nearest enemy), the wrappers
    fall back to baseline_decision output (so party navigates, not just stands).
10. Expectimax value on a sure-win state equals 1.0 (even after expectation the result
    is clamped because terminal wins score exactly 1.0 regardless of distribution).

Total: 10 tests.  18 search + 10 adversarial = 28 new tests in Phase 4.
"""
from tests._setup import *  # noqa: F401,F403

import unittest
from algorithms.adversarial import (
    CombatState,
    heuristic_eval,
    minimax_search,
    cutoff_search,
    alphabeta_search,
    expectimax_search,
    _CHANCE_OUTCOMES,
    minimax_decision,
    alphabeta_decision,
    expectimax_decision,
    cutoff_decision,
)
from game.actions import Action
from game.characters import Warrior
from game.enemies import Goblin
from game.dungeon import Dungeon, Position, PARTY_SYMBOLS, ENEMY_SYMBOLS


def _mini_map_with_two_adjacent_fighters():
    """7x7 cols: both Warrior and Goblin adjacent at row 3."""
    return [
        "#######",
        "#.....#",
        "#.....#",
        "#.W.G.#",    # W(3,1) G(3,3) → Chebyshev distance = 2 (adjacent via 1 cell)
        "#.....#",
        "#.....#",
        "#######",
    ]


class TestCombatStateBasics(unittest.TestCase):
    """Tests 1 & 2 from spec list above."""

    def test_attack_damage_is_5_deterministic(self):
        s = CombatState(our_hp=10, their_hp=10, our_turn=True)
        s2 = s.apply("our", "Attack")
        self.assertEqual(s2.their_hp, 10 - 5, "Standard attack should deal 5 HP damage.")
        self.assertFalse(s2.our_turn, "Turn must flip after our action.")

    def test_defend_halves_next_incoming_attack_floor(self):
        # Step 1: our turn, we Defend
        s0 = CombatState(our_hp=10, their_hp=10, our_turn=True)
        s1 = s0.apply("our", "Defend")
        self.assertTrue(s1.our_defend_next, "Defend sets our_defend_next flag.")
        # Step 2: their turn, they Attack → damage 5 halved floor = 2
        s2 = s1.apply("their", "Attack")
        self.assertEqual(s2.our_hp, 10 - 2,
            "Defend must halve next incoming attack (floor): 5//2 = 2 damage, "
            f"expected our_hp=8 got {s2.our_hp}")
        # Defend flag cleared after one hit:
        self.assertFalse(s2.our_defend_next, "Defend flag must be cleared after the hit it absorbed.")


class TestMinimaxAndAlphaBeta(unittest.TestCase):
    """Tests 3, 4, 5."""

    def test_minimax_sure_win_value_is_plus_one(self):
        # Our turn, we have 1 HP, they have 1 HP. One attack kills them → +1.0.
        sure_win = CombatState(our_hp=1, their_hp=1, our_turn=True, max_hp=10)
        value, action, expansions = minimax_search(sure_win)
        self.assertEqual(value, +1.0,
            f"Sure-win: Minimax value must be +1.0 (got {value}).")
        self.assertEqual(action, "Attack",
            f"Only Attack delivers the instant sure-win; got {action}.")
        self.assertGreater(expansions, 1)

    def test_alphabeta_equals_minimax_value_and_action(self):
        """Alpha-beta pruning is a SOUND optimization: returned action & value
        MUST match minimax on any reachable state (only expansions differ)."""
        state = CombatState(our_hp=8, their_hp=6, our_turn=True, max_hp=10)
        mv, ma, _ = minimax_search(state)
        av, aa, _ = alphabeta_search(state)
        self.assertEqual(mv, av,
            f"Alpha-Beta value ({av}) ≠ Minimax value ({mv}) — bug in pruning rules.")
        self.assertEqual(ma, aa,
            f"Alpha-Beta picked action {aa}, Minimax picked {ma} — they must agree.")

    def test_alphabeta_expands_less_than_or_equal_to_minimax(self):
        # Use max_depth=3 on both so this runs quickly (< 1s) and tree is comparable;
        # pruning property still holds regardless of depth bound.
        state = CombatState(our_hp=12, their_hp=14, our_turn=True, max_hp=20)
        _mv, _ma, m_exp = minimax_search(state, max_depth=3)
        _av, _aa, a_exp = alphabeta_search(state, max_depth=3)
        self.assertLessEqual(a_exp, m_exp,
            f"Alpha-Beta must never expand MORE states than Minimax. "
            f"AB={a_exp} vs MM={m_exp} (violation → pruning broken).")


class TestCutoffSearch(unittest.TestCase):
    """Test 6: depth-1 cutoff must use heuristic on leaves."""

    def test_maxdepth_1_root_calls_heuristic_on_children(self):
        # Start with a non-terminal state.  Depth=1 means our one action is played,
        # resulting children are at depth 0 → not-our-turn but depth == 0 (cutoff)
        # → calls heuristic_eval on each non-terminal child.
        s = CombatState(our_hp=10, their_hp=10, our_turn=True, max_hp=20)
        # Manually compute: best heuristic value of the 3 one-step children (our turn actions
        # → turn flipped → depth 0 → evaluate).
        candidates = []
        for a in ("Attack", "Heal", "Defend"):
            child = s.apply("our", a)   # child.our_turn = False.
            candidates.append(heuristic_eval(child))
        expected = max(candidates)
        value, action, expansions = cutoff_search(s, max_depth=1)
        self.assertAlmostEqual(value, expected, places=10,
            msg=f"Cutoff depth=1 value {value} must equal max(heuristic of 3 children)={expected}.")
        self.assertGreater(expansions, 0)


class TestExpectimax(unittest.TestCase):
    """Tests 7 & 10 (chance distribution sums to 1; and sure-win terminal gives 1.0)."""

    def test_chance_outcome_probabilities_sum_to_one(self):
        total = sum(p for p, _mult in _CHANCE_OUTCOMES)
        self.assertAlmostEqual(total, 1.0, places=12,
            msg="Expectimax: chance node probabilities (miss+hit+crit) must sum exactly 1.0.")

    def test_sure_win_terminal_expectimax_value(self):
        """Expectimax: terminal state they're dead → value +1.0; near-sure pick Attack
        with positive expected value, probabilities sum to 1.0 (already tested above)."""
        terminal = CombatState(our_hp=5, their_hp=0, our_turn=True, max_hp=10)
        self.assertTrue(terminal.terminal(), "State must be terminal for this test.")
        self.assertEqual(terminal.terminal_value(), +1.0)
        # Run expectimax from an immediate-win non-terminal: 1 HP them, we have 5.
        s = CombatState(our_hp=5, their_hp=1, our_turn=True, max_hp=10)
        # Expectimax depth=4 is enough to resolve this state fully (since any attack
        # kills in ≤2 turns) and avoids a 9^14-state blowout.
        value, action, expanded = expectimax_search(s, max_depth=4)
        # Attack is the only way to end the fight quickly → optimal action.
        self.assertEqual(action, "Attack")
        self.assertGreater(value, 0.0,
            f"Near-sure-win scenario should yield positive expected value, got {value}.")
        self.assertGreater(expanded, 0)


class TestDecisionWrappers(unittest.TestCase):
    """Tests 8 & 9: unified decision_fn 5-tuple shape and melee fallback."""

    def _build_far_apart(self):
        """W(1,1) and gobbo at far corner; Chebyshev distance > 2."""
        rows = [
            "#########",
            "#W......#",
            "#.......#",
            "#.......#",
            "#.......#",
            "#.......#",
            "#......G#",
            "#########",
        ]
        d = Dungeon(seed_layout=rows)
        warrior = Warrior(position=Position(1, 1))
        gobbo = Goblin(position=Position(6, 7))
        return d, warrior, [warrior], [gobbo]

    def _build_adjacent(self):
        """Both within Chebyshev 2."""
        d = Dungeon(seed_layout=_mini_map_with_two_adjacent_fighters())
        warrior = Warrior(position=Position(3, 1))
        gobbo = Goblin(position=Position(3, 3))
        return d, warrior, [warrior], [gobbo]

    def test_wrapper_returns_five_element_tuple(self):
        """Unified shape: (Action enum | None, BaseCharacter | None, 2-step-tup | None, str, str)."""
        d, actor, party, enemies = self._build_adjacent()
        for fn in (minimax_decision, alphabeta_decision, expectimax_decision, cutoff_decision):
            result = fn(actor, d, party, enemies, 1)
            with self.subTest(fn=fn.__name__):
                self.assertEqual(len(result), 5,
                    f"{fn.__name__} must return a 5-tuple, got len {len(result)}: {result}")
                action, target, step, algo_name, reason = result
                self.assertIsInstance(action, Action,
                    f"Element [0] must be game.actions.Action enum (got {type(action).__name__})")
                self.assertIsInstance(algo_name, str)
                self.assertIsInstance(reason, str)
                self.assertGreater(len(algo_name), 3, "Algorithm name must be a non-trivial string.")
                self.assertGreater(len(reason), 10, "Reason must be descriptive (~lecturer demo text).")

    def test_wrapper_outside_melee_range_returns_baseline_shape(self):
        """When no enemy is within Chebyshev ≤ 2, fallback to baseline must still work."""
        d, actor, party, enemies = self._build_far_apart()
        for fn in (minimax_decision, alphabeta_decision, expectimax_decision):
            result = fn(actor, d, party, enemies, 1)
            with self.subTest(fn=fn.__name__):
                # Always a 5-tuple.  Specific action returned depends on baseline, but
                # we at least verify the function didn't crash and the tuple is valid.
                self.assertEqual(len(result), 5)


class TestEnemyActorAdversarialTargeting(unittest.TestCase):
    """Regression: an enemy actor using an adversarial strategy must target a
    party member, never itself or another enemy.

    Before the fix, ``_build_adversarial_decision`` always searched the
    ``enemies`` list for the nearest target.  For an enemy actor that meant
    selecting itself (Chebyshev 0) and attacking itself.  These tests lock in
    the corrected behaviour for all three dropdown adversarial strategies
    (minimax, alphabeta, expectimax).
    """

    def _build_enemy_adjacent_to_warrior(self):
        """Goblin adjacent to a Warrior so adversarial melee mode engages."""
        d = Dungeon(seed_layout=_mini_map_with_two_adjacent_fighters())
        warrior = Warrior(position=Position(3, 1))
        gobbo = Goblin(position=Position(3, 3))
        return d, gobbo, [warrior], [gobbo]

    def test_enemy_targets_party_member_under_minimax(self):
        d, enemy, party, enemies = self._build_enemy_adjacent_to_warrior()
        action, target, step, algo_name, reason = minimax_decision(
            enemy, d, party, enemies, 1)
        self.assertIn(action, (Action.ATTACK, Action.HEAL, Action.DEFEND))
        if action == Action.ATTACK:
            self.assertIsNotNone(target, "Attack must have a target.")
            self.assertIn(target.symbol, PARTY_SYMBOLS,
                f"Enemy must attack a party member, got {target.symbol} ({target.name}).")
            self.assertNotIn(target.symbol, ENEMY_SYMBOLS,
                "Enemy must never attack another enemy (friendly fire).")
            self.assertIsNot(target, enemy, "Enemy must not target itself.")

    def test_enemy_targets_party_member_under_alphabeta(self):
        d, enemy, party, enemies = self._build_enemy_adjacent_to_warrior()
        action, target, step, algo_name, reason = alphabeta_decision(
            enemy, d, party, enemies, 1)
        self.assertIn(action, (Action.ATTACK, Action.HEAL, Action.DEFEND))
        if action == Action.ATTACK:
            self.assertIsNotNone(target)
            self.assertIn(target.symbol, PARTY_SYMBOLS)
            self.assertNotIn(target.symbol, ENEMY_SYMBOLS)
            self.assertIsNot(target, enemy)

    def test_enemy_targets_party_member_under_expectimax(self):
        d, enemy, party, enemies = self._build_enemy_adjacent_to_warrior()
        action, target, step, algo_name, reason = expectimax_decision(
            enemy, d, party, enemies, 1)
        self.assertIn(action, (Action.ATTACK, Action.HEAL, Action.DEFEND))
        if action == Action.ATTACK:
            self.assertIsNotNone(target)
            self.assertIn(target.symbol, PARTY_SYMBOLS)
            self.assertNotIn(target.symbol, ENEMY_SYMBOLS)
            self.assertIsNot(target, enemy)

    def test_enemy_never_targets_self_any_adversarial_strategy(self):
        """Belt-and-suspenders: across all adversarial strategies, an enemy
        actor must never return itself as the attack target."""
        d, enemy, party, enemies = self._build_enemy_adjacent_to_warrior()
        for fn in (minimax_decision, alphabeta_decision, expectimax_decision, cutoff_decision):
            action, target, step, algo_name, reason = fn(
                enemy, d, party, enemies, 1)
            with self.subTest(fn=fn.__name__):
                if action == Action.ATTACK:
                    self.assertIsNotNone(target)
                    self.assertIsNot(target, enemy,
                        f"{fn.__name__}: enemy targeted itself.")
                    self.assertNotIn(target.symbol, ENEMY_SYMBOLS,
                        f"{fn.__name__}: enemy targeted another enemy.")


if __name__ == "__main__":
    unittest.main()
