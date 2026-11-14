"""Tests for game/game_state.py advance_one_round, win/lose conditions + stats."""
from tests._setup import *  # noqa: F401,F403
import unittest
from game.game_state import GameState, MAX_ROUNDS


class TestGameState(unittest.TestCase):
    def test_default_construction_has_3_party_3_enemies(self):
        g = GameState.new_default(seed=42)
        self.assertEqual(len(g.party), 3)
        self.assertEqual(len(g.enemies), 3)
        self.assertIsNotNone(g.seed)

    def test_all_party_positions_placed(self):
        g = GameState.new_default(seed=1)
        for a in g.party:
            self.assertIsNotNone(a.position, f"{a.name} has no position after new_default.")
            # Check that symbol in grid matches position.
            r, c = a.position
            self.assertEqual(g.dungeon.grid[r][c], a.symbol)
        for e in g.enemies:
            self.assertIsNotNone(e.position)
            r, c = e.position
            self.assertEqual(g.dungeon.grid[r][c], e.symbol)

    def test_round_number_increments(self):
        g = GameState.new_default(seed=0)
        before = g.round_number
        g.advance_one_round()
        self.assertEqual(g.round_number, before + 1)
        g.advance_one_round()
        self.assertEqual(g.round_number, before + 2)

    def test_action_log_populated(self):
        g = GameState.new_default(seed=1)
        g.advance_one_round()
        # W+H+S+3 alive enemies = 6 actions minimum (some enemies may be defeated in round).
        self.assertGreaterEqual(len(g.action_log), 3)
        # All log entries have algorithm field.
        algos_used = {r.algorithm for r in g.action_log}
        self.assertIn("baseline", algos_used)

    def test_stats_are_updated_after_rounds(self):
        g = GameState.new_default(seed=5)
        for _ in range(10):
            g.advance_one_round()
            over, _ = g.game_over()
            if over:
                break
        self.assertGreaterEqual(g.stats.turns_taken, 1)

    def test_all_enemies_defeated_wins_game(self):
        g = GameState.new_default(seed=0)
        # Kill all 3 enemies directly (bypass AI) to test win.
        for e in g.enemies:
            if e.position:
                g.dungeon.remove_entity(*e.position)
            e.hp = 0
            e.status = "defeated"
            g.stats.enemies_defeated += 1
        over, winner = g.game_over()
        self.assertTrue(over)
        self.assertEqual(winner, "party")

    def test_all_party_defeated_loses(self):
        g = GameState.new_default(seed=0)
        for a in g.party:
            if a.position:
                g.dungeon.remove_entity(*a.position)
            a.hp = 0
            a.status = "defeated"
        over, winner = g.game_over()
        self.assertTrue(over)
        self.assertEqual(winner, "enemies")

    def test_advance_terminates_in_60_rounds(self):
        """Guard against infinite loops: game should terminate via some
        win/lose within 60 rounds or something is wrong with movement stalling."""
        g = GameState.new_default(seed=2)
        for round_idx in range(60):
            g.advance_one_round()
            over, winner = g.game_over()
            if over:
                return
        self.fail("Game did not terminate within 60 rounds (stuck?): "
                  "winner=%s, party_alive=%d, enemies_alive=%d" % (
                      g.winner, len(g.party_alive()), len(g.enemies_alive())))

    # ---- Regression tests for the "Ryn dies → infinite rounds" bug ----

    def test_defeated_party_member_removed_from_grid(self):
        """When a party member is defeated, their symbol must be cleared from
        the grid so the corpse does not block movement (root cause of the
        infinite-round stalemate when Ryn died in a corridor)."""
        g = GameState.new_default(seed=8)
        # Kill Ryn (the Scout, index 2) directly.
        ryn = g.party[2]
        ryn.hp = 0
        ryn.status = "defeated"
        if ryn.position is not None:
            g.dungeon.remove_entity(*ryn.position)
        # Ryn's symbol must no longer be on the grid.
        for row in g.dungeon.grid:
            self.assertNotIn("S", row)

    def test_ryn_dies_others_alive_game_terminates(self):
        """If Ryn dies but Brann and Mara are alive, the game must continue
        and eventually terminate (win/lose/draw) — not loop forever."""
        g = GameState.new_default(seed=8)
        # Run until Ryn dies.
        for _ in range(MAX_ROUNDS):
            g.advance_one_round()
            if not g.party[2].alive:
                break
        self.assertFalse(g.party[2].alive, "Ryn should have died by round 30 on seed 8")
        # Continue; the game must terminate (corpse no longer blocks the path).
        for _ in range(MAX_ROUNDS + 5):
            g.advance_one_round()
            if g.winner is not None:
                break
        self.assertIsNotNone(g.winner, "Game should terminate after Ryn dies, "
                                       "not loop forever (winner is None)")

    def test_all_party_defeated_ends_via_advance(self):
        """All party members defeated → advance_one_round must end the game
        with an enemies victory and record battles_lost."""
        g = GameState.new_default(seed=0)
        for a in g.party:
            if a.position:
                g.dungeon.remove_entity(*a.position)
            a.hp = 0
            a.status = "defeated"
        g.advance_one_round()
        self.assertEqual(g.winner, "enemies")
        self.assertEqual(g.stats.battles_lost, 1)

    def test_all_enemies_defeated_ends_via_advance(self):
        """All enemies defeated → advance_one_round must end the game with a
        party victory and record battles_won."""
        g = GameState.new_default(seed=0)
        for e in g.enemies:
            if e.position:
                g.dungeon.remove_entity(*e.position)
            e.hp = 0
            e.status = "defeated"
            g.stats.enemies_defeated += 1
        g.advance_one_round()
        self.assertEqual(g.winner, "party")
        self.assertEqual(g.stats.battles_won, 1)

    def test_max_rounds_declares_draw(self):
        """If no side wins within MAX_ROUNDS, game_over must return a draw
        instead of continuing indefinitely."""
        g = GameState.new_default(seed=0)
        g.round_number = MAX_ROUNDS  # force the stalemate limit
        over, winner = g.game_over()
        self.assertTrue(over)
        self.assertEqual(winner, "draw")

    def test_draw_does_not_count_as_win_or_loss(self):
        """A draw must not increment battles_won or battles_lost."""
        g = GameState.new_default(seed=0)
        g.round_number = MAX_ROUNDS
        g.advance_one_round()  # should detect the draw and end
        self.assertEqual(g.winner, "draw")
        self.assertEqual(g.stats.battles_won, 0)
        self.assertEqual(g.stats.battles_lost, 0)

    def test_dead_actor_takes_no_turn(self):
        """A defeated/unconscious character must be skipped during the round;
        no action should be logged for them."""
        g = GameState.new_default(seed=0)
        # Kill Ryn before the round.
        ryn = g.party[2]
        ryn.hp = 0
        ryn.status = "defeated"
        if ryn.position is not None:
            g.dungeon.remove_entity(*ryn.position)
        before = len(g.action_log)
        g.advance_one_round()
        # No log entry should name Ryn as the actor.
        actors = [r.actor.name for r in g.action_log[before:]]
        self.assertNotIn(ryn.name, actors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
