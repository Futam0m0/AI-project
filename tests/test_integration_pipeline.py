"""Integration pipeline tests (2 tests): headless game run + baseline log labels.

The adventure TXT / parser prototype was removed during cleanup; the game uses
its own self-contained dungeon and character data in ``game/``.  The previous
``test_adventure_txt_unchanged_while_game_runs`` integrity guard was a legacy
artifact that only verified the game did not modify a file it never reads, so
it was removed.
"""
from tests._setup import *  # noqa: F401,F403
import unittest
from game.game_state import GameState


class TestIntegrationPipeline(unittest.TestCase):
    def test_game_runs_headless(self):
        """A 10-round game should run without raising."""
        g = GameState.new_default(seed=7)
        for _ in range(10):
            g.advance_one_round()
            if g.game_over()[0]:
                break
        # Game state should have > 1 turn minimum.
        self.assertGreaterEqual(g.stats.turns_taken, 1)

    def test_log_entries_have_baseline_algorithm_attribute(self):
        g = GameState.new_default(seed=3)
        g.advance_one_round()
        algos = {r.algorithm for r in g.action_log}
        self.assertIn("baseline", algos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
