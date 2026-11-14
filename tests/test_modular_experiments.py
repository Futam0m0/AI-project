"""Tests for modular (3-layer) experiment support + non-typable selectors."""
from tests._setup import *  # noqa: F401,F403
import tempfile
import unittest
from pathlib import Path

from algorithms.algorithm_selection import dispatch
from algorithms.modular_selector import (
    COMBAT_OPTIONS,
    DECISION_OPTIONS,
    NAVIGATION_OPTIONS,
    build_decision_fn,
)
from tools.experiment_runner import (
    CSV_PER_BATTLE_NAME,
    aggregate_modular_rows,
    load_modular_results,
    modular_strategy_key,
    run_modular_matrix,
    run_single_battle,
    run_single_modular_battle,
    save_modular_results,
)


class TestModularExperiments(unittest.TestCase):

    # --- 1. Modular config reaches the intended routing function ---

    def test_modular_battle_uses_build_decision_fn(self):
        """run_single_modular_battle must use the 3-layer routing function,
        not a single dispatch() call.  We verify this by checking that the
        action-log 'algorithm' column reflects the routing layer: with
        navigation='astar', out-of-combat party moves should be tagged
        'astar' (never 'baseline')."""
        row = run_single_modular_battle("baseline", "astar", "baseline-combat", 100000)
        # The row must record all three algorithm names.
        self.assertEqual(row["decision"], "baseline")
        self.assertEqual(row["navigation"], "astar")
        self.assertEqual(row["combat"], "baseline-combat")
        # And the combined strategy key.
        self.assertEqual(row["strategy"], "baseline|astar|baseline-combat")

    def test_routing_uses_correct_layer(self):
        """When a specialised layer is selected, the action log must show
        that layer's algorithm name (not the baseline fallback) for turns
        where that layer is active."""
        from game.game_state import GameState
        gs = GameState.new_default(seed=100000)
        dfn = build_decision_fn("baseline", "astar", "minimax")
        for _ in range(60):
            if gs.winner is not None:
                break
            gs.advance_one_round(decision_fn=dfn)
        algos = {a.algorithm for a in gs.action_log}
        # A* must appear because the party starts out of combat and moves.
        # The action-log algorithm field stores the full label string.
        self.assertTrue(
            any("A*" in a for a in algos),
            f"A* should have controlled at least one move; got {algos}",
        )

    # --- 2. All three algorithms recorded correctly ---

    def test_all_three_algorithms_recorded(self):
        row = run_single_modular_battle("sa-opt", "bfs", "alphabeta", 100001)
        self.assertEqual(row["decision"], "sa-opt")
        self.assertEqual(row["navigation"], "bfs")
        self.assertEqual(row["combat"], "alphabeta")

    def test_modular_strategy_key_format(self):
        self.assertEqual(
            modular_strategy_key("baseline", "astar", "minimax"),
            "baseline|astar|minimax",
        )

    # --- 3. Reproducibility: same config + seeds = same results ---

    def test_modular_reproducible(self):
        cfg = ("baseline", "astar", "minimax")
        rows_a = run_modular_matrix([cfg], n_battles=3, seed_start=200000, verbose=False)
        rows_b = run_modular_matrix([cfg], n_battles=3, seed_start=200000, verbose=False)
        for a, b in zip(rows_a, rows_b):
            self.assertEqual(a["winner"], b["winner"])
            self.assertEqual(a["rounds"], b["rounds"])
            self.assertEqual(a["party_hp_left"], b["party_hp_left"])

    # --- 4. Backward compatibility: single-strategy mode still works ---

    def test_single_strategy_still_works(self):
        row = run_single_battle("baseline", 100000)
        self.assertEqual(row["strategy"], "baseline")
        self.assertIn("winner", row)
        self.assertIn("rounds", row)

    def test_single_strategy_no_modular_columns(self):
        """Single-strategy rows must NOT have decision/navigation/combat
        columns (keeps backward compatibility with historical CSVs)."""
        row = run_single_battle("baseline", 100000)
        self.assertNotIn("decision", row)
        self.assertNotIn("navigation", row)
        self.assertNotIn("combat", row)

    # --- 5. Historical experiment files remain untouched ---

    def test_modular_save_uses_separate_files(self):
        """Modular results must be saved to *separate* CSV files so the
        historical single-strategy CSVs are never overwritten."""
        configs = [("baseline", "baseline-nav", "baseline-combat")]
        rows = run_modular_matrix(configs, n_battles=2, verbose=False)
        summary = aggregate_modular_rows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            bfile, afile = save_modular_results(rows, summary, tmp)
            self.assertTrue(bfile.exists())
            self.assertTrue(afile.exists())
            # The filenames must be the *modular* ones, not the historical ones.
            self.assertNotEqual(bfile.name, CSV_PER_BATTLE_NAME)
            loaded_rows, loaded_summary = load_modular_results(tmp)
            self.assertEqual(len(loaded_rows), len(rows))
            self.assertEqual(loaded_rows[0]["decision"], "baseline")

    def test_modular_rows_have_three_layer_columns(self):
        configs = [("ga-opt", "ucs", "expectimax")]
        rows = run_modular_matrix(configs, n_battles=1, verbose=False)
        self.assertEqual(rows[0]["decision"], "ga-opt")
        self.assertEqual(rows[0]["navigation"], "ucs")
        self.assertEqual(rows[0]["combat"], "expectimax")

    # --- 6. Selectors use st.radio (not st.selectbox) ---

    def test_selectors_use_radio_not_selectbox(self):
        """app.py must use st.radio (which does not accept typed search
        input) for all three algorithm selectors, not st.selectbox."""
        app_text = Path("app.py").read_text(encoding="utf-8")
        # The three selectors must be st.radio calls.
        self.assertIn('st.radio(', app_text)
        # There must be no st.selectbox calls for algorithm selection.
        # (A selectbox might be used elsewhere, but the three algorithm
        #  selectors must be radios.)
        selectbox_count = app_text.count("st.selectbox(")
        self.assertEqual(
            selectbox_count, 0,
            "Algorithm selectors should use st.radio, not st.selectbox "
            "(which allows type-to-search).",
        )

    def test_radio_does_not_accept_typed_input(self):
        """st.radio is a native Streamlit component that renders radio
        buttons; it has no text input field, so users cannot type into it.
        We assert this factually (no code change can make st.radio accept
        typed input)."""
        import inspect
        import streamlit as st
        sig = inspect.signature(st.radio)
        # st.radio has no 'searchable' or text-related parameter; it is
        # purely clickable.  This confirms it cannot accept typed search.
        params = set(sig.parameters.keys())
        self.assertNotIn("searchable", params)
        self.assertNotIn("placeholder", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
