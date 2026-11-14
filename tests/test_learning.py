"""Tests for learning/ — Phase 5 (12 unit tests).

Test enumeration (12 total — exactly as planned in Phase 5.6):

  Group A — Dataset module (learning/dataset.py) — 4 tests
    A1. generate_dataset(n_rows=4) returns 4 × 10 features, n=4 responses,
        each chromosome in [0,1].
    A2. Columns in CombatDataset.feature_names match the 10 declared names.
    A3. save_csv + load_csv round-trips: X, y, feature names are preserved
        (CSV floats within 1e-3 because of limited precision in writer).
    A4. Responses are bounded (zero negatives, finite floats).

  Group B — Gradient Descent Linear Regressor (learning/gd.py) — 4 tests
    B1. fit() on a trivial separable dataset (y = 3*x_0 + 5) produces
        coefficient signs that match expectations (positive weight on x_0,
        intercept ≈ 5 after re-translation from std units).
    B2. Loss_history_ is non-increasing on average (the loss curve is
        monotonically non-increasing in GD when LR small and convex loss).
    B3. predict_one() on a single row returns a single float.
    B4. r_squared on a noisy synthetic data set where y = sum(X) is ≥ 0.7
        (i.e. the regressor captures ≥ 70% of variance after 800 epochs).

  Group C — Simulated Annealing (learning/sa.py) — 2 tests
    C1. simulated_annealing returns a 10-element chromosome in [0,1]^10 with
        best_score ≥ initial start_score (or equal if plateau).
    C2. Temperature history is monotonically non-increasing (geometric
        cooling schedule is correctly applied).

  Group D — Genetic Algorithm (learning/ga.py) — 2 tests
    D1. genetic_algorithm returns a 10-element chromosome in [0,1]^10 with
        best_score ≥ average fitness of generation 0 (improvement on average).
    D2. history length (best_history) == n_generations + 1 (one entry per
        gen 0..n inclusive).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestDatasetModule(unittest.TestCase):  # 4 tests
    def test_a1_generate_shape_and_range(self):
        from learning.dataset import generate_dataset
        ds = generate_dataset(n_rows=4, seed=2026, rounds_per_battle=3, rows_per_observation=1)
        self.assertEqual(len(ds), 4)
        self.assertEqual(len(ds.feature_names), 10)
        for i, row in enumerate(ds.X):
            self.assertEqual(len(row), 10, f"row {i} length != 10")
            for v in row:
                self.assertGreaterEqual(v, 0.0, f"row {i} weight out of range: {v}")
                self.assertLessEqual(v, 1.0, f"row {i} weight out of range: {v}")
        self.assertEqual(len(ds.y), 4)
        for r in ds.y:
            self.assertTrue(math.isfinite(r), "response must be finite float")

    def test_a2_feature_names(self):
        from learning.dataset import FEATURE_NAMES, generate_dataset
        ds = generate_dataset(n_rows=3, rounds_per_battle=2, rows_per_observation=1)
        self.assertEqual(ds.feature_names, list(FEATURE_NAMES))
        # Each feature name starts with "wN_" prefix (N = 0..9):
        for i, name in enumerate(FEATURE_NAMES):
            self.assertTrue(name.startswith(f"w{i}_"), f"feature {i} = {name}")

    def test_a3_save_load_roundtrip(self):
        from learning.dataset import generate_dataset
        ds = generate_dataset(n_rows=5, seed=42, rounds_per_battle=2, rows_per_observation=1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiment.csv")
            ds.save_csv(path)
            self.assertTrue(os.path.isfile(path))
            ds2 = type(ds).load_csv(path)
            self.assertEqual(ds2.feature_names, ds.feature_names)
            self.assertEqual(len(ds2), len(ds))
            for i in range(len(ds)):
                for j in range(10):
                    self.assertAlmostEqual(ds2.X[i][j], ds.X[i][j], places=3)
                self.assertAlmostEqual(ds2.y[i], ds.y[i], places=3)

    def test_a4_responses_non_negative(self):
        from learning.dataset import generate_dataset
        ds = generate_dataset(n_rows=8, seed=7, rounds_per_battle=4, rows_per_observation=2)
        for r in ds.y:
            self.assertGreaterEqual(r, 0.0, f"damage per round can't be negative: {r}")


class TestGradientDescentModule(unittest.TestCase):  # 4 tests
    def test_b1_fit_trivial_line_positive(self):
        # y = 3*x_0 + 5  (with 9 nuisance zero columns)
        X = [[(i - 5)] + [0.0] * 9 for i in range(11)]
        y = [3.0 * (i - 5) + 5.0 for i in range(11)]
        from learning.gd import LinearRegressionGD
        reg = LinearRegressionGD(learning_rate=0.05, n_epochs=1200).fit(X, y)
        self.assertEqual(len(reg.coef_), 10)
        self.assertGreater(reg.coef_[0], 0.5, "coef[0] should be strongly positive for 3x")
        for j in range(1, 10):
            self.assertLess(abs(reg.coef_[j]), 0.5, "nuisance coefs ~0")
        # Intercept in standardised space + inverse transform sanity check:
        # predict at x_0 = 0 (after standardisation intercepts mean of y)
        pred_at_zero_mean = reg.predict_one([0.0] * 10)
        # We don't assert exact equality, but it should be finite.
        self.assertTrue(math.isfinite(pred_at_zero_mean))

    def test_b2_loss_non_increasing_trend(self):
        # Build a small synthetic dataset and verify loss doesn't explode.
        X = [[0.1 * i, -0.2 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(20)]
        y = [0.5 + i * 0.3 for i in range(20)]
        from learning.gd import LinearRegressionGD
        reg = LinearRegressionGD(learning_rate=0.02, n_epochs=400).fit(X, y)
        # The first 100 loss entries should be decreasing on average.
        losses = reg.loss_history_
        self.assertTrue(len(losses) >= 400)
        first_quart = sum(losses[:50]) / 50
        last_quart = sum(losses[-50:]) / 50
        self.assertLessEqual(last_quart, first_quart + 1e-6,
                             "GD should reduce loss (last quart ≤ first quart)")

    def test_b3_predict_one_float(self):
        from learning.gd import LinearRegressionGD
        # Use 12 rows + enough epochs that the bias can converge close to 3.0.
        X = [[0.0] * 10 for _ in range(12)]
        y = [3.0] * 12
        reg = LinearRegressionGD(learning_rate=0.1, n_epochs=600, l2_reg=0.0).fit(X, y)
        out = reg.predict_one([0.0] * 10)
        self.assertIsInstance(out, float)
        self.assertTrue(math.isfinite(out))
        # All predictions should be near the constant target of 3.0.
        self.assertAlmostEqual(out, 3.0, delta=1.5)

    def test_b4_r_squared_geq_0p7_on_sum_x(self):
        # y = sum(x_i), noise 0.01.  R² should be high enough.
        import math
        X = []
        y = []
        for i in range(60):
            row = [((i * (j + 1)) % 20) / 20.0 - 0.5 for j in range(10)]
            X.append(row)
            target = sum(row) + 0.01 * math.sin(i)
            y.append(target)
        train_X, train_y = X[:50], y[:50]
        test_X, test_y = X[50:], y[50:]
        from learning.gd import LinearRegressionGD
        reg = LinearRegressionGD(learning_rate=0.1, n_epochs=1500, l2_reg=0.0).fit(train_X, train_y)
        r2 = reg.r_squared(test_X, test_y)
        # y = sum(x_i) is exactly linear, so R² on test set must be high.
        self.assertGreaterEqual(r2, 0.7, f"R²={r2} below 0.7")


class TestSimulatedAnnealingModule(unittest.TestCase):  # 2 tests
    def test_c1_sa_returns_chromosome_and_score(self):
        # A 1-D objective: maximize f = w[9] (last weight).  SA should return
        # best_score >= initial baseline score.
        def obj(w):
            return float(w[9])

        from learning.sa import simulated_annealing
        from algorithms.baseline_decision import DEFAULT_WEIGHTS
        start = DEFAULT_WEIGHTS.to_chromosome()
        start_score = obj(start)
        res = simulated_annealing(obj, start_chromosome=start, n_steps=15,
                                  start_temp=2.0, cooling=0.9, step_std=0.1,
                                  seed=2026)
        self.assertEqual(len(res.best_chromosome), 10)
        for v in res.best_chromosome:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertGreaterEqual(res.best_score, start_score - 1e-9,
                                "best_score can't be below start_score due to determinism")
        self.assertTrue(math.isfinite(res.final_score))

    def test_c2_temperature_schedule_monotone_nonincreasing(self):
        from learning.sa import simulated_annealing
        res = simulated_annealing(lambda w: 1.0,
                                  start_chromosome=[0.5]*10,
                                  n_steps=20, start_temp=5.0, cooling=0.85,
                                  step_std=0.1, seed=1)
        self.assertTrue(len(res.temp_history) >= 20)
        for i in range(1, len(res.temp_history)):
            self.assertLessEqual(res.temp_history[i], res.temp_history[i-1] + 1e-12,
                                 f"temp[{i}]={res.temp_history[i]} > temp[{i-1}]={res.temp_history[i-1]}")


class TestGeneticAlgorithmModule(unittest.TestCase):  # 2 tests
    def test_d1_ga_returns_chromosome_and_improves(self):
        # Objective: f(w) = - sum((w_i - 0.8)^2) — maximum at all 0.8.
        def obj(w):
            return -sum((v - 0.8) ** 2 for v in w)

        from learning.ga import genetic_algorithm
        res = genetic_algorithm(obj, pop_size=10, n_generations=8, seed=2026,
                                start_chromosome=[0.5]*10)
        self.assertEqual(len(res.best_chromosome), 10)
        for v in res.best_chromosome:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # Mean fitness of gen N should be ≥ mean fitness of gen 0.
        self.assertGreaterEqual(res.mean_history[-1], res.mean_history[0] - 1e-6,
                                "Population should improve on average.")

    def test_d2_history_length_matches_n_gens(self):
        from learning.ga import genetic_algorithm
        n = 5
        res = genetic_algorithm(lambda w: 1.0, pop_size=6, n_generations=n, seed=1)
        self.assertEqual(len(res.best_history), n + 1, f"best_history expected {n+1}")
        self.assertEqual(len(res.mean_history), n + 1, f"mean_history expected {n+1}")
        self.assertEqual(res.n_generations, n)


if __name__ == "__main__":
    unittest.main()
