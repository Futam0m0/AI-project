"""Phase 5.2 — Gradient Descent + Linear Regression, from scratch.

Pure-Python implementation (no NumPy or C bindings) so the gradient update
can be stepped through line-by-line against the course notebook pseudocode.

Model form
----------

Given a design matrix ``X`` (n rows × 10 columns of behaviour weights) and a
response ``y`` (damage per round), the prediction is::

    y_hat = X · θ + b

where ``θ`` is a length-10 coefficient vector and ``b`` is the bias (intercept).
Loss is mean-squared error::

    L(θ, b) = (1 / 2n) · Σ (y_hat_i - y_i)²

The factor of ``1/2`` cancels the 2 that comes out when differentiating and is
standard in textbook presentations.  Gradients::

    dL/dθ_j = (1/n) · Σ (y_hat_i - y_i) · X[i][j]
    dL/db   = (1/n) · Σ (y_hat_i - y_i)

Update rule (batch gradient descent)::

    θ ← θ - α · dL/dθ
    b ← b - α · dL/db

``X`` rows should be standardised (Z-score) before fitting so the learning
rate applies uniformly.  Standardisation is handled transparently by
:class:`LinearRegressionGD` with a ``fit`` method that memorises the
per-column mean/std for ``predict``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: Sequence[float], mean: float) -> float:
    if len(xs) <= 1:
        return 1.0
    var = sum((x - mean) ** 2 for x in xs) / float(len(xs))
    s = math.sqrt(var)
    return s if s > 1e-12 else 1.0


def z_score_normalise(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Z-score columns of ``X`` in place (mutates a copy) and return std-params.

    Returns (X_stdised, col_means, col_stds).
    """
    if not X or not X[0]:
        return ([], [], [])
    n_rows = len(X)
    n_cols = len(X[0])
    col_means = [_mean([X[i][c] for i in range(n_rows)]) for c in range(n_cols)]
    col_stds = [_std([X[i][c] for i in range(n_rows)], col_means[c]) for c in range(n_cols)]
    Xs = [[(X[i][c] - col_means[c]) / col_stds[c] for c in range(n_cols)] for i in range(n_rows)]
    return Xs, col_means, col_stds


@dataclass
class LinearRegressionGD:
    """Batch gradient-descent linear regressor.

    Parameters
    ----------
    learning_rate :
        Step size α in the GD update rule.  Default chosen so 1000-epoch
        convergence happens on small regression problems without exploding.
    n_epochs :
        Number of full passes over the dataset.
    l2_reg :
        Ridge-regression penalty coefficient on θ (stops extreme weights on
        small datasets where some columns are nearly collinear).
    """

    learning_rate: float = 0.05
    n_epochs: int = 1000
    l2_reg: float = 1e-4

    coef_: list[float] = field(default_factory=list)    # θ
    intercept_: float = 0.0                              # b
    col_means_: list[float] = field(default_factory=list)
    col_stds_: list[float] = field(default_factory=list)
    loss_history_: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Fitting API
    # ------------------------------------------------------------------
    def fit(self, X: list[list[float]], y: list[float]) -> "LinearRegressionGD":
        if len(X) != len(y):
            raise ValueError(f"X rows ({len(X)}) must equal y length ({len(y)}).")
        n = len(X)
        if n == 0:
            self.coef_ = []
            self.intercept_ = 0.0
            self.loss_history_ = []
            return self
        n_cols = len(X[0])
        if any(len(row) != n_cols for row in X):
            raise ValueError("All rows of X must have the same length.")

        Xs, mus, sigmas = z_score_normalise(X)
        self.col_means_ = mus
        self.col_stds_ = sigmas

        # Initialise parameters to zero (convention taught in class)
        self.coef_ = [0.0] * n_cols
        self.intercept_ = 0.0
        self.loss_history_ = []

        # --- gradient descent loop ---------------------------------------
        alpha = self.learning_rate
        l2 = self.l2_reg
        for epoch in range(self.n_epochs):
            # 1) Predict every row.
            preds = [self._predict_on_standardised(Xs[i]) for i in range(n)]
            # 2) Residual vector
            residuals = [preds[i] - y[i] for i in range(n)]
            # 3) Loss = 1/(2n) Σ res² + (l2/2) Σ θ²
            loss = sum(r * r for r in residuals) / (2.0 * n)
            loss += 0.5 * l2 * sum(th * th for th in self.coef_)
            self.loss_history_.append(loss)
            # 4) Gradient w.r.t each θ_j + bias
            inv_n = 1.0 / float(n)
            grad_theta = [0.0] * n_cols
            for j in range(n_cols):
                dot = 0.0
                for i in range(n):
                    dot += residuals[i] * Xs[i][j]
                grad_theta[j] = inv_n * dot + l2 * self.coef_[j]
            grad_bias = inv_n * sum(residuals)
            # 5) Update
            for j in range(n_cols):
                self.coef_[j] -= alpha * grad_theta[j]
            self.intercept_ -= alpha * grad_bias

        return self

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------
    def _predict_on_standardised(self, x_std_row: list[float]) -> float:
        return self.intercept_ + sum(t * v for t, v in zip(self.coef_, x_std_row))

    def predict_one(self, x_row: list[float]) -> float:
        """Predict damage-per-round for a single raw (unstandardised) row."""
        if len(x_row) != len(self.col_means_):
            raise ValueError(
                f"Row has {len(x_row)} cols; model expects {len(self.col_means_)}."
            )
        std = [(x_row[c] - self.col_means_[c]) / max(1e-12, self.col_stds_[c])
               for c in range(len(x_row))]
        return self._predict_on_standardised(std)

    def predict(self, X: list[list[float]]) -> list[float]:
        return [self.predict_one(row) for row in X]

    # ------------------------------------------------------------------
    # Goodness-of-fit helpers
    # ------------------------------------------------------------------
    def mean_squared_error(self, X: list[list[float]], y: list[float]) -> float:
        if len(X) != len(y) or not y:
            return 0.0
        preds = self.predict(X)
        return sum((preds[i] - y[i]) ** 2 for i in range(len(y))) / float(len(y))

    def r_squared(self, X: list[list[float]], y: list[float]) -> float:
        if len(X) != len(y) or not y:
            return 0.0
        preds = self.predict(X)
        mean_y = _mean(y)
        ss_res = sum((preds[i] - y[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot
