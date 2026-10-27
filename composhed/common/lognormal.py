"""Log-normal duration regression (CompoShed Steps 2/5 form) with a seeded Generator.

OLS on log duration with the training residual standard deviation retained;
sampling adds N(0, residual_std) in log space, optionally truncated to [lo, hi].
"""

import numpy as np
from scipy.special import ndtr, ndtri


class LogNormalDuration:
    def __init__(self) -> None:
        self.loc_offset = 0.0
        self.scale_mult = 1.0

    def fit(self, X: np.ndarray, t: np.ndarray, feature_names: list[str] | None = None) -> "LogNormalDuration":
        X1 = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        y = np.log(np.maximum(np.asarray(t, dtype=float), 1.0))
        # Small ridge keeps rank-deficient one-hot designs (tiny training sets) stable.
        A = X1.T @ X1 + 1e-6 * np.eye(X1.shape[1])
        self.beta_ = np.linalg.solve(A, X1.T @ y)
        self.residual_std_ = float(np.std(y - X1 @ self.beta_)) if len(y) > 1 else 0.5
        self.residual_std_ = max(self.residual_std_, 1e-3)
        self.feature_names_ = ["const"] + list(feature_names or [f"x{i}" for i in range(X1.shape[1] - 1)])
        return self

    def _mu(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.beta_[0] + X @ self.beta_[1:] + self.loc_offset

    @property
    def sigma(self) -> float:
        return self.residual_std_ * self.scale_mult

    def median(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._mu(X))

    def sample(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
        lo: np.ndarray | float = 1.0,
        hi: np.ndarray | float = np.inf,
    ) -> np.ndarray:
        mu = self._mu(X)
        lo = np.broadcast_to(np.maximum(np.asarray(lo, dtype=float), 1e-6), mu.shape)
        hi = np.broadcast_to(np.asarray(hi, dtype=float), mu.shape)
        a = ndtr((np.log(lo) - mu) / self.sigma)
        b = np.where(np.isfinite(hi), ndtr((np.log(np.maximum(hi, 1e-6)) - mu) / self.sigma), 1.0)
        u = np.clip(a + rng.random(mu.shape) * (b - a), 1e-12, 1 - 1e-12)
        t = np.exp(mu + self.sigma * ndtri(u))
        flat = (b - a) < 1e-12
        if flat.any():
            t = np.where(flat, lo + rng.random(mu.shape) * (np.minimum(hi, lo * 10) - lo), t)
        return np.clip(t, lo, hi)

    def validate(self, X: np.ndarray, t: np.ndarray) -> dict:
        t = np.maximum(np.asarray(t, dtype=float), 1.0)
        y = np.log(t)
        pred = self._mu(X) - self.loc_offset
        ss_tot = np.sum((y - y.mean()) ** 2)
        return {
            "n": int(len(t)),
            "log_r2": float(1 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
            "mae_minutes": float(np.mean(np.abs(np.exp(pred) - t))),
            "mean_actual": float(t.mean()),
        }

    def coef_table(self) -> dict:
        return {"residual_std": self.residual_std_, "beta": dict(zip(self.feature_names_, self.beta_.tolist()))}
