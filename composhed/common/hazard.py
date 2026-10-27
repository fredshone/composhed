"""Accelerated-failure-time duration models (CEMDAP-style hazard durations).

``log T = x·β + σ·ε`` with ε standard logistic (log-logistic hazard) or standard
minimum-extreme-value (Weibull hazard). Fitted by maximum likelihood (scipy
L-BFGS, analytic gradient); ``fit_best`` picks the distribution by AIC. Sampling
is by inverse CDF on the truncated range, so draws always respect the residual
window without rejection (CEM-5).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log1p

DISTRIBUTIONS = ("loglogistic", "weibull")


def _logpdf_and_score(eps: np.ndarray, distr: str) -> tuple[np.ndarray, np.ndarray]:
    if distr == "loglogistic":
        return -eps - 2.0 * np.logaddexp(0.0, -eps), -np.tanh(eps / 2.0)
    e = np.exp(np.minimum(eps, 50.0))
    return eps - e, 1.0 - e


def _cdf(eps: np.ndarray, distr: str) -> np.ndarray:
    if distr == "loglogistic":
        return expit(eps)
    return -np.expm1(-np.exp(np.minimum(eps, 50.0)))


def _ppf(u: np.ndarray, distr: str) -> np.ndarray:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    if distr == "loglogistic":
        return np.log(u) - log1p(-u)
    return np.log(-log1p(-u))


class AFTModel:
    def __init__(self, distr: str = "loglogistic", maxiter: int = 500) -> None:
        if distr not in DISTRIBUTIONS:
            raise ValueError(distr)
        self.distr = distr
        self.maxiter = maxiter
        self.loc_offset = 0.0
        self.scale_mult = 1.0

    @classmethod
    def fit_best(cls, X: np.ndarray, t: np.ndarray, feature_names: list[str] | None = None) -> "AFTModel":
        fits = [cls(d).fit(X, t, feature_names) for d in DISTRIBUTIONS]
        best = min(fits, key=lambda m: m.aic_)
        best.aic_all_ = {m.distr: m.aic_ for m in fits}
        return best

    def fit(self, X: np.ndarray, t: np.ndarray, feature_names: list[str] | None = None) -> "AFTModel":
        X1 = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        t = np.maximum(np.asarray(t, dtype=float), 1.0)
        y = np.log(t)
        N, K = X1.shape
        self.feature_names_ = ["const"] + list(feature_names or [f"x{i}" for i in range(K - 1)])

        def obj(theta):
            beta, log_s = theta[:K], float(np.clip(theta[K], -10.0, 10.0))
            s = np.exp(log_s)
            eps = (y - X1 @ beta) / s
            lp, score = _logpdf_and_score(eps, self.distr)
            ll = np.sum(lp) - N * log_s - np.sum(y)
            g_beta = X1.T @ (-score / s)
            g_ls = np.sum(-score * eps) - N
            return -ll / N, -np.concatenate([g_beta, [g_ls]]) / N

        beta0 = np.linalg.lstsq(X1, y, rcond=None)[0]
        s0 = max(np.std(y - X1 @ beta0), 1e-2)
        res = minimize(obj, np.concatenate([beta0, [np.log(s0)]]), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter})
        self.beta_ = res.x[:K]
        self.sigma_ = float(np.exp(np.clip(res.x[K], -10.0, 10.0)))
        self.llf_ = -float(res.fun) * N
        self.aic_ = 2 * (K + 1) - 2 * self.llf_
        self.converged_ = bool(res.success)
        return self

    def _mu(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.beta_[0] + X @ self.beta_[1:] + self.loc_offset

    @property
    def sigma(self) -> float:
        return self.sigma_ * self.scale_mult

    def median(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._mu(X) + self.sigma * _ppf(np.array(0.5), self.distr))

    def sample(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
        lo: np.ndarray | float = 1.0,
        hi: np.ndarray | float = np.inf,
    ) -> np.ndarray:
        """Draw durations truncated to [lo, hi] by inverse CDF."""
        mu = self._mu(X)
        lo = np.broadcast_to(np.maximum(np.asarray(lo, dtype=float), 1e-6), mu.shape)
        hi = np.broadcast_to(np.asarray(hi, dtype=float), mu.shape)
        a = _cdf((np.log(lo) - mu) / self.sigma, self.distr)
        b = np.where(np.isfinite(hi), _cdf((np.log(np.maximum(hi, 1e-6)) - mu) / self.sigma, self.distr), 1.0)
        u = a + rng.random(mu.shape) * (b - a)
        t = np.exp(mu + self.sigma * _ppf(u, self.distr))
        # Truncation window entirely in a numerically empty tail: uniform inside it.
        flat = (b - a) < 1e-12
        if flat.any():
            t = np.where(flat, lo + rng.random(mu.shape) * (np.minimum(hi, lo * 10) - lo), t)
        return np.clip(t, lo, hi)

    def validate(self, X: np.ndarray, t: np.ndarray) -> dict:
        t = np.maximum(np.asarray(t, dtype=float), 1.0)
        y = np.log(t)
        pred = np.log(self.median(X))
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return {
            "n": int(len(t)),
            "distr": self.distr,
            "log_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "mae_minutes": float(np.mean(np.abs(np.exp(pred) - t))),
            "mean_actual": float(t.mean()),
        }

    def coef_table(self) -> dict:
        return {"distr": self.distr, "sigma": self.sigma_, "beta": dict(zip(self.feature_names_, self.beta_.tolist()))}
