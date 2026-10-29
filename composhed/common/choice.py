"""Discrete choice models with a common fit / predict_proba / sample(rng) / validate API.

* ``MNL`` — multinomial logit with alternative-specific coefficients, optional
  availability, ridge-stabilised L-BFGS (numpy/scipy). statsmodels ``MNLogit`` is
  not used because it builds the full (K·J)² Hessian, which is infeasible for the
  DaySim day-pattern choice set (J ≈ 100).
* ``OrderedLogit`` — ordered logit (as CompoShed Step 3) with an analytic
  gradient; statsmodels ``OrderedModel`` uses numerical scores and is too slow
  for the many repeated fits here.
* ``ConditionalLogit`` — alternative-attribute logit for joint start/end choice:
  ``V_nj = f_j·γ + z_n·B·g_j + a_nj·δ + offset_j`` with sparse generic alternative
  features ``f_j`` (period dummies, duration bins), dense shape features ``g_j``
  interacted with person features ``z_n``, optional chooser-specific alternative
  attributes ``a_nj`` (e.g. timetable terms; supplied per chunk by a callable so the
  N × J × K array is never held in memory) and availability. ``b_mask`` restricts
  ``B`` to named (person feature, alternative attribute) pairs, as in an
  ActivitySim spec or a DaySim utility function, instead of every interaction.
* ``AltAttrMNL`` — ActivitySim ``interaction_simulate``-style logit over a fixed
  alternative table: alternative attributes (e.g. tour counts per purpose)
  interacted with person features, with the ``MNL`` interface.

Every model carries an ``offset`` vector over alternatives added to utilities
(log-probabilities for the ordered logit); the calibration stage adjusts it.
"""


import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def _log_softmax(V: np.ndarray, avail: np.ndarray | None) -> np.ndarray:
    if avail is not None:
        V = np.where(avail, V, -np.inf)
    lse = logsumexp(V, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):  # rows with nothing available give NaN, mapped to 0 by callers
        return V - lse


def _sample_rows(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draw of one column index per row of a probability matrix."""
    c = np.cumsum(P, axis=1)
    u = rng.random(len(P)) * c[:, -1]
    idx = (c <= u[:, None]).sum(axis=1)
    return np.minimum(idx, P.shape[1] - 1)


def _shares(labels, classes) -> dict:
    labels = list(labels)
    n = max(len(labels), 1)
    return {c: labels.count(c) / n for c in classes}


class MNL:
    """Multinomial logit with alternative-specific coefficients and availability."""

    def __init__(self, l2: float = 1e-4, maxiter: int = 500) -> None:
        self.l2 = l2
        self.maxiter = maxiter

    def fit(
        self,
        X: np.ndarray,
        y: list,
        classes: list | None = None,
        avail: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "MNL":
        y = list(y)
        self.classes_ = list(classes) if classes is not None else sorted(set(y), key=str)
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[v] for v in y], dtype=int)
        X1 = np.hstack([np.ones((len(X), 1)), np.asarray(X, dtype=float)])
        N, K = X1.shape
        J = len(self.classes_)
        self.feature_names_ = ["const"] + list(feature_names or [f"x{i}" for i in range(K - 1)])
        self.offset = np.zeros(J)
        if avail is not None:
            avail = avail.copy()
            avail[np.arange(N), yi] = True
        counts = np.bincount(yi, minlength=J).astype(float)
        if J == 1 or (counts > 0).sum() == 1:
            self.W_ = np.zeros((K, J))
            self.W_[0] = np.where(counts > 0, 0.0, -30.0)
            self.llf_ = 0.0
            self.llnull_ = 0.0
            return self
        Y = np.zeros((N, J))
        Y[np.arange(N), yi] = 1.0

        def obj(w):
            W = np.zeros((K, J))
            W[:, 1:] = w.reshape(K, J - 1)
            logP = _log_softmax(X1 @ W, avail)
            ll = np.sum(logP[np.arange(N), yi])
            P = np.exp(logP)
            G = X1.T @ (Y - P)
            f = -ll / N + 0.5 * self.l2 * np.sum(w * w)
            g = -G[:, 1:].ravel() / N + self.l2 * w
            return f, g

        w0 = np.zeros(K * (J - 1))
        # Start from the observed shares so rare alternatives do not dominate early steps.
        shares = np.maximum(counts / N, 1e-6)
        w0.reshape(K, J - 1)[0] = np.log(shares[1:] / shares[0])
        res = minimize(obj, w0, jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter})
        self.W_ = np.zeros((K, J))
        self.W_[:, 1:] = res.x.reshape(K, J - 1)
        self.converged_ = bool(res.success)
        logP = _log_softmax(X1 @ self.W_, avail)
        self.llf_ = float(np.sum(logP[np.arange(N), yi]))
        self.train_shares_ = counts / N
        p0 = np.maximum(self.train_shares_, 1e-12)
        self.llnull_ = float(np.sum(counts * np.log(p0)))
        return self

    @property
    def prsquared(self) -> float:
        return 1.0 - self.llf_ / self.llnull_ if self.llnull_ < 0 else float("nan")

    def coef_table(self) -> dict:
        return {
            str(c): dict(zip(self.feature_names_, self.W_[:, j].tolist()))
            for j, c in enumerate(self.classes_)
        }

    def predict_proba(self, X: np.ndarray, avail: np.ndarray | None = None) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X1 = np.hstack([np.ones((len(X), 1)), X])
        P = np.exp(_log_softmax(X1 @ self.W_ + self.offset, avail))
        return np.nan_to_num(P)

    def sample(self, X: np.ndarray, rng: np.random.Generator, avail: np.ndarray | None = None) -> list:
        P = self.predict_proba(X, avail)
        idx = _sample_rows(P, rng)
        return [self.classes_[i] for i in idx]

    def validate(self, X: np.ndarray, y: list, avail: np.ndarray | None = None) -> dict:
        y = list(y)
        P = self.predict_proba(X, avail)
        known = np.array([v in self.classes_ for v in y])
        yi = np.array([self.classes_.index(v) if v in self.classes_ else 0 for v in y])
        acc = float(np.mean((P.argmax(axis=1) == yi) & known)) if len(y) else float("nan")
        ll = float(np.sum(np.log(np.maximum(P[np.arange(len(y)), yi], 1e-300))[known]))
        p0 = np.maximum(getattr(self, "train_shares_", np.full(len(self.classes_), 1 / len(self.classes_))), 1e-12)
        ll0 = float(np.sum(np.log(p0[yi])[known]))
        pred_share = P.mean(axis=0) if len(y) else np.zeros(len(self.classes_))
        obs = _shares(y, self.classes_)
        return {
            "n": len(y),
            "accuracy": acc,
            "prsquared": 1.0 - ll / ll0 if ll0 < 0 else float("nan"),
            "shares": {str(c): {"obs": obs[c], "pred": float(pred_share[j])} for j, c in enumerate(self.classes_)},
        }


class OrderedLogit:
    """Ordered logit over integer counts with a numpy/scipy analytic-gradient MLE.

    ``P(y ≤ c_j) = F(τ_j − x·β)`` with increasing thresholds
    ``τ_j = τ_0 + Σ_{m≤j} exp(δ_m)``. statsmodels ``OrderedModel`` (CompoShed
    Step 3) uses numerical score vectors, which is too slow for the repeated
    fits here; the likelihood is identical. Degenerate data (a single observed
    category or a failed fit) falls back to empirical frequencies.
    """

    def __init__(self, max_k: int | None = None, l2: float = 1e-4, maxiter: int = 500) -> None:
        self.max_k = max_k
        self.l2 = l2
        self.maxiter = maxiter

    def fit(self, X: np.ndarray, y, feature_names: list[str] | None = None) -> "OrderedLogit":
        y = np.asarray(y, dtype=int)
        if self.max_k is not None:
            y = np.minimum(y, self.max_k)
        X = np.asarray(X, dtype=float)
        self.classes_ = sorted(np.unique(y).tolist())
        self.offset = np.zeros(len(self.classes_))
        self.train_shares_ = np.array([(y == c).mean() for c in self.classes_])
        self.feature_names_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        self.keep_ = X.var(axis=0) > 1e-10 if len(X) else np.zeros(X.shape[1], bool)
        self.beta_ = None
        J = len(self.classes_)
        if J < 2:
            return self
        Xk = X[:, self.keep_]
        N, K = Xk.shape
        yi = np.searchsorted(self.classes_, y)
        rows = np.arange(N)

        def unpack(theta):
            beta, c0, delta = theta[:K], theta[K], theta[K + 1:]
            tau = c0 + np.concatenate([[0.0], np.cumsum(np.exp(delta))])
            return beta, tau, delta

        def obj(theta):
            beta, tau, delta = unpack(theta)
            xb = Xk @ beta
            ext = np.concatenate([[-np.inf], tau, [np.inf]])
            a = ext[yi + 1] - xb  # upper
            b = ext[yi] - xb  # lower
            Fa, Fb = expit(a), expit(b)
            p = np.maximum(Fa - Fb, 1e-300)
            fa, fb = Fa * (1 - Fa), Fb * (1 - Fb)
            ll = np.sum(np.log(p))
            g_beta = Xk.T @ (-(fa - fb) / p)
            G_tau = np.zeros((N, J - 1))
            up = yi < J - 1
            G_tau[rows[up], yi[up]] += fa[up] / p[up]
            lo = yi > 0
            G_tau[rows[lo], yi[lo] - 1] -= fb[lo] / p[lo]
            col = G_tau.sum(axis=0)
            g_c0 = col.sum()
            g_delta = np.exp(delta) * np.cumsum(col[::-1])[::-1][1:]
            grad = np.concatenate([g_beta, [g_c0], g_delta])
            pen = 0.5 * self.l2 * np.sum(beta * beta)
            return -ll / N + pen, -grad / N + np.concatenate([self.l2 * beta, np.zeros(J - 1)])

        cum = np.clip(np.cumsum(self.train_shares_)[:-1], 1e-6, 1 - 1e-6)
        tau0 = np.log(cum / (1 - cum))
        theta0 = np.concatenate([np.zeros(K), [tau0[0]], np.log(np.maximum(np.diff(tau0), 1e-3))])
        res = minimize(obj, theta0, jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter})
        beta, tau, _ = unpack(res.x)
        if np.all(np.isfinite(beta)) and np.all(np.isfinite(tau)):
            self.beta_, self.thresholds_ = beta, tau
            self.llf_ = -float(res.fun - 0.5 * self.l2 * np.sum(beta * beta)) * N
            self.llnull_ = float(np.sum(N * self.train_shares_ * np.log(np.maximum(self.train_shares_, 1e-12))))
            self.converged_ = bool(res.success)
        return self

    @property
    def prsquared(self) -> float:
        if self.beta_ is None or getattr(self, "llnull_", 0) >= 0:
            return float("nan")
        return 1.0 - self.llf_ / self.llnull_

    def coef_table(self) -> dict:
        if self.beta_ is None:
            return {"frequencies": dict(zip(map(str, self.classes_), self.train_shares_.tolist()))}
        names = [n for n, k in zip(self.feature_names_, self.keep_) if k]
        return {"beta": dict(zip(names, self.beta_.tolist())), "thresholds": self.thresholds_.tolist()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.beta_ is None:
            P = np.tile(self.train_shares_, (len(X), 1))
        else:
            xb = X[:, self.keep_] @ self.beta_
            cdf = expit(self.thresholds_[None, :] - xb[:, None])
            cdf = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
            P = np.clip(np.diff(cdf, axis=1), 0.0, None)
        P = P * np.exp(self.offset)[None, :]
        s = P.sum(axis=1, keepdims=True)
        return np.where(s > 0, P / np.where(s > 0, s, 1.0), 1.0 / P.shape[1])

    def sample(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
        max_allowed: np.ndarray | int | None = None,
        min_allowed: np.ndarray | int | None = None,
    ) -> np.ndarray:
        P = self.predict_proba(X)
        cats = np.array(self.classes_)
        if max_allowed is not None:
            P = np.where(cats[None, :] <= np.broadcast_to(max_allowed, (len(P),))[:, None], P, 0.0)
        if min_allowed is not None:
            P = np.where(cats[None, :] >= np.broadcast_to(min_allowed, (len(P),))[:, None], P, 0.0)
        s = P.sum(axis=1, keepdims=True)
        empty = s.ravel() <= 0
        P = np.where(s > 0, P / np.where(s > 0, s, 1.0), 0.0)
        out = cats[_sample_rows(np.where(empty[:, None], 1.0, P), rng)]
        if empty.any():
            # No category within bounds: take the bound nearest the support.
            lo = np.broadcast_to(min_allowed if min_allowed is not None else cats[0], (len(P),))
            hi = np.broadcast_to(max_allowed if max_allowed is not None else cats[-1], (len(P),))
            out[empty] = np.clip(cats[0], lo[empty], hi[empty])
        return out

    def validate(self, X: np.ndarray, y) -> dict:
        y = np.asarray(y, dtype=int)
        if self.max_k is not None:
            y = np.minimum(y, self.max_k)
        P = self.predict_proba(X)
        cats = np.array(self.classes_)
        pred = cats[P.argmax(axis=1)] if len(y) else np.array([])
        exp_count = P @ cats if len(y) else np.array([])
        return {
            "n": int(len(y)),
            "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
            "prsquared": self.prsquared,
            "mae": float(np.mean(np.abs(exp_count - y))) if len(y) else float("nan"),
            "shares": {str(c): {"obs": float((y == c).mean()) if len(y) else 0.0,
                                "pred": float(P[:, j].mean()) if len(y) else 0.0}
                       for j, c in enumerate(self.classes_)},
        }


class ConditionalLogit:
    """Logit over a fixed alternative table with alternative attributes and availability.

    ``dyn`` (optional, every method): callable ``dyn(s, e) -> (e - s, J, n_dyn)`` returning the
    chooser-specific alternative attributes of rows ``s:e``. ``b_mask`` (Kz × Kg, bool):
    only these entries of ``B`` are estimated; the rest are fixed at zero.
    """

    def __init__(
        self,
        F: sp.spmatrix | np.ndarray,
        G: np.ndarray,
        alt_names: list | None = None,
        l2: float = 1e-3,
        maxiter: int = 300,
        chunk: int = 4000,
        n_dyn: int = 0,
        b_mask: np.ndarray | None = None,
    ) -> None:
        self.F = sp.csr_matrix(F)
        self.G = np.asarray(G, dtype=float)
        self.J = self.F.shape[0]
        self.alt_names = list(alt_names) if alt_names is not None else list(range(self.J))
        self.l2 = l2
        self.maxiter = maxiter
        self.chunk = chunk
        self.n_dyn = n_dyn
        self.b_mask = None if b_mask is None else np.asarray(b_mask, dtype=bool)
        self.offset = np.zeros(self.J)
        self.gamma_ = np.zeros(self.F.shape[1])
        self.delta_ = np.zeros(n_dyn)
        self.B_ = None

    def _utilities(self, Z: np.ndarray, gamma: np.ndarray, B: np.ndarray, A: np.ndarray | None = None,
                   delta: np.ndarray | None = None) -> np.ndarray:
        V = (self.F @ gamma)[None, :] + (Z @ B) @ self.G.T + self.offset[None, :]
        if A is not None and len(delta):
            V = V + A @ delta
        return V

    def _dyn(self, dyn, s: int, e: int) -> np.ndarray | None:
        if dyn is None or not getattr(self, "n_dyn", 0):  # bundles saved before n_dyn existed
            return None
        return dyn(s, e)

    def fit(self, Z: np.ndarray, y: np.ndarray, avail: np.ndarray | None = None, dyn=None) -> "ConditionalLogit":
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=int)
        N, Kz = Z.shape
        Kf, Kg, Ka = self.F.shape[1], self.G.shape[1], self.n_dyn
        if avail is not None:
            avail = avail.copy()
            self.forced_available_ = int((~avail[np.arange(N), y]).sum())
            avail[np.arange(N), y] = True
        Fy = np.asarray(self.F[y].todense())
        Gy = self.G[y]
        FT = self.F.T.tocsr()
        # Chooser-specific attributes are recomputed per chunk (N × J × Ka can exceed memory).
        Ay = np.zeros(Ka)
        for s in range(0, N, self.chunk):
            A = self._dyn(dyn, s, min(N, s + self.chunk))
            if A is not None:
                Ay += A[np.arange(len(A)), y[s:s + len(A)]].sum(axis=0)

        def obj(theta):
            gamma = theta[:Kf]
            B = theta[Kf:Kf + Kz * Kg].reshape(Kz, Kg)
            delta = theta[Kf + Kz * Kg:]
            ll = 0.0
            g_gamma = Fy.sum(axis=0).copy()
            g_B = Z.T @ Gy
            g_delta = Ay.copy()
            for s in range(0, N, self.chunk):
                e = min(N, s + self.chunk)
                A = self._dyn(dyn, s, e)
                V = self._utilities(Z[s:e], gamma, B, A, delta)
                logP = _log_softmax(V, None if avail is None else avail[s:e])
                ll += np.sum(logP[np.arange(e - s), y[s:e]])
                P = np.exp(logP)
                g_gamma -= np.asarray((FT @ P.T).sum(axis=1)).ravel()
                g_B -= Z[s:e].T @ (P @ self.G)
                if A is not None:
                    g_delta -= np.einsum("nj,njk->k", P, A)
            f = -ll / N + 0.5 * self.l2 * np.sum(theta * theta)
            g = -np.concatenate([g_gamma, g_B.ravel(), g_delta]) / N + self.l2 * theta
            return f, g

        mask = getattr(self, "b_mask", None)
        if mask is None:
            free = np.ones(Kf + Kz * Kg + Ka, dtype=bool)
        else:
            if mask.shape != (Kz, Kg):
                raise ValueError(f"b_mask has shape {mask.shape}, expected {(Kz, Kg)}")
            free = np.concatenate([np.ones(Kf, bool), mask.ravel(), np.ones(Ka, bool)])

        def obj_free(x):
            theta = np.zeros(len(free))
            theta[free] = x
            f, g = obj(theta)
            return f, g[free]

        res = minimize(obj_free, np.zeros(int(free.sum())), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter})
        theta = np.zeros(len(free))
        theta[free] = res.x
        self.gamma_ = theta[:Kf]
        self.B_ = theta[Kf:Kf + Kz * Kg].reshape(Kz, Kg)
        self.delta_ = theta[Kf + Kz * Kg:]
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.n_params_ = int(free.sum())
        self.llf_ = -float(res.fun - 0.5 * self.l2 * np.sum(res.x * res.x)) * N
        if avail is None:
            self.llnull_ = -N * np.log(self.J)
        else:
            self.llnull_ = -float(np.sum(np.log(avail.sum(axis=1))))
        return self

    @property
    def prsquared(self) -> float:
        return 1.0 - self.llf_ / self.llnull_

    def predict_proba(self, Z: np.ndarray, avail: np.ndarray | None = None, dyn=None) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        out = np.empty((len(Z), self.J))
        for s in range(0, len(Z), self.chunk):
            e = min(len(Z), s + self.chunk)
            V = self._utilities(Z[s:e], self.gamma_, self.B_, self._dyn(dyn, s, e), getattr(self, "delta_", None))
            out[s:e] = np.exp(_log_softmax(V, None if avail is None else avail[s:e]))
        return np.nan_to_num(out)

    def sample(self, Z: np.ndarray, rng: np.random.Generator, avail: np.ndarray | None = None, dyn=None) -> np.ndarray:
        """Sample an alternative index per row; rows with no available alternative return -1."""
        P = self.predict_proba(Z, avail, dyn)
        empty = P.sum(axis=1) <= 0
        idx = _sample_rows(np.where(empty[:, None], 1.0, P), rng)
        idx[empty] = -1
        return idx

    def validate(self, Z: np.ndarray, y: np.ndarray, avail: np.ndarray | None = None, dyn=None) -> dict:
        y = np.asarray(y, dtype=int)
        if avail is not None:
            avail = avail.copy()
            avail[np.arange(len(y)), y] = True
        P = self.predict_proba(Z, avail, dyn)
        ll = float(np.sum(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300))))
        ll0 = -float(np.sum(np.log(avail.sum(axis=1)))) if avail is not None else -len(y) * np.log(self.J)
        return {
            "n": int(len(y)),
            "accuracy": float((P.argmax(axis=1) == y).mean()) if len(y) else float("nan"),
            "prsquared": 1.0 - ll / ll0 if ll0 < 0 else float("nan"),
        }


class AltAttrMNL:
    """Logit over a fixed alternative table whose utilities are built from alternative attributes.

    ``V_nj = a_j·γ + x_n·B·c_j + offset_j``: every alternative attribute (column of
    ``attrs``) gets a generic constant, and every interaction attribute ``c_j``
    (``inter_attrs``; default the same table) an interaction with each person
    feature, as an ActivitySim ``interaction_simulate`` spec or a DaySim utility
    function does. The parameter count is ``n_attrs + K · n_inter`` instead of the
    ``(K + 1) · (J − 1)`` of an alternative-specific MNL. ``offset`` is alternative-specific, so calibration can still move every share.
    Same interface as ``MNL`` (``classes_``, ``offset``, ``fit`` / ``predict_proba`` /
    ``sample`` / ``validate``).
    """

    def __init__(self, attrs: dict, attr_names: list[str], l2: float = 1e-4, maxiter: int = 500,
                 inter_attrs: dict | None = None, inter_names: list[str] | None = None) -> None:
        self.attrs = {k: np.asarray(v, dtype=float) for k, v in attrs.items()}
        self.attr_names = list(attr_names)
        self.inter_attrs = None if inter_attrs is None else {k: np.asarray(v, dtype=float) for k, v in inter_attrs.items()}
        self.inter_names = list(inter_names) if inter_names is not None else self.attr_names
        self.l2 = l2
        self.maxiter = maxiter

    @property
    def offset(self) -> np.ndarray:
        return self.clogit_.offset

    @offset.setter
    def offset(self, value) -> None:
        self.clogit_.offset = np.asarray(value, dtype=float)

    def fit(self, X: np.ndarray, y: list, classes: list | None = None, avail: np.ndarray | None = None,
            feature_names: list[str] | None = None) -> "AltAttrMNL":
        y = list(y)
        self.classes_ = list(classes) if classes is not None else sorted(set(y), key=str)
        A = np.array([self.attrs[c] for c in self.classes_])
        inter = getattr(self, "inter_attrs", None)
        C = A if inter is None else np.array([inter[c] for c in self.classes_])
        self.clogit_ = ConditionalLogit(A, C, alt_names=self.classes_, l2=self.l2, maxiter=self.maxiter)
        X = np.asarray(X, dtype=float)
        self.feature_names_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        yi = np.array([self.classes_.index(v) for v in y], dtype=int)
        self.clogit_.fit(X, yi, avail)
        counts = np.bincount(yi, minlength=len(self.classes_)).astype(float)
        self.train_shares_ = counts / max(len(y), 1)
        self.llf_ = self.clogit_.llf_
        p0 = np.maximum(self.train_shares_, 1e-12)
        self.llnull_ = float(np.sum(counts * np.log(p0)))
        self.converged_ = self.clogit_.converged_
        self.n_params_ = A.shape[1] + C.shape[1] * X.shape[1]
        return self

    @property
    def prsquared(self) -> float:
        return 1.0 - self.llf_ / self.llnull_ if self.llnull_ < 0 else float("nan")

    def coef_table(self) -> dict:
        out = {"const": dict(zip(self.attr_names, self.clogit_.gamma_.tolist()))}
        names = getattr(self, "inter_names", self.attr_names)
        for k, name in enumerate(self.feature_names_):
            out[name] = dict(zip(names, self.clogit_.B_[k].tolist()))
        return out

    def predict_proba(self, X: np.ndarray, avail: np.ndarray | None = None) -> np.ndarray:
        return self.clogit_.predict_proba(np.atleast_2d(np.asarray(X, dtype=float)), avail)

    def sample(self, X: np.ndarray, rng: np.random.Generator, avail: np.ndarray | None = None) -> list:
        idx = _sample_rows(self.predict_proba(X, avail), rng)
        return [self.classes_[i] for i in idx]

    validate = MNL.validate
