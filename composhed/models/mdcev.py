"""MDCEV variant — replaces Steps 1–5 with a single Biogeme GammaProfile model."""

import re

import numpy as np
import pandas as pd
import biogeme.database as bio_db
from biogeme.mdcev import GammaProfile
from biogeme.expressions import Beta, Variable

from composhed.data import LABEL_COLS, encode_features


def _safe(s: str) -> str:
    """Sanitise a string for use as a Biogeme Beta parameter name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


class MDCEVModel:
    """Multiple Discrete-Continuous Extreme Value model for 24-hour time allocation.

    Predicts time spent on each of 8 activity types simultaneously, replacing the
    four sequential models (Steps 1–5) in the baseline CompSched pipeline.

    One activity per type: MDCEV gives total time per type; each chosen type appears
    once in the schedule. Documented simplification.
    """

    TYPES = ["home", "work", "education", "shop", "visit", "escort", "medical", "other"]
    BUDGET = 1440.0

    def fit(
        self,
        records: list[dict],
        feature_names: list[str],
        X: np.ndarray | None = None,
    ) -> "MDCEVModel":
        """Estimate MDCEV parameters from training records.

        Parameters
        ----------
        records:
            Per-person dicts as returned by ``build_training_dataset``.
        feature_names:
            Ordered list of one-hot feature column names (from ``encode_features``).
        X:
            Pre-computed feature matrix ``(N, len(feature_names))``. If provided,
            the internal ``encode_features`` call is skipped.
        """
        self.feature_names_ = feature_names
        K = len(self.TYPES)
        N = len(records)

        # ---- 1. Encode label features ----------------------------------------
        if X is None:
            X, _ = encode_features(records, LABEL_COLS, feature_names=feature_names)

        # ---- 2. Extract time allocations per activity type -------------------
        DISC_TYPES = ["shop", "visit", "escort", "medical", "other"]

        # Pre-aggregate disc_activities per person to avoid repeated inner loops
        disc_agg: list[dict[str, float]] = []
        for r in records:
            agg: dict[str, float] = {}
            for a, d in r["disc_activities"]:
                agg[a] = agg.get(a, 0.0) + d
            disc_agg.append(agg)

        t_matrix = np.zeros((N, K), dtype=np.float64)
        t_matrix[:, 0] = [r["total_home"] for r in records]
        t_matrix[:, 1] = [
            r["mandatory_duration"] if r["mandatory_type"] == "work" else 0.0
            for r in records
        ]
        t_matrix[:, 2] = [
            r["mandatory_duration"] if r["mandatory_type"] == "education" else 0.0
            for r in records
        ]
        for j, atype in enumerate(DISC_TYPES, start=3):
            t_matrix[:, j] = [agg.get(atype, 0.0) for agg in disc_agg]

        # ---- 3. Compute n_chosen per person ----------------------------------
        n_chosen = (t_matrix > 0).sum(axis=1).astype(int)

        # ---- 4. Build Pandas DataFrame for Biogeme ---------------------------
        feat_df = pd.DataFrame(X, columns=feature_names)
        for i, atype in enumerate(self.TYPES):
            feat_df[f"t_{atype}"] = t_matrix[:, i]
        feat_df["n_chosen"] = n_chosen

        database = bio_db.Database("mdcev_train", feat_df)

        # ---- 5. Define Biogeme utility expressions ---------------------------
        V: dict[int, object] = {}
        gamma: dict[int, object] = {}
        for i, atype in enumerate(self.TYPES):
            s = _safe(atype)
            util = Beta(f"cte_{s}", 0, None, None, 0)
            for feat in feature_names:
                util = util + Beta(f"b_{s}_{_safe(feat)}", 0, None, None, 0) * Variable(feat)
            V[i] = util
            gamma[i] = Beta(f"gamma_{s}", 1, 1e-4, None, 0)

        scale = Beta("scale", 1, 1e-4, None, 0)

        # ---- 6. Estimate ----------------------------------------------------
        self.biogeme_model_ = GammaProfile(
            model_name="mdcev",
            baseline_utilities=V,
            gamma_parameters=gamma,
            scale_parameter=scale,
        )
        consumed = {i: Variable(f"t_{atype}") for i, atype in enumerate(self.TYPES)}
        self.biogeme_model_.estimate_parameters(
            database=database,
            number_of_chosen_alternatives=Variable("n_chosen"),
            consumed_quantities=consumed,
            generate_html=False,
            generate_netcdf=False,
            generate_yaml=False,
        )
        self._build_numpy_params()
        return self

    def _build_expressions(self) -> tuple[dict, dict, object]:
        """Re-create V, gamma, scale biogeme expressions from feature_names_."""
        V: dict[int, object] = {}
        gamma: dict[int, object] = {}
        for i, atype in enumerate(self.TYPES):
            s = _safe(atype)
            util = Beta(f"cte_{s}", 0, None, None, 0)
            for feat in self.feature_names_:
                util = util + Beta(f"b_{s}_{_safe(feat)}", 0, None, None, 0) * Variable(feat)
            V[i] = util
            gamma[i] = Beta(f"gamma_{s}", 1, 1e-4, None, 0)
        scale = Beta("scale", 1, 1e-4, None, 0)
        return V, gamma, scale

    def _build_numpy_params(self) -> None:
        """Cache fitted Biogeme parameters as numpy arrays for fast vectorised inference."""
        bv = self.biogeme_model_.estimation_results.get_beta_values()
        self._intercepts = np.array([bv[f"cte_{_safe(t)}"] for t in self.TYPES])
        self._B = np.array(
            [
                [bv.get(f"b_{_safe(t)}_{_safe(f)}", 0.0) for f in self.feature_names_]
                for t in self.TYPES
            ]
        )  # (K, n_features)
        self._gammas = np.array([bv[f"gamma_{_safe(t)}"] for t in self.TYPES])
        self._scale = float(bv["scale"])

    def __getstate__(self) -> dict:
        raw = self.biogeme_model_._estimation_results.raw_estimation_results
        return {"feature_names_": self.feature_names_, "raw": raw}

    def __setstate__(self, state: dict) -> None:
        from biogeme.results_processing.estimation_results import EstimationResults

        self.feature_names_ = state["feature_names_"]
        V, gamma, scale = self._build_expressions()
        self.biogeme_model_ = GammaProfile(
            model_name="mdcev",
            baseline_utilities=V,
            gamma_parameters=gamma,
            scale_parameter=scale,
        )
        self.biogeme_model_.estimation_results = EstimationResults(state["raw"])
        self._build_numpy_params()

    def _numpy_forecast(self, X: np.ndarray, epsilons: np.ndarray) -> np.ndarray:
        """Vectorised Gamma-profile MDCEV allocation (Pinjari-Bhat 2021).

        Parameters
        ----------
        X:
            ``(N, n_features)`` feature matrix.
        epsilons:
            ``(N, K)`` raw Gumbel draws (divided by scale internally).

        Returns
        -------
        ``(N, K)`` allocation matrix; each row sums to ``BUDGET``.
        """
        N = X.shape[0]
        K = len(self.TYPES)

        # psi_k = exp(V_k + eps_k / scale), V_k = intercept_k + B_k · x
        psi = np.exp(X @ self._B.T + self._intercepts + epsilons / self._scale)  # (N, K)

        # Sort by psi descending — Pinjari-Bhat chosen-set identification
        order = np.argsort(-psi, axis=1)  # (N, K)
        psi_s = np.take_along_axis(psi, order, axis=1)
        gamma_s = self._gammas[order]

        # lambda(m) = cum(gamma * psi) / (BUDGET + cum(gamma)) if first m chosen
        cum_gp = np.cumsum(gamma_s * psi_s, axis=1)  # (N, K)
        cum_g = np.cumsum(gamma_s, axis=1)  # (N, K)
        lam_m = cum_gp / (self.BUDGET + cum_g)  # (N, K)

        # Chosen set is a prefix: psi_s[:,m] > lam_m[:,m]
        cond = psi_s > lam_m  # (N, K)
        all_ch = cond.all(axis=1)
        n_ch = np.where(all_ch, K, np.argmin(cond, axis=1))
        n_ch = np.maximum(n_ch, 1)

        lam = lam_m[np.arange(N), n_ch - 1]  # (N,)

        # x_k = gamma_k * (psi_k / lambda - 1) for chosen, 0 for unchosen
        x_s = gamma_s * (psi_s / lam[:, None] - 1)
        mask = np.arange(K)[None, :] < n_ch[:, None]
        x_s = np.where(mask, np.maximum(x_s, 0.0), 0.0)

        # Un-sort, normalise to exact BUDGET
        x = np.empty_like(x_s)
        x[np.arange(N)[:, None], order] = x_s
        x = x / x.sum(axis=1, keepdims=True) * self.BUDGET
        return x

    def sample_batch(self, X: np.ndarray) -> list[dict[str, float]]:
        """Sample time allocations for N persons using vectorised NumPy forecast.

        Parameters
        ----------
        X:
            2-D float array of shape ``(N, len(feature_names_))``.

        Returns
        -------
        List of N dicts, each mapping activity type to allocated minutes (sum ≈ 1440).
        """
        eps = np.random.gumbel(0, 1, size=(X.shape[0], len(self.TYPES)))
        x = self._numpy_forecast(X, eps)
        K = len(self.TYPES)
        return [{self.TYPES[i]: float(x[n, i]) for i in range(K)} for n in range(x.shape[0])]

    def sample(self, x_label: np.ndarray) -> dict[str, float]:
        """Sample one time allocation for a single person.

        Parameters
        ----------
        x_label:
            1-D float array of shape ``(len(feature_names_),)``.

        Returns
        -------
        dict mapping each activity type to allocated minutes (8 entries, sum ≈ 1440).
        """
        return self.sample_batch(x_label.reshape(1, -1))[0]
