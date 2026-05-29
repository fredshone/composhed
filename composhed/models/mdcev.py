"""MDCEV variant — replaces Steps 1–5 with a single Biogeme GammaProfile model."""

import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import biogeme.database as bio_db
from biogeme.mdcev import GammaProfile
from biogeme.expressions import Beta, Variable

from composhed.data import LABEL_COLS, encode_features


def _safe(s: str) -> str:
    """Sanitise a string for use as a Biogeme Beta parameter name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


class MDCEVModel:
    """Multiple Discrete-Continuous Extreme Value model for 24-hour time allocation.

    Predicts time spent on each enumerated activity occurrence simultaneously.
    Alternatives are enumerated per-type: home_0, home_1, work_0, shop_0, shop_1, etc.
    The list of alternatives (TYPES) is set dynamically at fit time.
    """

    BUDGET = 1440.0

    def fit(
        self,
        records: list[dict],
        feature_names: list[str],
        types: list[str] | None = None,
        X: np.ndarray | None = None,
        fix_gamma: bool = True,
    ) -> "MDCEVModel":
        """Estimate MDCEV parameters from training records.

        Parameters
        ----------
        records:
            Per-person dicts as returned by ``build_enumerated_dataset``.
            Each must contain ``enumerated_durations: dict[str, float]``.
        feature_names:
            Ordered list of one-hot feature column names (from ``encode_features``).
        types:
            Ordered list of enumerated alternative names (e.g. ["home_0", "home_1",
            "work_0", ...]). Must be provided when using enumerated records.
        X:
            Pre-computed feature matrix ``(N, len(feature_names))``. If provided,
            the internal ``encode_features`` call is skipped.
        fix_gamma:
            If True (default), gamma parameters are fixed to the conditional mean
            duration per activity type and not estimated. This prevents the
            degenerate optimum (gammas at bounds, intercepts ±20) that arises
            when gammas are free. If False, gammas are estimated freely within
            [1, 1440].
        """
        if types is None:
            raise ValueError("types must be provided: list of enumerated alternative names")
        self.TYPES = list(types)
        self.feature_names_ = feature_names
        K = len(self.TYPES)
        N = len(records)

        # ---- 1. Encode label features ----------------------------------------
        if X is None:
            X, _ = encode_features(records, LABEL_COLS, feature_names=feature_names)

        # ---- 2. Extract time allocations from enumerated_durations -----------
        t_matrix = np.zeros((N, K), dtype=np.float64)
        for n, rec in enumerate(records):
            td = rec["total_durations"]
            for j, atype in enumerate(self.TYPES):
                t_matrix[n, j] = td.get(atype, 0.0)

        # ---- 3. Compute n_chosen per person ----------------------------------
        n_chosen = (t_matrix > 0).sum(axis=1).astype(int)

        # ---- 4. Build Pandas DataFrame for Biogeme ---------------------------
        feat_df = pd.DataFrame(X, columns=feature_names)
        for i, atype in enumerate(self.TYPES):
            feat_df[f"t_{atype}"] = t_matrix[:, i]
        feat_df["n_chosen"] = n_chosen

        database = bio_db.Database("mdcev_train", feat_df)

        # ---- 4b. Data-driven gamma: fix to conditional mean duration ---------
        # Free-gamma estimation converges to a degenerate optimum (gamma_home→0,
        # gamma_work→1440, intercepts ±20) regardless of starting point.
        # Fixing gamma to the observed conditional mean duration removes that
        # degree of freedom: the optimizer only needs to find intercepts and
        # feature coefficients that reproduce participation rates, and the
        # resulting intercept differences stay within ±2 rather than ±20.
        home_idx = self.TYPES.index("home") if "home" in self.TYPES else 0
        gamma_fixed = np.array([
            float(np.clip(
                np.mean(t_matrix[t_matrix[:, j] > 0, j]) if (t_matrix[:, j] > 0).sum() > 0 else 100.0,
                1.0, 1440.0,
            ))
            for j in range(K)
        ])
        mean_home = float(gamma_fixed[home_idx])
        cte_inits = np.clip(
            np.log(np.maximum(gamma_fixed, 1.0) / max(mean_home, 1.0)) / 2.0,
            -5.0, 5.0,
        )
        if fix_gamma:
            self._gamma_fixed = gamma_fixed  # stored for __getstate__ and serialisation
        else:
            self._gamma_fixed = None

        # ---- 4c. Clear Biogeme cache so optimisation always runs fresh -------
        for f in glob.glob("__mdcev*.iter") + glob.glob("mdcev*.pickle"):
            try:
                os.remove(f)
            except OSError:
                pass

        # ---- 5. Define Biogeme utility expressions ---------------------------
        V: dict[int, object] = {}
        gamma: dict[int, object] = {}
        for i, atype in enumerate(self.TYPES):
            s = _safe(atype)
            util = Beta(f"cte_{s}", float(cte_inits[i]), None, None, 0)
            for feat in feature_names:
                util = util + Beta(f"b_{s}_{_safe(feat)}", 0, None, None, 0) * Variable(
                    feat
                )
            V[i] = util
            if fix_gamma:
                # last arg=1 means the parameter is not estimated by Biogeme
                gamma[i] = Beta(f"gamma_{s}", float(gamma_fixed[i]), None, None, 1)
            else:
                gamma[i] = Beta(f"gamma_{s}", float(gamma_fixed[i]), 1.0, 1440, 0)

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

    def _build_expressions(
        self, gamma_values: list[float] | None = None
    ) -> tuple[dict, dict, object]:
        """Re-create V, gamma, scale biogeme expressions from feature_names_.

        gamma_values: if provided (e.g. from __setstate__), gammas are fixed at
        those values; otherwise falls back to free gammas (legacy models).
        """
        V: dict[int, object] = {}
        gamma: dict[int, object] = {}
        for i, atype in enumerate(self.TYPES):
            s = _safe(atype)
            util = Beta(f"cte_{s}", 0, None, None, 0)
            for feat in self.feature_names_:
                util = util + Beta(f"b_{s}_{_safe(feat)}", 0, None, None, 0) * Variable(
                    feat
                )
            V[i] = util
            if gamma_values is not None:
                gamma[i] = Beta(f"gamma_{s}", float(gamma_values[i]), None, None, 1)
            else:
                gamma[i] = Beta(f"gamma_{s}", 100, 1.0, 1440, 0)
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
        # Fixed-gamma models omit gamma keys from get_beta_values(); use _gamma_fixed.
        if getattr(self, "_gamma_fixed", None) is not None:
            self._gammas = self._gamma_fixed.copy()
        else:
            self._gammas = np.array([bv[f"gamma_{_safe(t)}"] for t in self.TYPES])
        self._scale = float(bv["scale"])

    def calibrate(
        self,
        X: np.ndarray,
        target_participation: dict[str, float],
        n_draws: int = 100,
        subsample: int = 5000,
        max_iter: int = 50,
        tol: float = 0.005,
        lr: float = 0.1,
        delta_cap: float = 0.3,
        skip_threshold: float = 0.99,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Adjust intercepts (ASCs) to match target participation rates.

        Uses iterative log-ratio updates (standard ASC calibration):
          delta = clip(lr * (log(target) - log(pred)), ±delta_cap)

        Types with target >= skip_threshold (e.g. home, always participates) are
        excluded — they are structurally guaranteed and calibrating them destabilises
        the coupled budget model by inflating lambda beyond other types' psi values.

        Offsets accumulate in _calibration_offsets and are persisted through
        serialisation so a loaded model retains calibration.

        Returns dict mapping each activity type to its cumulative offset.
        """
        type_idx = {t: i for i, t in enumerate(self.TYPES)}
        # Exclude structural types whose target is essentially 1 — calibrating
        # these drives their intercept to an extreme and collapses the budget.
        targets = {
            t: v for t, v in target_participation.items()
            if t in type_idx and v < skip_threshold
        }
        if not targets:
            return {}

        N = X.shape[0]
        rng = np.random.default_rng()
        idx = rng.choice(N, min(subsample, N), replace=False)
        X_cal = X[idx]

        offsets = np.zeros(len(self.TYPES))

        for it in range(max_iter):
            eps = np.random.gumbel(0, 1, size=(n_draws, X_cal.shape[0], len(self.TYPES)))
            draws = np.stack([self._numpy_forecast(X_cal, eps[d]) for d in range(n_draws)])
            pred = (draws > 0).mean(axis=(0, 1))  # (K,)

            gaps = {t: abs(v - float(pred[type_idx[t]])) for t, v in targets.items()}
            max_gap = max(gaps.values())

            if verbose:
                parts = "  ".join(
                    f"{t}={float(pred[type_idx[t]]):.3f}(tgt={targets[t]:.3f})"
                    for t in self.TYPES
                    if t in targets
                )
                print(f"    calib iter {it:2d}  max_gap={max_gap:.4f}  {parts}")

            if max_gap < tol:
                break

            for atype, target in targets.items():
                j = type_idx[atype]
                p = float(np.clip(pred[j], 1e-6, 1 - 1e-6))
                t_clip = float(np.clip(target, 1e-6, 1 - 1e-6))
                delta = np.clip(lr * (np.log(t_clip) - np.log(p)), -delta_cap, delta_cap)
                self._intercepts[j] += delta
                offsets[j] += delta

        self._calibration_offsets = (
            getattr(self, "_calibration_offsets", np.zeros(len(self.TYPES))) + offsets
        )
        return {t: float(offsets[type_idx[t]]) for t in self.TYPES if t in type_idx}

    def __getstate__(self) -> dict:
        raw = self.biogeme_model_._estimation_results.raw_estimation_results
        return {
            "feature_names_": self.feature_names_,
            "TYPES": self.TYPES,
            "raw": raw,
            "calibration_offsets": getattr(
                self, "_calibration_offsets", np.zeros(len(self.TYPES))
            ),
            "gamma_fixed": getattr(self, "_gamma_fixed", None),
        }

    def __setstate__(self, state: dict) -> None:
        from biogeme.results_processing.estimation_results import EstimationResults

        self.feature_names_ = state["feature_names_"]
        self.TYPES = state["TYPES"]
        gamma_values = state.get("gamma_fixed")
        if gamma_values is not None:
            self._gamma_fixed = np.asarray(gamma_values)
        V, gamma, scale = self._build_expressions(
            gamma_values=list(gamma_values) if gamma_values is not None else None
        )
        self.biogeme_model_ = GammaProfile(
            model_name="mdcev",
            baseline_utilities=V,
            gamma_parameters=gamma,
            scale_parameter=scale,
        )
        self.biogeme_model_.estimation_results = EstimationResults(state["raw"])
        self._build_numpy_params()
        # For fixed-gamma models _build_numpy_params reads the fixed values from
        # Biogeme correctly; override to be safe in case raw doesn't include them.
        if gamma_values is not None:
            self._gammas = np.asarray(gamma_values)
        # Re-apply calibration offsets baked in after estimation
        offsets = state.get("calibration_offsets", np.zeros(len(self.TYPES)))
        self._calibration_offsets = np.asarray(offsets)
        self._intercepts += self._calibration_offsets

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N = X.shape[0]
        K = len(self.TYPES)

        X_t = torch.tensor(X, dtype=torch.float64, device=device)
        eps_t = torch.tensor(epsilons, dtype=torch.float64, device=device)
        B_t = torch.tensor(self._B, dtype=torch.float64, device=device)
        intercepts_t = torch.tensor(
            self._intercepts, dtype=torch.float64, device=device
        )
        gammas_t = torch.tensor(self._gammas, dtype=torch.float64, device=device)

        # psi_k = exp(V_k + eps_k / scale), V_k = intercept_k + B_k · x
        psi = torch.exp(X_t @ B_t.T + intercepts_t + eps_t / self._scale)  # (N, K)

        # Sort by psi descending — Pinjari-Bhat chosen-set identification
        order = torch.argsort(-psi, dim=1)  # (N, K)
        psi_s = torch.gather(psi, 1, order)
        gamma_s = gammas_t.unsqueeze(0).expand(N, K).gather(1, order)  # (N, K)

        # lambda(m) = cum(gamma * psi) / (BUDGET + cum(gamma)) if first m chosen
        cum_gp = torch.cumsum(gamma_s * psi_s, dim=1)  # (N, K)
        cum_g = torch.cumsum(gamma_s, dim=1)  # (N, K)
        lam_m = cum_gp / (self.BUDGET + cum_g)  # (N, K)

        # Chosen set is a prefix: psi_s[:,m] > lam_m[:,m]
        cond = psi_s > lam_m  # (N, K)
        all_ch = cond.all(dim=1)
        n_ch = torch.where(
            all_ch, torch.tensor(K, device=device), torch.argmin(cond.long(), dim=1)
        )
        n_ch = torch.clamp(n_ch, min=1)

        lam = lam_m[torch.arange(N, device=device), n_ch - 1]  # (N,)

        # x_k = gamma_k * (psi_k / lambda - 1) for chosen, 0 for unchosen
        x_s = gamma_s * (psi_s / lam.unsqueeze(1) - 1)
        mask = torch.arange(K, device=device).unsqueeze(0) < n_ch.unsqueeze(1)
        x_s = torch.where(mask, torch.clamp(x_s, min=0.0), torch.zeros_like(x_s))

        # Un-sort, normalise to exact BUDGET
        x = torch.zeros_like(x_s)
        x.scatter_(1, order, x_s)
        x = x / x.sum(dim=1, keepdim=True) * self.BUDGET
        return x.cpu().numpy()

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
        return [
            {self.TYPES[i]: float(x[n, i]) for i in range(K)} for n in range(x.shape[0])
        ]

    def validate(self, records: list[dict], X: np.ndarray, n_draws: int = 50) -> dict:
        """In-sample participation rates and mean durations: actual vs predicted.

        Uses n_draws Gumbel samples and averages participation/duration estimates
        across them to reduce single-draw noise.
        """
        N = len(records)
        t_actual = np.zeros((N, len(self.TYPES)), dtype=np.float64)
        for n, rec in enumerate(records):
            td = rec["total_durations"]
            for j, atype in enumerate(self.TYPES):
                t_actual[n, j] = td.get(atype, 0.0)

        eps = np.random.gumbel(0, 1, size=(n_draws, N, len(self.TYPES)))
        draws = np.stack([self._numpy_forecast(X, eps[d]) for d in range(n_draws)])
        # draws: (n_draws, N, K)
        t_pred_mean = draws.mean(axis=0)           # mean allocation per person
        part_pred = (draws > 0).mean(axis=(0, 1))  # mean participation rate per type

        # Conditional mean duration per draw then averaged — this matches how
        # mean_duration_actual is computed (mean duration given participation).
        cond_dur_pred = np.zeros(len(self.TYPES))
        for j in range(len(self.TYPES)):
            per_draw = [
                float(draws[d, draws[d, :, j] > 0, j].mean())
                for d in range(n_draws)
                if (draws[d, :, j] > 0).any()
            ]
            cond_dur_pred[j] = float(np.mean(per_draw)) if per_draw else 0.0

        result: dict = {}
        for j, atype in enumerate(self.TYPES):
            act_mask = t_actual[:, j] > 0
            result[atype] = {
                "participation_actual": float(act_mask.mean()),
                "participation_pred": float(part_pred[j]),
                "mean_duration_actual": (
                    float(t_actual[act_mask, j].mean()) if act_mask.any() else 0.0
                ),
                "mean_duration_pred": cond_dur_pred[j],
            }

        try:
            ll = float(self.biogeme_model_.estimation_results.data.logLike)
            ll_null = float(self.biogeme_model_.estimation_results.data.nullLogLike)
            result["log_likelihood"] = ll
            result["pseudo_r2"] = 1.0 - ll / ll_null if ll_null != 0 else float("nan")
        except Exception:
            pass

        return result

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
