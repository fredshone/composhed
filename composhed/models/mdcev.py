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

    def fit(self, records: list[dict], feature_names: list[str]) -> "MDCEVModel":
        """Estimate MDCEV parameters from training records.

        Parameters
        ----------
        records:
            Per-person dicts as returned by ``build_training_dataset``.
        feature_names:
            Ordered list of one-hot feature column names (from ``encode_features``).
        """
        self.feature_names_ = feature_names
        K = len(self.TYPES)

        # ---- 1. Encode label features ----------------------------------------
        X, _ = encode_features(records, LABEL_COLS, feature_names=feature_names)

        # ---- 2. Extract time allocations per activity type -------------------
        t_cols: dict[str, list[float]] = {f"t_{a}": [] for a in self.TYPES}
        for r in records:
            for atype in self.TYPES:
                if atype == "home":
                    t_cols["t_home"].append(float(r["total_home"]))
                elif atype in ("work", "education"):
                    if r["mandatory_type"] == atype:
                        t_cols[f"t_{atype}"].append(float(r["mandatory_duration"]))
                    else:
                        t_cols[f"t_{atype}"].append(0.0)
                else:
                    total = sum(d for a, d in r["disc_activities"] if a == atype)
                    t_cols[f"t_{atype}"].append(float(total))

        # ---- 3. Compute n_chosen per person ----------------------------------
        t_matrix = np.column_stack([t_cols[f"t_{a}"] for a in self.TYPES])
        n_chosen = (t_matrix > 0).sum(axis=1).astype(int)

        # ---- 4. Build Pandas DataFrame for Biogeme ---------------------------
        feat_df = pd.DataFrame(X, columns=feature_names)
        for atype in self.TYPES:
            feat_df[f"t_{atype}"] = t_cols[f"t_{atype}"]
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
        return self

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
        row = dict(zip(self.feature_names_, x_label.tolist()))
        df = pd.DataFrame([row])
        database = bio_db.Database("mdcev_gen", df)

        K = len(self.TYPES)
        epsilons = [np.random.gumbel(0, 1, size=(1, K))]
        results = self.biogeme_model_.forecast(
            database, total_budget=self.BUDGET, epsilons=epsilons
        )
        # results[0] is a (1, K) DataFrame; columns are integer alternative indices
        alloc = results[0].iloc[0]
        return {self.TYPES[i]: float(alloc.iloc[i]) for i in range(K)}
