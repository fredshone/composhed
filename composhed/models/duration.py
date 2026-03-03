"""Step 5 — Non-mandatory activity duration per type (log-normal OLS)."""

import numpy as np
from sklearn.linear_model import LinearRegression

from compsched.data import LABEL_COLS, encode_features

DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]


class ActivityDurationModel:
    """Log-normal OLS regression per discretionary activity type."""

    def fit(
        self,
        slot_records: list[dict],
        feature_names: list[str],
    ) -> "ActivityDurationModel":
        self.models_: dict[str, LinearRegression] = {}
        self.residual_stds_: dict[str, float] = {}
        self.feature_names_ = feature_names

        X_base, _ = encode_features(slot_records, LABEL_COLS, feature_names=feature_names)
        remaining = np.array(
            [float(r["remaining_budget"]) for r in slot_records], dtype=np.float64
        )
        X_all = np.hstack([X_base, remaining.reshape(-1, 1)])
        y_all = np.array([float(r["duration"]) for r in slot_records], dtype=np.float64)
        atypes = [r["atype"] for r in slot_records]

        for atype in DISC_TYPES:
            mask = np.array([a == atype for a in atypes])
            if mask.sum() < 5:
                continue
            Xs = X_all[mask]
            ys = np.log(np.clip(y_all[mask], 1.0, None))
            model = LinearRegression().fit(Xs, ys)
            residuals = ys - model.predict(Xs)
            self.models_[atype] = model
            self.residual_stds_[atype] = float(np.std(residuals))

        return self

    def sample(
        self,
        atype: str,
        x_label: np.ndarray,
        remaining_budget: float,
    ) -> float:
        """Sample duration for one activity. x_label: pre-encoded label row (1D)."""
        if atype not in self.models_:
            return float(np.clip(np.random.exponential(60.0), 10.0, remaining_budget))
        model = self.models_[atype]
        std = self.residual_stds_[atype]
        x = np.hstack([x_label, [remaining_budget]]).reshape(1, -1)
        log_pred = model.predict(x)[0]
        dur = np.exp(np.random.normal(log_pred, std))
        return float(np.clip(dur, 10.0, max(10.0, remaining_budget)))
