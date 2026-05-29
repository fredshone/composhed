"""Step 5 — Discretionary activity duration per type (log-normal OLS)."""

import numpy as np
from sklearn.linear_model import LinearRegression

from composhed.data import LABEL_COLS, encode_features

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
            [float(r["remaining_budget"]) / 1440.0 for r in slot_records], dtype=np.float64
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
        x = np.hstack([x_label, [remaining_budget / 1440.0]]).reshape(1, -1)
        log_pred = model.predict(x)[0]
        dur = np.exp(np.random.normal(log_pred, std))
        return float(np.clip(dur, 10.0, max(10.0, remaining_budget)))

    def validate(self, slot_records: list[dict], feature_names: list[str]) -> dict:
        """In-sample log-R² and MAE in minutes per activity type."""
        X_base, _ = encode_features(slot_records, LABEL_COLS, feature_names=feature_names)
        remaining = np.array(
            [float(r["remaining_budget"]) / 1440.0 for r in slot_records], dtype=np.float64
        )
        X_all = np.hstack([X_base, remaining.reshape(-1, 1)])
        y_all = np.array([float(r["duration"]) for r in slot_records], dtype=np.float64)
        atypes = [r["atype"] for r in slot_records]

        result: dict = {}
        for atype in DISC_TYPES:
            if atype not in self.models_:
                continue
            mask = np.array([a == atype for a in atypes])
            if not mask.any():
                continue
            ys = y_all[mask]
            log_ys = np.log(np.clip(ys, 1.0, None))
            log_pred = self.models_[atype].predict(X_all[mask])
            ss_res = float(np.sum((log_ys - log_pred) ** 2))
            ss_tot = float(np.sum((log_ys - log_ys.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(np.exp(log_pred) - ys)))
            result[atype] = {"n": int(mask.sum()), "log_r2": r2, "mae_minutes": mae}
        return result
