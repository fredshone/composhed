"""Step 2 — Mandatory activity duration (log-normal OLS)."""

import numpy as np
from sklearn.linear_model import LinearRegression


class MandatoryDurationModel:
    """Log-normal regression for work/education duration."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MandatoryDurationModel":
        log_y = np.log(np.clip(y, 1.0, None))
        self.model_ = LinearRegression().fit(X, log_y)
        residuals = log_y - self.model_.predict(X)
        self.residual_std_ = float(np.std(residuals))
        return self

    def sample(self, X: np.ndarray) -> np.ndarray:
        log_pred = self.model_.predict(X)
        noise = np.random.normal(0.0, self.residual_std_, size=len(log_pred))
        return np.clip(np.exp(log_pred + noise), 30.0, 600.0)

    def validate(self, X: np.ndarray, y: np.ndarray, is_edu: np.ndarray | None = None) -> dict:
        """In-sample fit: log-R², MAE in minutes; split by work vs education."""
        log_y = np.log(np.clip(y, 1.0, None))
        log_pred = self.model_.predict(X)
        ss_res = float(np.sum((log_y - log_pred) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(np.exp(log_pred) - y)))
        result: dict = {"n": len(y), "log_r2": r2, "mae_minutes": mae}
        if is_edu is not None:
            for label, mask in [("work", is_edu == 0), ("edu", is_edu == 1)]:
                mask = mask.astype(bool)
                if mask.sum() > 0:
                    lp, ly, yv = log_pred[mask], log_y[mask], y[mask]
                    ss_r = float(np.sum((ly - lp) ** 2))
                    ss_t = float(np.sum((ly - ly.mean()) ** 2))
                    result[label] = {
                        "n": int(mask.sum()),
                        "log_r2": 1.0 - ss_r / ss_t if ss_t > 0 else float("nan"),
                        "mae_minutes": float(np.mean(np.abs(np.exp(lp) - yv))),
                    }
        return result
