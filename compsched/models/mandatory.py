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
