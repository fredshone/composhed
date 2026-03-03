"""Step 1 — Daily Activity Pattern classifier (Multinomial Logit)."""

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit


class DAPModel:
    """MNLogit classifier for DAP type: D / H / W / WD."""

    CLASSES = ["D", "H", "W", "WD"]  # indices 0–3

    def fit(self, X: np.ndarray, y_labels: list[str]) -> "DAPModel":
        class_map = {c: i for i, c in enumerate(self.CLASSES)}
        y = np.array([class_map[l] for l in y_labels], dtype=int)
        X_const = sm.add_constant(X, has_constant="add")
        self.result_ = MNLogit(y, X_const).fit(disp=False, maxiter=200)
        self.n_features_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_const = sm.add_constant(X, has_constant="add")
        probs = self.result_.predict(X_const)  # (n, 4)
        # Ensure sums to 1 (floating point safety)
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def sample(self, X: np.ndarray) -> list[str]:
        probs = self.predict_proba(X)
        return [
            self.CLASSES[np.random.choice(len(self.CLASSES), p=probs[i])]
            for i in range(len(probs))
        ]
