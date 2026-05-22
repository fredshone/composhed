"""Step 1 — Daily Activity Pattern classifier (Multinomial Logit)."""

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit


class DAPModel:
    """MNLogit classifier for DAP type: D / E / ED / H / W / WD."""

    CLASSES = ["D", "E", "ED", "H", "W", "WD"]

    def fit(self, X: np.ndarray, y_labels: list[str]) -> "DAPModel":
        # Only use classes that actually appear in the training data
        self._seen_classes_ = sorted(set(y_labels), key=self.CLASSES.index)
        class_map = {c: i for i, c in enumerate(self._seen_classes_)}
        y = np.array([class_map[l] for l in y_labels], dtype=int)
        X_const = sm.add_constant(X, has_constant="add")
        self.result_ = MNLogit(y, X_const).fit(disp=False, maxiter=200)
        self.n_features_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_const = sm.add_constant(X, has_constant="add")
        raw = self.result_.predict(X_const)  # (n, len(_seen_classes_))
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = np.clip(raw, 0, None)
        s = raw.sum(axis=1, keepdims=True)
        raw = np.where(s > 0, raw / s, 1.0 / raw.shape[1])

        # Expand to full CLASSES shape, placing 0 for absent classes
        n = X.shape[0]
        probs = np.zeros((n, len(self.CLASSES)), dtype=np.float64)
        for j, cls in enumerate(self._seen_classes_):
            probs[:, self.CLASSES.index(cls)] = raw[:, j]
        # Renormalise (absent classes stay 0; present classes already sum to 1)
        return probs

    def sample(self, X: np.ndarray) -> list[str]:
        probs = self.predict_proba(X)
        return [
            self.CLASSES[np.random.choice(len(self.CLASSES), p=probs[i])]
            for i in range(len(probs))
        ]
