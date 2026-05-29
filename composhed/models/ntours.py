"""Step 3 — Number of discretionary tours (ordered logit)."""

import warnings

import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel


class NToursModel:
    """Ordered logit for count of discretionary activities (0–4)."""

    MAX_K = 4

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NToursModel":
        y_clipped = np.clip(y, 0, self.MAX_K).astype(int)
        # Track the actual category values so sample() can map indices → values
        self.categories_ = np.unique(y_clipped)
        # OrderedModel rejects constant/near-constant columns — drop them
        col_var = X.var(axis=0)
        self.keep_cols_ = col_var > 1e-10
        X_clean = X[:, self.keep_cols_]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = OrderedModel(y_clipped, X_clean, distr="logit").fit(
                method="bfgs", disp=False
            )
        return self

    def sample(self, X: np.ndarray, max_allowed: np.ndarray) -> np.ndarray:
        """Sample n_disc for each row; cap at max_allowed[i]."""
        X_clean = X[:, self.keep_cols_]
        probs = self.result_.predict(X_clean)  # (n, n_categories)
        results = np.zeros(len(probs), dtype=int)
        cats = self.categories_  # actual integer values, e.g. [1, 2, 3, 4]
        for i in range(len(probs)):
            p = probs[i].copy()
            p = np.where(np.isfinite(p), p, 0.0)
            p = np.clip(p, 0.0, None)
            # Zero out categories above the budget cap
            cap = int(max_allowed[i])
            p[cats > cap] = 0.0
            total = p.sum()
            if total <= 0:
                # Fallback: lowest valid category
                results[i] = cats[0]
                continue
            p /= total
            results[i] = cats[np.random.choice(len(cats), p=p)]
        return results

    def validate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """In-sample fit: accuracy, MAE, per-count accuracy."""
        y_clipped = np.clip(y, 0, self.MAX_K).astype(int)
        X_clean = X[:, self.keep_cols_]
        probs = self.result_.predict(X_clean)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        s = probs.sum(axis=1, keepdims=True)
        probs = np.where(s > 0, probs / s, 1.0 / probs.shape[1])
        cats = self.categories_
        pred = cats[probs.argmax(axis=1)]  # map index → actual category value
        acc = float((pred == y_clipped).mean())
        mae = float(np.mean(np.abs(pred.astype(float) - y_clipped.astype(float))))
        per_count = {}
        for k in cats:
            mask = y_clipped == k
            if mask.sum() > 0:
                per_count[int(k)] = {
                    "acc": float((pred[mask] == k).mean()),
                    "n": int(mask.sum()),
                }
        return {"n": len(y), "accuracy": acc, "mae": mae, "per_count": per_count}
