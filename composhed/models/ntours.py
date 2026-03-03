"""Step 3 — Number of non-mandatory tours (ordered logit)."""

import warnings

import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel


class NToursModel:
    """Ordered logit for count of discretionary activities (0–4)."""

    MAX_K = 4

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NToursModel":
        y_clipped = np.clip(y, 0, self.MAX_K).astype(int)
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
        probs = self.result_.predict(X_clean)  # (n, MAX_K+1)
        results = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            p = probs[i].copy()
            # Replace NaN/inf with 0
            p = np.where(np.isfinite(p), p, 0.0)
            p = np.clip(p, 0.0, None)
            cap = int(max_allowed[i])
            if cap < self.MAX_K:
                p[cap + 1 :] = 0.0
            total = p.sum()
            if total <= 0:
                p[0] = 1.0
            else:
                p /= total
            results[i] = np.random.choice(len(p), p=p)
        return results
