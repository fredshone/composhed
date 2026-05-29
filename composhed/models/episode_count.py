"""Per-type episode count model — ordered logit on log(total_duration)."""

import warnings

import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel


class EpisodeCountModel:
    """For each activity type, predict number of episodes given total allocated minutes.

    Fits a separate ordered logit per type: n_episodes ~ f(log(total_dur)).
    Interface matches the episode_model.sample(atype, total) used in assembly._split_duration.
    """

    MAX_EPISODES = 6
    MIN_OBS = 30

    def fit(self, records: list[dict]) -> "EpisodeCountModel":
        """Fit per-type ordered logit from derive_type_records output."""
        from collections import defaultdict

        data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

        for r in records:
            td = r["total_durations"]
            ec = r["episode_counts"]
            for atype, total in td.items():
                if total <= 0:
                    continue
                n = min(int(ec.get(atype, 1)), self.MAX_EPISODES)
                n = max(n, 1)
                data[atype][0].append(np.log(max(1.0, float(total))))
                data[atype][1].append(n)

        self.models_: dict[str, object] = {}
        self.categories_: dict[str, np.ndarray] = {}
        self.fallback_: dict[str, int] = {}

        for atype, (X_log, y_raw) in data.items():
            y = np.array(y_raw, dtype=int)
            cats = np.unique(y)
            self.categories_[atype] = cats
            # Modal category as fallback
            counts = np.bincount(y - y.min())
            self.fallback_[atype] = int(cats[int(np.argmax(counts))])

            if len(cats) < 2 or len(y) < self.MIN_OBS:
                continue

            X = np.array(X_log).reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = OrderedModel(y, X, distr="logit").fit(
                        method="bfgs", disp=False
                    )
                    self.models_[atype] = result
                except Exception:
                    pass

        return self

    def sample(self, atype: str, total_dur: float) -> int:
        """Sample number of episodes given allocated minutes. Returns 0 if total_dur <= 0."""
        if total_dur <= 0:
            return 0
        fallback = self.fallback_.get(atype, 1)
        if atype not in self.models_:
            return max(1, fallback)
        X = np.array([[np.log(max(1.0, float(total_dur)))]])
        try:
            probs = self.models_[atype].predict(X)[0]
            probs = np.where(np.isfinite(probs), probs, 0.0)
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s <= 0:
                return max(1, fallback)
            probs /= s
            cats = self.categories_[atype]
            return int(cats[np.random.choice(len(cats), p=probs)])
        except Exception:
            return max(1, fallback)

    def validate(self, records: list[dict]) -> dict:
        """In-sample MAE and mean episode counts per type."""
        from collections import defaultdict

        data: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r in records:
            td = r["total_durations"]
            ec = r["episode_counts"]
            for atype, total in td.items():
                if total <= 0:
                    continue
                n = min(int(ec.get(atype, 1)), self.MAX_EPISODES)
                data[atype][0].append(float(total))
                data[atype][1].append(max(n, 1))

        result: dict = {}
        for atype, (totals, y_true) in data.items():
            y_pred = np.array([self.sample(atype, t) for t in totals])
            y_true_arr = np.array(y_true)
            result[atype] = {
                "n": len(y_true),
                "mae": float(np.mean(np.abs(y_pred - y_true_arr))),
                "mean_actual": float(np.mean(y_true_arr)),
                "mean_pred": float(np.mean(y_pred)),
            }
        return result
