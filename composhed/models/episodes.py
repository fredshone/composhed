"""Episode count model — Poisson regression for n episodes per activity type."""

from collections import defaultdict

import numpy as np


class EpisodeCountModel:
    """Per-type Poisson regression: n_episodes ~ Poisson(exp(α + β·log(duration)))."""

    DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]

    def fit(self, records: list[dict]) -> "EpisodeCountModel":
        type_obs: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for r in records:
            counts: dict[str, int] = defaultdict(int)
            durs: dict[str, float] = defaultdict(float)
            for atype, dur in r["disc_activities"]:
                if atype in self.DISC_TYPES:
                    counts[atype] += 1
                    durs[atype] += float(dur)
            for atype in self.DISC_TYPES:
                if counts[atype] > 0:
                    type_obs[atype].append((durs[atype], counts[atype]))

        self.params_: dict[str, tuple[float, float]] = {}
        for atype in self.DISC_TYPES:
            obs = type_obs.get(atype, [])
            if len(obs) < 3:
                self.params_[atype] = (0.0, 0.0)
                continue
            durations = np.array([d for d, _ in obs], dtype=np.float64)
            counts_arr = np.array([n for _, n in obs], dtype=np.float64)
            log_d = np.log(np.clip(durations, 1.0, None))
            log_n = np.log(np.clip(counts_arr, 1.0, None))
            var_d = float(np.var(log_d))
            beta = float(np.cov(log_d, log_n)[0, 1] / var_d) if var_d > 0 else 0.0
            alpha = float(np.mean(log_n) - beta * np.mean(log_d))
            self.params_[atype] = (alpha, beta)

        return self

    def sample(self, atype: str, total_duration: float) -> int:
        """Sample n episodes; always ≥ 1; capped so each episode ≥ 10 min."""
        if atype not in self.params_ or total_duration <= 0:
            return 1
        alpha, beta = self.params_[atype]
        lam = np.exp(alpha + beta * np.log(max(1.0, total_duration)))
        n = int(np.random.poisson(max(1e-6, lam)))
        n = max(1, n)
        n = min(n, max(1, int(total_duration) // 10))
        return n
