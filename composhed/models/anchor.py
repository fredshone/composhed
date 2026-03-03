"""Step 6 timing — KDE anchor models, Beta home split, logistic before/after work."""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression

from compsched.data import LABEL_COLS, encode_features

DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]


class AnchorTimingModel:
    """Fit and sample timing-related distributions from training data."""

    def fit(self, records: list[dict], feature_names: list[str]) -> "AnchorTimingModel":
        self.feature_names_ = feature_names
        self._fit_work_start_kdes(records)
        self._fit_first_dep_kdes(records)
        self._fit_home_split(records)
        self._fit_before_work(records, feature_names)
        return self

    # ------------------------------------------------------------------
    # KDEs for anchor timing
    # ------------------------------------------------------------------

    def _fit_work_start_kdes(self, records: list[dict]) -> None:
        """One KDE per work_status for W/WD schedules."""
        from collections import defaultdict

        by_status: dict[str, list[float]] = defaultdict(list)
        all_starts: list[float] = []
        for r in records:
            if r["dap"] in ("W", "WD") and r["work_start"] is not None:
                ws = float(r["work_start"])
                by_status[str(r["work_status"])].append(ws)
                all_starts.append(ws)

        self.work_start_kdes_: dict[str, gaussian_kde] = {}
        for status, vals in by_status.items():
            if len(vals) >= 3:
                self.work_start_kdes_[status] = gaussian_kde(vals)
        self._global_work_kde = gaussian_kde(all_starts) if all_starts else None

    def _fit_first_dep_kdes(self, records: list[dict]) -> None:
        """One KDE per work_status for D schedules."""
        from collections import defaultdict

        by_status: dict[str, list[float]] = defaultdict(list)
        all_deps: list[float] = []
        for r in records:
            if r["dap"] == "D" and r["first_departure"] is not None:
                fd = float(r["first_departure"])
                by_status[str(r["work_status"])].append(fd)
                all_deps.append(fd)

        self.first_dep_kdes_: dict[str, gaussian_kde] = {}
        for status, vals in by_status.items():
            if len(vals) >= 3:
                self.first_dep_kdes_[status] = gaussian_kde(vals)
        self._global_dep_kde = gaussian_kde(all_deps) if all_deps else None

    def _fit_home_split(self, records: list[dict]) -> None:
        """Fit Beta on home_ratio (home_morning / total_home)."""
        from scipy.stats import beta

        ratios = np.array(
            [r["home_ratio"] for r in records if r["total_home"] > 0],
            dtype=np.float64,
        )
        ratios = np.clip(ratios, 0.01, 0.99)
        if len(ratios) >= 10:
            a, b, _, _ = beta.fit(ratios, floc=0.0, fscale=1.0)
        else:
            a, b = 2.0, 2.0
        self.beta_a_ = float(a)
        self.beta_b_ = float(b)

    def _fit_before_work(self, records: list[dict], feature_names: list[str]) -> None:
        """Logistic regression: P(before_work | labels, work_start, atype)."""
        rows_for_fit: list[dict] = []
        y_vals: list[int] = []

        for r in records:
            if r["dap"] != "WD" or not r["disc_activities"]:
                continue
            ws = r["work_start"]
            flags = r["before_work_flags"]
            for (atype, _), flag in zip(r["disc_activities"], flags):
                row = {k: r[k] for k in LABEL_COLS}
                row["_work_start"] = float(ws) if ws is not None else 0.0
                row["_atype"] = atype
                rows_for_fit.append(row)
                y_vals.append(int(flag))

        if len(y_vals) < 10 or len(set(y_vals)) < 2:
            self.before_work_model_ = None
            return

        # Encode: label one-hots + work_start + atype dummies
        X_label, _ = encode_features(rows_for_fit, LABEL_COLS, feature_names=feature_names)
        work_starts = np.array(
            [r["_work_start"] for r in rows_for_fit], dtype=np.float64
        ).reshape(-1, 1)
        atype_dummies = np.zeros((len(rows_for_fit), len(DISC_TYPES)), dtype=np.float64)
        for i, r in enumerate(rows_for_fit):
            if r["_atype"] in DISC_TYPES:
                atype_dummies[i, DISC_TYPES.index(r["_atype"])] = 1.0
        X = np.hstack([X_label, work_starts, atype_dummies])
        y = np.array(y_vals, dtype=int)

        self.before_work_model_ = LogisticRegression(max_iter=500).fit(X, y)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_work_start(self, work_status: str) -> float:
        kde = self.work_start_kdes_.get(work_status, self._global_work_kde)
        if kde is None:
            return float(np.random.uniform(480, 600))
        val = float(kde.resample(1)[0, 0])
        return float(np.clip(val, 0.0, 1380.0))

    def sample_first_departure(self, work_status: str) -> float:
        kde = self.first_dep_kdes_.get(work_status, self._global_dep_kde)
        if kde is None:
            return float(np.random.uniform(480, 720))
        val = float(kde.resample(1)[0, 0])
        return float(np.clip(val, 0.0, 1380.0))

    def sample_home_ratio(self) -> float:
        from scipy.stats import beta

        return float(beta.rvs(self.beta_a_, self.beta_b_))

    def sample_before_work_flags(
        self,
        x_label: np.ndarray,
        disc_activities: list[tuple[str, int]],
        work_start: float,
    ) -> list[bool]:
        """Sample before-work flag for each disc activity."""
        if self.before_work_model_ is None or not disc_activities:
            return [False] * len(disc_activities)

        flags = []
        for atype, _ in disc_activities:
            atype_row = np.zeros(len(DISC_TYPES), dtype=np.float64)
            if atype in DISC_TYPES:
                atype_row[DISC_TYPES.index(atype)] = 1.0
            ws_arr = np.array([[work_start]], dtype=np.float64)
            x = np.hstack([x_label.reshape(1, -1), ws_arr, atype_row.reshape(1, -1)])
            p_before = self.before_work_model_.predict_proba(x)[0, 1]
            flags.append(bool(np.random.random() < p_before))
        return flags
