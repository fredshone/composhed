"""Step 6 timing — KDE anchor models, Beta home split, logistic before/after work."""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression

from composhed.data import LABEL_COLS, encode_features

DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]

# Ordered placement classes for tour-placement model (index = class integer)
PLACEMENT_CLASSES = [
    "mandatory_outbound",
    "mandatory_inbound",
    "separate_before",
    "separate_after",
]
PLACEMENT_IDX = {name: i for i, name in enumerate(PLACEMENT_CLASSES)}


class AnchorTimingModel:
    """Fit and sample timing-related distributions from training data."""

    def fit(self, records: list[dict], feature_names: list[str], scaler) -> "AnchorTimingModel":
        self.feature_names_ = feature_names
        self.scaler_ = scaler
        self._fit_work_start_kdes(records)
        self._fit_first_dep_kdes(records)
        self._fit_home_split(records)
        self._fit_tour_placement(records, feature_names, scaler)
        self._fit_tour_grouping(records, feature_names, scaler)
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
            if r["dap"] in ("W", "WD", "E", "ED") and r["work_start"] is not None:
                ws = float(r["work_start"])
                by_status[str(r["employment"])].append(ws)
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
                by_status[str(r["employment"])].append(fd)
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

    def _fit_tour_placement(self, records: list[dict], feature_names: list[str], scaler) -> None:
        """4-class MNL: placement of each disc activity relative to mandatory tour.

        Classes: mandatory_outbound, mandatory_inbound, separate_before, separate_after.
        Features: label one-hots + work_start/1440 + atype dummies.
        """
        rows_for_fit: list[dict] = []
        y_vals: list[int] = []

        for r in records:
            if r["dap"] not in ("WD", "ED") or not r["disc_activities"]:
                continue
            ws = r["work_start"]
            placements = r.get("tour_placement", [])
            for (atype, _), placement in zip(r["disc_activities"], placements):
                if placement not in PLACEMENT_IDX:
                    continue
                row = {k: r[k] for k in LABEL_COLS}
                row["_work_start"] = float(ws) if ws is not None else 0.0
                row["_atype"] = atype
                rows_for_fit.append(row)
                y_vals.append(PLACEMENT_IDX[placement])

        if len(y_vals) < 10 or len(set(y_vals)) < 2:
            self.tour_placement_model_ = None
            return

        X_label, _ = encode_features(rows_for_fit, LABEL_COLS, feature_names=feature_names)
        X_label = scaler.transform(X_label)
        work_starts = np.array(
            [r["_work_start"] for r in rows_for_fit], dtype=np.float64
        ).reshape(-1, 1) / 1440.0
        atype_dummies = np.zeros((len(rows_for_fit), len(DISC_TYPES)), dtype=np.float64)
        for i, r in enumerate(rows_for_fit):
            if r["_atype"] in DISC_TYPES:
                atype_dummies[i, DISC_TYPES.index(r["_atype"])] = 1.0
        X = np.hstack([X_label, work_starts, atype_dummies])
        y = np.array(y_vals, dtype=int)

        self.tour_placement_model_ = LogisticRegression(max_iter=2000).fit(X, y)

    def _fit_tour_grouping(self, records: list[dict], feature_names: list[str], scaler) -> None:
        """Binary logistic: P(same tour as next activity | labels, slot, atype pair, budget).

        Trained on D-schedule records with ≥2 disc activities.
        """
        rows_for_fit: list[dict] = []
        y_vals: list[int] = []

        for r in records:
            if r["dap"] != "D" or r["n_disc"] < 2:
                continue
            disc = r["disc_activities"]
            same_tour = r.get("same_tour_as_next", [])
            remaining = float(sum(d for _, d in disc))

            for i in range(len(disc) - 1):
                atype_curr, dur_curr = disc[i]
                atype_next, _ = disc[i + 1]
                remaining -= dur_curr
                row = {k: r[k] for k in LABEL_COLS}
                row["_slot"] = float(i + 1)
                row["_atype_curr"] = atype_curr
                row["_atype_next"] = atype_next
                row["_remaining"] = remaining / 1440.0
                rows_for_fit.append(row)
                y_vals.append(int(same_tour[i]) if i < len(same_tour) else 0)

        if len(y_vals) < 10 or len(set(y_vals)) < 2:
            self.tour_grouping_model_ = None
            return

        X_label, _ = encode_features(rows_for_fit, LABEL_COLS, feature_names=feature_names)
        X_label = scaler.transform(X_label)
        slot = np.array([r["_slot"] for r in rows_for_fit]).reshape(-1, 1) / 5.0
        curr_dummies = np.zeros((len(rows_for_fit), len(DISC_TYPES)), dtype=np.float64)
        next_dummies = np.zeros((len(rows_for_fit), len(DISC_TYPES)), dtype=np.float64)
        remaining = np.array([r["_remaining"] for r in rows_for_fit]).reshape(-1, 1)
        for i, r in enumerate(rows_for_fit):
            if r["_atype_curr"] in DISC_TYPES:
                curr_dummies[i, DISC_TYPES.index(r["_atype_curr"])] = 1.0
            if r["_atype_next"] in DISC_TYPES:
                next_dummies[i, DISC_TYPES.index(r["_atype_next"])] = 1.0
        X = np.hstack([X_label, slot, curr_dummies, next_dummies, remaining])
        y = np.array(y_vals, dtype=int)

        self.tour_grouping_model_ = LogisticRegression(max_iter=2000).fit(X, y)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_work_start(self, employment: str) -> float:
        kde = self.work_start_kdes_.get(employment, self._global_work_kde)
        if kde is None:
            return float(np.random.uniform(480, 600))
        val = float(kde.resample(1)[0, 0])
        return float(np.clip(val, 0.0, 1380.0))

    def sample_first_departure(self, employment: str) -> float:
        kde = self.first_dep_kdes_.get(employment, self._global_dep_kde)
        if kde is None:
            return float(np.random.uniform(480, 720))
        val = float(kde.resample(1)[0, 0])
        return float(np.clip(val, 0.0, 1380.0))

    def sample_home_ratio(self) -> float:
        from scipy.stats import beta

        return float(beta.rvs(self.beta_a_, self.beta_b_))

    def sample_tour_placements(
        self,
        x_label: np.ndarray,
        disc_activities: list[tuple[str, int]],
        work_start: float,
    ) -> list[str]:
        """Sample tour placement label for each WD/ED disc activity.

        Returns list of strings from PLACEMENT_CLASSES, one per activity.
        Falls back to "mandatory_inbound" if model unavailable.
        """
        if not disc_activities:
            return []
        if self.tour_placement_model_ is None:
            return ["mandatory_inbound"] * len(disc_activities)

        x_label_scaled = self.scaler_.transform(x_label.reshape(1, -1))[0]
        ws_norm = work_start / 1440.0
        results = []
        for atype, _ in disc_activities:
            atype_row = np.zeros(len(DISC_TYPES), dtype=np.float64)
            if atype in DISC_TYPES:
                atype_row[DISC_TYPES.index(atype)] = 1.0
            x = np.hstack([
                x_label_scaled,
                [ws_norm],
                atype_row,
            ]).reshape(1, -1)
            # predict_proba returns probs for model.classes_ (subset of PLACEMENT_IDX)
            probs_subset = self.tour_placement_model_.predict_proba(x)[0]
            # Map to full PLACEMENT_CLASSES distribution
            full_probs = np.zeros(len(PLACEMENT_CLASSES), dtype=np.float64)
            for j, cls_int in enumerate(self.tour_placement_model_.classes_):
                full_probs[cls_int] += probs_subset[j]
            full_probs /= full_probs.sum()
            idx = int(np.random.choice(len(PLACEMENT_CLASSES), p=full_probs))
            results.append(PLACEMENT_CLASSES[idx])
        return results

    def validate(self, records: list[dict], feature_names: list[str], scaler) -> dict:
        """In-sample accuracy for tour placement (4-class) and tour grouping (binary)."""
        result: dict = {}

        # --- Tour placement ---
        if getattr(self, "tour_placement_model_", None) is not None:
            rows_fp: list[dict] = []
            y_fp: list[int] = []
            for r in records:
                if r["dap"] not in ("WD", "ED") or not r["disc_activities"]:
                    continue
                ws = r["work_start"]
                for (atype, _), placement in zip(
                    r["disc_activities"], r.get("tour_placement", [])
                ):
                    if placement not in PLACEMENT_IDX:
                        continue
                    row = {k: r[k] for k in LABEL_COLS}
                    row["_work_start"] = float(ws) if ws is not None else 0.0
                    row["_atype"] = atype
                    rows_fp.append(row)
                    y_fp.append(PLACEMENT_IDX[placement])
            if rows_fp:
                X_label, _ = encode_features(rows_fp, LABEL_COLS, feature_names=feature_names)
                X_label = scaler.transform(X_label)
                work_starts = (
                    np.array([r["_work_start"] for r in rows_fp]).reshape(-1, 1) / 1440.0
                )
                atype_dummies = np.zeros((len(rows_fp), len(DISC_TYPES)), dtype=np.float64)
                for i, r in enumerate(rows_fp):
                    if r["_atype"] in DISC_TYPES:
                        atype_dummies[i, DISC_TYPES.index(r["_atype"])] = 1.0
                X = np.hstack([X_label, work_starts, atype_dummies])
                y = np.array(y_fp, dtype=int)
                pred = self.tour_placement_model_.predict(X)
                per_class = {}
                for i, cls in enumerate(PLACEMENT_CLASSES):
                    mask = y == i
                    if mask.sum() > 0:
                        per_class[cls] = {
                            "acc": float((pred[mask] == i).mean()),
                            "n": int(mask.sum()),
                        }
                result["tour_placement"] = {
                    "n": len(y),
                    "accuracy": float((pred == y).mean()),
                    "per_class": per_class,
                }

        # --- Tour grouping ---
        if getattr(self, "tour_grouping_model_", None) is not None:
            rows_tg: list[dict] = []
            y_tg: list[int] = []
            for r in records:
                if r["dap"] != "D" or r["n_disc"] < 2:
                    continue
                disc = r["disc_activities"]
                same_tour = r.get("same_tour_as_next", [])
                remaining = float(sum(d for _, d in disc))
                for i in range(len(disc) - 1):
                    atype_curr, dur_curr = disc[i]
                    atype_next, _ = disc[i + 1]
                    remaining -= dur_curr
                    row = {k: r[k] for k in LABEL_COLS}
                    row["_slot"] = float(i + 1)
                    row["_atype_curr"] = atype_curr
                    row["_atype_next"] = atype_next
                    row["_remaining"] = remaining / 1440.0
                    rows_tg.append(row)
                    y_tg.append(int(same_tour[i]) if i < len(same_tour) else 0)
            if rows_tg:
                X_label, _ = encode_features(rows_tg, LABEL_COLS, feature_names=feature_names)
                X_label = scaler.transform(X_label)
                slot = np.array([r["_slot"] for r in rows_tg]).reshape(-1, 1) / 5.0
                curr_d = np.zeros((len(rows_tg), len(DISC_TYPES)), dtype=np.float64)
                next_d = np.zeros((len(rows_tg), len(DISC_TYPES)), dtype=np.float64)
                rem = np.array([r["_remaining"] for r in rows_tg]).reshape(-1, 1)
                for i, r in enumerate(rows_tg):
                    if r["_atype_curr"] in DISC_TYPES:
                        curr_d[i, DISC_TYPES.index(r["_atype_curr"])] = 1.0
                    if r["_atype_next"] in DISC_TYPES:
                        next_d[i, DISC_TYPES.index(r["_atype_next"])] = 1.0
                X = np.hstack([X_label, slot, curr_d, next_d, rem])
                y = np.array(y_tg, dtype=int)
                pred = self.tour_grouping_model_.predict(X)
                result["tour_grouping"] = {
                    "n": len(y),
                    "accuracy": float((pred == y).mean()),
                }

        return result

    def sample_tour_groupings(
        self,
        x_label: np.ndarray,
        disc_activities: list[tuple[str, int]],
    ) -> list[bool]:
        """Sample same-tour flag for each consecutive disc pair in D schedules.

        Returns list of bools of length len(disc_activities) - 1.
        True = next activity is on the same tour; False = new separate home-based tour.
        Falls back to False (all separate tours) if model unavailable.
        """
        n = len(disc_activities)
        if n < 2:
            return []
        if self.tour_grouping_model_ is None:
            return [False] * (n - 1)

        x_label_scaled = self.scaler_.transform(x_label.reshape(1, -1))[0]
        flags = []
        remaining = float(sum(d for _, d in disc_activities))
        for i in range(n - 1):
            atype_curr, dur_curr = disc_activities[i]
            atype_next, _ = disc_activities[i + 1]
            remaining -= dur_curr
            curr_row = np.zeros(len(DISC_TYPES), dtype=np.float64)
            next_row = np.zeros(len(DISC_TYPES), dtype=np.float64)
            if atype_curr in DISC_TYPES:
                curr_row[DISC_TYPES.index(atype_curr)] = 1.0
            if atype_next in DISC_TYPES:
                next_row[DISC_TYPES.index(atype_next)] = 1.0
            x = np.hstack([
                x_label_scaled,
                [(i + 1) / 5.0],
                curr_row,
                next_row,
                [remaining / 1440.0],
            ]).reshape(1, -1)
            p_same = self.tour_grouping_model_.predict_proba(x)[0, 1]
            flags.append(bool(np.random.random() < p_same))
        return flags
