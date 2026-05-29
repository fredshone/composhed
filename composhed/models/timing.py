"""Start time model for enumerated MDCEV activity scheduling."""

import numpy as np
from sklearn.linear_model import LinearRegression

from composhed.data import LABEL_COLS, encode_features


class StartTimeModel:
    """Predict observed start times for enumerated activity occurrences.

    Fits a separate LinearRegression per enumerated activity key (e.g. work_0,
    home_1, shop_0) predicting start time in minutes from person features and
    number of tours in the day. Used during assembly to order activities within
    and across tour slots.

    home_0 is excluded (trivially starts at 0). The last home in a schedule
    (home_{n-1}) is also excluded — its start is determined by what precedes it,
    not independently predictable.

    At sample time, predicted start times are approximate; they serve as ordering
    signals, not hard constraints.
    """

    def fit(
        self,
        records: list[dict],
        feature_names: list[str],
        X: np.ndarray | None = None,
    ) -> "StartTimeModel":
        self.feature_names_ = feature_names

        if X is None:
            X, _ = encode_features(records, LABEL_COLS, feature_names=feature_names)

        # Collect (x_extended, start_time) per enumerated activity key.
        # x_extended = [person_features | n_tours]
        data_per_act: dict[str, tuple[list, list]] = {}

        for i, rec in enumerate(records):
            n_tours = rec["n_tours"]
            x_ext = np.append(X[i], float(n_tours))
            n_home = sum(1 for k in rec["enumerated_durations"] if k.startswith("home_"))

            for act_enum, start in rec["enumerated_starts"].items():
                if act_enum == "home_0":
                    continue
                if act_enum not in data_per_act:
                    data_per_act[act_enum] = ([], [])
                data_per_act[act_enum][0].append(x_ext)
                data_per_act[act_enum][1].append(float(start))

        self.models_: dict[str, tuple] = {}
        self.fallback_means_: dict[str, float] = {}

        for act_enum, (Xrows, y) in data_per_act.items():
            yarr = np.array(y)
            self.fallback_means_[act_enum] = float(yarr.mean())
            if len(Xrows) < 5:
                continue
            Xarr = np.array(Xrows)
            model = LinearRegression().fit(Xarr, yarr)
            residuals = yarr - model.predict(Xarr)
            self.models_[act_enum] = (model, float(np.std(residuals)))

        return self

    def sample(self, act_enum: str, x_person: np.ndarray, n_tours: int) -> float:
        """Sample a start time for one enumerated activity, in minutes [0, 1439]."""
        if act_enum == "home_0":
            return 0.0

        if act_enum in self.models_:
            model, residual_std = self.models_[act_enum]
            x_ext = np.append(x_person, float(n_tours)).reshape(1, -1)
            pred = float(model.predict(x_ext)[0])
            if residual_std > 0:
                pred += float(np.random.normal(0.0, residual_std))
            return float(np.clip(pred, 0.0, 1439.0))

        if act_enum in self.fallback_means_:
            return float(np.clip(self.fallback_means_[act_enum], 0.0, 1439.0))

        return float(np.random.uniform(0.0, 1440.0))
