"""Step 4 — Non-mandatory activity type per slot (MNLogit per slot)."""

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

from compsched.data import LABEL_COLS, encode_features

DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]  # sorted → stable indices


class ActivityTypeModel:
    """Three MNLogit models keyed by slot: '1', '2', '3+'."""

    SLOTS = ["1", "2", "3+"]

    def fit(
        self,
        slot_records: list[dict],
        feature_names: list[str],
    ) -> "ActivityTypeModel":
        """Fit one MNLogit per slot.

        slot_records: list of dicts from build_training_dataset (raw label values).
        feature_names: label feature names from encode_features (for alignment).
        """
        self.models_: dict[str, object] = {}
        self.feature_names_ = feature_names

        # Encode label features for all slot records
        X_base, _ = encode_features(slot_records, LABEL_COLS, feature_names=feature_names)
        extra = np.column_stack(
            [
                [float(r["dap_WD"]) for r in slot_records],
                [float(r["slot_numeric"]) for r in slot_records],
                [float(r["remaining_budget"]) for r in slot_records],
            ]
        )
        X_all = np.hstack([X_base, extra])
        y_all = np.array([DISC_TYPES.index(r["atype"]) for r in slot_records], dtype=int)
        slot_keys = [r["slot_key"] for r in slot_records]

        for slot in self.SLOTS:
            mask = np.array([k == slot for k in slot_keys])
            if mask.sum() < 10:
                self.models_[slot] = None
                continue
            Xs, ys = X_all[mask], y_all[mask]
            if len(np.unique(ys)) < 2:
                self.models_[slot] = None
                continue
            Xs_const = sm.add_constant(Xs, has_constant="add")
            try:
                self.models_[slot] = MNLogit(ys, Xs_const).fit(disp=False, maxiter=200)
            except Exception:
                self.models_[slot] = None

        return self

    def sample_slot(
        self,
        x_label: np.ndarray,
        slot: str,
        dap_WD: int,
        slot_numeric: int,
        remaining_budget: float,
    ) -> str:
        """Sample one activity type. x_label: pre-encoded label row (1D)."""
        model = self.models_.get(slot) or self.models_.get("3+")
        if model is None:
            return np.random.choice(DISC_TYPES)

        extra = np.array([float(dap_WD), float(slot_numeric), float(remaining_budget)])
        x = np.hstack([x_label, extra]).reshape(1, -1)
        x_const = sm.add_constant(x, has_constant="add")
        probs = model.predict(x_const)[0]
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            return np.random.choice(DISC_TYPES)
        probs /= total
        # MNLogit probs may be fewer than len(DISC_TYPES) if some classes absent
        if len(probs) < len(DISC_TYPES):
            # Pad with zeros (shouldn't happen with full training data)
            probs = np.pad(probs, (0, len(DISC_TYPES) - len(probs)))
            probs /= probs.sum()
        return DISC_TYPES[np.random.choice(len(DISC_TYPES), p=probs)]
