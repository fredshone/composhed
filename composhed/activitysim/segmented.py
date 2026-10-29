"""Segment-specific MNLs (ActivitySim coefficient files per person type / tour purpose).

Each segment with at least ``n_min`` observations gets its own model; smaller
segments share a pooled model fitted on all observations (pooling is logged).
``segmented=False`` (AB-5) uses the pooled model for everyone. ``factory`` builds
each segment's model (default: an alternative-specific ``MNL``; ActivitySim's
non-mandatory tour frequency and stop frequency use ``AltAttrMNL``).
"""

import numpy as np

from composhed.common.choice import MNL

POOLED = "__pooled__"


class SegmentedMNL:
    def __init__(self, n_min: int = 200, segmented: bool = True, l2: float = 1e-4, factory=None) -> None:
        self.n_min = n_min
        self.segmented = segmented
        self.l2 = l2
        self.factory = factory

    def _new(self):
        return self.factory() if self.factory is not None else MNL(self.l2)

    def fit(self, X: np.ndarray, y: list, segments: list, classes: list, avail: np.ndarray | None = None,
            feature_names: list[str] | None = None) -> "SegmentedMNL":
        segments = np.asarray(segments, dtype=object)
        y = list(y)
        self.classes_ = list(classes)
        self.models: dict = {POOLED: self._new().fit(X, y, classes, avail, feature_names)}
        self.pooled_segments: list = []
        for seg in sorted(set(segments.tolist()), key=str):
            sel = segments == seg
            if self.segmented and sel.sum() >= self.n_min:
                self.models[seg] = self._new().fit(
                    X[sel], [v for v, s in zip(y, sel) if s], classes,
                    None if avail is None else avail[sel], feature_names)
            else:
                self.pooled_segments.append(seg)
        return self

    def key(self, segment) -> str:
        return segment if segment in self.models else POOLED

    def predict_proba(self, X: np.ndarray, segments: list, avail: np.ndarray | None = None) -> np.ndarray:
        X = np.atleast_2d(X)
        P = np.zeros((len(X), len(self.classes_)))
        keys = np.array([self.key(s) for s in segments], dtype=object)
        for k in set(keys.tolist()):
            sel = keys == k
            P[sel] = self.models[k].predict_proba(X[sel], None if avail is None else avail[sel])
        return P

    def sample(self, X: np.ndarray, segments: list, rng: np.random.Generator, avail: np.ndarray | None = None) -> list:
        keys = [self.key(s) for s in segments]
        if len(set(keys)) == 1:
            return self.models[keys[0]].sample(X, rng, avail)
        out = []
        for i, k in enumerate(keys):
            out += self.models[k].sample(X[i:i + 1], rng, None if avail is None else avail[i:i + 1])
        return out

    def validate(self, X: np.ndarray, y: list, segments: list, avail: np.ndarray | None = None) -> dict:
        keys = np.array([self.key(s) for s in segments], dtype=object)
        groups = {}
        for k in sorted(set(keys.tolist()), key=str):
            sel = keys == k
            groups[str(k)] = self.models[k].validate(X[sel], [v for v, s in zip(y, sel) if s],
                                                     None if avail is None else avail[sel])
        P = self.predict_proba(X, segments, avail)
        yi = np.array([self.classes_.index(v) for v in y])
        return {"n": len(y), "accuracy": float((P.argmax(axis=1) == yi).mean()) if len(y) else float("nan"),
                "pooled_segments": [str(s) for s in self.pooled_segments], "groups": groups}
