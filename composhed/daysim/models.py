"""DaySim-aligned component models (SPECIFICATION §6.2).

| Component | Spec | Model |
|---|---|---|
| day pattern (#work, #edu, #non-mandatory tours by purpose, stops-on-mandatory) | DAY-1 | MNL over enumerated patterns with support ≥ ``min_support`` |
| tour time of day (arrival × departure period pair) | DAY-2/4 | ``TourTOD`` per group (work, education, non-mandatory) with availability |
| stops per half-tour {0,1,2,3+} | DAY-5 | MNL per direction |
| stop purpose | DAY-6 | MNL per direction (stop index as covariate) |
| stop duration | DAY-7 | log-normal per purpose, truncated to the half-tour window |
| AB-2 timing | — | ``IndependentTiming`` per group |
"""

from collections import Counter

import numpy as np

from composhed.common.choice import MNL
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import PeriodGrid
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD
from composhed.daysim.data import NM_PURPOSES, TOUR_PURPOSES, DaysimRecord, pattern_label, tod_group
from composhed.data import DISC_ACTS

GROUPS = ["work", "education", "nonmandatory"]
DIRECTIONS = ["outbound", "inbound"]
STOP_PURPOSES = sorted(DISC_ACTS)
MAX_STOPS = 3
DAY = 1440


def _onehot(value, levels):
    return [1.0 if value == lv else 0.0 for lv in levels[1:]]


def z_tod(xl, purpose, rank, n_tours, nth_of_purpose):
    return np.concatenate([[1.0], xl, _onehot(purpose, NM_PURPOSES),
                           [rank / 5.0, n_tours / 6.0, float(nth_of_purpose > 0)]])


def x_nstops(xl, purpose, start, end, stops_flag):
    mand = purpose in ("work", "education")
    return np.concatenate([xl, _onehot(purpose, TOUR_PURPOSES),
                           [(end - start) / DAY, start / DAY, float(mand and stops_flag)]])


def x_stop_purpose(xl, tour_purpose, index, tour_duration):
    return np.concatenate([xl, _onehot(tour_purpose, TOUR_PURPOSES), _onehot(min(index, 2), [0, 1, 2]),
                           [tour_duration / DAY]])


def x_stop_duration(xl, direction, tour_purpose, tour_duration):
    return np.concatenate([xl, [float(direction == "inbound"), float(tour_purpose in ("work", "education")),
                                tour_duration / DAY]])


def tour_z_rows(xl, purposes: list[str]) -> list[np.ndarray]:
    seen: Counter = Counter()
    rows = []
    for rank, p in enumerate(purposes):
        rows.append(z_tod(xl, p, rank, len(purposes), seen[p]))
        seen[p] += 1
    return rows


class DaysimModels:
    def __init__(self, time_resolution: int = 30, min_support: int = 30, home_inter: int = 10,
                 tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.min_support = min_support
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    # ---- day pattern -------------------------------------------------------------
    def _pattern_alternatives(self, patterns: list[tuple]) -> None:
        counts = Counter(patterns)
        keep = [p for p, c in counts.most_common() if c >= self.min_support]
        if not keep:
            keep = [counts.most_common(1)[0][0]]
        self.pattern_vectors = {pattern_label(p): p for p in keep}
        self.pattern_counts = {pattern_label(p): counts[p] for p in keep}
        self.pattern_merged = sum(c for p, c in counts.items() if p not in set(keep))

    def map_pattern(self, vec: tuple) -> str:
        """Rare patterns map to the nearest supported pattern by L1 (ties: most frequent)."""
        lab = pattern_label(vec)
        if lab in self.pattern_vectors:
            return lab
        return min(self.pattern_vectors, key=lambda k: (
            sum(abs(a - b) for a, b in zip(vec, self.pattern_vectors[k])), -self.pattern_counts[k]))

    # ---- designs -------------------------------------------------------------------
    def tod_design(self, records: list[DaysimRecord], X: np.ndarray) -> dict:
        out = {g: {"Z": [], "starts": [], "ends": [], "masks": [], "X": []} for g in GROUPS}
        for r, xl in zip(records, X):
            tw = TimeWindow()
            zs = tour_z_rows(xl, [t.purpose for t in r.tours])
            for t, z in zip(r.tours, zs):
                g = tod_group(t.purpose)
                d = out[g]
                d["Z"].append(z)
                d["X"].append(xl)
                d["starts"].append(t.start)
                d["ends"].append(t.end)
                d["masks"].append(self.tod[g].mask(tw) if self.tod.get(g) is not None else None)
                tw.reserve(t.start, t.end)
        return out

    def stop_design(self, records: list[DaysimRecord], X: np.ndarray) -> dict:
        d = {"nstops": {k: ([], []) for k in DIRECTIONS}, "purpose": {k: ([], []) for k in DIRECTIONS},
             "duration": {p: ([], []) for p in STOP_PURPOSES}}
        for r, xl in zip(records, X):
            flag = r.pattern[-1]
            for t in r.tours:
                for direction, stops in (("outbound", t.outbound), ("inbound", t.inbound)):
                    d["nstops"][direction][0].append(x_nstops(xl, t.purpose, t.start, t.end, flag))
                    d["nstops"][direction][1].append(min(len(stops), MAX_STOPS))
                    for k, s in enumerate(stops):
                        d["purpose"][direction][0].append(x_stop_purpose(xl, t.purpose, k, t.duration))
                        d["purpose"][direction][1].append(s.act)
                        d["duration"][s.act][0].append(x_stop_duration(xl, direction, t.purpose, t.duration))
                        d["duration"][s.act][1].append(max(s.duration, 1))
        return d

    # ---- fit -------------------------------------------------------------------------
    def fit(self, records: list[DaysimRecord], X: np.ndarray, feature_names: list[str]) -> "DaysimModels":
        L = X.shape[1]
        self._pattern_alternatives([r.pattern for r in records])
        y_pat = [self.map_pattern(r.pattern) for r in records]
        self.pattern = MNL(maxiter=300).fit(X, y_pat, classes=list(self.pattern_vectors), feature_names=feature_names)
        print(f"  DAY-1 patterns: {len(self.pattern_vectors)} alternatives (support ≥ {self.min_support}); "
              f"{self.pattern_merged} persons mapped to nearest pattern")

        grid = PeriodGrid.uniform(self.time_resolution)
        self.tod = {g: TourTOD(grid, buffer=self.home_inter, maxiter=self.tod_maxiter) for g in GROUPS}
        design = self.tod_design(records, X)
        self.indep = {}
        for g in GROUPS:
            d = design[g]
            if not d["starts"]:
                self.tod[g] = None
                continue
            print(f"  DAY-2 TOD [{g}]: {len(d['starts']):,} tours × {len(self.tod[g].ps)} period pairs", flush=True)
            self.tod[g].fit(np.array(d["Z"]), np.array(d["starts"]), np.array(d["ends"]),
                            np.array(d["masks"]), key=g)
            self.indep[g] = IndependentTiming().fit(np.array(d["X"]), np.array(d["starts"]), np.array(d["ends"]))

        sd = self.stop_design(records, X)
        nsw = L + len(TOUR_PURPOSES) - 1 + 3
        self.nstops = {k: MNL().fit(np.array(sd["nstops"][k][0]).reshape(-1, nsw), sd["nstops"][k][1],
                                    classes=list(range(MAX_STOPS + 1))) for k in DIRECTIONS}
        psw = L + len(TOUR_PURPOSES) - 1 + 2 + 1
        self.stop_purpose = {}
        for k in DIRECTIONS:
            Xp, yp = sd["purpose"][k]
            self.stop_purpose[k] = MNL().fit(np.array(Xp).reshape(-1, psw), yp, classes=STOP_PURPOSES) if yp else None
        dw = L + 3
        all_X = [x for p in STOP_PURPOSES for x in sd["duration"][p][0]]
        all_t = [t for p in STOP_PURPOSES for t in sd["duration"][p][1]]
        pooled = LogNormalDuration().fit(np.array(all_X).reshape(-1, dw), np.array(all_t)) if all_t else None
        self.stop_duration = {}
        for p in STOP_PURPOSES:
            Xp, tp = sd["duration"][p]
            self.stop_duration[p] = (LogNormalDuration().fit(np.array(Xp), np.array(tp)) if len(tp) >= 10
                                     else pooled)
        return self

    # ---- validate ----------------------------------------------------------------------
    def validate(self, records: list[DaysimRecord], X: np.ndarray) -> dict:
        out = {"DAY-1 day pattern": self.pattern.validate(X, [self.map_pattern(r.pattern) for r in records])}
        out["DAY-1 day pattern"]["shares"] = dict(sorted(
            out["DAY-1 day pattern"]["shares"].items(), key=lambda kv: -kv[1]["obs"])[:8])
        design = self.tod_design(records, X)
        for g in GROUPS:
            d = design[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"DAY-2 TOD {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                             np.array(d["ends"]), np.array(d["masks"]))
        sd = self.stop_design(records, X)
        L = X.shape[1]
        nsw = L + len(TOUR_PURPOSES) - 1 + 3
        psw = nsw
        for k in DIRECTIONS:
            out[f"DAY-5 stops {k}"] = self.nstops[k].validate(np.array(sd["nstops"][k][0]).reshape(-1, nsw),
                                                              sd["nstops"][k][1])
            Xp, yp = sd["purpose"][k]
            if yp and self.stop_purpose[k] is not None:
                out[f"DAY-6 stop purpose {k}"] = self.stop_purpose[k].validate(np.array(Xp).reshape(-1, psw), yp)
        groups = {}
        for p in STOP_PURPOSES:
            Xp, tp = sd["duration"][p]
            if tp and self.stop_duration.get(p) is not None:
                groups[p] = self.stop_duration[p].validate(np.array(Xp), np.array(tp))
        out["DAY-7 stop duration (log-normal)"] = {"n": sum(m["n"] for m in groups.values()), "groups": groups}
        return out
