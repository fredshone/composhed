"""ActivitySim-aligned component models (SPECIFICATION §7.2).

| Component | Spec | Model |
|---|---|---|
| person type | ASIM-1 | rules (``persontype.py``) |
| CDAP M / N / H | ASIM-2 | MNL per person type |
| mandatory tour frequency | ASIM-3 | MNL per person type over observed alternatives |
| mandatory tour scheduling | ASIM-4 | ``TourTOD`` on the hourly ``tdd`` table (work, school), person-type terms |
| non-mandatory tour frequency (≤ 4 tours) | ASIM-5 | MNL per person type over observed purpose-count combinations |
| non-mandatory tour scheduling | ASIM-6 | ``TourTOD`` with purpose terms, timetable masking, fixed purpose order |
| stop frequency (out × in, ≤ 3 × 3) | ASIM-7 | MNL per tour purpose over observed alternatives |
| trip purpose | ASIM-8 | probability lookup (tour purpose, direction, stop index) |
| trip scheduling | ASIM-9 | probability lookup over departure periods (tour class, direction, stop index) |
"""

from collections import Counter

import numpy as np

from composhed.activitysim.data import (
    MAX_STOPS,
    MTF_ALTS,
    NM_ORDER,
    TOUR_ORDER,
    AsimRecord,
    nm_label,
)
from composhed.activitysim.persontype import PERSON_TYPES
from composhed.activitysim.segmented import SegmentedMNL
from composhed.common.periods import IntraPeriodSampler, PeriodGrid
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD
from composhed.data import DISC_ACTS

GROUPS = ["work", "school", "nonmandatory"]
STOP_ACTS = sorted(DISC_ACTS)
DAY = 1440


def tod_group(purpose: str) -> str:
    return purpose if purpose in ("work", "school") else "nonmandatory"


def _onehot(value, levels):
    return [1.0 if value == lv else 0.0 for lv in levels[1:]]


def z_tod(xl, ptype, purpose, nth, segmented=True):
    pt = _onehot(ptype, PERSON_TYPES) if segmented else [0.0] * (len(PERSON_TYPES) - 1)
    return np.concatenate([[1.0], xl, pt, _onehot(purpose, NM_ORDER), [float(nth > 0)]])


def x_nmtf(xl, cdap, n_mand):
    return np.concatenate([xl, [float(cdap == "M"), n_mand / 2.0]])


def x_stopfreq(xl, start, end):
    return np.concatenate([xl, [(end - start) / DAY, start / DAY]])


def stop_alt(n_out: int, n_in: int) -> str:
    return f"{min(n_out, MAX_STOPS)}out_{min(n_in, MAX_STOPS)}in"


def tour_class(purpose: str) -> str:
    return "mandatory" if purpose in ("work", "school") else "nonmandatory"


class Lookup:
    """Conditional probability table with back-off to coarser keys (ASIM-8/9)."""

    def __init__(self, outcomes: list, min_count: int = 20) -> None:
        self.outcomes = list(outcomes)
        self.min_count = min_count
        self.tables: dict = {}

    def fit(self, keys: list[tuple], values: list) -> "Lookup":
        counts: dict = {}
        for key, v in zip(keys, values):
            for depth in range(len(key) + 1):
                counts.setdefault(key[:depth], Counter())[v] += 1
        for k, c in counts.items():
            n = sum(c.values())
            if n >= self.min_count or k == ():
                self.tables[k] = (np.array([c[o] for o in self.outcomes], dtype=float) / n, n)
        return self

    def probs(self, key: tuple) -> np.ndarray:
        for depth in range(len(key), -1, -1):
            if key[:depth] in self.tables:
                return self.tables[key[:depth]][0]
        return np.full(len(self.outcomes), 1.0 / len(self.outcomes))

    def as_dict(self) -> dict:
        return {"|".join(map(str, k)) or "*": {"n": n, "p": dict(zip(map(str, self.outcomes), p.round(4).tolist()))}
                for k, (p, n) in self.tables.items()}


class AsimModels:
    def __init__(self, time_resolution: int = 60, segmented: bool = True, n_min: int = 200,
                 min_support: int = 30, home_inter: int = 10, tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.segmented = segmented
        self.n_min = n_min
        self.min_support = min_support
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    # ---- non-mandatory frequency alternatives --------------------------------------
    def map_nm(self, vec: tuple) -> str:
        lab = nm_label(vec)
        if lab in self.nm_vectors:
            return lab
        return min(self.nm_vectors, key=lambda k: (
            sum(abs(a - b) for a, b in zip(vec, self.nm_vectors[k])), -self.nm_counts[k]))

    def _nm_avail(self, cdaps: list[str]) -> np.ndarray:
        zero = np.array([sum(v) == 0 for v in self.nm_vectors.values()])
        return np.array([~zero if c == "N" else np.ones(len(zero), bool) for c in cdaps]).reshape(len(cdaps), -1)

    # ---- designs ------------------------------------------------------------------------
    def tod_design(self, records: list[AsimRecord], X: np.ndarray) -> dict:
        out = {g: {"Z": [], "X": [], "starts": [], "ends": [], "masks": []} for g in GROUPS}
        for r, xl in zip(records, X):
            tw = TimeWindow()
            seen: Counter = Counter()
            for t in r.tours:
                g = tod_group(t.purpose)
                d = out[g]
                d["Z"].append(z_tod(xl, r.ptype, t.purpose, seen[t.purpose], self.segmented))
                d["X"].append(xl)
                d["starts"].append(t.start)
                d["ends"].append(t.end)
                d["masks"].append(self.tod[g].mask(tw) if self.tod.get(g) is not None else None)
                tw.reserve(t.start, t.end)
                seen[t.purpose] += 1
        return out

    # ---- fit ---------------------------------------------------------------------------------
    def fit(self, records: list[AsimRecord], X: np.ndarray, feature_names: list[str]) -> "AsimModels":
        seg = SegmentedMNL
        ptypes = [r.ptype for r in records]
        self.cdap_classes = [c for c in ("M", "N", "H") if any(r.cdap == c for r in records)]
        self.cdap = seg(self.n_min, self.segmented).fit(X, [r.cdap for r in records], ptypes, self.cdap_classes,
                                                        feature_names=feature_names)
        m_idx = [i for i, r in enumerate(records) if r.cdap == "M"]
        self.mtf_classes = [a for a in MTF_ALTS if any(records[i].mtf == a for i in m_idx)]
        self.mtf = seg(self.n_min, self.segmented).fit(
            X[m_idx], [records[i].mtf for i in m_idx], [ptypes[i] for i in m_idx], self.mtf_classes,
            feature_names=feature_names) if m_idx else None

        # Non-mandatory frequency over observed combinations with support >= min_support.
        mn_idx = [i for i, r in enumerate(records) if r.cdap in ("M", "N")]
        counts = Counter(records[i].nm for i in mn_idx)
        keep = [v for v, c in counts.most_common() if c >= self.min_support] or [counts.most_common(1)[0][0]]
        if not any(sum(v) == 0 for v in keep):
            keep.append((0,) * len(NM_ORDER))
        self.nm_vectors = {nm_label(v): v for v in keep}
        self.nm_counts = {nm_label(v): counts.get(v, 0) for v in keep}
        self.nm_mapped = sum(c for v, c in counts.items() if nm_label(v) not in self.nm_vectors)
        Xn = np.array([x_nmtf(X[i], records[i].cdap, sum(t.purpose in ("work", "school") for t in records[i].tours))
                       for i in mn_idx]).reshape(len(mn_idx), -1)
        yn = [self.map_nm(records[i].nm) for i in mn_idx]
        self.nmtf = seg(self.n_min, self.segmented).fit(
            Xn, yn, [ptypes[i] for i in mn_idx], list(self.nm_vectors),
            avail=self._nm_avail([records[i].cdap for i in mn_idx])) if mn_idx else None
        print(f"  ASIM-5 non-mandatory frequency: {len(self.nm_vectors)} alternatives; "
              f"{self.nm_mapped} persons mapped to nearest")

        grid = PeriodGrid.with_early_bin(self.time_resolution)
        self.tod = {g: TourTOD(grid, buffer=self.home_inter, maxiter=self.tod_maxiter) for g in GROUPS}
        design = self.tod_design(records, X)
        self.indep = {}
        for g in GROUPS:
            d = design[g]
            if not d["starts"]:
                self.tod[g] = None
                continue
            print(f"  ASIM-4/6 tdd [{g}]: {len(d['starts']):,} tours × {len(self.tod[g].ps)} alternatives", flush=True)
            self.tod[g].fit(np.array(d["Z"]), np.array(d["starts"]), np.array(d["ends"]), np.array(d["masks"]), key=g)
            self.indep[g] = IndependentTiming().fit(np.array(d["X"]), np.array(d["starts"]), np.array(d["ends"]))

        # Stop frequency per tour purpose.
        sx, sy, sp = [], [], []
        trip_keys, trip_acts, dep_keys, dep_periods, dep_minutes = [], [], [], [], {}
        for r, xl in zip(records, X):
            for t in r.tours:
                sx.append(x_stopfreq(xl, t.start, t.end))
                sy.append(stop_alt(len(t.outbound), len(t.inbound)))
                sp.append(t.purpose)
                for direction, stops in (("outbound", t.outbound), ("inbound", t.inbound)):
                    for k, (act, s, dur) in enumerate(stops):
                        trip_keys.append((t.purpose, direction, min(k, 2)))
                        trip_acts.append(act)
                    # Departures: from each outbound stop; from the primary and each inbound stop but the last.
                    if direction == "outbound":
                        deps = [s + dur for _, s, dur in stops]
                    else:
                        deps = [t.primary_end] + [s + dur for _, s, dur in stops[:-1]] if stops else []
                    for k, dep in enumerate(deps):
                        key = (tour_class(t.purpose), direction, min(k, 2))
                        dep_keys.append(key)
                        dep_periods.append(int(grid.end_period_of(dep)))
                        dep_minutes.setdefault(direction, []).append(dep)
        self.stop_classes = sorted(set(sy)) or ["0out_0in"]
        self.stopfreq = seg(self.n_min, True).fit(np.array(sx).reshape(len(sx), -1), sy, sp, self.stop_classes) \
            if sx else None
        self.trip_purpose = Lookup(STOP_ACTS).fit(trip_keys, trip_acts)
        self.trip_departure = Lookup(list(range(grid.n))).fit(dep_keys, dep_periods)
        self.trip_intra = IntraPeriodSampler(grid).fit(dep_minutes, end=True)
        self.grid = grid
        return self

    # ---- validate ------------------------------------------------------------------------
    def validate(self, records: list[AsimRecord], X: np.ndarray) -> dict:
        ptypes = [r.ptype for r in records]
        out = {"ASIM-2 CDAP": self.cdap.validate(X, [r.cdap for r in records], ptypes)}
        m_idx = [i for i, r in enumerate(records) if r.cdap == "M"]
        if self.mtf is not None and m_idx:
            out["ASIM-3 mandatory tour frequency"] = self.mtf.validate(
                X[m_idx], [records[i].mtf for i in m_idx], [ptypes[i] for i in m_idx])
        mn_idx = [i for i, r in enumerate(records) if r.cdap in ("M", "N")]
        if self.nmtf is not None and mn_idx:
            Xn = np.array([x_nmtf(X[i], records[i].cdap, sum(t.purpose in ("work", "school") for t in records[i].tours))
                           for i in mn_idx])
            out["ASIM-5 non-mandatory tour frequency"] = self.nmtf.validate(
                Xn, [self.map_nm(records[i].nm) for i in mn_idx], [ptypes[i] for i in mn_idx],
                self._nm_avail([records[i].cdap for i in mn_idx]))
        design = self.tod_design(records, X)
        for g in GROUPS:
            d = design[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"ASIM-4/6 tdd {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                                np.array(d["ends"]), np.array(d["masks"]))
        if self.stopfreq is not None:
            sx = [x_stopfreq(xl, t.start, t.end) for r, xl in zip(records, X) for t in r.tours]
            sy = [stop_alt(len(t.outbound), len(t.inbound)) for r in records for t in r.tours]
            sp = [t.purpose for r in records for t in r.tours]
            sy = [v if v in self.stop_classes else "0out_0in" for v in sy]
            if sx:
                out["ASIM-7 stop frequency"] = self.stopfreq.validate(np.array(sx), sy, sp)
        out["ASIM-8 trip purpose lookup"] = {"n": sum(n for k, (_, n) in self.trip_purpose.tables.items() if k == ()),
                                             "cells": len(self.trip_purpose.tables)}
        out["ASIM-9 trip departure lookup"] = {"n": sum(n for k, (_, n) in self.trip_departure.tables.items()
                                                        if k == ()), "cells": len(self.trip_departure.tables)}
        return out
