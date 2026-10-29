"""ActivitySim-aligned component models (SPECIFICATION §7.2).

| Component | Spec | Model |
|---|---|---|
| person type | ASIM-1 | rules (``persontype.py``) |
| CDAP M / N / H | ASIM-2 | MNL per person type |
| mandatory tour frequency | ASIM-3 | MNL per person type over observed alternatives |
| mandatory tour scheduling | ASIM-4 | ``TourTOD`` on the ``tdd`` table (work, school) with the MTC spec (``tod_spec``) and timetable terms |
| non-mandatory tour frequency | ASIM-5 | ``AltAttrMNL`` per person type (interaction_simulate style) over all purpose-count combinations (escort 0–2, other purposes 0–1), after mandatory scheduling, with the log maximum residual window; then the extension lookup (0–2 extra tours per purpose) |
| non-mandatory tour scheduling | ASIM-6 | ``TourTOD`` with the MTC non-mandatory spec and timetable terms, timetable masking, fixed purpose order |
| stop frequency (out × in, ≤ 3 × 3) | ASIM-7 | ``AltAttrMNL`` per tour purpose over the 16 alternatives: outbound, inbound and total-stop constants; covariates on any / outbound / inbound stops |
| trip purpose | ASIM-8 | probability lookup (tour purpose, direction, tour-hour band, person type) |
| trip scheduling | ASIM-9 | probability lookup over departure periods (direction, tour class, tour hour, tour purpose, trip index from the anchor) |

Back-off in the lookups drops the trailing key elements first.
"""

from collections import Counter
from functools import partial

import numpy as np

from composhed.activitysim.data import (
    MAX_EXTRA,
    MAX_STOPS,
    MTF_ALTS,
    NM_CAPS,
    NM_ORDER,
    AsimRecord,
    AsimTour,
    mtf_tours,
    nm_label,
)
from composhed.activitysim.persontype import PERSON_TYPES
from composhed.activitysim.segmented import SegmentedMNL
from composhed.activitysim.tod_spec import SPECS
from composhed.common.choice import AltAttrMNL
from composhed.common.periods import PeriodGrid
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD, tw_state
from composhed.data import DISC_ACTS

GROUPS = ["work", "school", "nonmandatory"]
STOP_ACTS = sorted(DISC_ACTS)
DAY = 1440


def tod_group(purpose: str) -> str:
    return purpose if purpose in ("work", "school") else "nonmandatory"


def phase(purpose: str) -> str:
    return "mandatory" if purpose in ("work", "school") else "nonmandatory"


def income_levels(values) -> list[str]:
    """Income bands in increasing order (by the lower bound of an ``a-b`` label; others last)."""
    def lower(v):
        try:
            return float(str(v).split("-")[0])
        except ValueError:
            return float("inf")
    return sorted(set(values), key=lambda v: (lower(v), str(v)))


def z_tod(spec_z, labels: dict, ptype: str, purpose: str, ctx: dict, levels: list[str], segmented: bool = True):
    """Scheduling covariates, named as in ``tod_spec.Z_NAMES``.

    ``ctx``: ``idx`` / ``n_in_phase`` (tour number among the mandatory or among the
    non-mandatory tours: MTC ``tour_num`` / ``tour_count``), ``same_idx`` /
    ``same_count`` (``tour_type_num`` / ``tour_type_count``), ``mtf``, ``n_mand``,
    ``n_escort``, ``n_non_escort``. Income enters as the band rank (MTC: income in
    thousands); "high income" is the top band.
    """
    inc = labels.get("hh_income")
    rank = levels.index(inc) if inc in levels else (len(levels) - 1) / 2
    v = {"const": 1.0, "is_escort": float(purpose == "escort"), "is_not_escort": float(purpose != "escort"),
         "income": float(rank), "high_income": float(inc == levels[-1]) if levels else 0.0,
         "rural": float(labels.get("hh_zone") == "rural"),
         "saturday": float(labels.get("day") == "saturday"), "sunday": float(labels.get("day") == "sunday"),
         "first_of_2plus": float(ctx["idx"] == 0 and ctx["n_in_phase"] > 1), "subsequent": float(ctx["idx"] > 0),
         "work_and_school_worker": float(ctx.get("mtf") == "work_and_school" and ptype in ("FTW",)),
         "work_and_school_student": float(ctx.get("mtf") == "work_and_school" and ptype not in ("FTW",)),
         f"purpose_{purpose}": 1.0,
         "num_mand": float(ctx.get("n_mand", 0)), "num_escort": float(ctx.get("n_escort", 0)),
         "num_non_escort": float(ctx.get("n_non_escort", 0)),
         "first_of_2plus_same": float(ctx.get("same_idx", 0) == 0 and ctx.get("same_count", 1) > 1),
         "subsequent_same": float(ctx.get("same_idx", 0) > 0)}
    if segmented:
        v[f"ptype_{ptype}"] = 1.0
    return spec_z(v)


def tour_contexts(purposes: list[str], mtf: str | None, n_mand: int, nm_purposes: list[str]) -> list[dict]:
    """``z_tod`` context of each tour of one scheduling phase, in scheduling order."""
    out, seen = [], Counter()
    count = Counter(purposes)
    for i, p in enumerate(purposes):
        out.append({"idx": i, "n_in_phase": len(purposes), "same_idx": seen[p], "same_count": count[p], "mtf": mtf,
                    "n_mand": n_mand, "n_escort": nm_purposes.count("escort"),
                    "n_non_escort": sum(q != "escort" for q in nm_purposes)})
        seen[p] += 1
    return out


def residual_window(intervals, buffer: int) -> float:
    """log(1 + hours) of the largest window left free by ``intervals`` (ActivitySim "Logged Maximum Residual Window")."""
    tw = TimeWindow()
    for s, e in intervals:
        if tw.is_free(s, e):
            tw.reserve(s, e)
    w = tw.largest_window(buffer)
    return float(np.log1p((w[1] - w[0]) / 60.0)) if w else 0.0


def x_nmtf(xl, cdap, n_mand, log_window):
    return np.concatenate([xl, [float(cdap == "M"), n_mand / 2.0, log_window]])


MAX_NM_BASE = sum(NM_CAPS.values())


def nm_alternatives() -> list[tuple]:
    """All purpose-count vectors with escort 0–2 and every other purpose 0–1 (prototype_mtc's alternatives file)."""
    ranges = [range(NM_CAPS[p] + 1) for p in NM_ORDER]
    return [tuple(v) for v in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(NM_ORDER), -1).T.tolist()]


def nm_attrs(vec: tuple) -> list[float]:
    """Alternative attributes of a purpose-count vector: tours per purpose, two escort tours, total-tour dummies."""
    tot = sum(vec)
    return ([float(n) for n in vec] + [float(vec[NM_ORDER.index("escort")] >= 2)]
            + [float(tot == k) for k in range(1, MAX_NM_BASE + 1)])


NM_ATTR_NAMES = ([f"n_{p}" for p in NM_ORDER] + ["2plus_escort"]
                 + [f"tot_{k}" for k in range(1, MAX_NM_BASE + 1)])
STOP_ALTS = [f"{o}out_{i}in" for o in range(MAX_STOPS + 1) for i in range(MAX_STOPS + 1)]
STOP_ATTR_NAMES = ([f"{k}out" for k in range(1, MAX_STOPS + 1)] + [f"{k}in" for k in range(1, MAX_STOPS + 1)]
                   + ["total_2", "total_3", "total_4plus"])
STOP_INTER_NAMES = ["any_stop", "any_outbound", "any_inbound"]


def _stop_counts(alt: str) -> tuple[int, int]:
    return int(alt[0]), int(alt.split("_")[1][0])


def stop_attrs(alt: str) -> list[float]:
    """Outbound, inbound and total-stop constants of an ``NoutMin`` alternative (MTC ``stop_frequency_*.csv``)."""
    n_out, n_in = _stop_counts(alt)
    tot = n_out + n_in
    return ([float(n_out == k) for k in range(1, MAX_STOPS + 1)] + [float(n_in == k) for k in range(1, MAX_STOPS + 1)]
            + [float(tot == 2), float(tot == 3), float(tot >= 4)])


def stop_inter_attrs(alt: str) -> list[float]:
    """What the covariates act on: any stop, any outbound stop, any inbound stop (MTC's shared coefficients)."""
    n_out, n_in = _stop_counts(alt)
    return [float(n_out + n_in > 0), float(n_out > 0), float(n_in > 0)]


def hour_band(minute: float) -> str:
    """Tour-hour range for the trip purpose lookup (ActivitySim ``depart_range_start/end``)."""
    h = minute / 60.0
    return "early" if h < 9 else ("midday" if h < 16 else "late")


def departure_keys(t: AsimTour, grid: PeriodGrid) -> list[tuple[tuple, float, float]]:
    """Observed trip departures of a tour: (lookup key, departure, neighbouring event) (ASIM-9).

    Outbound: departures from each stop, indexed forwards from the tour start.
    Inbound: departures from the primary activity and each stop but the last,
    indexed backwards from the tour end (ActivitySim schedules inbound trips in
    reverse). The anchor ("tour hour") is the start period outbound and the end
    period inbound. The neighbouring event is the previous departure (or tour
    start) outbound and the next departure (or tour end) inbound. Tours whose departures are not increasing (stops inside a
    fragmented work block, docs/tour_rules.md) give no records.
    """
    out_deps = [s + d for _, s, d in t.outbound]
    in_deps = ([t.primary_end] + [s + d for _, s, d in t.inbound[:-1]]) if t.inbound else []
    seq = [t.start] + out_deps + in_deps + [t.end]
    if any(b <= a for a, b in zip(seq, seq[1:])) and (out_deps or in_deps):
        return []
    cls = tour_class(t.purpose)
    out = []
    a_out, a_in = int(grid.period_of(t.start)), int(grid.end_period_of(t.end))
    for k, dep in enumerate(out_deps):
        prev = out_deps[k - 1] if k else t.start
        out.append((("outbound", cls, a_out, t.purpose, min(k, 2)), dep, prev))
    n = len(in_deps)
    for k, dep in enumerate(in_deps):
        nxt = in_deps[k + 1] if k + 1 < n else t.end
        out.append((("inbound", cls, a_in, t.purpose, min(n - 1 - k, 2)), dep, nxt))
    return out


STOPFREQ_EXTRA = ["am_peak_departure", "evening_arrival", "duration_11h_plus", "duration_9h_plus", "duration_3h_plus",
                  "n_work_tours", "n_school_tours", "n_escort_tours"]


def x_stopfreq(xl, start, end, tour_counts: dict):
    """Labels plus MTC's tour terms: departure 06:00–07:59, arrival from 19:00, duration bands (hours as
    ActivitySim's period indices) and the person's work, school and escort tours."""
    s, e = int(start // 60), int(end // 60)
    d = e - s
    return np.concatenate([xl, [float(5 < s < 8), float(e > 18), float(d > 10), float(d > 8), float(d > 2),
                                tour_counts.get("work", 0), tour_counts.get("school", 0),
                                tour_counts.get("escort", 0)]])


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
                 home_inter: int = 10, tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.segmented = segmented
        self.n_min = n_min
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    # ---- non-mandatory frequency alternatives --------------------------------------
    def map_nm(self, vec: tuple) -> str:
        """Records already hold the capped (pre-extension) vector, which is always an alternative."""
        return nm_label(vec)

    def z(self, group: str, labels: dict, ptype: str, purpose: str, ctx: dict) -> np.ndarray:
        return z_tod(self.tod[group].z, labels, ptype, purpose, ctx, self.income_levels, self.segmented)

    def _nm_avail(self, cdaps: list[str]) -> np.ndarray:
        zero = np.array([sum(v) == 0 for v in self.nm_vectors.values()])
        return np.array([~zero if c == "N" else np.ones(len(zero), bool) for c in cdaps]).reshape(len(cdaps), -1)

    # ---- designs ------------------------------------------------------------------------
    def tod_design(self, records: list[AsimRecord], X: np.ndarray) -> dict:
        """Scheduling observations in chain order; masks and timetable terms from the earlier observed tours."""
        out = {g: {"Z": [], "X": [], "starts": [], "ends": [], "masks": [], "states": []} for g in GROUPS}
        for r, xl in zip(records, X):
            tw = TimeWindow()
            mand = [t.purpose for t in r.tours if phase(t.purpose) == "mandatory"]
            nm = [t.purpose for t in r.tours if phase(t.purpose) != "mandatory"]
            ctxs = (tour_contexts(mand, r.mtf, len(mand), nm) + tour_contexts(nm, r.mtf, len(mand), nm))
            for t, ctx in zip(r.tours, ctxs):
                g = tod_group(t.purpose)
                d = out[g]
                if self.tod.get(g) is None:
                    continue
                d["Z"].append(self.z(g, r.labels, r.ptype, t.purpose, ctx))
                d["X"].append(xl)
                d["starts"].append(t.start)
                d["ends"].append(t.end)
                d["masks"].append(self.tod[g].mask(tw))
                d["states"].append(tw_state(tw, ctx["n_in_phase"] - ctx["idx"] - 1))
                tw.reserve(t.start, t.end)
        return out

    def nmtf_design(self, records: list[AsimRecord], X: np.ndarray, idx: list[int]) -> np.ndarray:
        """NMTF covariates from the observed mandatory tours (count as chosen by MTF, residual window)."""
        rows = []
        for i in idx:
            r = records[i]
            mand = [(t.start, t.end) for t in r.tours if phase(t.purpose) == "mandatory"]
            n_mand = len(mtf_tours(r.mtf)) if r.mtf else 0
            rows.append(x_nmtf(X[i], r.cdap, n_mand, residual_window(mand, self.home_inter)))
        return np.array(rows).reshape(len(idx), -1)

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

        self.income_levels = income_levels(r.labels.get("hh_income") for r in records)
        # Non-mandatory frequency over all capped purpose-count combinations, then the extension lookup.
        mn_idx = [i for i, r in enumerate(records) if r.cdap in ("M", "N")]
        self.nm_vectors = {nm_label(v): v for v in nm_alternatives()}
        ext_keys, ext_vals = [], []
        for i in mn_idx:
            r = records[i]
            for p, n, extra in zip(NM_ORDER, r.nm, r.nm_extra):
                if n > 0:
                    ext_keys.append((p, r.cdap == "M", r.ptype))
                    ext_vals.append(extra)
        self.nm_extension = Lookup(list(range(MAX_EXTRA + 1))).fit(ext_keys, ext_vals)
        self.nm_mapped = sum(1 for i in mn_idx if any(records[i].nm_extra))
        Xn = self.nmtf_design(records, X, mn_idx)
        yn = [self.map_nm(records[i].nm) for i in mn_idx]
        nm_factory = partial(AltAttrMNL, {k: nm_attrs(v) for k, v in self.nm_vectors.items()}, NM_ATTR_NAMES)
        self.nmtf = SegmentedMNL(self.n_min, self.segmented, factory=nm_factory).fit(
            Xn, yn, [ptypes[i] for i in mn_idx], list(self.nm_vectors),
            avail=self._nm_avail([records[i].cdap for i in mn_idx]),
            feature_names=feature_names + ["cdap_M", "n_mandatory/2", "log_max_residual_window"]) if mn_idx else None
        print(f"  ASIM-5 non-mandatory frequency: {len(self.nm_vectors)} alternatives, "
              f"{len(NM_ATTR_NAMES)} alternative attributes; extension adds tours for {self.nm_mapped} persons")

        # ActivitySim's sub-hourly examples (SEMCOG, ARC, MWCOG) start their periods at 03:00; MTC's hourly at 05:00.
        grid = PeriodGrid.with_early_bin(self.time_resolution, early_end=300 if self.time_resolution == 60 else 180)
        self.tod = {g: TourTOD(grid, SPECS[g], buffer=self.home_inter, maxiter=self.tod_maxiter, dynamic="asim")
                    for g in GROUPS}
        design = self.tod_design(records, X)
        self.indep = {}
        for g in GROUPS:
            d = design[g]
            if not d["starts"]:
                self.tod[g] = None
                continue
            print(f"  ASIM-4/6 tdd [{g}]: {len(d['starts']):,} tours × {len(self.tod[g].ps)} alternatives", flush=True)
            self.tod[g].fit(np.array(d["Z"]), np.array(d["starts"]), np.array(d["ends"]), np.array(d["masks"]),
                            states=d["states"])
            self.indep[g] = IndependentTiming().fit(np.array(d["X"]), np.array(d["starts"]), np.array(d["ends"]))

        # Stop frequency per tour purpose; trip purpose and departure lookups.
        sx, sy, sp = [], [], []
        trip_keys, trip_acts, dep_keys, dep_periods = [], [], [], []
        self.departure_records_skipped = 0
        for r, xl in zip(records, X):
            counts = Counter(t.purpose for t in r.tours)
            for t in r.tours:
                sx.append(x_stopfreq(xl, t.start, t.end, counts))
                sy.append(stop_alt(len(t.outbound), len(t.inbound)))
                sp.append(t.purpose)
                for direction, stops in (("outbound", t.outbound), ("inbound", t.inbound)):
                    anchor = t.start if direction == "outbound" else t.end
                    for act, _, _ in stops:
                        trip_keys.append((t.purpose, direction, hour_band(anchor), r.ptype))
                        trip_acts.append(act)
                deps = departure_keys(t, grid)
                if not deps and (t.outbound or t.inbound):
                    self.departure_records_skipped += 1
                for key, dep, _ in deps:
                    dep_keys.append(key)
                    dep_periods.append(int(grid.end_period_of(dep)))
        self.stop_classes = STOP_ALTS
        stop_factory = partial(AltAttrMNL, {a: stop_attrs(a) for a in STOP_ALTS}, STOP_ATTR_NAMES,
                               inter_attrs={a: stop_inter_attrs(a) for a in STOP_ALTS}, inter_names=STOP_INTER_NAMES)
        self.stopfreq = SegmentedMNL(self.n_min, True, factory=stop_factory).fit(
            np.array(sx).reshape(len(sx), -1), sy, sp, self.stop_classes,
            feature_names=feature_names + STOPFREQ_EXTRA) if sx else None
        self.trip_purpose = Lookup(STOP_ACTS).fit(trip_keys, trip_acts)
        self.trip_departure = Lookup(list(range(grid.n))).fit(dep_keys, dep_periods)
        print(f"  ASIM-9 trip departure: {len(dep_keys):,} departures, {len(self.trip_departure.tables)} lookup cells; "
              f"{self.departure_records_skipped} tours with non-increasing departures skipped")
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
            Xn = self.nmtf_design(records, X, mn_idx)
            out["ASIM-5 non-mandatory tour frequency"] = self.nmtf.validate(
                Xn, [self.map_nm(records[i].nm) for i in mn_idx], [ptypes[i] for i in mn_idx],
                self._nm_avail([records[i].cdap for i in mn_idx]))
        design = self.tod_design(records, X)
        for g in GROUPS:
            d = design[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"ASIM-4/6 tdd {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                                np.array(d["ends"]), np.array(d["masks"]), d["states"])
        if self.stopfreq is not None:
            sx = [x_stopfreq(xl, t.start, t.end, Counter(u.purpose for u in r.tours))
                  for r, xl in zip(records, X) for t in r.tours]
            sy = [stop_alt(len(t.outbound), len(t.inbound)) for r in records for t in r.tours]
            sp = [t.purpose for r in records for t in r.tours]
            if sx:
                out["ASIM-7 stop frequency"] = self.stopfreq.validate(np.array(sx), sy, sp)
        out["ASIM-8 trip purpose lookup"] = {"n": sum(n for k, (_, n) in self.trip_purpose.tables.items() if k == ()),
                                             "cells": len(self.trip_purpose.tables)}
        out["ASIM-9 trip departure lookup"] = {"n": sum(n for k, (_, n) in self.trip_departure.tables.items()
                                                        if k == ()), "cells": len(self.trip_departure.tables)}
        return out
