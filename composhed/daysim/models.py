"""DaySim-aligned component models (SPECIFICATION §6.2), after the DaySim source (RSGInc/DaySim, Default models).

| Component | DaySim model | Model here |
|---|---|---|
| day pattern: which tour purposes and which stop purposes the day has | ``IndividualPersonDayPatternModel`` | ``AltAttrMNL`` over all 1,324 feasible patterns: tour and stop constants, tour–stop, tour–tour, stop–stop and tour-count × stop-count combination constants; person features on purpose presence, number of tours, number of stops, any non-mandatory tour / stop |
| exact number of tours per purpose (1, 2, 3+) | ``PersonExactNumberOfToursModel`` | MNL per purpose on person features and the pattern |
| tour time: arrival and departure at the primary destination | ``WorkTourTimeModel`` / ``SchoolTourTimeModel`` / ``OtherHomeBasedTourTimeModel`` | ``TourTOD`` over half-hour period pairs with the DaySim spec (``tod_spec``) and time-window terms |
| intermediate stop generation (next stop's purpose, or none) | ``IntermediateStopGenerationModel`` | MNL over none + 5 stop purposes, purposes available only if the pattern has them; per half-tour, working outwards from the primary destination |
| trip time (period of each stop's outer boundary) | ``TripTimeModel`` | ``TripTimeModel`` below: conditional logit over half-hour periods with direction-specific period constants, duration constants and shift terms; uniform minute within the period |
| AB-2 timing | — | ``IndependentTiming`` per group |
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp

from composhed.activitysim.persontype import PERSON_TYPES
from composhed.common.choice import MNL, AltAttrMNL, ConditionalLogit
from composhed.common.periods import PeriodGrid
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD, tw_state
from composhed.daysim.data import (
    MAX_TOURS_PER_PURPOSE,
    NM_PURPOSES,
    STOP_PURPOSES,
    TOUR_PURPOSES,
    DaysimRecord,
    enumerate_patterns,
    pattern_label,
    tod_group,
)
from composhed.daysim.tod_spec import SPECS, priority_flags

GROUPS = ["work", "education", "nonmandatory"]
DIRECTIONS = ["outbound", "inbound"]
MAX_STOPS = 4  # per half-tour
NONE = "none"
STOPGEN_CLASSES = [NONE] + STOP_PURPOSES
DAY = 1440
NT = len(TOUR_PURPOSES)


def _onehot(value, levels):
    return [1.0 if value == lv else 0.0 for lv in levels[1:]]


def person_x(xl: np.ndarray, ptype: str) -> np.ndarray:
    """Person features of the DaySim models: person type (full-time worker reference) and the eight labels."""
    return np.concatenate([_onehot(ptype, PERSON_TYPES), xl])


# ---- day pattern -------------------------------------------------------------------------------------
def pattern_attrs(vec: tuple) -> list[float]:
    """Constants of a pattern (IndividualPersonDayPatternModel terms 100p, 100p+1, 1000+, 1100+, 1200+, 1300+)."""
    tours, stops = vec[:NT], vec[NT:]
    nt, ns = sum(tours), min(sum(stops), 3)
    tour_stop = [float(t and s) for t in tours for s in stops]
    tour_tour = [float(tours[i] and tours[j]) for i in range(NT) for j in range(i + 1, NT)]
    stop_stop = [float(stops[i] and stops[j]) for i in range(len(stops)) for j in range(i + 1, len(stops))]
    counts = [float(nt == a and ns == b) for a in range(1, 4) for b in range(1, 4)]
    return [float(t) for t in tours] + [float(s) for s in stops] + tour_stop + tour_tour + stop_stop + counts


PATTERN_ATTR_NAMES = ([f"tour_{p}" for p in TOUR_PURPOSES] + [f"stop_{p}" for p in STOP_PURPOSES]
                      + [f"tour_{p}*stop_{q}" for p in TOUR_PURPOSES for q in STOP_PURPOSES]
                      + [f"tour_{TOUR_PURPOSES[i]}*tour_{TOUR_PURPOSES[j]}" for i in range(NT) for j in range(i + 1, NT)]
                      + [f"stop_{STOP_PURPOSES[i]}*stop_{STOP_PURPOSES[j]}" for i in range(5) for j in range(i + 1, 5)]
                      + [f"{a}tours_{b}stops" for a in range(1, 4) for b in range(1, 4)])


def pattern_inter_attrs(vec: tuple) -> list[float]:
    """What person features act on: purpose present (tour or stop), 2 / 3 tours, 2 / 3 / 4 stops, any
    non-mandatory tour, any stop (the purpose, tour-count, stop-count, tour and stop components)."""
    tours, stops = vec[:NT], vec[NT:]
    present = [float(t or (p in STOP_PURPOSES and stops[STOP_PURPOSES.index(p)]))
               for p, t in zip(TOUR_PURPOSES, tours)]
    nt, ns = sum(tours), sum(stops)
    return present + [float(nt == 2), float(nt == 3), float(ns == 2), float(ns == 3), float(ns == 4),
                      float(sum(tours[2:]) > 0), float(ns > 0)]


PATTERN_INTER_NAMES = ([f"present_{p}" for p in TOUR_PURPOSES]
                       + ["tours_2", "tours_3", "stops_2", "stops_3", "stops_4", "any_nm_tour", "any_stop"])


def x_exact(xp: np.ndarray, vec: tuple) -> np.ndarray:
    return np.concatenate([xp, np.asarray(vec, dtype=float)])


# ---- tour time ------------------------------------------------------------------------------------------
def z_tod_values(labels: dict, ptype: str, purpose: str, rank: int, n_tours: int, same_idx: int, same_count: int,
                 vec: tuple, levels: list[str]) -> dict:
    inc = labels.get("hh_income")
    v = {"const": 1.0, f"ptype_{ptype}": 1.0, f"purpose_{purpose}": 1.0,
         "income_low": float(bool(levels) and inc == levels[0]), "income_high": float(bool(levels) and inc == levels[-1]),
         "n_stop_purposes": float(sum(vec[NT:])), "escort_stops": float(vec[NT + STOP_PURPOSES.index("escort")]),
         "saturday": float(labels.get("day") == "saturday"), "sunday": float(labels.get("day") == "sunday")}
    v |= priority_flags(rank, n_tours, same_idx, same_count)
    return v


def tour_ranks(purposes: list[str]) -> list[tuple[int, int, int, int]]:
    """(rank, n_tours, same-purpose index, same-purpose count) of each tour in priority order."""
    count, seen, out = Counter(purposes), Counter(), []
    for i, p in enumerate(purposes):
        out.append((i, len(purposes), seen[p], count[p]))
        seen[p] += 1
    return out


# ---- stop generation --------------------------------------------------------------------------------------
TOD_FLAGS = [(7, 9), (9, 11), (11, 13), (13, 15), (15, 17), (19, 21), (21, 23)]  # 17–19 reference; 23–7 below


def x_stopgen(xp, tour_purpose: str, outbound: bool, k: int, window_hours: float, anchor: float,
              remaining_tours: int, stops_so_far: int, female: bool) -> np.ndarray:
    """IntermediateStopGenerationModel covariates: tour purpose, direction, trip number in the half-tour
    (× direction), before the mandatory destination, remaining tours, free hours on this side of the tour,
    time of day of the tour anchor (arrival outbound, departure inbound), stops simulated so far."""
    h = anchor / 60.0
    trip = [float(k == j) for j in range(1, MAX_STOPS + 1)]
    tod = [float(a <= h < b) for a, b in TOD_FLAGS] + [float(h >= 23 or h < 7)]
    return np.concatenate([xp, _onehot(tour_purpose, TOUR_PURPOSES),
                           [float(outbound)], [t * outbound for t in trip], [t * (not outbound) for t in trip],
                           [float(outbound and tour_purpose in ("work", "education")), float(remaining_tours),
                            window_hours, float(stops_so_far), float(female)], tod])


# ---- trip time --------------------------------------------------------------------------------------------
OUT_GROUPS = [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]
IN_GROUPS = [(0, 7), (7, 10), (10, 13), (13, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 21), (21, 25)]
TT_DUR = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 9), (9, 12), (12, 14), (14, 18), (18, 99)]
TT_FLAGS = ([f"ptype_{p}" for p in PERSON_TYPES if p != "FTW"]
            + ["outbound", "work_outbound", "work_inbound", "nonwork_inbound"]
            + [f"stop_{p}" for p in STOP_PURPOSES if p != "leisure"] + ["saturday", "sunday"])
TT_DYN = ([f"duration_{a}_{b}" for a, b in TT_DUR[1:]] + [f"{f}*duration" for f in TT_FLAGS]
          + ["log_period_fraction_free", "trip_remaining_per_adjacent_hour", "tours_left_per_free_hour"])


def tt_flags(ptype: str, outbound: bool, tour_purpose: str, stop_purpose: str, labels: dict) -> dict:
    work = tour_purpose == "work"
    return {f"ptype_{ptype}": 1.0, "outbound": float(outbound), "work_outbound": float(work and outbound),
            "work_inbound": float(work and not outbound), "nonwork_inbound": float(not work and not outbound),
            f"stop_{stop_purpose}": 1.0, "saturday": float(labels.get("day") == "saturday"),
            "sunday": float(labels.get("day") == "sunday")}


class TripTimeModel:
    """DaySim ``TripTimeModel``: the period of a stop's outer boundary, working outwards from the primary activity.

    Outbound, the chosen period holds the stop's start (the departure from the
    previous place); inbound, the stop's end. The alternatives are the periods
    with a free minute between the anchor (the inner boundary) and the edge of
    the free window. Utility: direction-specific period constants (terms 11–30),
    the period shift (terms 42–56, 132–156) and the duration from the anchor
    period (terms 31–55, 131–155) by person type, direction × work tour and stop
    purpose, the log share of the period that is free (term 92), and the
    trip-remaining and tours-remaining window terms (97–99).
    """

    def __init__(self, grid: PeriodGrid, l2: float = 1e-3, maxiter: int = 300) -> None:
        self.grid = grid
        n = grid.n
        self.mid = grid.midpoints()
        h = self.mid / 60.0
        G = {"shift": np.arange(n) * 48.0 / n}
        for a, b in OUT_GROUPS:
            G[f"out_{a}_{b}"] = ((h >= a) & (h < b)).astype(float)
        for a, b in IN_GROUPS:
            G[f"in_{a}_{b}"] = ((h >= a) & (h < b)).astype(float)
        self.g_names = list(G)
        self.z_names = ["outbound", "inbound"] + TT_FLAGS
        mask = np.zeros((len(self.z_names), len(self.g_names)), dtype=bool)
        zi, gi = self.z_names.index, self.g_names.index
        for a, b in OUT_GROUPS[:2] + OUT_GROUPS[3:]:  # 08:00–09:00 reference
            mask[zi("outbound"), gi(f"out_{a}_{b}")] = True
        for a, b in IN_GROUPS[:6] + IN_GROUPS[7:]:  # 17:00–18:00 reference
            mask[zi("inbound"), gi(f"in_{a}_{b}")] = True
        for f in TT_FLAGS:
            mask[zi(f), gi("shift")] = True
        self.clogit = ConditionalLogit(sp.csr_matrix((n, 0)), np.column_stack([G[k] for k in self.g_names]),
                                       l2=l2, maxiter=maxiter, n_dyn=len(TT_DYN), b_mask=mask)
        self.offset = self.clogit.offset

    def z(self, flags: dict) -> np.ndarray:
        return np.array([float(flags.get(k, 0.0)) for k in self.z_names])

    def _feasible_minutes(self, anchor: int, lo: int, hi: int, outbound: bool) -> tuple[np.ndarray, np.ndarray]:
        """Per period, the range of feasible boundary minutes: outbound starts in [lo, anchor - 1] (start convention),
        inbound ends in [anchor + 1, hi] (end convention, minutes (p_lo, p_hi])."""
        g = self.grid
        if outbound:
            a = np.maximum(g.edges[:-1], lo)
            b = np.minimum(g.edges[1:] - 1, anchor - 1)
        else:
            a = np.maximum(g.edges[:-1] + 1, anchor + 1)
            b = np.minimum(g.edges[1:], hi)
        return a, b

    def attributes(self, states: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """states: (anchor, lo, hi, outbound, flags dict, trip_remaining, tours_left, free_minutes).
        Returns (availability (n, J), chooser-specific attributes (n, J, len(TT_DYN)))."""
        n, J = len(states), self.grid.n
        avail = np.zeros((n, J), dtype=bool)
        A = np.zeros((n, J, len(TT_DYN)))
        width = np.diff(self.grid.edges).astype(float)
        anchor_period = lambda m, out: int(self.grid.end_period_of(m) if out else self.grid.period_of(m))
        nd = len(TT_DUR) - 1
        for i, (anchor, lo, hi, outbound, flags, trip_rem, tours_left, free) in enumerate(states):
            a, b = self._feasible_minutes(int(anchor), int(lo), int(hi), outbound)
            nfree = np.maximum(b - a + 1, 0)
            avail[i] = nfree > 0
            dur = np.abs(self.mid - self.mid[anchor_period(anchor, outbound)]) / 60.0
            for k, (x, y) in enumerate(TT_DUR[1:]):
                A[i, :, k] = (dur >= x) & (dur < y)
            for k, f in enumerate(TT_FLAGS):
                A[i, :, nd + k] = flags.get(f, 0.0) * dur
            A[i, :, -3] = np.log(np.maximum(nfree, 1) / width)
            adj = ((self.grid.edges[:-1] - lo) if outbound else (hi - self.grid.edges[1:])) / 60.0
            A[i, :, -2] = trip_rem / (1.0 + np.maximum(adj, 0.0))
            A[i, :, -1] = tours_left / (1.0 + max(free, 0) / 60.0)
        return avail, A

    def fit(self, Z: np.ndarray, states: list[tuple], chosen: np.ndarray) -> "TripTimeModel":
        avail, A = self.attributes(states)
        self.clogit.fit(Z, chosen, avail, lambda s, e: A[s:e])
        return self

    def validate(self, Z, states, chosen) -> dict:
        avail, A = self.attributes(states)
        out = self.clogit.validate(Z, chosen, avail, lambda s, e: A[s:e])
        out["coefficients"] = dict(zip(TT_DYN, self.clogit.delta_.round(4).tolist()))
        return out

    def sample(self, z: np.ndarray, state: tuple, rng: np.random.Generator) -> int | None:
        """A boundary minute, uniform within the chosen period's feasible minutes; None if nothing is free."""
        avail, A = self.attributes([state])
        if not avail.any():
            return None
        j = int(self.clogit.sample(z[None, :], rng, avail, lambda s, e: A[s:e])[0])
        if j < 0:
            return None
        anchor, lo, hi, outbound = (int(state[0]), int(state[1]), int(state[2]), state[3])
        a, b = self._feasible_minutes(anchor, lo, hi, outbound)
        return int(rng.integers(a[j], b[j] + 1))


def window_around(tw: TimeWindow, start: int, end: int, buffer: int) -> tuple[int, int]:
    """Usable window (with the home buffer to other tours) containing [start, end)."""
    for a, b in tw.usable_windows(buffer):
        if a <= start and end <= b:
            return a, b
    return start, end


# ---- observed stops ----------------------------------------------------------------------------------------
def half_tour_stops(t, outbound: bool) -> list[tuple[str, int, int]]:
    """Stops of a half-tour ordered outwards from the primary: (purpose, inner boundary, outer boundary)."""
    if outbound:
        out, inner = [], t.primary_start
        for s in reversed(t.outbound):
            out.append((s.act, inner, s.start))
            inner = s.start
        return out
    out, inner = [], t.primary_start + t.primary_duration
    for s in t.inbound:
        out.append((s.act, inner, s.start + s.duration))
        inner = s.start + s.duration
    return out


class DaysimModels:
    def __init__(self, time_resolution: int = 30, home_inter: int = 10, tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    def map_pattern(self, vec: tuple) -> str:
        return pattern_label(vec)

    # ---- designs ---------------------------------------------------------------------------------------
    def _walk(self, records: list[DaysimRecord], X: np.ndarray):
        """Replay each observed day in DaySim order (tours by priority, each tour's stops outwards from the primary),
        yielding the design rows of the tour-time, stop-generation and trip-time models."""
        tod = {g: {"Z": [], "X": [], "starts": [], "ends": [], "masks": [], "states": []} for g in GROUPS}
        sg = {"X": [], "y": [], "avail": []}
        tt = {"Z": [], "states": [], "chosen": []}
        for r, xl in zip(records, X):
            xp = person_x(xl, r.ptype)
            female = r.labels.get("sex") == "female"
            tw = TimeWindow()
            purposes = [t.purpose for t in r.tours]
            stops_left = sum(r.model_pattern[NT:])
            stops_so_far = 0
            pattern_stops = {p for p, f in zip(STOP_PURPOSES, r.pattern[NT:]) if f}
            for t, (rank, n, si, sc) in zip(r.tours, tour_ranks(purposes)):
                g = tod_group(t.purpose)
                p_end = t.primary_start + t.primary_duration
                if self.tod.get(g) is not None:
                    d = tod[g]
                    d["Z"].append(self.tod[g].z(z_tod_values(r.labels, r.ptype, t.purpose, rank, n, si, sc,
                                                             r.model_pattern, self.income_levels)))
                    d["X"].append(xl)
                    d["starts"].append(t.primary_start)
                    d["ends"].append(p_end)
                    d["masks"].append(self.tod[g].mask(tw))
                    d["states"].append(tw_state(tw, n - rank - 1, rank > 0, stops_left))
                lo, hi = window_around(tw, t.start, t.end, self.home_inter)
                for direction in DIRECTIONS:
                    outbound = direction == "outbound"
                    stops = half_tour_stops(t, outbound)
                    anchor = t.primary_start if outbound else p_end
                    prev = t.purpose
                    for k in range(len(stops) + 1):
                        choice = stops[k][0] if k < len(stops) else NONE
                        wh = ((anchor - lo) if outbound else (hi - anchor)) / 60.0
                        sg["X"].append(x_stopgen(xp, t.purpose, outbound, k, wh, anchor, n - rank - 1,
                                                 stops_so_far, female))
                        sg["y"].append(choice)
                        sg["avail"].append([c == NONE or c == choice or (c in pattern_stops and c != prev)
                                            for c in STOPGEN_CLASSES])
                        if k == len(stops):
                            break
                        purpose, inner, outer = stops[k]
                        flags = tt_flags(r.ptype, outbound, t.purpose, purpose, r.labels)
                        tt["Z"].append(self.trip_time.z(flags))
                        tt["states"].append((inner, min(lo, outer), max(hi, outer), outbound, flags,
                                             float(k + 1 < len(stops)), n - rank - 1, tw.remaining()))
                        tt["chosen"].append(int(self.grid.period_of(outer) if outbound
                                                else self.grid.end_period_of(outer)))
                        anchor, prev = outer, purpose
                        stops_so_far += 1
                stops_left = max(stops_left - len(t.stops), 0)
                if tw.is_free(t.start, t.end):
                    tw.reserve(t.start, t.end)
        return tod, sg, tt

    # ---- fit -----------------------------------------------------------------------------------------------
    def fit(self, records: list[DaysimRecord], X: np.ndarray, feature_names: list[str]) -> "DaysimModels":
        from composhed.activitysim.models import income_levels

        self.income_levels = income_levels(r.labels.get("hh_income") for r in records)
        XP = np.array([person_x(xl, r.ptype) for r, xl in zip(records, X)]).reshape(len(records), -1)
        self.person_feature_names = [f"ptype_{p}" for p in PERSON_TYPES[1:]] + list(feature_names)

        patterns = enumerate_patterns()
        self.pattern_vectors = {pattern_label(v): v for v in patterns}
        self.pattern_counts = Counter(pattern_label(r.model_pattern) for r in records)
        self.pattern_merged = sum(r.pattern != r.model_pattern for r in records)
        self.pattern = AltAttrMNL({k: pattern_attrs(v) for k, v in self.pattern_vectors.items()}, PATTERN_ATTR_NAMES,
                                  maxiter=300, inter_attrs={k: pattern_inter_attrs(v) for k, v in self.pattern_vectors.items()},
                                  inter_names=PATTERN_INTER_NAMES).fit(
            XP, [pattern_label(r.model_pattern) for r in records], list(self.pattern_vectors),
            feature_names=self.person_feature_names)
        print(f"  DAY-1 day pattern: {len(patterns)} alternatives; {self.pattern_merged} persons mapped to the "
              f"nearest feasible pattern")

        self.exact = {}
        for p in TOUR_PURPOSES:
            idx = [i for i, r in enumerate(records) if r.counts[p] > 0]
            y = [min(records[i].counts[p], MAX_TOURS_PER_PURPOSE) for i in idx]
            if len(set(y)) > 1:
                Xe = np.array([x_exact(XP[i], records[i].model_pattern) for i in idx])
                self.exact[p] = MNL(maxiter=300).fit(Xe, y, classes=list(range(1, MAX_TOURS_PER_PURPOSE + 1)))
            else:
                self.exact[p] = y[0] if y else 1

        self.grid = PeriodGrid.uniform(self.time_resolution)
        self.tod = {g: TourTOD(self.grid, SPECS[g], buffer=self.home_inter, maxiter=self.tod_maxiter, dynamic="daysim")
                    for g in GROUPS}
        self.trip_time = TripTimeModel(self.grid)
        tod, sg, tt = self._walk(records, X)
        self.indep = {}
        for g in GROUPS:
            d = tod[g]
            if not d["starts"]:
                self.tod[g] = None
                continue
            print(f"  DAY-2 tour time [{g}]: {len(d['starts']):,} tours × {len(self.tod[g].ps)} period pairs", flush=True)
            self.tod[g].fit(np.array(d["Z"]), np.array(d["starts"]), np.array(d["ends"]), np.array(d["masks"]),
                            states=d["states"])
            self.indep[g] = IndependentTiming().fit(np.array(d["X"]), np.array(d["starts"]), np.array(d["ends"]))
        self.stopgen = MNL(maxiter=300).fit(np.array(sg["X"]), sg["y"], classes=STOPGEN_CLASSES,
                                            avail=np.array(sg["avail"]))
        print(f"  DAY-5 stop generation: {len(sg['y']):,} choices; DAY-6 trip time: {len(tt['chosen']):,} stops",
              flush=True)
        if tt["chosen"]:
            self.trip_time.fit(np.array(tt["Z"]), tt["states"], np.array(tt["chosen"]))
        else:
            self.trip_time = None
        return self

    # ---- validate -------------------------------------------------------------------------------------------
    def validate(self, records: list[DaysimRecord], X: np.ndarray) -> dict:
        XP = np.array([person_x(xl, r.ptype) for r, xl in zip(records, X)]).reshape(len(records), -1)
        out = {"DAY-1 day pattern": self.pattern.validate(XP, [pattern_label(r.model_pattern) for r in records])}
        out["DAY-1 day pattern"]["shares"] = dict(sorted(
            out["DAY-1 day pattern"]["shares"].items(), key=lambda kv: -kv[1]["obs"])[:8])
        for p, m in self.exact.items():
            if isinstance(m, MNL):
                idx = [i for i, r in enumerate(records) if r.counts[p] > 0]
                if idx:
                    out[f"DAY-1b exact tours {p}"] = m.validate(
                        np.array([x_exact(XP[i], records[i].model_pattern) for i in idx]),
                        [min(records[i].counts[p], MAX_TOURS_PER_PURPOSE) for i in idx])
        tod, sg, tt = self._walk(records, X)
        for g in GROUPS:
            d = tod[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"DAY-2 tour time {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                                   np.array(d["ends"]), np.array(d["masks"]),
                                                                   d["states"])
        if sg["y"]:
            out["DAY-5 stop generation"] = self.stopgen.validate(np.array(sg["X"]), sg["y"], np.array(sg["avail"]))
        if tt["chosen"] and self.trip_time is not None:
            out["DAY-6 trip time"] = self.trip_time.validate(np.array(tt["Z"]), tt["states"], np.array(tt["chosen"]))
        return out
