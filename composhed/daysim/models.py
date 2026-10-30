"""DaySim-aligned component models (SPECIFICATION §6.2), after the DaySim source (RSGInc/DaySim, Default models).

| Component | DaySim model | Model here |
|---|---|---|
| day pattern: which tour purposes and which stop purposes the day has | ``IndividualPersonDayPatternModel`` | ``AltAttrMNL`` over all 1,324 feasible patterns with DaySim's terms: tour and stop constants; tour–stop, tour–tour, stop–stop and tour-count × stop-count constants; the person terms on each purpose's presence; one person vector × log(tours) for 2+ tour purposes and one × log(stops) for 2+ stop purposes; the tour and stop components; logsum (accessibility) terms; DaySim's worker / student availability |
| exact number of tours per purpose (1, 2, 3+) | ``PersonExactNumberOfToursModel`` | ``SpecLogit`` per purpose: the person terms and the day's tours and stops on 2 and 3+ (shared), constants and logsum per alternative |
| tour time: arrival and departure at the primary destination | ``WorkTourTimeModel`` / ``SchoolTourTimeModel`` / ``OtherHomeBasedTourTimeModel`` | ``TourTOD`` over half-hour period pairs with the DaySim spec (``tod_spec``) and time-window terms |
| intermediate stop generation (next stop's purpose, or none) | ``IntermediateStopGenerationModel`` | ``SpecLogit`` with DaySim's terms, each on the stop purposes the source gives it; purposes available only if the pattern has them |
| trip time (period of each stop's outer boundary) | ``TripTimeModel`` | ``TripTimeModel`` below |
| AB-2 timing | — | ``IndependentTiming`` per group |

Person variables come from ``sourcevars`` (the DaySim terms with an NTS
counterpart). Saturday and Sunday terms are the only additions.
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp

from composhed.activitysim.persontype import PERSON_TYPES
from composhed.common.choice import AltAttrMNL, ConditionalLogit
from composhed.common.periods import PeriodGrid
from composhed.common.spec import Spec, SpecLogit
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD, tw_state
from composhed.daysim.data import (
    MAX_TOURS_PER_PURPOSE,
    STOP_PURPOSES,
    TOUR_PURPOSES,
    DaysimRecord,
    available,
    enumerate_patterns,
    is_student,
    is_worker,
    pattern_label,
    tod_group,
)
from composhed.daysim.sourcevars import PERSON_TERMS, DaysimVars
from composhed.daysim.tod_spec import SPECS, priority_flags

GROUPS = ["work", "education", "nonmandatory"]
DIRECTIONS = ["outbound", "inbound"]
MAX_STOPS = 4  # per half-tour
NONE = "none"
STOPGEN_CLASSES = [NONE] + STOP_PURPOSES
DAY = 1440
NT = len(TOUR_PURPOSES)
# DaySim purposes behind the NTS ones (data.PRIORITY): visit takes DaySim's meal slot, leisure social / recreation.
DAYSIM_NAME = {"work": "work", "education": "school", "escort": "escort", "other": "pb", "shop": "shop",
               "visit": "meal", "leisure": "social"}


# ---- day pattern -------------------------------------------------------------------------------------
PATTERN_X = PERSON_TERMS + ["mixed_density_child_5_15", "saturday", "sunday"]
COMPONENT_TERMS = ["cars_per_driver", "mixed_density", "mixed_density_child_5_15", "logsum"]  # terms 1401–1406


def pattern_x(dv: DaysimVars, labels: dict, ptype: str) -> np.ndarray:
    v = dv.person(labels, ptype)
    v["mixed_density_child_5_15"] = v["mixed_density"] * v["child_5_15"]
    return np.array([v[k] for k in PATTERN_X])


def pattern_attrs(vec: tuple) -> list[float]:
    """Constants of a pattern: tour and stop ASCs (100p, 100p+1), tour–stop (1000+; DaySim pairs every tour purpose
    with every stop purpose but social/recreation), tour–tour (1100+), stop–stop (1200+), tours × stops (1300+)."""
    tours, stops = vec[:NT], vec[NT:]
    nt, ns = sum(tours), min(sum(stops), 3)
    tour_stop = [float(t and s) for t in tours for s in stops[:-1]]
    tour_tour = [float(tours[i] and tours[j]) for i in range(NT) for j in range(i + 1, NT)]
    stop_stop = [float(stops[i] and stops[j]) for i in range(len(stops)) for j in range(i + 1, len(stops))]
    counts = [float(nt == a and ns == b) for a in range(1, 4) for b in range(1, 4)]
    return [float(t) for t in tours] + [float(s) for s in stops] + tour_stop + tour_tour + stop_stop + counts


PATTERN_ATTR_NAMES = ([f"tour_{p}" for p in TOUR_PURPOSES] + [f"stop_{p}" for p in STOP_PURPOSES]
                      + [f"tour_{p}*stop_{q}" for p in TOUR_PURPOSES for q in STOP_PURPOSES[:-1]]
                      + [f"tour_{TOUR_PURPOSES[i]}*tour_{TOUR_PURPOSES[j]}" for i in range(NT) for j in range(i + 1, NT)]
                      + [f"stop_{STOP_PURPOSES[i]}*stop_{STOP_PURPOSES[j]}" for i in range(5) for j in range(i + 1, 5)]
                      + [f"{a}tours_{b}stops" for a in range(1, 4) for b in range(1, 4)])


def pattern_inter_attrs(vec: tuple) -> list[float]:
    """What the person variables act on: each purpose's presence (tour or stop; DaySim components 1–7), each tour and
    stop (logsum terms 100p+29 / 100p+31), log(tour purposes) and log(stop purposes) when 2+ (the shared multi-tour
    and multi-stop components), any non-mandatory tour and any stop (the tour and stop components)."""
    tours, stops = vec[:NT], vec[NT:]
    present = [float(t or (p in STOP_PURPOSES and stops[STOP_PURPOSES.index(p)]))
               for p, t in zip(TOUR_PURPOSES, tours)]
    nt, ns = sum(tours), sum(stops)
    return (present + [float(t) for t in tours] + [float(s) for s in stops]
            + [np.log(nt) if nt > 1 else 0.0, np.log(ns) if ns > 1 else 0.0, float(sum(tours[2:]) > 0), float(ns > 0)])


PATTERN_INTER_NAMES = ([f"present_{p}" for p in TOUR_PURPOSES] + [f"tour_{p}" for p in TOUR_PURPOSES]
                       + [f"stop_{p}" for p in STOP_PURPOSES] + ["log_tours", "log_stops", "any_nm_tour", "any_stop"])


def pattern_b_mask() -> np.ndarray:
    mask = np.zeros((len(PATTERN_X), len(PATTERN_INTER_NAMES)), dtype=bool)
    xi, ci = PATTERN_X.index, PATTERN_INTER_NAMES.index
    for p in TOUR_PURPOSES:
        for t in PERSON_TERMS + ["saturday", "sunday"]:
            mask[xi(t), ci(f"present_{p}")] = True
        mask[xi("logsum"), ci(f"tour_{p}")] = True
    for q in STOP_PURPOSES:
        mask[xi("logsum"), ci(f"stop_{q}")] = True
    for c in ("log_tours", "log_stops"):
        for t in PERSON_TERMS:
            mask[xi(t), ci(c)] = True
    for c in ("any_nm_tour", "any_stop"):
        for t in COMPONENT_TERMS:
            mask[xi(t), ci(c)] = True
    return mask


def pattern_avail(records_or_flags, vectors: list[tuple]) -> np.ndarray:
    """DaySim availability per person: (worker, student) flags → allowed patterns."""
    return np.array([[available(v, w, s) for v in vectors] for w, s in records_or_flags], dtype=bool)


# ---- exact number of tours --------------------------------------------------------------------------
def exact_spec() -> Spec:
    spec = Spec(alternatives=[1, 2, 3], rows=[])
    for t in PERSON_TERMS + [f"tours_{p}" for p in TOUR_PURPOSES] + [f"stops_{q}" for q in STOP_PURPOSES] \
            + ["saturday", "sunday"]:
        spec.add(t, t, {2: f"b_{t}", 3: f"b_{t}"})
    for a in (2, 3):
        spec.add(f"asc_{a}", "1", {a: f"asc_{a}"})
        spec.add(f"logsum_{a}", "logsum", {a: f"logsum_{a}"})
    return spec


def exact_env(person: list[dict], vecs: list[tuple]) -> dict:
    n = len(person)
    env = {k: np.array([p[k] for p in person], dtype=float).reshape(n, 1) for k in PERSON_TERMS + ["saturday", "sunday"]}
    for i, p in enumerate(TOUR_PURPOSES):
        env[f"tours_{p}"] = np.array([v[i] for v in vecs], dtype=float).reshape(n, 1)
    for i, q in enumerate(STOP_PURPOSES):
        env[f"stops_{q}"] = np.array([v[NT + i] for v in vecs], dtype=float).reshape(n, 1)
    return env


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
TOD_BANDS = {"t7_9": (7, 9), "t9_11": (9, 11), "t11_13": (11, 13), "t13_15": (13, 15), "t15_17": (15, 17),
             "t19_21": (19, 21), "t21_23": (21, 23)}
TOUR_FLAGS = [f"{DAYSIM_NAME[p]}_tour" for p in TOUR_PURPOSES]
STOPGEN_VARS = (["one_trip", "two_trips", "three_trips", "four_trips", "five_trips", "origin_half", "dest_half",
                 "home_based_tours", "simulated_tours", "before_mandatory", "remaining_tours", "duration"]
                + TOUR_FLAGS + [f"sim_{DAYSIM_NAME[q]}_stops" for q in STOP_PURPOSES] + list(TOD_BANDS)
                + ["t23_7", "adult_male", "retired", "driving_age_student", "nonworking_adult", "child_5_15",
                   "child_under_5", "saturday", "sunday"])


def stopgen_vars(person: dict, tour_purpose: str, outbound: bool, k: int, window_hours: float, anchor: float,
                 n_tours: int, rank: int, sim_stops: Counter) -> dict:
    """``IntermediateStopGenerationModel`` inputs for the stop after ``k`` stops of this half-tour (trip ``k + 1``).
    The time-of-day flags use the tour's arrival (outbound) or departure (inbound) at the primary destination."""
    h = anchor / 60.0
    trip = k + 1
    v = {name: float(trip == i) for i, name in enumerate(["one_trip", "two_trips", "three_trips", "four_trips",
                                                           "five_trips"], start=1)}
    v |= {"origin_half": float(outbound), "dest_half": float(not outbound), "home_based_tours": float(n_tours),
          "simulated_tours": float(rank > 0),
          "before_mandatory": float(outbound and tour_purpose in ("work", "education")),
          "remaining_tours": float(n_tours - rank - 1), "duration": float(window_hours)}
    v |= {f"{DAYSIM_NAME[p]}_tour": float(tour_purpose == p) for p in TOUR_PURPOSES}
    v |= {f"sim_{DAYSIM_NAME[q]}_stops": float(sim_stops[q]) for q in STOP_PURPOSES}
    v |= {name: float(a <= h < b) for name, (a, b) in TOD_BANDS.items()}
    v["t23_7"] = float(h >= 23 or h < 3)  # DaySim's day starts at 03:00: "from 11 PM" covers 23:00–03:00
    v |= {k_: person[k_] for k_ in ("adult_male", "retired", "driving_age_student", "nonworking_adult", "child_5_15",
                                    "child_under_5", "saturday", "sunday")}
    return v


def stopgen_spec() -> Spec:
    """``IntermediateStopGenerationModel`` terms, each on the alternatives DaySim gives it (term numbers in labels).
    Dropped: transit / walk / bike / HOV tour mode, not-home-based tours, parcel connectivity, households with
    children, one-person households, part-time workers (no NTS counterpart)."""
    spec = Spec(alternatives=STOPGEN_CLASSES, rows=[])
    for i, n in enumerate(["two_trips", "three_trips", "four_trips", "five_trips"]):
        spec.add(f"{1 + i}", f"{n} * origin_half", {NONE: f"none_{n}_origin"})
        spec.add(f"{5 + i}", f"{n} * dest_half", {NONE: f"none_{n}_destination"})
    for term, var in ((9, "home_based_tours"), (10, "simulated_tours"), (12, "before_mandatory"), (40, "school_tour")):
        spec.add(str(term), var, {NONE: f"none_{var}"})
    ws = "work_tour + school_tour"
    day_vars = {"escort": "escort", "other": "pb", "shop": "shop", "visit": "meal", "leisure": "social"}
    tour_terms = {  # stop purpose -> tour-purpose flags it has terms for
        "escort": [ws, "escort_tour"],
        "other": [ws, "escort_tour", "pb_tour", "shop_tour", "meal_tour", "social_tour"],
        "shop": [ws, "escort_tour", "pb_tour", "shop_tour", "meal_tour", "social_tour"],
        "visit": ["work_tour", "school_tour", "escort_tour", "pb_tour", "shop_tour", "meal_tour", "social_tour"],
        "leisure": [ws, "escort_tour", "pb_tour", "shop_tour", "meal_tour", "social_tour"],
    }
    tod_terms = {
        "escort": ["t7_9", "t9_11 + t11_13 + t13_15 + t15_17"],
        "other": ["t7_9 + t19_21 + t21_23 + t23_7", "t9_11 + t11_13 + t13_15 + t15_17"],
        "shop": ["t7_9 + t21_23 + t23_7", "t11_13 + t13_15 + t15_17"],
        "visit": ["t7_9 + t23_7", "t11_13 + t13_15", "t19_21"],
        "leisure": ["t7_9 + t23_7", "t11_13 + t13_15 + t15_17"],
    }
    for q in STOP_PURPOSES:
        d = day_vars[q]
        for f in tour_terms[q]:
            spec.add(f"{q}:{f}", f, {q: f"{d}_{f.replace(' + ', '_or_')}"})
        for var in ("origin_half", f"sim_{d}_stops", "remaining_tours", "duration"):
            spec.add(f"{q}:{var}", var, {q: f"{d}_{var}"})
        for j, e in enumerate(tod_terms[q]):
            spec.add(f"{q}:tod{j}", e, {q: f"{d}_tod_{e.replace(' + ', '_or_')}"})
        for day in ("saturday", "sunday"):
            spec.add(f"{q}:{day}", day, {q: f"{d}_{day}"})
    spec.add("escort:one_trip", "one_trip", {"escort": "escort_one_trip"})
    spec.add("visit:one_trip", "one_trip", {"visit": "meal_one_trip"})
    spec.add("visit:ptype_a", "retired + driving_age_student", {"visit": "meal_pt_retired_driving"})
    spec.add("visit:ptype_b", "nonworking_adult + child_5_15 + child_under_5", {"visit": "meal_nonworker_children"})
    spec.add("work stop: adult male", "adult_male", {})  # term 175 is on work stops, which do not exist here
    spec.rows = [r for r in spec.rows if r.cells]
    return spec


def rows_env(rows: list[dict], names: list[str]) -> dict:
    n = len(rows)
    return {k: np.array([r[k] for r in rows], dtype=float).reshape(n, 1) for k in names}


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
    def __init__(self, time_resolution: int = 30, home_inter: int = 1, tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    def flags(self, labels: dict, ptype: str) -> tuple[bool, bool]:
        return is_worker(labels), is_student(labels, ptype)

    # ---- designs ---------------------------------------------------------------------------------------
    def _walk(self, records: list[DaysimRecord]):
        """Replay each observed day in DaySim order (tours by priority, each tour's stops outwards from the primary),
        yielding the design rows of the tour-time, stop-generation and trip-time models."""
        tod = {g: {"Z": [], "X": [], "starts": [], "ends": [], "masks": [], "states": []} for g in GROUPS}
        sg = {"rows": [], "y": [], "avail": []}
        tt = {"Z": [], "states": [], "chosen": []}
        for r in records:
            person = self.dv.person(r.labels, r.ptype)
            tw = TimeWindow()
            purposes = [t.purpose for t in r.tours]
            stops_left = sum(r.pattern[NT:])
            sim_stops: Counter = Counter()
            pattern_stops = {p for p, f in zip(STOP_PURPOSES, r.pattern[NT:]) if f}
            for t, (rank, n, si, sc) in zip(r.tours, tour_ranks(purposes)):
                g = tod_group(t.purpose)
                p_end = t.primary_start + t.primary_duration
                if self.tod.get(g) is not None:
                    d = tod[g]
                    d["Z"].append(self.tod[g].z(z_tod_values(r.labels, r.ptype, t.purpose, rank, n, si, sc,
                                                             r.pattern, self.income_levels)))
                    d["X"].append(self.xl_of(r))
                    d["starts"].append(t.primary_start)
                    d["ends"].append(p_end)
                    d["masks"].append(self.tod[g].mask(tw))
                    d["states"].append(tw_state(tw, n - rank - 1, rank > 0, stops_left))
                lo, hi = window_around(tw, t.start, t.end, self.home_inter)
                for direction in DIRECTIONS:
                    outbound = direction == "outbound"
                    stops = half_tour_stops(t, outbound)
                    anchor0 = t.primary_start if outbound else p_end
                    anchor = anchor0
                    for k in range(len(stops) + 1):
                        choice = stops[k][0] if k < len(stops) else NONE
                        wh = ((anchor - lo) if outbound else (hi - anchor)) / 60.0
                        sg["rows"].append(stopgen_vars(person, t.purpose, outbound, k, wh, anchor0, n, rank, sim_stops))
                        sg["y"].append(choice)
                        sg["avail"].append([c == NONE or c == choice or c in pattern_stops for c in STOPGEN_CLASSES])
                        if k == len(stops):
                            break
                        purpose, inner, outer = stops[k]
                        flags = tt_flags(r.ptype, outbound, t.purpose, purpose, r.labels)
                        tt["Z"].append(self.trip_time.z(flags))
                        tt["states"].append((inner, min(lo, outer), max(hi, outer), outbound, flags,
                                             float(k + 1 < len(stops)), n - rank - 1, tw.remaining()))
                        tt["chosen"].append(int(self.grid.period_of(outer) if outbound
                                                else self.grid.end_period_of(outer)))
                        anchor = outer
                        sim_stops[purpose] += 1
                stops_left = max(stops_left - len(t.stops), 0)
                if tw.is_free(t.start, t.end):
                    tw.reserve(t.start, t.end)
        return tod, sg, tt

    def xl_of(self, r) -> np.ndarray:
        return self._xl[id(r)]

    # ---- fit -----------------------------------------------------------------------------------------------
    def fit(self, records: list[DaysimRecord], X: np.ndarray, feature_names: list[str]) -> "DaysimModels":
        self.dv = DaysimVars([r.labels for r in records])
        self._xl = {id(r): x for r, x in zip(records, X)}
        self.income_levels = sorted(self.dv.income_rank, key=self.dv.income_rank.get)

        patterns = enumerate_patterns()
        self.pattern_vectors = {pattern_label(v): v for v in patterns}
        est = [r for r in records if r.in_choice_set]
        self.pattern_left_out = len(records) - len(est)
        XP = np.array([pattern_x(self.dv, r.labels, r.ptype) for r in est])
        self.pattern = AltAttrMNL({k: pattern_attrs(v) for k, v in self.pattern_vectors.items()}, PATTERN_ATTR_NAMES,
                                  maxiter=300,
                                  inter_attrs={k: pattern_inter_attrs(v) for k, v in self.pattern_vectors.items()},
                                  inter_names=PATTERN_INTER_NAMES, b_mask=pattern_b_mask()).fit(
            XP, [pattern_label(r.pattern) for r in est], list(self.pattern_vectors),
            avail=pattern_avail([self.flags(r.labels, r.ptype) for r in est], patterns), feature_names=PATTERN_X)
        print(f"  DAY-1 day pattern: {len(patterns)} alternatives, {self.pattern.n_params_} parameters; "
              f"{self.pattern_left_out} days outside DaySim's choice set left out", flush=True)

        self.exact = {}
        for p in TOUR_PURPOSES:
            idx = [r for r in records if r.counts[p] > 0]
            y = [min(r.counts[p], MAX_TOURS_PER_PURPOSE) for r in idx]
            if len(set(y)) > 1:
                env = exact_env([self.dv.person(r.labels, r.ptype) for r in idx], [r.pattern for r in idx])
                self.exact[p] = SpecLogit(exact_spec(), maxiter=300).fit(env, len(idx), y)
            else:
                self.exact[p] = y[0] if y else 1

        self.grid = PeriodGrid.uniform(self.time_resolution)
        self.tod = {g: TourTOD(self.grid, SPECS[g], buffer=self.home_inter, maxiter=self.tod_maxiter, dynamic="daysim")
                    for g in GROUPS}
        self.trip_time = TripTimeModel(self.grid)
        tod, sg, tt = self._walk(records)
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
        self.stopgen = SpecLogit(stopgen_spec(), maxiter=300).fit(rows_env(sg["rows"], STOPGEN_VARS), len(sg["y"]),
                                                                  sg["y"], np.array(sg["avail"]))
        print(f"  DAY-5 stop generation: {len(sg['y']):,} choices, {self.stopgen.n_params} parameters; "
              f"DAY-6 trip time: {len(tt['chosen']):,} stops", flush=True)
        if tt["chosen"]:
            self.trip_time.fit(np.array(tt["Z"]), tt["states"], np.array(tt["chosen"]))
        else:
            self.trip_time = None
        self._xl = {}
        return self

    def n_params(self) -> dict:
        return {"pattern": int(self.pattern.n_params_),
                "exact": {p: m.n_params for p, m in self.exact.items() if isinstance(m, SpecLogit)},
                "tour_time": {g: getattr(t.clogit, "n_params_", None) for g, t in self.tod.items() if t is not None},
                "stop_generation": self.stopgen.n_params,
                "trip_time": getattr(self.trip_time.clogit, "n_params_", None) if self.trip_time else None}

    # ---- validate -------------------------------------------------------------------------------------------
    def validate(self, records: list[DaysimRecord], X: np.ndarray) -> dict:
        self._xl = {id(r): x for r, x in zip(records, X)}
        est = [r for r in records if r.in_choice_set]
        XP = np.array([pattern_x(self.dv, r.labels, r.ptype) for r in est]).reshape(len(est), -1)
        out = {"DAY-1 day pattern": self.pattern.validate(
            XP, [pattern_label(r.pattern) for r in est],
            pattern_avail([self.flags(r.labels, r.ptype) for r in est], list(self.pattern_vectors.values())))}
        out["DAY-1 day pattern"]["shares"] = dict(sorted(
            out["DAY-1 day pattern"]["shares"].items(), key=lambda kv: -kv[1]["obs"])[:8])
        out["DAY-1 day pattern"]["n_params"] = int(self.pattern.n_params_)
        out["DAY-1 day pattern"]["left_out"] = len(records) - len(est)
        for p, m in self.exact.items():
            if isinstance(m, SpecLogit):
                idx = [r for r in records if r.counts[p] > 0]
                if idx:
                    env = exact_env([self.dv.person(r.labels, r.ptype) for r in idx], [r.pattern for r in idx])
                    out[f"DAY-1b exact tours {p}"] = m.validate(env, len(idx),
                                                                [min(r.counts[p], MAX_TOURS_PER_PURPOSE) for r in idx])
        tod, sg, tt = self._walk(records)
        for g in GROUPS:
            d = tod[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"DAY-2 tour time {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                                   np.array(d["ends"]), np.array(d["masks"]),
                                                                   d["states"])
        if sg["y"]:
            out["DAY-5 stop generation"] = self.stopgen.validate(rows_env(sg["rows"], STOPGEN_VARS), len(sg["y"]),
                                                                 sg["y"], np.array(sg["avail"]))
        if tt["chosen"] and self.trip_time is not None:
            out["DAY-6 trip time"] = self.trip_time.validate(np.array(tt["Z"]), tt["states"], np.array(tt["chosen"]))
        self._xl = {}
        return out
