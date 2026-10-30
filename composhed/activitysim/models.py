"""ActivitySim-aligned component models (SPECIFICATION §7.2), from the ``prototype_mtc`` v1.6.0 configs.

| Component | Model |
|---|---|
| person type | rules (``persontype.py``) |
| CDAP M / N / H | ``SpecLogit`` on ``cdap_indiv_and_hhsize1.csv`` (one model, person-type terms) |
| mandatory tour frequency | ``SpecLogit`` on ``mandatory_tour_frequency.csv`` (one model, its availability rows) |
| mandatory tour scheduling | ``TourTOD`` with the MTC spec (``tod_spec``) and timetable terms |
| non-mandatory tour frequency | ``SpecLogit`` on ``non_mandatory_tour_frequency.csv`` with one coefficient file per person type (``interaction_simulate`` over the 48 alternatives without eat-out), after mandatory scheduling; then the extension lookup |
| non-mandatory tour scheduling | ``TourTOD`` with the MTC non-mandatory spec |
| stop frequency | ``SpecLogit`` on ``stop_frequency_<purpose>.csv`` per tour purpose (school tours of university students use ``univ``) |
| trip purpose / trip scheduling | probability lookups estimated on NTS |

The spec files are vendored in ``specs/``. Rows whose variables NTS lacks are
dropped (``sourcevars``); every remaining coefficient is re-estimated, except
those the files constrain. The only rows added are Saturday and Sunday terms in
CDAP, non-mandatory tour frequency and stop frequency.
"""

from collections import Counter
from pathlib import Path

import numpy as np

from composhed.activitysim.data import (
    MAX_EXTRA,
    MAX_STOPS,
    NM_ORDER,
    AsimRecord,
    AsimTour,
    mtf_tours,
    nm_label,
)
from composhed.activitysim.sourcevars import (
    INCOME_K,
    MTC_SEGMENT,
    PERSON_VARS,
    PersonVars,
    subset,
)
from composhed.activitysim.tod_spec import SPECS
from composhed.common.periods import PeriodGrid
from composhed.common.spec import SpecLogit, read_spec
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import IndependentTiming, TourTOD, tw_state
from composhed.data import DISC_ACTS

SPEC_DIR = Path(__file__).parent / "specs"
GROUPS = ["work", "school", "nonmandatory"]
STOP_ACTS = sorted(DISC_ACTS)
DAY = 1440
STOP_ALTS = [f"{o}out_{i}in" for o in range(MAX_STOPS + 1) for i in range(MAX_STOPS + 1)]
NM_VARS = {"num_mand", "log_max_window", "escort", "shopping", "othmaint", "othdiscr", "eatout", "social", "tot_tours"}
TOUR_VARS = {"start", "end", "duration", "num_work_tours", "num_school_tours", "num_univ_tours", "num_escort_tours",
             "num_shop_tours", "num_maint_tours", "num_eatout_tours", "num_social_tours", "is_social"}
STOP_SEGMENTS = ["work", "school", "univ", "escort", "shopping", "othmaint", "othdiscr", "social"]


def tod_group(purpose: str) -> str:
    return purpose if purpose in ("work", "school") else "nonmandatory"


def phase(purpose: str) -> str:
    return "mandatory" if purpose in ("work", "school") else "nonmandatory"


# ---- specs --------------------------------------------------------------------------------------------
def cdap_spec():
    spec = read_spec(str(SPEC_DIR / "cdap_indiv_and_hhsize1.csv"), coefficients=str(SPEC_DIR / "cdap_coefficients.csv"))
    for day in ("saturday", "sunday"):
        spec.add(f"{day}", day, {"M": f"coef_{day}_M", "N": f"coef_{day}_N"})
    return spec.restrict(PERSON_VARS)


def mtf_spec():
    return read_spec(str(SPEC_DIR / "mandatory_tour_frequency.csv"),
                     coefficients=str(SPEC_DIR / "mandatory_tour_frequency_coefficients.csv")).restrict(PERSON_VARS)


def nm_alternatives() -> list[tuple]:
    """``non_mandatory_tour_frequency_alternatives.csv`` without eat-out (NTS has none): 48 purpose-count vectors."""
    import csv

    out = []
    with open(SPEC_DIR / "non_mandatory_tour_frequency_alternatives.csv") as f:
        for r in csv.DictReader(f):
            if int(r["eatout"]) == 0:
                out.append(tuple(int(r[p]) for p in NM_ORDER))
    return out


def nmtf_spec(segment: str, labels: list[str]):
    spec = read_spec(str(SPEC_DIR / "non_mandatory_tour_frequency.csv"), columns=[segment],
                     coefficients=str(SPEC_DIR / f"non_mandatory_tour_frequency_coefficients_{segment}.csv"))
    spec.alternatives = list(labels)
    for p in NM_ORDER:
        for day in ("saturday", "sunday"):
            spec.add(f"{day}_{p}", f"{day} * ({p} >= 1)", {"*": f"coef_{day}_{p}"})
    return spec.restrict(PERSON_VARS | NM_VARS)


def stopfreq_spec(segment: str):
    spec = read_spec(str(SPEC_DIR / f"stop_frequency_{segment}.csv"),
                     coefficients=str(SPEC_DIR / f"stop_frequency_coefficients_{segment}.csv"))
    any_stop = [a for a in STOP_ALTS if a != "0out_0in"]
    for day in ("saturday", "sunday"):
        spec.add(day, day, {a: f"coef_{day}" for a in any_stop})
    return spec.restrict(PERSON_VARS | TOUR_VARS)


def nm_attr_env(vectors: list[tuple]) -> dict:
    A = np.array(vectors, dtype=float)
    env = {p: A[:, i][None, :] for i, p in enumerate(NM_ORDER)}
    env["eatout"] = np.zeros((1, len(vectors)))
    env["tot_tours"] = A.sum(axis=1)[None, :]
    return env


def tour_env(purposes: list[str], starts, ends, counts: list[Counter], univ: list[bool]) -> dict:
    """Stop-frequency tour variables: ActivitySim hour indices (5–23) and the person's tours by purpose."""
    n = len(purposes)
    col = lambda v: np.asarray(v, dtype=float).reshape(n, 1)
    s = np.clip(np.floor(np.asarray(starts, float) / 60.0), 5, 23)
    e = np.clip(np.floor(np.asarray(ends, float) / 60.0), 5, 23)
    return {"start": col(s), "end": col(e), "duration": col(e - s),
            "num_work_tours": col([c["work"] for c in counts]),
            "num_school_tours": col([0 if u else c["school"] for c, u in zip(counts, univ)]),
            "num_univ_tours": col([c["school"] if u else 0 for c, u in zip(counts, univ)]),
            "num_escort_tours": col([c["escort"] for c in counts]),
            "num_shop_tours": col([c["shopping"] for c in counts]),
            "num_maint_tours": col([c["othmaint"] for c in counts]),
            "num_social_tours": col([c["social"] for c in counts]),
            "num_eatout_tours": np.zeros((n, 1)),
            "is_social": col([p == "social" for p in purposes])}


def stop_segment(purpose: str, ptype: str) -> str:
    return "univ" if purpose == "school" and ptype == "UNIV" else purpose


def z_tod(spec_z, labels: dict, ptype: str, purpose: str, ctx: dict, income_k: float, segmented: bool = True):
    """Scheduling covariates, named as in ``tod_spec.Z_NAMES``.

    ``ctx``: ``idx`` / ``n_in_phase`` (tour number among the mandatory or among the
    non-mandatory tours: MTC ``tour_num`` / ``tour_count``), ``same_idx`` /
    ``same_count`` (``tour_type_num`` / ``tour_type_count``), ``mtf``, ``n_mand``,
    ``n_escort``, ``n_non_escort``. Income is MTC's ``income_in_thousands``
    (``sourcevars``); "high income" is above 100.
    """
    v = {"const": 1.0, "is_escort": float(purpose == "escort"), "is_not_escort": float(purpose != "escort"),
         "income": float(income_k), "high_income": float(income_k > 100),
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
    def __init__(self, time_resolution: int = 60, segmented: bool = True, home_inter: int = 1,
                 tod_maxiter: int = 200) -> None:
        self.time_resolution = time_resolution
        self.segmented = segmented
        self.home_inter = home_inter
        self.tod_maxiter = tod_maxiter

    # ---- variables -------------------------------------------------------------------------------
    def person_env(self, label_rows: list[dict]) -> dict:
        return self.pv(label_rows)

    def income_k(self, labels: dict) -> float:
        return INCOME_K[min(self.pv.income_rank.get(str(labels.get("hh_income")), 2), len(INCOME_K) - 1)]

    def z(self, group: str, labels: dict, ptype: str, purpose: str, ctx: dict) -> np.ndarray:
        return z_tod(self.tod[group].z, labels, ptype, purpose, ctx, self.income_k(labels), self.segmented)

    def nm_segment(self, ptype: str) -> str:
        return MTC_SEGMENT[ptype] if self.segmented else "PTYPE_FULL"

    def nm_env(self, env: dict, n_mand, log_window) -> dict:
        n = len(env["ptype"])
        return env | self.nm_attrs | {"num_mand": np.asarray(n_mand, float).reshape(n, 1),
                                      "log_max_window": np.asarray(log_window, float).reshape(n, 1)}

    # ---- designs ------------------------------------------------------------------------------------
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
                if tw.is_free(t.start, t.end):
                    tw.reserve(t.start, t.end)
        return out

    def _nmtf_data(self, records: list[AsimRecord], env: dict) -> dict:
        """Per segment: (env, n, chosen labels) for the M and N days."""
        out = {}
        idx = [i for i, r in enumerate(records) if r.cdap in ("M", "N")]
        for seg in sorted({self.nm_segment(records[i].ptype) for i in idx}):
            sel = [i for i in idx if self.nm_segment(records[i].ptype) == seg]
            n_mand = [len(mtf_tours(records[i].mtf)) if records[i].mtf else 0 for i in sel]
            win = [residual_window([(t.start, t.end) for t in records[i].tours if phase(t.purpose) == "mandatory"],
                                   self.home_inter) for i in sel]
            out[seg] = (self.nm_env(subset(env, sel), n_mand, win), len(sel), [nm_label(records[i].nm) for i in sel])
        return out

    def _stopfreq_data(self, records: list[AsimRecord], env: dict) -> dict:
        rows: dict = {}
        for i, r in enumerate(records):
            counts = Counter(t.purpose for t in r.tours)
            for t in r.tours:
                rows.setdefault(stop_segment(t.purpose, r.ptype), []).append(
                    (i, t.purpose, t.start, t.end, counts, r.ptype == "UNIV", stop_alt(len(t.outbound), len(t.inbound))))
        out = {}
        for seg, rs in rows.items():
            e = subset(env, [x[0] for x in rs]) | tour_env([x[1] for x in rs], [x[2] for x in rs], [x[3] for x in rs],
                                                          [x[4] for x in rs], [x[5] for x in rs])
            out[seg] = (e, len(rs), [x[6] for x in rs])
        return out

    # ---- fit -------------------------------------------------------------------------------------------
    def fit(self, records: list[AsimRecord], X: np.ndarray, feature_names: list[str]) -> "AsimModels":
        self.pv = PersonVars([r.labels for r in records])
        env = self.person_env([r.labels for r in records])
        n = len(records)
        self.cdap = SpecLogit(cdap_spec()).fit(env, n, [r.cdap for r in records])
        m_idx = [i for i, r in enumerate(records) if r.cdap == "M"]
        self.mtf = SpecLogit(mtf_spec()).fit(subset(env, m_idx), len(m_idx), [records[i].mtf for i in m_idx])
        print(f"  ASIM-2 CDAP: {self.cdap.n_params} parameters, {self.cdap.n_unavailable_chosen_} days with an "
              f"unavailable choice left out; ASIM-3 MTF: {self.mtf.n_params} parameters, "
              f"{self.mtf.n_unavailable_chosen_} left out", flush=True)

        # Non-mandatory frequency over the 48 alternatives (one spec, one coefficient file per person type),
        # then the extension lookup.
        vectors = nm_alternatives()
        self.nm_vectors = {nm_label(v): v for v in vectors}
        self.nm_attrs = nm_attr_env(vectors)
        ext_keys, ext_vals = [], []
        for r in records:
            if r.cdap in ("M", "N"):
                for p, k, extra in zip(NM_ORDER, r.nm, r.nm_extra):
                    if k > 0:
                        ext_keys.append((p, r.cdap == "M", r.ptype))
                        ext_vals.append(extra)
        self.nm_extension = Lookup(list(range(MAX_EXTRA + 1))).fit(ext_keys, ext_vals)
        self.nmtf = {}
        for seg, (e, k, y) in self._nmtf_data(records, env).items():
            self.nmtf[seg] = SpecLogit(nmtf_spec(seg, list(self.nm_vectors))).fit(e, k, y)
        print("  ASIM-5 non-mandatory frequency: "
              + ", ".join(f"{s.replace('PTYPE_', '')} {m.n_params}" for s, m in self.nmtf.items()) + " parameters",
              flush=True)

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

        self.stopfreq = {seg: SpecLogit(stopfreq_spec(seg)).fit(e, k, y)
                         for seg, (e, k, y) in self._stopfreq_data(records, env).items() if seg in STOP_SEGMENTS}
        print("  ASIM-7 stop frequency: " + ", ".join(f"{s} {m.n_params}" for s, m in self.stopfreq.items())
              + " parameters", flush=True)

        trip_keys, trip_acts, dep_keys, dep_periods = [], [], [], []
        self.departure_records_skipped = 0
        for r in records:
            for t in r.tours:
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
        self.trip_purpose = Lookup(STOP_ACTS).fit(trip_keys, trip_acts)
        self.trip_departure = Lookup(list(range(grid.n))).fit(dep_keys, dep_periods)
        print(f"  ASIM-9 trip departure: {len(dep_keys):,} departures, {len(self.trip_departure.tables)} lookup cells; "
              f"{self.departure_records_skipped} tours with non-increasing departures skipped")
        self.grid = grid
        return self

    def n_params(self) -> dict:
        out = {"cdap": self.cdap.n_params, "mtf": self.mtf.n_params,
               "nmtf": {s: m.n_params for s, m in self.nmtf.items()},
               "stop_frequency": {s: m.n_params for s, m in self.stopfreq.items()},
               "tour_scheduling": {g: getattr(t.clogit, "n_params_", None) for g, t in self.tod.items()
                                   if t is not None}}
        return out

    # ---- validate ------------------------------------------------------------------------------------
    def validate(self, records: list[AsimRecord], X: np.ndarray) -> dict:
        env = self.person_env([r.labels for r in records])
        out = {"ASIM-2 CDAP": self.cdap.validate(env, len(records), [r.cdap for r in records])}
        m_idx = [i for i, r in enumerate(records) if r.cdap == "M"]
        if m_idx:
            out["ASIM-3 mandatory tour frequency"] = self.mtf.validate(subset(env, m_idx), len(m_idx),
                                                                       [records[i].mtf for i in m_idx])
        for seg, (e, k, y) in self._nmtf_data(records, env).items():
            if seg in self.nmtf:
                out[f"ASIM-5 non-mandatory tour frequency {seg}"] = self.nmtf[seg].validate(e, k, y)
        design = self.tod_design(records, X)
        for g in GROUPS:
            d = design[g]
            if self.tod.get(g) is not None and d["starts"]:
                out[f"ASIM-4/6 tdd {g}"] = self.tod[g].validate(np.array(d["Z"]), np.array(d["starts"]),
                                                                np.array(d["ends"]), np.array(d["masks"]), d["states"])
        for seg, (e, k, y) in self._stopfreq_data(records, env).items():
            if seg in self.stopfreq:
                out[f"ASIM-7 stop frequency {seg}"] = self.stopfreq[seg].validate(e, k, y)
        out["ASIM-8 trip purpose lookup"] = {"n": sum(n for k, (_, n) in self.trip_purpose.tables.items() if k == ()),
                                             "cells": len(self.trip_purpose.tables)}
        out["ASIM-9 trip departure lookup"] = {"n": sum(n for k, (_, n) in self.trip_departure.tables.items()
                                                        if k == ()), "cells": len(self.trip_departure.tables)}
        return out
