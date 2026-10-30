"""ActivitySim-aligned generation chain (ASIM-1 … ASIM-11).

Order as ActivitySim's pipeline: CDAP → mandatory tour frequency → mandatory
tour scheduling → non-mandatory tour frequency (seeing the residual time window)
and its extension → non-mandatory tour scheduling → stop frequency → trip
purpose → trip scheduling. Trip departures are chosen as periods; the minute is
drawn uniformly from the feasible minutes of the period. A stop may have
the purpose of the activity next to it; the two are written as separate
episodes.
"""

from collections import Counter

import numpy as np

from composhed.activitysim.data import NM_ORDER, NTS_ACT, WORKER_TYPES, mtf_tours
from composhed.activitysim.models import (
    STOP_ACTS,
    AsimModels,
    hour_band,
    residual_window,
    stop_segment,
    tod_group,
    tour_class,
    tour_contexts,
    tour_env,
)
from composhed.activitysim.persontype import person_type
from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.tourgen import PlannedTour, place_tours, tours_to_rows


class AsimChain:
    def __init__(self, models: AsimModels, timing: str = "tod", consistency: str = "constraint",
                 max_resample: int = 50) -> None:
        self.m = models
        self.timing = timing
        self.consistency = consistency
        self.max_resample = max_resample

    def _plan(self, purposes: list[str], attrs: dict, ptype: str, mtf, n_mand: int, nm: list[str]) -> list[PlannedTour]:
        """Tours of one scheduling phase, numbered within the phase (first of 2+, look-ahead)."""
        m = self.m
        purposes = [p for p in purposes if m.tod.get(tod_group(p)) is not None]
        return [PlannedTour(p, tod_group(p), m.z(tod_group(p), attrs, ptype, p, ctx),
                            n_remaining=ctx["n_in_phase"] - ctx["idx"] - 1)
                for p, ctx in zip(purposes, tour_contexts(purposes, mtf, n_mand, nm))]

    def _place(self, tours, xl, rng, report, prior=()) -> list[PlannedTour]:
        m = self.m
        return place_tours(tours, m.tod, m.indep, xl, rng, report, m.home_inter, self.timing,
                           self.consistency, self.max_resample, prior=prior)

    def generate(self, xl: np.ndarray, attrs: dict, rng: np.random.Generator,
                 report: GenerationReport) -> tuple[list[dict], dict]:
        m = self.m
        ptype = person_type(attrs["age"], attrs["employment"])
        env = m.person_env([attrs])
        cdap = m.cdap.sample(env, 1, rng)[0]
        trace = {"ptype": ptype, "cdap": cdap, "mtf": None, "nm": None, "tours": [], "stop_alts": []}
        if cdap == "H":
            return home_day(), trace
        mand: list[str] = []
        if cdap == "M":
            trace["mtf"] = m.mtf.sample(env, 1, rng)[0]
            mand = mtf_tours(trace["mtf"], ptype in WORKER_TYPES)
        placed = self._place(self._plan(mand, attrs, ptype, trace["mtf"], len(mand), []), xl, rng, report)
        seg = m.nm_segment(ptype)
        if seg in m.nmtf:
            window = residual_window([(int(t.start), int(t.end)) for t in placed], m.home_inter)
            label = m.nmtf[seg].sample(m.nm_env(env, [len(mand)], [window]), 1, rng)[0]
            trace["nm"] = label
            nm = []
            for p, n in zip(NM_ORDER, m.nm_vectors[label]):
                if n:
                    pr = m.nm_extension.probs((p, cdap == "M", ptype))
                    n += int(rng.choice(len(pr), p=pr / pr.sum()))
                nm += [p] * n
            placed = self._place(self._plan(nm, attrs, ptype, trace["mtf"], len(mand), nm), xl, rng, report,
                                 prior=placed)
        counts = Counter(t.purpose for t in placed)
        for t in placed:
            self._stops(env, ptype, t, counts, rng, report, trace)
            trace["tours"].append((t.group, t.start, t.end))
        for t in placed:
            t.purpose = NTS_ACT[t.purpose]
        return tours_to_rows(placed, report), trace

    def _stops(self, env, ptype, t: PlannedTour, counts, rng, report, trace) -> None:
        m = self.m
        seg = stop_segment(t.purpose, ptype)
        alt = "0out_0in"
        if seg in m.stopfreq:
            e = env | tour_env([t.purpose], [t.start], [t.end], [counts], [ptype == "UNIV"])
            alt = m.stopfreq[seg].sample(e, 1, rng)[0]
        trace["stop_alts"].append((t.purpose, alt))
        n_out, n_in = int(alt[0]), int(alt.split("_")[1][0])
        dur = int(t.end - t.start)
        while n_out + n_in + 1 > dur:
            if n_out >= n_in:
                n_out -= 1
            else:
                n_in -= 1
            report.stops_dropped += 1
        S, E = float(t.start), float(t.end)
        cls = tour_class(t.purpose)
        a_out, a_in = int(m.grid.period_of(S)), int(m.grid.end_period_of(E))
        # Outbound, forwards from the tour start: departure from each stop; stop k occupies [d_{k-1}, d_k).
        deps, cursor = [], S
        for k in range(n_out):
            hi = E - (n_out - k - 1) - n_in - 1  # leave >= 1 min per later stop and the primary activity
            key = ("outbound", cls, a_out, t.purpose, min(k, 2))
            cursor = self._departure(key, cursor + 1, hi, rng, report)
            deps.append(cursor)
        prev = S
        for d in deps:
            t.outbound.append((self._purpose(t.purpose, "outbound", S, ptype, rng), d - prev))
            prev = d
        # Inbound, backwards from the tour end: departures from the primary and from each stop but the last.
        in_deps, cursor = [0.0] * n_in, E
        for b in range(n_in):
            k = n_in - 1 - b
            lo = prev + 1 + k  # leave >= 1 min for the primary and each earlier inbound stop
            key = ("inbound", cls, a_in, t.purpose, min(b, 2))
            cursor = self._departure(key, lo, cursor - 1, rng, report)
            in_deps[k] = cursor
        bounds = in_deps + [E]
        for k in range(n_in):
            t.inbound.append((self._purpose(t.purpose, "inbound", E, ptype, rng), bounds[k + 1] - bounds[k]))

    def _purpose(self, tour_purpose, direction, anchor, ptype, rng) -> str:
        """ASIM-8: stop purpose from the lookup."""
        p = self.m.trip_purpose.probs((tour_purpose, direction, hour_band(anchor), ptype))
        return STOP_ACTS[int(rng.choice(len(STOP_ACTS), p=p / p.sum()))]

    def _departure(self, key: tuple, lo: float, hi: float, rng, report) -> float:
        """ASIM-9: departure period from the lookup, restricted to [lo, hi]; uniform minute within the period.

        Probability outside the feasible periods is removed and the rest renormalised,
        which is ActivitySim's fail-and-retry with unlimited retries; when no feasible
        period has probability, the period nearest the anchor is used (as
        ``choose_most_initial``) and counted.
        """
        grid = self.m.grid
        lo_i, hi_i = int(np.ceil(lo)), int(np.floor(hi))
        if hi_i <= lo_i:
            return float(lo_i)
        p = self.m.trip_departure.probs(key).copy()
        first, last = int(grid.end_period_of(lo_i)), int(grid.end_period_of(hi_i))
        p[:first] = 0.0
        p[last + 1:] = 0.0
        if p.sum() <= 0:
            report.count("trip_departure_failfix")
            period = first if key[0] == "outbound" else last
        else:
            period = int(rng.choice(len(p), p=p / p.sum()))
        # Departure minutes of a period are (lo, hi] (end convention), here restricted to [lo_i, hi_i].
        a = max(lo_i, int(grid.lo(period)) + 1)
        b = min(hi_i, int(grid.hi(period)))
        return float(rng.integers(a, b + 1))
