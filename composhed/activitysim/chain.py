"""ActivitySim-aligned generation chain (ASIM-1 … ASIM-11).

Order as ActivitySim's pipeline: CDAP → mandatory tour frequency → mandatory
tour scheduling → non-mandatory tour frequency (seeing the residual time window)
and its extension → non-mandatory tour scheduling → stop frequency → trip
purpose → trip scheduling. As in ActivitySim, every time is a period (hourly,
as prototype_mtc: 05:00 to 23:00, the first period holding every earlier time);
it is written at hh:30. A tour may start in the period another ends, so the home
stay between them can last zero minutes, and a stop whose trips depart in one
period lasts zero minutes. A stop may have the purpose of the activity next to
it; the two are written as separate episodes.

Trip scheduling follows ``trip_scheduling.py`` (``prototype_mtc`` settings:
``MAX_ITERATIONS: 100``, ``FAILFIX: choose_most_initial``) with ``logic_version:
2``: departure probabilities outside a trip's feasible periods are not
renormalised but count as failure; a leg with a failed trip is drawn again, up
to 100 times; on the last try a failed trip leaves at its earliest (outbound)
or latest (inbound) period. Inbound trips may not leave before the last
outbound departure drawn in the same try; a retried inbound leg whose outbound
leg is not retried with it starts from the tour start period (version 1 left
it without an earliest period, so it failed every time).
"""

from collections import Counter

import numpy as np

from composhed.activitysim.data import NM_ORDER, NTS_ACT, WORKER_TYPES, mtf_tours
from composhed.activitysim.models import (
    STOP_ACTS,
    AsimModels,
    after_previous,
    extension_eligible,
    log_max_window,
    purpose_range,
    stop_segment,
    timetable_kw,
    tod_group,
    tour_contexts,
    tour_env,
    tour_start_hour,
)
from composhed.activitysim.persontype import person_type
from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.tourgen import PlannedTour, place_tours, tours_to_rows

MAX_ITERATIONS = 100  # trip_scheduling.yaml


class AsimChain:
    def __init__(self, models: AsimModels, timing: str = "tod", consistency: str = "constraint",
                 max_resample: int = 50) -> None:
        self.m = models
        self.timing = timing
        self.consistency = consistency
        self.max_resample = max_resample

    def _plan(self, purposes: list[str], attrs: dict, ptype: str, mtf, n_mand: int, nm: list[str]) -> list[PlannedTour]:
        """Tours of one scheduling phase, numbered within the phase (first of 2+, timetable terms)."""
        m = self.m
        purposes = [p for p in purposes if m.tod.get(tod_group(p)) is not None]
        return [PlannedTour(p, tod_group(p), m.z(tod_group(p), attrs, ptype, p, ctx),
                            state_kw=timetable_kw(ctx, tod_group(p) != "nonmandatory"),
                            after_previous=after_previous(ctx, tod_group(p) != "nonmandatory"))
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
            window = log_max_window([(int(t.start), int(t.end)) for t in placed], m.grid)
            label = m.nmtf[seg].sample(m.nm_env(env, [len(mand)], [window]), 1, rng)[0]
            trace["nm"] = label
            nm = []
            vec = m.nm_vectors[label]
            for p, n, ok in zip(NM_ORDER, vec, extension_eligible(vec)):
                if ok:
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
        S, E = float(t.start), float(t.end)
        grid, pts = m.grid, m.grid.points()
        a_out, a_in = int(grid.period_of(S)), int(grid.end_period_of(E))
        out_p, in_p = self._schedule_trips(seg, a_out, a_in, n_out, n_in, rng, report)

        def at(period, lo, hi):
            return float(min(max(pts[period], lo), hi))

        # Stop k of a leg lasts from the departure before it to its own departure. Times are clamped to stay in
        # order: ActivitySim can schedule an outbound trip after an inbound one when only one leg was retried.
        prev, deps = S, []
        for p in out_p:
            prev = at(p, prev, E)
            deps.append(prev)
        if any(o > i for o in out_p for i in in_p):
            report.count("trip_departure_out_of_order")
        hour = tour_start_hour(grid, S)
        prev_t = S
        for d in deps:
            t.outbound.append((self._purpose(seg, True, hour, ptype, rng), d - prev_t))
            prev_t = d
        in_deps, nxt = [0.0] * n_in, E
        for k in range(n_in - 1, -1, -1):
            nxt = at(in_p[k], prev_t, nxt)
            in_deps[k] = nxt
        bounds = in_deps + [E]
        for k in range(n_in):
            t.inbound.append((self._purpose(seg, False, hour, ptype, rng), bounds[k + 1] - bounds[k]))

    def _purpose(self, segment, outbound, start_hour, ptype, rng) -> str:
        """ASIM-8: stop purpose from the lookup (``trip_purpose_probs.csv``'s keys)."""
        p = self.m.trip_purpose.probs((segment, outbound, purpose_range(segment, outbound, start_hour), ptype))
        return STOP_ACTS[int(rng.choice(len(STOP_ACTS), p=p / p.sum()))]

    def _schedule_trips(self, seg, a_out, a_in, n_out, n_in, rng, report) -> tuple[list[int], list[int]]:
        """ASIM-9: departure periods of outbound trips 2..n_out+1 and inbound trips 1..n_in (``trip_scheduling``)."""
        out_p: list[int] = []
        in_p: list[int] = []
        out_todo, in_todo = n_out > 0, n_in > 0
        for it in range(1, MAX_ITERATIONS + 1):
            if not (out_todo or in_todo):
                break
            last = it == MAX_ITERATIONS
            # Inbound trips may not leave before the last outbound departure of this iteration (outbound trip 1
            # leaves at the tour start). Logic version 2: when only the inbound leg is retried, the tour start.
            earliest_in = a_out
            if out_todo:
                out_p, failed = self._leg(seg, True, a_out, n_out, a_out, a_in, last, rng, report)
                earliest_in = max([a_out] + [p for p in out_p if p is not None])
                out_todo = failed and not last
                if it > 1:
                    report.count("trip_leg_retries")
            if in_todo:
                in_p, failed = self._leg(seg, False, a_in, n_in, earliest_in, a_in, last, rng, report)
                in_todo = failed and not last
                if it > 1:
                    report.count("trip_leg_retries")
        return [int(p) for p in out_p], [int(p) for p in in_p]

    def _leg(self, seg, outbound, tour_hour, n, earliest, latest, last, rng, report):
        """One leg: outbound trips in ascending trip_num, each constraining the next one's earliest period;
        inbound trips in descending trip_num, each constraining the previous one's latest. Probability outside
        [earliest, latest] is a failure (not renormalised)."""
        choices: list = [None] * n
        failed = False
        order = range(n) if outbound else range(n - 1, -1, -1)
        lo, hi = earliest, latest
        for k in order:
            trip_num = k + 2 if outbound else k + 1
            c = None
            p = self.m.trip_departure.probs((seg, outbound, tour_hour, trip_num)).copy()
            p[:lo] = 0.0
            p[hi + 1:] = 0.0
            u = rng.random()
            cum = np.cumsum(p)
            if u < cum[-1]:
                c = int(np.searchsorted(cum, u, side="right"))
            if c is None:
                failed = True
                if last:  # FAILFIX choose_most_initial
                    c = lo if outbound else hi
                    report.count("trip_departure_failfix")
            choices[k] = c
            if c is not None:
                if outbound:
                    lo = c
                else:
                    hi = c
        return choices, failed
