"""DaySim-aligned generation chain, in DaySim's ``ChoiceModelRunner`` order.

Day pattern → exact number of tours per purpose → for each tour in priority
order: tour time (arrival and departure at the primary destination) → its
half-tours, each working outwards from the primary destination: generate the
next stop's purpose (or none), then time the current stop's outer boundary
with the trip time model → reserve the whole tour in the time window → next
tour. Minutes are drawn uniformly inside the chosen periods.

As DaySim, a person-day in which a tour or stop cannot be placed is simulated
again (up to ``max_resample`` times, counted as ``person_day_resimulated``);
after that the unplaceable tours and stops are dropped (counted). A stop
may have the purpose of the activity next to it; the two are written as
separate episodes.
"""

import copy
from collections import Counter

import numpy as np

from composhed.activitysim.persontype import person_type
from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import layout_rescale
from composhed.common.tourgen import PlannedTour, tours_to_rows
from composhed.daysim.data import STOP_PURPOSES, TOUR_PURPOSES, pattern_tours, tod_group
from composhed.daysim.models import (
    DIRECTIONS,
    MAX_STOPS,
    NONE,
    NT,
    STOPGEN_CLASSES,
    STOPGEN_VARS,
    DaysimModels,
    exact_env,
    pattern_avail,
    pattern_x,
    rows_env,
    stopgen_vars,
    tour_ranks,
    tt_flags,
    window_around,
    z_tod_values,
)


class DaysimChain:
    def __init__(self, models: DaysimModels, timing: str = "tod", consistency: str = "constraint",
                 max_resample: int = 50) -> None:
        self.m = models
        self.timing = timing
        self.consistency = consistency
        self.max_resample = max_resample

    def generate(self, xl: np.ndarray, attrs: dict, rng: np.random.Generator,
                 report: GenerationReport) -> tuple[list[dict], dict]:
        before = copy.deepcopy(report.__dict__)
        for attempt in range(self.max_resample + 1):
            rows, trace = self._person_day(xl, attrs, rng, report)
            failed = (report.tours_dropped, report.stops_dropped) != (before["tours_dropped"], before["stops_dropped"])
            if not failed or attempt == self.max_resample:
                return rows, trace
            report.__dict__.update(copy.deepcopy(before))
            report.count("person_day_resimulated")
            before = copy.deepcopy(report.__dict__)
        return rows, trace

    def _person_day(self, xl, attrs, rng, report) -> tuple[list[dict], dict]:
        m = self.m
        ptype = person_type(attrs["age"], attrs["employment"])
        person = m.dv.person(attrs, ptype)
        vectors = list(m.pattern_vectors.values())
        avail = pattern_avail([m.flags(attrs, ptype)], vectors)
        label = m.pattern.sample(pattern_x(m.dv, attrs, ptype)[None, :], rng, avail)[0]
        vec = m.pattern_vectors[label]
        trace = {"pattern": label, "exact": [], "tours": [], "stopgen": []}
        counts = {}
        for p, f in zip(TOUR_PURPOSES, vec[:NT]):
            if f:
                model = m.exact[p]
                counts[p] = model if isinstance(model, int) else int(model.sample(exact_env([person], [vec]), 1, rng)[0])
                trace["exact"].append((p, counts[p]))
        purposes = pattern_tours(vec, counts)
        if not purposes:
            return home_day(), trace
        constrained = self.consistency == "constraint"
        tw = TimeWindow()
        stops_left = sum(vec[NT:])
        sim_stops: Counter = Counter()
        pattern_stops = {p for p, f in zip(STOP_PURPOSES, vec[NT:]) if f}
        placed: list[PlannedTour] = []
        for purpose, (rank, n, si, sc) in zip(purposes, tour_ranks(purposes)):
            g = tod_group(purpose)
            tod = m.tod.get(g)
            if tod is None:
                report.tours_dropped += 1
                continue
            z = tod.z(z_tod_values(attrs, ptype, purpose, rank, n, si, sc, vec, m.income_levels))
            interval = self._tour_time(g, z, xl, tw if constrained else TimeWindow(), rng, report,
                                       n - rank - 1, rank > 0, stops_left)
            if interval is None:
                report.tours_dropped += 1
                continue
            arr, dep = interval
            lo, hi = window_around(tw if constrained else TimeWindow(), arr, dep, m.home_inter)
            t = PlannedTour(purpose, g, z)
            sides = {}
            for direction in DIRECTIONS:
                outbound = direction == "outbound"
                sides[direction] = self._half_tour(person, ptype, attrs, purpose, outbound, arr if outbound else dep,
                                                   lo, hi, n, rank, sim_stops, pattern_stops, tw, rng, report, trace)
            # Outbound stops were generated outwards; lay them out in time order.
            out = list(reversed(sides["outbound"]))
            t.start = float(out[0][1]) if out else float(arr)
            t.end = float(sides["inbound"][-1][2]) if sides["inbound"] else float(dep)
            t.outbound = [(p, float(e - s)) for p, s, e in out]
            t.inbound = [(p, float(e - s)) for p, s, e in sides["inbound"]]
            stops_left = max(stops_left - len(out) - len(sides["inbound"]), 0)
            if constrained:
                tw.reserve(int(t.start), int(t.end))
            placed.append(t)
            trace["tours"].append((g, arr, dep))
        if not constrained and placed:
            self._rescale(placed, report)
        placed.sort(key=lambda t: t.start)
        return tours_to_rows(placed, report), trace

    def _tour_time(self, g, z, xl, tw, rng, report, n_remaining, later, stops_left):
        m = self.m
        if self.timing == "independent":
            interval, used = m.indep[g].sample(xl, tw, rng, m.home_inter, self.max_resample)
            report.resamples += used if interval is not None else self.max_resample
            return interval
        tod = m.tod[g]
        if not tod.mask(tw).any():
            return None
        for _ in range(self.max_resample + 1):
            interval = tod.sample(z, tw, rng, n_remaining=n_remaining, later=later, stops_left=stops_left)
            if interval is not None:
                return interval
            report.resamples += 1
        return None

    def _half_tour(self, person, ptype, attrs, tour_purpose, outbound, anchor, lo, hi, n_tours, rank, sim_stops,
                   pattern_stops, tw, rng, report, trace) -> list[tuple[str, int, int]]:
        """Stops of one half-tour, outwards from the primary: [(purpose, start, end)]."""
        m = self.m
        out = []
        anchor0 = anchor
        avail = np.array([[c == NONE or c in pattern_stops for c in STOPGEN_CLASSES]])

        def next_purpose(k, anchor_):
            if k >= MAX_STOPS:
                return NONE
            wh = ((anchor_ - lo) if outbound else (hi - anchor_)) / 60.0
            v = stopgen_vars(person, tour_purpose, outbound, k, wh, anchor0, n_tours, rank, sim_stops)
            choice = m.stopgen.sample(rows_env([v], STOPGEN_VARS), 1, rng, avail)[0]
            trace["stopgen"].append(choice)
            return choice

        purpose = next_purpose(0, anchor)
        k = 0
        while purpose != NONE:
            sim_stops[purpose] += 1
            # DaySim generates the next stop before timing this one (the trip-remaining term needs it).
            following = next_purpose(k + 1, anchor)
            flags = tt_flags(ptype, outbound, tour_purpose, purpose, attrs)
            state = (anchor, lo, hi, outbound, flags, float(following != NONE), n_tours - rank - 1, tw.remaining())
            outer = m.trip_time.sample(m.trip_time.z(flags), state, rng) if m.trip_time is not None else None
            if outer is None:
                report.stops_dropped += 1
                break
            out.append((purpose, outer, anchor) if outbound else (purpose, anchor, outer))
            anchor = outer
            purpose = following
            k += 1
        return out

    def _rescale(self, placed: list[PlannedTour], report) -> None:
        """AB-3: shift and shrink overlapping tours; stops keep their share of their tour."""
        new, changed = layout_rescale([(t.start, t.end) for t in placed], self.m.home_inter)
        placed.sort(key=lambda t: t.start)
        for t, (s, e) in zip(placed, new):
            f = (e - s) / max(t.end - t.start, 1e-9)
            t.outbound = [(p, d * f) for p, d in t.outbound]
            t.inbound = [(p, d * f) for p, d in t.inbound]
            t.start, t.end = s, e
        if changed:
            report.rescales += 1
