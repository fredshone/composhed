"""DaySim-aligned generation chain, in DaySim's ``ChoiceModelRunner`` order.

Day pattern → exact number of tours per purpose → for each tour in priority
order: tour time (arrival and departure at the primary destination) → its
half-tours, each working outwards from the primary destination: generate the
next stop's purpose (or none), then time the current stop's outer boundary
with the trip time model → reserve the whole tour in the time window → next
tour. Minutes are drawn uniformly inside the chosen periods.

A stop's purpose excludes the purpose of the activity next to it (the primary
or the previous stop): without locations two neighbouring activities of one
type would merge into one. A stop whose boundary has no free minute is dropped
(counted).
"""

import numpy as np

from composhed.activitysim.persontype import person_type
from composhed.common.calibration import share_dict
from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import layout_rescale
from composhed.common.tourgen import PlannedTour, tours_to_rows
from composhed.daysim.data import (
    MAX_TOURS_PER_PURPOSE,
    STOP_PURPOSES,
    TOUR_PURPOSES,
    pattern_label,
    pattern_tours,
    tod_group,
)
from composhed.daysim.models import (
    DIRECTIONS,
    GROUPS,
    MAX_STOPS,
    NONE,
    NT,
    STOPGEN_CLASSES,
    DaysimModels,
    half_tour_stops,
    person_x,
    tour_ranks,
    tt_flags,
    window_around,
    x_exact,
    x_stopgen,
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
        m = self.m
        ptype = person_type(attrs["age"], attrs["employment"])
        xp = person_x(xl, ptype)
        label = m.pattern.sample(xp[None, :], rng)[0]
        vec = m.pattern_vectors[label]
        trace = {"pattern": label, "exact": [], "tours": [], "stopgen": []}
        counts = {}
        for p, f in zip(TOUR_PURPOSES, vec[:NT]):
            if f:
                model = m.exact[p]
                counts[p] = int(model.sample(x_exact(xp, vec)[None, :], rng)[0]) if not isinstance(model, int) else model
                trace["exact"].append((p, counts[p]))
        purposes = pattern_tours(vec, counts)
        if not purposes:
            return home_day(), trace
        constrained = self.consistency == "constraint"
        tw = TimeWindow()
        stops_left = sum(vec[NT:])
        stops_so_far = 0
        pattern_stops = {p for p, f in zip(STOP_PURPOSES, vec[NT:]) if f}
        female = attrs.get("sex") == "female"
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
                stops, k_used = self._half_tour(xp, ptype, attrs, purpose, outbound, arr if outbound else dep, lo, hi,
                                                n - rank - 1, stops_so_far, female, pattern_stops, tw, rng, report,
                                                trace)
                sides[direction] = stops
                stops_so_far += k_used
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

    def _half_tour(self, xp, ptype, attrs, tour_purpose, outbound, anchor, lo, hi, remaining_tours, stops_so_far,
                   female, pattern_stops, tw, rng, report, trace) -> tuple[list[tuple[str, int, int]], int]:
        """Stops of one half-tour, outwards from the primary: [(purpose, start, end)]."""
        m = self.m
        out = []
        prev = tour_purpose

        def next_purpose(k, anchor_, prev_):
            if k >= MAX_STOPS:
                return NONE
            wh = ((anchor_ - lo) if outbound else (hi - anchor_)) / 60.0
            x = x_stopgen(xp, tour_purpose, outbound, k, wh, anchor_, remaining_tours, stops_so_far + len(out), female)
            avail = np.array([[c == NONE or (c in pattern_stops and c != prev_) for c in STOPGEN_CLASSES]])
            choice = m.stopgen.sample(x[None, :], rng, avail)[0]
            trace["stopgen"].append(choice)
            return choice

        purpose = next_purpose(0, anchor, prev)
        k = 0
        while purpose != NONE:
            # DaySim generates the next stop before timing this one (the trip-remaining term needs it).
            following = next_purpose(k + 1, anchor, purpose)
            flags = tt_flags(ptype, outbound, tour_purpose, purpose, attrs)
            state = (anchor, lo, hi, outbound, flags, float(following != NONE), remaining_tours, tw.remaining())
            outer = m.trip_time.sample(m.trip_time.z(flags), state, rng) if m.trip_time is not None else None
            if outer is None:
                report.stops_dropped += 1
                break
            out.append((purpose, outer, anchor) if outbound else (purpose, anchor, outer))
            anchor = outer
            purpose = following
            k += 1
        return out, len(out)

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


# ---- calibration statistics --------------------------------------------------------------------
def record_trace(rec, models: DaysimModels) -> dict:
    stopgen = []
    for t in rec.tours:
        for direction in DIRECTIONS:
            stops = half_tour_stops(t, direction == "outbound")
            stopgen += [s[0] for s in stops[:MAX_STOPS]] + ([NONE] if len(stops) < MAX_STOPS else [])
    return {
        "pattern": pattern_label(rec.model_pattern),
        "exact": [(p, min(c, MAX_TOURS_PER_PURPOSE)) for p, c in rec.counts.items() if c > 0],
        "tours": [(tod_group(t.purpose), t.primary_start, t.primary_start + t.primary_duration) for t in rec.tours],
        "stopgen": stopgen,
    }


def trace_stats(traces: list[dict], models: DaysimModels, timing: str = "tod") -> dict[str, dict]:
    out: dict[str, dict] = {"pattern": share_dict([t["pattern"] for t in traces], list(models.pattern_vectors)),
                            "stopgen": share_dict([c for t in traces for c in t["stopgen"]], STOPGEN_CLASSES)}
    for p in TOUR_PURPOSES:
        vals = [c for t in traces for q, c in t["exact"] if q == p]
        if vals and not isinstance(models.exact.get(p), int):
            out[f"exact_{p}"] = share_dict(vals, list(range(1, MAX_TOURS_PER_PURPOSE + 1)))
    tours = [x for t in traces for x in t["tours"]]
    for g in GROUPS:
        tod = models.tod.get(g)
        gt = [x for x in tours if x[0] == g]
        if tod is None or not gt:
            continue
        if timing == "tod":
            out[f"tod_start_{g}"] = share_dict([int(tod.grid.period_of(x[1])) for x in gt], range(tod.grid.n))
            out[f"tod_end_{g}"] = share_dict([int(tod.grid.end_period_of(x[2])) for x in gt], range(tod.grid.n))
            # Measured on the chosen period pair, the space the duration constants act on.
            out[f"tod_duration_{g}"] = share_dict([int(tod.dbin[tod.alt_of(x[1], x[2])]) for x in gt],
                                                  range(tod.n_dur_bins))
        else:
            out[f"tour_duration_mean_{g}"] = {"mean": float(np.mean([x[2] - x[1] for x in gt]))}
    return out


def array_accessors(arr: np.ndarray):
    def get() -> dict:
        return {i: float(v) for i, v in enumerate(arr)}

    def set_(d: dict) -> None:
        for i in range(len(arr)):
            arr[i] = d[i]

    return get, set_
