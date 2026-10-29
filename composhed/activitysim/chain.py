"""ActivitySim-aligned generation chain (ASIM-1 … ASIM-11).

Order as ActivitySim's pipeline: CDAP → mandatory tour frequency → mandatory
tour scheduling → non-mandatory tour frequency (seeing the residual time window)
and its extension → non-mandatory tour scheduling → stop frequency → trip
purpose → trip scheduling. Trip departures are chosen as periods; the minute is
drawn uniformly from the feasible minutes of the period.
"""

from collections import Counter

import numpy as np

from composhed.activitysim.data import NM_ORDER, NTS_ACT, WORKER_TYPES, mtf_tours
from composhed.activitysim.models import (
    GROUPS,
    STOP_ACTS,
    AsimModels,
    hour_band,
    residual_window,
    tod_group,
    tour_class,
    tour_contexts,
    x_nmtf,
    x_stopfreq,
)
from composhed.activitysim.persontype import person_type
from composhed.common.calibration import share_dict
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
        x1 = xl[None, :]
        cdap = m.cdap.sample(x1, [ptype], rng)[0]
        trace = {"ptype": ptype, "cdap": cdap, "mtf": None, "nm": None, "tours": [], "stop_alts": []}
        if cdap == "H":
            return home_day(), trace
        mand: list[str] = []
        if cdap == "M" and m.mtf is not None:
            trace["mtf"] = m.mtf.sample(x1, [ptype], rng)[0]
            mand = mtf_tours(trace["mtf"], ptype in WORKER_TYPES)
        placed = self._place(self._plan(mand, attrs, ptype, trace["mtf"], len(mand), []), xl, rng, report)
        if m.nmtf is not None:
            window = residual_window([(int(t.start), int(t.end)) for t in placed], m.home_inter)
            avail = m._nm_avail([cdap])
            label = m.nmtf.sample(x_nmtf(xl, cdap, len(mand), window)[None, :], [ptype], rng, avail)[0]
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
            self._stops(xl, ptype, t, counts, rng, report, trace)
            trace["tours"].append((t.group, t.start, t.end))
        for t in placed:
            t.purpose = NTS_ACT[t.purpose]
        return tours_to_rows(placed, report), trace

    def _stops(self, xl, ptype, t: PlannedTour, counts, rng, report, trace) -> None:
        m = self.m
        alt = m.stopfreq.sample(x_stopfreq(xl, t.start, t.end, counts)[None, :], [t.purpose], rng)[0] \
            if m.stopfreq is not None else "0out_0in"
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
        primary = NTS_ACT[t.purpose]
        prev, prev_act = S, None
        for k, d in enumerate(deps):
            avoid = {prev_act} | ({primary} if k == n_out - 1 else set())
            act = self._purpose(t.purpose, "outbound", S, ptype, avoid, rng)
            t.outbound.append((act, d - prev))
            prev, prev_act = d, act
        # Inbound, backwards from the tour end: departures from the primary and from each stop but the last.
        in_deps, cursor = [0.0] * n_in, E
        for b in range(n_in):
            k = n_in - 1 - b
            lo = prev + 1 + k  # leave >= 1 min for the primary and each earlier inbound stop
            key = ("inbound", cls, a_in, t.purpose, min(b, 2))
            cursor = self._departure(key, lo, cursor - 1, rng, report)
            in_deps[k] = cursor
        bounds = in_deps + [E]
        prev_act = primary
        for k in range(n_in):
            act = self._purpose(t.purpose, "inbound", E, ptype, {prev_act}, rng)
            t.inbound.append((act, bounds[k + 1] - bounds[k]))
            prev_act = act

    def _purpose(self, tour_purpose, direction, anchor, ptype, avoid: set, rng) -> str:
        """ASIM-8: stop purpose from the lookup, excluding the purpose of the adjacent activity.

        Two adjacent episodes of one purpose would be merged into one in the
        location-free schedule output, silently removing the stop.
        """
        p = self.m.trip_purpose.probs((tour_purpose, direction, hour_band(anchor), ptype)).copy()
        ok = np.array([a not in avoid for a in STOP_ACTS])
        p = np.where(ok, p, 0.0)
        if p.sum() <= 0:
            p = ok.astype(float)
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


# ---- calibration statistics ------------------------------------------------------------------------
def record_trace(rec, models: AsimModels) -> dict:
    from composhed.activitysim.models import stop_alt

    return {"ptype": rec.ptype, "cdap": rec.cdap, "mtf": rec.mtf,
            "nm": models.map_nm(rec.nm) if rec.cdap in ("M", "N") else None,
            "tours": [(tod_group(t.purpose), t.start, t.end) for t in rec.tours],
            "stop_alts": [(t.purpose, stop_alt(len(t.outbound), len(t.inbound))) for t in rec.tours]}


def trace_stats(traces: list[dict], models: AsimModels, timing: str = "tod") -> dict[str, dict]:
    out: dict[str, dict] = {}
    m = models
    for key in m.cdap.models:
        sel = [t for t in traces if m.cdap.key(t["ptype"]) == key]
        if sel:
            out[f"cdap:{key}"] = share_dict([t["cdap"] for t in sel], m.cdap_classes)
    if m.mtf is not None:
        for key in m.mtf.models:
            sel = [t for t in traces if t["mtf"] is not None and m.mtf.key(t["ptype"]) == key]
            if sel:
                out[f"mtf:{key}"] = share_dict([t["mtf"] for t in sel], m.mtf_classes)
    if m.nmtf is not None:
        for key in m.nmtf.models:
            sel = [t for t in traces if t["nm"] is not None and m.nmtf.key(t["ptype"]) == key]
            if sel:
                out[f"nmtf:{key}"] = share_dict([t["nm"] for t in sel], list(m.nm_vectors))
    if m.stopfreq is not None:
        alts = [a for t in traces for a in t["stop_alts"]]
        for key in m.stopfreq.models:
            sel = [a for p, a in alts if m.stopfreq.key(p) == key]
            if sel:
                out[f"stopfreq:{key}"] = share_dict(sel, m.stop_classes)
    tours = [x for t in traces for x in t["tours"]]
    for g in GROUPS:
        tod = m.tod.get(g)
        gt = [x for x in tours if x[0] == g]
        if tod is None or not gt:
            continue
        if timing == "tod":
            out[f"tod_start_{g}"] = share_dict([int(tod.grid.period_of(x[1])) for x in gt], range(tod.grid.n))
            out[f"tod_end_{g}"] = share_dict([int(tod.grid.end_period_of(x[2])) for x in gt], range(tod.grid.n))
            out[f"tod_duration_{g}"] = share_dict([int(tod.dbin[tod.alt_of(x[1], x[2])]) for x in gt],
                                                  range(tod.n_dur_bins))
        else:
            out[f"tour_duration_mean_{g}"] = {"mean": float(np.mean([x[2] - x[1] for x in gt]))}
    return out
