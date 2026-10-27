"""ActivitySim-aligned generation chain (ASIM-1 … ASIM-11)."""

from collections import Counter

import numpy as np

from composhed.activitysim.data import NM_ORDER, NTS_ACT, mtf_tours, nm_tours
from composhed.activitysim.models import (
    GROUPS,
    STOP_ACTS,
    AsimModels,
    tod_group,
    tour_class,
    x_nmtf,
    x_stopfreq,
    z_tod,
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
            mand = mtf_tours(trace["mtf"])
        nm: list[str] = []
        if m.nmtf is not None:
            avail = m._nm_avail([cdap])
            label = m.nmtf.sample(x_nmtf(xl, cdap, len(mand))[None, :], [ptype], rng, avail)[0]
            trace["nm"] = label
            nm = nm_tours(m.nm_vectors[label])
        purposes = mand + sorted(nm, key=NM_ORDER.index)
        seen: Counter = Counter()
        tours = []
        for p in purposes:
            if m.tod.get(tod_group(p)) is None:
                continue
            tours.append(PlannedTour(p, tod_group(p), z_tod(xl, ptype, p, seen[p], m.segmented)))
            seen[p] += 1
        placed = place_tours(tours, m.tod, m.indep, xl, rng, report, m.home_inter, self.timing,
                             self.consistency, self.max_resample)
        for t in placed:
            self._stops(xl, t, rng, report, trace)
            trace["tours"].append((t.group, t.start, t.end))
        for t in placed:
            t.purpose = NTS_ACT[t.purpose]
        return tours_to_rows(placed, report), trace

    def _stops(self, xl, t: PlannedTour, rng, report, trace) -> None:
        m = self.m
        alt = m.stopfreq.sample(x_stopfreq(xl, t.start, t.end)[None, :], [t.purpose], rng)[0] \
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
        # Outbound: departure from each stop; stop k occupies [d_{k-1}, d_k), d_0 = tour start.
        deps, cursor = [], S
        for k in range(n_out):
            hi = E - (n_out - k - 1) - n_in - 1  # leave >= 1 min per later stop and the primary activity
            cursor = self._departure(cls, "outbound", k, cursor + 1, hi, rng)
            deps.append(cursor)
        prev = S
        for k, d in enumerate(deps):
            act = self._purpose(t.purpose, "outbound", k, rng)
            t.outbound.append((act, d - prev))
            prev = d
        # Inbound: departure from the primary activity and from each inbound stop but the last.
        in_deps, cursor = [], prev
        for k in range(n_in):
            hi = E - (n_in - k)
            cursor = self._departure(cls, "inbound", k, cursor + 1, hi, rng)
            in_deps.append(cursor)
        bounds = in_deps + [E]
        for k in range(n_in):
            act = self._purpose(t.purpose, "inbound", k, rng)
            t.inbound.append((act, bounds[k + 1] - bounds[k]))

    def _purpose(self, tour_purpose, direction, k, rng) -> str:
        p = self.m.trip_purpose.probs((tour_purpose, direction, min(k, 2)))
        return STOP_ACTS[int(rng.choice(len(STOP_ACTS), p=p))]

    def _departure(self, cls, direction, k, lo: float, hi: float, rng) -> float:
        """ASIM-9: departure period from the lookup, restricted to [lo, hi]; minute within period."""
        grid = self.m.grid
        lo_i, hi_i = int(np.ceil(lo)), int(np.floor(hi))
        if hi_i <= lo_i:
            return float(lo_i)
        p = self.m.trip_departure.probs((cls, direction, min(k, 2))).copy()
        first, last = int(grid.end_period_of(lo_i)), int(grid.end_period_of(hi_i))
        p[:first] = 0.0
        p[last + 1:] = 0.0
        if p.sum() <= 0:
            p[first:last + 1] = 1.0
        period = int(rng.choice(len(p), p=p / p.sum()))
        return float(self.m.trip_intra.sample(direction, period, lo_i - 1, hi_i, rng, end=True))


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
