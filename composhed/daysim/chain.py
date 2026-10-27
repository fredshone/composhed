"""DaySim-aligned generation chain (DAY-1 … DAY-9)."""

import numpy as np

from composhed.common.calibration import share_dict
from composhed.common.report import GenerationReport
from composhed.common.tourgen import PlannedTour, fit_stop_durations, place_tours, tours_to_rows
from composhed.daysim.data import pattern_tours, tod_group
from composhed.daysim.models import (
    DIRECTIONS,
    GROUPS,
    MAX_STOPS,
    STOP_PURPOSES,
    DaysimModels,
    tour_z_rows,
    x_nstops,
    x_stop_duration,
    x_stop_purpose,
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
        label = m.pattern.sample(xl[None, :], rng)[0]
        vec = m.pattern_vectors[label]
        flag = vec[-1]
        purposes = pattern_tours(vec)
        tours = [PlannedTour(p, tod_group(p), z) for p, z in zip(purposes, tour_z_rows(xl, purposes))]
        tours = [t for t in tours if m.tod.get(t.group) is not None]
        placed = place_tours(tours, m.tod, m.indep, xl, rng, report, m.home_inter, self.timing,
                             self.consistency, self.max_resample)
        trace = {"pattern": label, "tours": [], "stop_purposes": []}
        need_mand_stop = bool(flag)
        for t in placed:
            n_out, n_in = self._n_stops(xl, t, flag, need_mand_stop, rng)
            if t.purpose in ("work", "education") and n_out + n_in > 0:
                need_mand_stop = False
            # Every stop and the primary activity need at least one minute (DAY-9: drop, counted).
            while n_out + n_in + 1 > int(t.duration):
                if n_out >= n_in:
                    n_out -= 1
                else:
                    n_in -= 1
                report.stops_dropped += 1
            self._fill_stops(xl, t, n_out, n_in, rng, report, trace)
            trace["tours"].append((t.group, t.start, t.end, n_out, n_in))
        return tours_to_rows(placed, report), trace

    def _n_stops(self, xl, t, flag, need_mand_stop, rng) -> tuple[int, int]:
        mand = t.purpose in ("work", "education")
        if mand and not flag:
            return 0, 0
        x = x_nstops(xl, t.purpose, t.start, t.end, flag)[None, :]
        for _ in range(20):
            n_out = int(self.m.nstops["outbound"].sample(x, rng)[0])
            n_in = int(self.m.nstops["inbound"].sample(x, rng)[0])
            if not (mand and need_mand_stop) or n_out + n_in > 0:
                return n_out, n_in
        return n_out, max(n_in, 1)

    def _fill_stops(self, xl, t, n_out, n_in, rng, report, trace) -> None:
        truncate = self.consistency == "constraint"
        dur = t.duration
        remaining = dur - 1.0  # keep >= 1 min of primary activity
        n_total = n_out + n_in
        sampled: dict[str, list] = {"outbound": [], "inbound": []}
        k_all = 0
        for direction, n in (("outbound", n_out), ("inbound", n_in)):
            for k in range(n):
                model = self.m.stop_purpose[direction] or self.m.stop_purpose["outbound" if direction == "inbound" else "inbound"]
                purpose = model.sample(x_stop_purpose(xl, t.purpose, k, dur)[None, :], rng)[0]
                reserve = n_total - k_all - 1
                hi = remaining - reserve if truncate else np.inf
                d = float(self.m.stop_duration[purpose].sample(
                    x_stop_duration(xl, direction, t.purpose, dur)[None, :], rng, lo=1.0, hi=max(hi, 1.0))[0])
                sampled[direction].append((purpose, d))
                trace["stop_purposes"].append(purpose)
                remaining -= d
                k_all += 1
        if not truncate:
            durs = fit_stop_durations([d for _, d in sampled["outbound"] + sampled["inbound"]], dur - 1.0, False, report)
            flat = [p for p, _ in sampled["outbound"] + sampled["inbound"]]
            rescaled = list(zip(flat, durs))
            sampled = {"outbound": rescaled[:n_out], "inbound": rescaled[n_out:]}
        t.outbound, t.inbound = sampled["outbound"], sampled["inbound"]


# ---- calibration statistics --------------------------------------------------------------------
def record_trace(rec, models: DaysimModels) -> dict:
    return {
        "pattern": models.map_pattern(rec.pattern),
        "tours": [(tod_group(t.purpose), t.start, t.end, len(t.outbound), len(t.inbound)) for t in rec.tours],
        "stop_purposes": [s.act for t in rec.tours for s in t.stops],
    }


def trace_stats(traces: list[dict], models: DaysimModels, timing: str = "tod") -> dict[str, dict]:
    out: dict[str, dict] = {"pattern": share_dict([t["pattern"] for t in traces], list(models.pattern_vectors))}
    tours = [x for t in traces for x in t["tours"]]
    for direction, col in (("outbound", 3), ("inbound", 4)):
        out[f"stops_{direction}"] = share_dict([min(x[col], MAX_STOPS) for x in tours], range(MAX_STOPS + 1))
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
