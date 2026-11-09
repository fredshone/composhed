"""CEMDAP-aligned generation chain (CEM-1 … CEM-9).

As CEMDAP (Bhat et al., 2004), a worker's day rests on two pegs: the work-based
duration is drawn first and then the work start given it. The day is then
filled period by period: the before-work period (outbound commute stops, tours
before work), the after-work period (inbound commute stops, tours after work)
and the work-based period (work stays and work-based tours). Time is continuous
(minutes since the start of the day) and nothing has a minimum duration.

CEMDAP checks every predicted duration against the time still available. Here
every duration is drawn from its model truncated to the time left in its period,
so the day closes at 1440 by construction (CEM-5, CEM-6, CEM-9). The first and
last home stays of a mandatory day, the last work stay and the last home stay of
a non-mandatory day take the residual (CEM-8).

With ``consistency="rescale"`` (AB-3) durations are drawn untruncated and any
overflowing period is rescaled proportionally, as CompoShed does; a person whose
draw cannot be completed is redrawn (up to ``max_resample``). Both are counted.
"""

import numpy as np

from composhed.cemdap.data import DAY, MAX_NM_TOURS, MAX_POSITION_COUNT
from composhed.cemdap.models import (
    CemdapModels,
    x_counts,
    x_duration,
    x_first_departure,
    x_home,
    x_start,
    x_stop,
    x_tour_size,
)
from composhed.common.report import GenerationReport
from composhed.common.schedule import finalise


class Infeasible(RuntimeError):
    pass


class CemdapChain:
    def __init__(self, models: CemdapModels, consistency: str = "constraint", max_resample: int = 50) -> None:
        self.m = models
        self.consistency = consistency
        self.max_resample = max_resample

    # ---- helpers -----------------------------------------------------------------------
    @property
    def _truncate(self) -> bool:
        return self.consistency == "constraint"

    def _draw(self, model, x, rng, hi: float) -> float:
        """A duration in [0, hi] (untruncated above with ``consistency="rescale"``)."""
        hi = max(hi, 0.0)
        if model is None:
            # Component never observed in training (tiny data sets): uniform on a short window.
            return float(rng.uniform(0.0, min(hi, 60.0)))
        return float(model.sample(x[None, :], rng, lo=0.0, hi=hi if self._truncate else np.inf)[0])

    def _stops(self, xl, position, n, resid, rng, trace) -> tuple[list[tuple[str, float]], float]:
        """Sample n stops at ``position``; returns (stops, residual left)."""
        out = []
        for k in range(n):
            xs = x_stop(xl, position, k, resid)
            purpose = self.m.stop_purpose.sample(xs[None, :], rng)[0]
            dur = self._draw(self.m.stop_duration.get(purpose), xs, rng, resid)
            out.append((purpose, dur))
            trace["stops"].append((purpose, dur))
            resid -= dur
        return out, resid

    def _tour(self, xl, position, ti, resid, rng, trace):
        n = int(self.m.tour_size.sample(x_tour_size(xl, position, ti)[None, :], rng, min_allowed=1)[0])
        trace["tour_sizes"].append(n)
        return self._stops(xl, position, n, resid, rng, trace)

    def _stay(self, xl, side, resid, rng, trace) -> float:
        """Home stay between tours, or the work stay before a work-based tour."""
        h = self._draw(self.m.home, x_home(xl, side, resid), rng, resid)
        trace["home_stays"].append(h)
        return h

    # ---- person ----------------------------------------------------------------------------
    def generate(self, xl: np.ndarray, attrs: dict, rng: np.random.Generator,
                 report: GenerationReport) -> tuple[list[dict], dict]:
        for attempt in range(self.max_resample + 1):
            try:
                seq, trace = self._attempt(xl, str(attrs["employment"]), rng, report)
                break
            except Infeasible:
                report.resamples += 1
        else:
            raise Infeasible("no feasible draw within max_resample")
        return finalise(seq, report), trace

    def _attempt(self, xl, employment, rng, report):
        trace = {"mand": None, "stops": [], "tour_sizes": [], "home_stays": [], "counts": {}}
        work = self.m.work.sample(xl[None, :], rng)[0] == 1
        edu = (not work) and self.m.edu.sample(xl[None, :], rng)[0] == 1
        if (work or edu) and self.m.start is not None:
            return self._mandatory_day(xl, employment, "work" if work else "education", rng, report, trace), trace
        return self._nonmandatory_day(xl, rng, report, trace), trace

    def _mandatory_day(self, xl, employment, mand_type, rng, report, trace):
        is_edu = mand_type == "education"
        if self.m.independent_timing:
            kde = self.m.kde_start.get(employment) or self.m.kde_start.get("__all__")
            s = float(kde.resample(1, seed=rng)[0, 0]) if kde is not None else self.m.const_start
            start = float(np.clip(s, 0, DAY))
            dur = float(self.m.indep_duration.sample(x_duration(xl, is_edu)[None, :], rng, lo=0.0,
                                                     hi=DAY - start)[0])
        else:
            dur = self._draw(self.m.duration, x_duration(xl, is_edu), rng, DAY)
            start = self._draw(self.m.start, x_start(xl, is_edu, dur), rng, DAY - dur)
        if start + dur > DAY:
            raise Infeasible("work-based period overflows the day")
        trace["mand"] = (mand_type, start, dur)

        before, after = start, DAY - start - dur
        xc = x_counts(xl, is_edu, start, dur)[None, :]
        cnt = {k: int(self.m.counts[k].sample(xc, rng, max_allowed=MAX_POSITION_COUNT)[0])
               for k in ("n_out", "n_pre", "n_in", "n_post", "n_wb")}
        trace["counts"] = cnt

        # Before-work period, in generation order: outbound stops, then pre-tours each followed by a home stay.
        resid = before
        outbound, resid = self._stops(xl, "outbound", cnt["n_out"], resid, rng, trace)
        pre_tours: list[tuple[list, float]] = []
        for ti in range(cnt["n_pre"]):
            stops, resid = self._tour(xl, "pre_tour", ti, resid, rng, trace)
            h = self._stay(xl, "before", resid, rng, trace)
            resid -= h
            pre_tours.append((stops, h))
        before_items = [d for _, d in outbound] + [d for st, h in pre_tours for d in [h] + [x for _, x in st]]
        # After-work period: inbound stops, then each post-tour preceded by a home stay.
        resid = after
        inbound, resid = self._stops(xl, "inbound", cnt["n_in"], resid, rng, trace)
        post_tours: list[tuple[float, list]] = []
        for ti in range(cnt["n_post"]):
            h = self._stay(xl, "after", resid, rng, trace)
            resid -= h
            stops, resid = self._tour(xl, "post_tour", ti, resid, rng, trace)
            post_tours.append((h, stops))
        after_items = [d for _, d in inbound] + [d for h, st in post_tours for d in [h] + [x for _, x in st]]
        # Work-based period: each work-based tour preceded by a work stay; the last work stay is the residual.
        resid = dur
        wb_tours: list[tuple[float, list]] = []
        for ti in range(cnt["n_wb"]):
            w = self._stay(xl, "work", resid, rng, trace)
            resid -= w
            stops, resid = self._tour(xl, "wb_tour", ti, resid, rng, trace)
            wb_tours.append((w, stops))
        wb_items = [d for w, st in wb_tours for d in [w] + [x for _, x in st]]

        scale_b = self._fit_side(before_items, before, report)
        scale_a = self._fit_side(after_items, after, report)
        scale_w = self._fit_side(wb_items, dur, report)

        seq: list[tuple[str, float]] = []
        seq.append(("home", start - sum(before_items) * scale_b))
        for stops, h in pre_tours:
            seq += [(p, d * scale_b) for p, d in stops] + [("home", h * scale_b)]
        seq += [(p, d * scale_b) for p, d in outbound]
        for w, stops in wb_tours:
            seq += [(mand_type, w * scale_w)] + [(p, d * scale_w) for p, d in stops]
        seq.append((mand_type, dur - sum(wb_items) * scale_w))
        seq += [(p, d * scale_a) for p, d in inbound]
        for h, stops in post_tours:
            seq += [("home", h * scale_a)] + [(p, d * scale_a) for p, d in stops]
        seq.append(("home", DAY - start - dur - sum(after_items) * scale_a))
        return seq

    def _fit_side(self, items: list[float], budget: float, report: GenerationReport) -> float:
        total = sum(items)
        if total <= budget + 1e-9:
            return 1.0
        if self._truncate:
            raise Infeasible("period overflow")
        report.rescales += 1
        return budget / total

    def _nonmandatory_day(self, xl, rng, report, trace):
        n = int(self.m.nm_tours.sample(xl[None, :], rng, max_allowed=MAX_NM_TOURS)[0])
        trace["nm_tours"] = n
        if n == 0:
            return [("home", float(DAY))]
        if self.m.first_departure is not None:
            home0 = self._draw(self.m.first_departure, x_first_departure(xl, n), rng, DAY)
        else:
            home0 = float(rng.uniform(0.0, DAY))
        home0 = min(home0, float(DAY))
        trace["first_departure"] = home0
        resid = DAY - home0
        tours: list[tuple[list, float]] = []
        for ti in range(n):
            stops, resid = self._tour(xl, "nm_tour", ti, resid, rng, trace)
            h = 0.0
            if ti < n - 1:
                h = self._stay(xl, "nm", resid, rng, trace)
                resid -= h
            tours.append((stops, h))
        items = [d for st, h in tours for d in [h] + [x for _, x in st]]
        scale = self._fit_side(items, DAY - home0, report)
        seq = [("home", home0)]
        for ti, (stops, h) in enumerate(tours):
            seq += [(p, d * scale) for p, d in stops]
            if ti < n - 1:
                seq.append(("home", h * scale))
        seq.append(("home", DAY - home0 - sum(items) * scale))
        return seq
