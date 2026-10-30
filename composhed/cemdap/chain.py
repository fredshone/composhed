"""CEMDAP-aligned generation chain (CEM-1 … CEM-9).

Every duration is drawn from its model *truncated* to the residual window that
is left once a minimum is reserved for the items still to come (1 min per stop,
``HOME_INTER`` per between-tour home stay, 1 min for the first / last home), so
the day closes at 1440 by construction (CEM-5, CEM-6, CEM-9). Counts are capped
at what can still fit. The first home stay of a mandatory day and the last home
stay of every day take the residual (CEM-8).

If a person still cannot be completed, the whole person is resampled (up to
``max_resample``); proportional rescaling is the last resort. Both events are
counted. With ``consistency="rescale"`` (AB-3) durations are drawn untruncated
and any overflowing side of the day is rescaled proportionally, as CompoShed does.
"""

import numpy as np

from composhed.cemdap.data import (
    DAY,
    MAND_START_ORIGIN,
    MAX_NM_TOURS,
    MAX_POSITION_COUNT,
)
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

HOME_INTER = 10


class Infeasible(RuntimeError):
    pass


class CemdapChain:
    def __init__(self, models: CemdapModels, consistency: str = "constraint", max_resample: int = 50,
                 home_inter: int = HOME_INTER) -> None:
        self.m = models
        self.consistency = consistency
        self.max_resample = max_resample
        self.home_inter = home_inter

    # ---- helpers -----------------------------------------------------------------------
    @property
    def _truncate(self) -> bool:
        return self.consistency == "constraint"

    def _draw(self, model, x, rng, lo: float, hi: float) -> float:
        if self._truncate and hi < lo:
            raise Infeasible("empty duration window")
        if model is None:
            # Component never observed in training (tiny data sets): uniform on a short window.
            return float(rng.uniform(lo, max(lo, min(hi, lo + 60.0))))
        return float(model.sample(x[None, :], rng, lo=lo, hi=hi if self._truncate else np.inf)[0])

    def _stops(self, xl, position, n, resid, reserve_after, rng, trace) -> tuple[list[tuple[str, float]], float]:
        """Sample n stops at ``position``; returns (stops, residual left)."""
        out = []
        for k in range(n):
            xs = x_stop(xl, position, k, resid)
            purpose = self.m.stop_purpose.sample(xs[None, :], rng)[0]
            reserve = (n - k - 1) + reserve_after
            dur = self._draw(self.m.stop_duration.get(purpose), xs, rng, 1.0, resid - reserve)
            out.append((purpose, dur))
            trace["stops"].append((purpose, dur))
            resid -= dur
        return out, resid

    def _tour(self, xl, position, ti, resid, reserve_after, rng, trace):
        max_n = int(max(1, min(4, resid - reserve_after)))
        n = int(self.m.tour_size.sample(x_tour_size(xl, position, ti)[None, :], rng, max_allowed=max_n,
                                        min_allowed=1)[0])
        trace["tour_sizes"].append(n)
        return self._stops(xl, position, n, resid, reserve_after, rng, trace)

    def _home(self, xl, side, resid, reserve_after, rng, trace) -> float:
        lo = float(self.home_inter)
        h = self._draw(self.m.home, x_home(xl, side, resid), rng, lo, resid - reserve_after)
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
        hi_inter = self.home_inter
        is_edu = mand_type == "education"
        if self.m.independent_timing:
            kde = self.m.kde_start.get(employment) or self.m.kde_start.get("__all__")
            s = float(kde.resample(1, seed=rng)[0, 0]) if kde is not None else self.m.const_start
            start = float(np.clip(s, 1, DAY - 3))
            dur = self._draw(self.m.indep_duration, x_start(xl, is_edu), rng, 1.0, DAY - start - 2)
            dur = min(dur, DAY - start - 2)
        else:
            t = float(self.m.start.sample(x_start(xl, is_edu)[None, :], rng, lo=1.0,
                                           hi=DAY - 3 - MAND_START_ORIGIN)[0])
            start = t + MAND_START_ORIGIN
            dur = float(self.m.duration.sample(x_duration(xl, is_edu, start)[None, :], rng, lo=1.0,
                                               hi=DAY - start - 2)[0])
        trace["mand"] = (mand_type, start, dur)

        before = start - 1.0  # minutes available before the block, keeping >= 1 min of first home
        after = DAY - start - dur - 1.0
        xc = x_counts(xl, is_edu, start, dur)[None, :]
        cnt = {}
        cnt["n_out"] = int(self.m.counts["n_out"].sample(xc, rng, max_allowed=min(MAX_POSITION_COUNT, int(before)))[0])
        cnt["n_pre"] = int(self.m.counts["n_pre"].sample(
            xc, rng, max_allowed=min(MAX_POSITION_COUNT, int((before - cnt["n_out"]) // (1 + hi_inter))))[0])
        cnt["n_in"] = int(self.m.counts["n_in"].sample(xc, rng, max_allowed=min(MAX_POSITION_COUNT, int(after)))[0])
        cnt["n_post"] = int(self.m.counts["n_post"].sample(
            xc, rng, max_allowed=min(MAX_POSITION_COUNT, int((after - cnt["n_in"]) // (1 + hi_inter))))[0])
        trace["counts"] = cnt

        # Before side, in generation order: outbound stops, then pre-tours each followed by a home stay.
        resid = before
        outbound, resid = self._stops(xl, "outbound", cnt["n_out"], resid, cnt["n_pre"] * (1 + hi_inter), rng, trace)
        pre_tours: list[tuple[list, float]] = []
        for ti in range(cnt["n_pre"]):
            reserve = (cnt["n_pre"] - ti - 1) * (1 + hi_inter)
            stops, resid = self._tour(xl, "pre_tour", ti, resid, hi_inter + reserve, rng, trace)
            h = self._home(xl, "before", resid, reserve, rng, trace)
            resid -= h
            pre_tours.append((stops, h))
        before_items = [d for _, d in outbound] + [d for st, h in pre_tours for d in [h] + [x for _, x in st]]
        # After side: inbound stops, then each post-tour preceded by a home stay.
        resid = after
        inbound, resid = self._stops(xl, "inbound", cnt["n_in"], resid, cnt["n_post"] * (1 + hi_inter), rng, trace)
        post_tours: list[tuple[float, list]] = []
        for ti in range(cnt["n_post"]):
            reserve = (cnt["n_post"] - ti - 1) * (1 + hi_inter)
            h = self._home(xl, "after", resid, 1 + reserve, rng, trace)
            resid -= h
            stops, resid = self._tour(xl, "post_tour", ti, resid, reserve, rng, trace)
            post_tours.append((h, stops))
        after_items = [d for _, d in inbound] + [d for h, st in post_tours for d in [h] + [x for _, x in st]]

        scale_b = self._fit_side(before_items, before, report)
        scale_a = self._fit_side(after_items, after, report)

        seq: list[tuple[str, float]] = []
        b_used = sum(before_items) * scale_b
        seq.append(("home", start - b_used))
        for stops, h in pre_tours:
            seq += [(p, d * scale_b) for p, d in stops] + [("home", h * scale_b)]
        seq += [(p, d * scale_b) for p, d in outbound]
        seq.append((mand_type, dur))
        seq += [(p, d * scale_a) for p, d in inbound]
        for h, stops in post_tours:
            seq += [("home", h * scale_a)] + [(p, d * scale_a) for p, d in stops]
        a_used = sum(after_items) * scale_a
        seq.append(("home", DAY - start - dur - a_used))
        return seq

    def _fit_side(self, items: list[float], budget: float, report: GenerationReport) -> float:
        total = sum(items)
        if total <= budget + 1e-9:
            return 1.0
        if self._truncate:
            raise Infeasible("side overflow")
        report.rescales += 1
        return budget / total

    def _nonmandatory_day(self, xl, rng, report, trace):
        hi_inter = self.home_inter
        max_tours = int(min(MAX_NM_TOURS, (DAY - 2 + hi_inter) // (1 + hi_inter)))
        n = int(self.m.nm_tours.sample(xl[None, :], rng, max_allowed=max_tours)[0])
        trace["nm_tours"] = n
        if n == 0:
            return [("home", float(DAY))]
        reserve_all = n * 1 + (n - 1) * hi_inter + 1
        if self.m.first_departure is not None:
            home0 = self._draw(self.m.first_departure, x_first_departure(xl, n), rng, 1.0, DAY - reserve_all)
        else:
            home0 = float(rng.uniform(1.0, DAY - reserve_all))
        trace["first_departure"] = home0
        resid = DAY - home0 - 1.0
        tours: list[tuple[list, float]] = []
        for ti in range(n):
            reserve = (n - ti - 1) * (1 + hi_inter)
            last = ti == n - 1
            stops, resid = self._tour(xl, "nm_tour", ti, resid, (0 if last else hi_inter) + reserve, rng, trace)
            h = 0.0
            if not last:
                h = self._home(xl, "nm", resid, reserve, rng, trace)
                resid -= h
            tours.append((stops, h))
        items = [d for st, h in tours for d in [h] + [x for _, x in st]]
        scale = self._fit_side(items, DAY - home0 - 1.0, report)
        seq = [("home", home0)]
        for stops, h in tours:
            seq += [(p, d * scale) for p, d in stops]
            if h:
                seq.append(("home", h * scale))
        seq.append(("home", DAY - home0 - sum(items) * scale))
        return seq
