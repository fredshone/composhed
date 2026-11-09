"""Joint start/end period choice with time-window availability (DAY-2/4, ASIM-4/6).

A ``TourTOD`` model is a ``ConditionalLogit`` over all (start period, end period)
pairs of a ``PeriodGrid`` with end ≥ start. Its utility is a ``TodSpec`` built by
the variant from the period pairs: named alternative attributes (grouped
departure / arrival / duration constants, linear "shift" variables, time-band
flags) and the (person feature, alternative attribute) pairs that get a
coefficient. ActivitySim's spec is ``activitysim.tod_spec`` (prototype_mtc
``tour_scheduling_*.csv``), DaySim's ``daysim.tod_spec`` (``*TourTimeModel.cs``).

Chooser-specific timetable terms (``dynamic``):

* ``"asim"`` (ASIM-4/6): ActivitySim's timetable rows, computed on its period
  timetable (``asim_state``): a previously scheduled tour ends in the departure
  period / begins in the arrival period (``previous_tour_ends`` /
  ``previous_tour_begins``); whether a free period is adjacent before the
  departure / after the arrival (``adjacent_window_before`` / ``_after`` > 0, as
  the specs use them), once for the first of 2+ tours and once for later tours;
  and tours still to schedule divided by the periods left free
  (``remaining_periods_available``);
* ``"daysim"`` (DAY-2/4): log of the free minutes in the arrival and departure
  periods, adjacent free hours before arrival / after departure for later tours,
  stop purposes still to place / (1 + adjacent free hours), and tours still to
  schedule / (1 + free hours before and after), capped at 0.3 (WorkTourTimeModel
  terms 91–100).

A chosen pair is turned into minutes in one of two ways, matching the source's
precision:

* minutes (DaySim): each minute is drawn uniformly from the free minutes of its
  period inside the window that made the pair available, as DaySim's
  ``TimeWindow.GetMinuteSpan`` does. Availability comes from
  ``TimeWindow.period_availability_mask``; placed tours never overlap and keep
  ``buffer`` minutes of home between them;
* ``points=True`` (ActivitySim, which schedules in periods and stops there):
  each period is written at one minute, its midpoint, so a tour starting and
  ending in one period lasts zero minutes. A pair is available when its interval
  overlaps no placed tour; as in ActivitySim's timetable, a tour may start in the
  period another ends. ``min_start`` (minutes) also removes pairs starting in an
  earlier period, for the specs' "subsequent tour must start after previous tour
  ends" rows.

``IndependentTiming`` is the AB-2 ablation (CompoShed-style KDE start plus an
independent log-normal duration) and ``layout_rescale`` the AB-3 ablation.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde

from composhed.common.choice import ConditionalLogit
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import DAY, PeriodGrid
from composhed.common.timewindow import TimeWindow

DYN_NAMES = {
    "asim": ["previous_tour_ends_in_departure_period", "previous_tour_begins_in_arrival_period",
             "adjacent_window_before_first_of_2plus", "adjacent_window_after_first_of_2plus",
             "adjacent_window_before_second_plus", "adjacent_window_after_second_plus",
             "tours_to_schedule_per_remaining_period"],
    "daysim": ["log_free_minutes_arrival_period", "log_free_minutes_departure_period",
               "adjacent_hours_before_later_tour", "adjacent_hours_after_later_tour",
               "stops_left_per_adjacent_hour_before", "stops_left_per_adjacent_hour_after",
               "tours_left_per_free_hours", "tours_left_per_max_free_hours"],
}
DYN_PRECOMPUTE_BYTES = 12e9  # chooser-specific attributes are precomputed for estimation below this size (float32)


@dataclass
class TodSpec:
    """Utility of a ``TourTOD``: ``V_nj = sum over terms (z, g) of beta · z_n[z] · g_j[g]``.

    ``g``: alternative attribute name → (J,) column; ``terms``: the (person feature,
    alternative attribute) pairs with a coefficient (a constant is ``("const", g)``);
    ``z_names``: person features, in the order of the ``z`` vectors; ``dbin`` (J,)
    groups the alternatives by duration band (the calibration constants).
    """

    z_names: list[str]
    g: dict
    terms: list[tuple[str, str]]
    dbin: np.ndarray
    n_dbin: int

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        g_names = list(self.g)
        G = np.column_stack([np.asarray(self.g[k], dtype=float) for k in g_names])
        mask = np.zeros((len(self.z_names), len(g_names)), dtype=bool)
        for z, gname in self.terms:
            mask[self.z_names.index(z), g_names.index(gname)] = True
        keep = mask.any(axis=0)
        self.g_names = [k for k, kk in zip(g_names, keep) if kk]
        return G[:, keep], mask[:, keep]

    def z(self, values: dict) -> np.ndarray:
        """Person-feature vector from a {name: value} dict (missing names are 0)."""
        return np.array([float(values.get(k, 0.0)) for k in self.z_names])


def tw_state(tw: TimeWindow, n_remaining: int = 0, later: bool = False, stops_left: int = 0) -> tuple:
    """Compact timetable state for the dynamic terms.

    (reserved intervals, tours still to schedule, free minutes, later tour flag, stop purposes still to place).
    """
    return tw.busy_intervals(), int(n_remaining), tw.remaining(), bool(later), int(stops_left)


# ActivitySim timetable period states (core/timetable.py).
I_EMPTY, I_START, I_END, I_START_END, I_MIDDLE = 0, 2, 4, 6, 7


def asim_state(tw: TimeWindow, grid: PeriodGrid, first: bool = False, second: bool = False,
               numerator: float = 0.0) -> tuple:
    """ActivitySim timetable row of the tours reserved in ``tw`` (one state per period, as ``assign`` builds it by
    OR-ing tour footprints), with the scheduling context of the tour being chosen: first of 2+ tours, a later
    tour, and the numerator of the tours-to-schedule term."""
    w = np.zeros(grid.n, dtype=np.int8)
    for s, e in tw.tours:
        ps = int(grid.period_of(min(s, DAY - 1)))
        pe = ps if e == s else int(grid.end_period_of(e))
        if pe > ps:
            w[ps] |= I_START
            w[ps + 1:pe] |= I_MIDDLE
            w[pe] |= I_END
        else:
            w[ps] |= I_START_END
    return w, bool(first), bool(second), float(numerator)


class TourTOD:
    def __init__(self, grid: PeriodGrid, spec_fn, buffer: int = 0, l2: float = 1e-3, maxiter: int = 300,
                 dynamic: str | None = None, points: bool = False, **spec_kwargs) -> None:
        self.grid = grid
        self.buffer = buffer
        self.points = points
        self.dynamic = dynamic
        self.ps, self.pe = grid.pairs()
        self.start_lo, self.start_hi = grid.lo(self.ps), grid.hi(self.ps)
        self.end_lo, self.end_hi = grid.lo(self.pe), grid.hi(self.pe)
        self.mid_s = (self.start_lo + self.start_hi) / 2.0
        self.mid_e = (self.end_lo + self.end_hi) / 2.0
        self.pt_s, self.pt_e = grid.points()[self.ps], grid.points()[self.pe]
        self.spec: TodSpec = spec_fn(grid, self.ps, self.pe, **spec_kwargs)
        G, mask = self.spec.matrices()
        J = len(self.ps)
        n_dyn = len(DYN_NAMES[dynamic]) if dynamic else 0
        self.clogit = ConditionalLogit(sp.csr_matrix((J, 0)), G, l2=l2, maxiter=maxiter, n_dyn=n_dyn, b_mask=mask,
                                       chunk=1000 if J > 500 else 4000)
        self.dbin = self.spec.dbin
        self.n_dur_bins = self.spec.n_dbin
        self.alt_index = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(self.ps, self.pe))}
        # Calibration constants on start-period, end-period and duration-band marginals.
        self.start_offset = np.zeros(grid.n)
        self.end_offset = np.zeros(grid.n)
        self.dur_offset = np.zeros(self.n_dur_bins)

    def z(self, values: dict) -> np.ndarray:
        return self.spec.z(values)

    # ---- alternatives --------------------------------------------------------
    def alt_of(self, start: int, end: int) -> int:
        return self.alt_index[(int(self.grid.period_of(start)), int(self.grid.end_period_of(end)))]

    def mask(self, tw: TimeWindow, min_duration: int = 1, min_start: float | None = None) -> np.ndarray:
        if self.points:
            m = point_mask(tw, self.pt_s, self.pt_e)
        else:
            m = tw.period_availability_mask(
                self.start_lo, self.start_hi, self.end_lo, self.end_hi, self.buffer, min_duration
            )
        if min_start is not None:
            m &= self.ps >= self.grid.period_of(min_start)
        return m

    # ---- chooser-specific timetable terms -------------------------------------
    def dyn_fn(self, states: list[tuple]):
        """Callable for ``ConditionalLogit``: rows s:e of ``states`` (see ``tw_state``) → (n, J, n_dyn)."""
        kind = getattr(self, "dynamic", None)
        if not kind:
            return None
        if kind == "asim":
            return self._dyn_asim(states)
        R = max([len(st[0]) for st in states] + [1])
        S = np.full((len(states), R), np.nan)
        E = np.full((len(states), R), np.nan)
        for i, st in enumerate(states):
            for r, (a, b) in enumerate(st[0]):
                S[i, r], E[i, r] = a, b
        nrem = np.array([st[1] for st in states], dtype=float)
        later = np.array([float(st[3]) if len(st) > 3 else 0.0 for st in states])
        nstops = np.array([float(st[4]) if len(st) > 4 else 0.0 for st in states])
        return self._dyn_daysim(S, E, nrem, later, nstops)

    def state(self, tw: TimeWindow, **kw) -> tuple:
        """Timetable state for the dynamic terms (``asim_state`` or ``tw_state``)."""
        if self.dynamic == "asim":
            return asim_state(tw, self.grid, **kw)
        return tw_state(tw, **kw)

    def _dyn_asim(self, states):
        W = np.stack([st[0] for st in states]) if states else np.zeros((0, self.grid.n), dtype=np.int8)
        first = np.array([st[1] for st in states], dtype=float)
        second = np.array([st[2] for st in states], dtype=float)
        num = np.array([st[3] for st in states], dtype=float)
        n, P = W.shape
        free = W != I_MIDDLE
        # Free periods adjacent before / after each period, up to a period in the middle of a tour or the day edge;
        # the specs use whether there is one (``_adjacent_window_before@tt.adjacent_window_before(...) > 0``).
        adj_b, adj_a = np.zeros((n, P)), np.zeros((n, P))
        for p in range(1, P):
            adj_b[:, p] = np.where(free[:, p - 1], adj_b[:, p - 1] + 1, 0)
        for p in range(P - 2, -1, -1):
            adj_a[:, p] = np.where(free[:, p + 1], adj_a[:, p + 1] + 1, 0)
        ends = np.isin(W, (I_END, I_START_END))
        begins = np.isin(W, (I_START, I_START_END))
        n_free = free.sum(axis=1).astype(float)
        ps, pe = self.ps, self.pe
        interior = np.maximum(pe - ps - 1, 0)

        def dyn(s: int, e: int) -> np.ndarray:
            ab, aa = adj_b[s:e][:, ps] > 0, adj_a[s:e][:, pe] > 0
            f, sec = first[s:e, None], second[s:e, None]
            # ActivitySim divides by the periods left; a tour leaving none gets 1 (it would divide by zero).
            remaining = np.maximum(n_free[s:e, None] - interior[None, :], 1.0)
            return np.stack([ends[s:e][:, ps], begins[s:e][:, pe], f * ab, f * aa, sec * ab, sec * aa,
                             num[s:e, None] / remaining], axis=2).astype(float)

        return dyn

    def _dyn_daysim(self, S, E, nrem, later, nstops):
        """DaySim tour-time terms 91–100, with the reserved intervals ``[S, E)`` plus the home buffer as busy."""
        a_lo, a_hi, d_lo, d_hi = self.start_lo, self.start_hi, self.end_lo, self.end_hi
        buf = self.buffer
        valid = ~np.isnan(S)
        # Busy spans including the home buffer, clipped to the day.
        Bs = np.where(valid, np.maximum(np.nan_to_num(S) - buf, 0), 0.0)
        Be = np.where(valid, np.minimum(np.nan_to_num(E) + buf, 1440), 0.0)

        def overlap(lo, hi, s, e):
            """Busy minutes of [lo, hi) given busy spans (n, 1, R): (n, J)."""
            return np.clip(np.minimum(hi[None, :, None], Be[s:e, None, :]) - np.maximum(lo[None, :, None], Bs[s:e, None, :]),
                           0, None).sum(axis=2)

        def dyn(s: int, e: int) -> np.ndarray:
            v = valid[s:e, None, :]
            free_a = (a_hi - a_lo)[None, :] - overlap(a_lo, a_hi, s, e)
            free_d = (d_hi - d_lo)[None, :] - overlap(d_lo, d_hi, s, e)
            # Adjacent free minutes: from the end of the latest busy span before arrival / to the next one after departure.
            before_end = np.where(v & (Be[s:e, None, :] <= a_lo[None, :, None]), Be[s:e, None, :], -np.inf).max(axis=2)
            after_start = np.where(v & (Bs[s:e, None, :] >= d_hi[None, :, None]), Bs[s:e, None, :], np.inf).min(axis=2)
            adj_b = (a_lo[None, :] - np.where(np.isfinite(before_end), before_end, 0.0)) / 60.0
            adj_a = (np.where(np.isfinite(after_start), after_start, 1440.0) - d_hi[None, :]) / 60.0
            busy_before = overlap(np.zeros_like(a_lo), a_lo, s, e)
            busy_after = overlap(d_hi, np.full_like(d_hi, 1440), s, e)
            tot_b = (a_lo[None, :] - busy_before) / 60.0
            tot_a = (1440 - d_hi[None, :] - busy_after) / 60.0
            lat = later[s:e, None]
            ns = nstops[s:e, None]
            nr = nrem[s:e, None]
            out = np.stack([
                np.log(np.maximum(free_a, 1.0)), np.log(np.maximum(free_d, 1.0)),
                lat * adj_b, lat * adj_a,
                ns / (1.0 + adj_b), ns / (1.0 + adj_a),
                np.minimum(0.3, nr / (1.0 + tot_b + tot_a)),
                np.minimum(0.3, nr / (1.0 + np.maximum(tot_b, tot_a))),
            ], axis=2)
            return out

        return dyn

    def _sync_offsets(self) -> None:
        self.clogit.offset = self.start_offset[self.ps] + self.end_offset[self.pe] + self.dur_offset[self.dbin]

    def _estimation_dyn(self, states: list[tuple] | None):
        """Chooser-specific attributes for estimation, precomputed once when they fit in memory."""
        if states is None or not self.dynamic:
            return None
        dyn = self.dyn_fn(states)
        n, J, k = len(states), len(self.ps), len(DYN_NAMES[self.dynamic])
        if n * J * k * 4 > DYN_PRECOMPUTE_BYTES:
            return dyn
        A = np.empty((n, J, k), dtype=np.float32)
        for s in range(0, n, 2000):
            A[s:s + 2000] = dyn(s, min(n, s + 2000))
        return lambda s, e: A[s:e]

    # ---- estimation ------------------------------------------------------------
    def fit(self, Z: np.ndarray, starts: np.ndarray, ends: np.ndarray, masks: np.ndarray,
            states: list[tuple] | None = None) -> "TourTOD":
        y = np.array([self.alt_of(s, e) for s, e in zip(starts, ends)], dtype=int)
        self.clogit.fit(Z, y, masks, self._estimation_dyn(states))
        return self

    def coef_table(self) -> dict:
        B = self.clogit.B_
        out = {f"{z}*{g}": float(B[i, j]) for i, z in enumerate(self.spec.z_names)
               for j, g in enumerate(self.spec.g_names) if self.clogit.b_mask[i, j]}
        if self.dynamic:
            out.update(dict(zip(DYN_NAMES[self.dynamic], self.clogit.delta_.tolist())))
        return out

    def validate(self, Z: np.ndarray, starts: np.ndarray, ends: np.ndarray, masks: np.ndarray,
                 states: list[tuple] | None = None) -> dict:
        y = np.array([self.alt_of(s, e) for s, e in zip(starts, ends)], dtype=int)
        dyn = self.dyn_fn(states) if states is not None else None
        m = self.clogit.validate(Z, y, masks, dyn)
        P = self.clogit.predict_proba(Z, masks, dyn) if len(y) else np.zeros((0, len(self.ps)))
        m["n_params"] = getattr(self.clogit, "n_params_", None)
        m["mean_start_obs"] = float(np.mean(starts)) if len(y) else float("nan")
        m["mean_start_pred"] = float((P @ self.mid_s).mean()) if len(y) else float("nan")
        m["forced_available"] = getattr(self.clogit, "forced_available_", 0)
        if self.dynamic:
            m["timetable_terms"] = dict(zip(DYN_NAMES[self.dynamic], self.clogit.delta_.round(4).tolist()))
        return m

    # ---- generation ------------------------------------------------------------
    def sample(
        self, z: np.ndarray, tw: TimeWindow, rng: np.random.Generator, min_duration: int = 1,
        use_mask: bool = True, min_start: float | None = None, **state_kw,
    ) -> tuple[int, int] | None:
        """Sample an interval [start, end) in minutes; None if no pair is available.

        ``state_kw``: the tour's context for the timetable terms (``tw_state`` / ``asim_state`` arguments).
        """
        self._sync_offsets()
        avail = self.mask(tw, min_duration, min_start) if use_mask else np.ones(len(self.ps), dtype=bool)
        if not avail.any():
            return None
        dyn = self.dyn_fn([self.state(tw, **state_kw)])
        j = int(self.clogit.sample(z[None, :], rng, avail[None, :], dyn)[0])
        if j < 0:
            return None
        return self.realise(j, tw if use_mask else TimeWindow(), rng, min_duration)

    def realise(self, j: int, tw: TimeWindow, rng: np.random.Generator, min_duration: int = 1) -> tuple[int, int] | None:
        """Minutes for pair ``j``: the period points, or start uniform over its feasible minutes and then end
        uniform given the start."""
        if self.points:
            return int(self.pt_s[j]), int(self.pt_e[j])
        sl, sh, el, eh = (int(v[j]) for v in (self.start_lo, self.start_hi, self.end_lo, self.end_hi))
        for a, b in tw.usable_windows(self.buffer):
            s_min, e_max = max(sl, a), min(eh, b)
            # The end must fall in its period, (el, eh], and at least min_duration after the start.
            s_max = min(sh - 1, e_max - min_duration)
            if s_min > s_max or e_max <= el:
                continue
            s = int(rng.integers(s_min, s_max + 1))
            e_lo = max(el + 1, s + min_duration)
            if e_lo > e_max:
                continue
            e = int(rng.integers(e_lo, e_max + 1))
            return s, e
        return None


def point_mask(tw: TimeWindow, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Intervals [s, e) of whole minutes that overlap no reserved tour and span no zero-length one; a zero-length
    one must not fall strictly inside a reserved tour."""
    busy = tw.busy
    cs = np.concatenate([[0], np.cumsum(busy)])
    s, e = np.asarray(s, dtype=int), np.asarray(e, dtype=int)
    inside = busy[s] & busy[np.maximum(s - 1, 0)] & (s > 0)
    free = cs[e] == cs[s]
    for p in tw.points:
        free &= ~((s < p) & (p < e))
    return np.where(e > s, free, ~inside)


class IndependentTiming:
    """AB-2: KDE start minute plus independent log-normal duration (CompoShed style)."""

    def fit(self, X: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> "IndependentTiming":
        starts = np.asarray(starts, dtype=float)
        jitter = np.random.default_rng(0).normal(0, 1e-3, len(starts))
        self.kde_ = gaussian_kde(starts + jitter) if len(np.unique(starts)) > 1 else None
        self.const_start_ = float(starts.mean()) if len(starts) else 480.0
        self.dur_ = LogNormalDuration().fit(X, np.asarray(ends) - starts)
        return self

    def sample(self, x: np.ndarray, tw: TimeWindow, rng: np.random.Generator, buffer: int,
               max_resample: int, use_mask: bool = True) -> tuple[tuple[int, int] | None, int]:
        """Returns (interval or None, resamples used)."""
        for attempt in range(max_resample + 1):
            s = float(self.kde_.resample(1, seed=rng)[0, 0]) if self.kde_ is not None else self.const_start_
            d = float(self.dur_.sample(x[None, :], rng, lo=1.0, hi=1438.0)[0])
            s = int(np.clip(round(s), 1, 1438))
            e = int(min(1439, s + max(1, round(d))))
            if e <= s:
                continue
            if not use_mask:
                return (s, e), attempt
            for a, b in tw.usable_windows(buffer):
                if a <= s and e <= b:
                    return (s, e), attempt
        return None, max_resample


def layout_rescale(intervals: list[tuple[float, float]], buffer: int) -> tuple[list[tuple[float, float]], bool]:
    """AB-3: make overlapping tour intervals feasible by shifting and proportional rescaling.

    Tours keep their sampled order (by start). Each starts no earlier than the
    previous end plus ``buffer``; if the day then overflows, all tour durations are
    scaled down so the sequence fits in [1, 1439]. Returns (intervals, changed).
    """
    if not intervals:
        return [], False
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    ivs = [intervals[i] for i in order]
    durs = np.array([max(e - s, 1.0) for s, e in ivs])
    gaps = buffer * (len(ivs) - 1)
    avail = 1438.0 - gaps
    changed = False
    if durs.sum() > avail:
        durs = durs * (avail / durs.sum())
        changed = True
    out = []
    cursor = 1.0
    for (s, _), d in zip(ivs, durs):
        start = max(s, cursor)
        if start != s:
            changed = True
        out.append([start, start + d])
        cursor = start + d + buffer
    overflow = out[-1][1] - 1439.0
    if overflow > 0:
        changed = True
        # Pull tours back towards the morning, keeping order and buffers.
        limit = 1439.0
        for k in reversed(range(len(out))):
            d = out[k][1] - out[k][0]
            end = min(out[k][1], limit)
            out[k] = [end - d, end]
            limit = end - d - buffer
    return [(float(s), float(e)) for s, e in out], changed
