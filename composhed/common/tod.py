"""Joint start/end period choice with time-window availability (DAY-2/4, ASIM-4/6).

A ``TourTOD`` model is a ``ConditionalLogit`` over all (start period, end period)
pairs of a ``PeriodGrid`` with end ≥ start. Its utility is a ``TodSpec`` built by
the variant from the period pairs: named alternative attributes (grouped
departure / arrival / duration constants, linear "shift" variables, time-band
flags) and the (person feature, alternative attribute) pairs that get a
coefficient. ActivitySim's spec is ``activitysim.tod_spec`` (prototype_mtc
``tour_scheduling_*.csv``), DaySim's ``daysim.tod_spec`` (``*TourTimeModel.cs``).

Chooser-specific timetable terms (``dynamic``):

* ``"asim"`` (ASIM-4/6): the alternative departs in the period a previous tour
  ended / arrives in the period a later tour starts (ActivitySim
  ``previous_tour_ends`` / ``previous_tour_begins``), the gap to the adjacent tour
  before / after (``adjacent_window_before`` / ``_after``) and the look-ahead term
  ``tours still to schedule / free hours left``;
* ``"daysim"`` (DAY-2/4): log of the free minutes in the arrival and departure
  periods, adjacent free hours before arrival / after departure for later tours,
  stop purposes still to place / (1 + adjacent free hours), and tours still to
  schedule / (1 + free hours before and after), capped at 0.3 (WorkTourTimeModel
  terms 91–100).

Availability comes from ``TimeWindow.period_availability_mask``. A chosen pair is
turned into minutes by drawing each minute uniformly from the free minutes of
its period inside the window that made the pair available, as DaySim's
``TimeWindow.GetMinuteSpan`` does (ActivitySim stops at periods). Placed tours
never overlap and keep ``buffer`` minutes of home between them.

``IndependentTiming`` is the AB-2 ablation (CompoShed-style KDE start plus an
independent log-normal duration) and ``layout_rescale`` the AB-3 ablation.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde

from composhed.common.choice import ConditionalLogit
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import PeriodGrid
from composhed.common.timewindow import TimeWindow

DYN_NAMES = {
    "asim": ["prev_tour_ends_in_start_period", "next_tour_starts_in_end_period", "remaining_tours_per_free_hour",
             "gap_to_previous_tour", "gap_to_next_tour"],
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


class TourTOD:
    def __init__(self, grid: PeriodGrid, spec_fn, buffer: int = 10, l2: float = 1e-3, maxiter: int = 300,
                 dynamic: str | None = None, **spec_kwargs) -> None:
        self.grid = grid
        self.buffer = buffer
        self.dynamic = dynamic
        self.ps, self.pe = grid.pairs()
        self.start_lo, self.start_hi = grid.lo(self.ps), grid.hi(self.ps)
        self.end_lo, self.end_hi = grid.lo(self.pe), grid.hi(self.pe)
        self.mid_s = (self.start_lo + self.start_hi) / 2.0
        self.mid_e = (self.end_lo + self.end_hi) / 2.0
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

    def mask(self, tw: TimeWindow, min_duration: int = 1) -> np.ndarray:
        return tw.period_availability_mask(
            self.start_lo, self.start_hi, self.end_lo, self.end_hi, self.buffer, min_duration
        )

    # ---- chooser-specific timetable terms -------------------------------------
    def dyn_fn(self, states: list[tuple]):
        """Callable for ``ConditionalLogit``: rows s:e of ``states`` (see ``tw_state``) → (n, J, n_dyn)."""
        kind = getattr(self, "dynamic", None)
        if not kind:
            return None
        R = max([len(st[0]) for st in states] + [1])
        S = np.full((len(states), R), np.nan)
        E = np.full((len(states), R), np.nan)
        for i, st in enumerate(states):
            for r, (a, b) in enumerate(st[0]):
                S[i, r], E[i, r] = a, b
        nrem = np.array([st[1] for st in states], dtype=float)
        free = np.array([st[2] for st in states], dtype=float)
        later = np.array([float(st[3]) if len(st) > 3 else 0.0 for st in states])
        nstops = np.array([float(st[4]) if len(st) > 4 else 0.0 for st in states])
        if kind == "asim":
            return self._dyn_asim(S, E, nrem, free)
        return self._dyn_daysim(S, E, nrem, later, nstops)

    def _dyn_asim(self, S, E, nrem, free):
        valid = ~np.isnan(S)
        startp = np.where(valid, self.grid.period_of(np.nan_to_num(S)), -1)
        endp = np.where(valid, self.grid.end_period_of(np.nan_to_num(E, nan=1.0)), -1)
        ps, pe, ms, me = self.ps, self.pe, self.mid_s, self.mid_e
        dur = np.maximum(me - ms, 0.0)

        def dyn(s: int, e: int) -> np.ndarray:
            v = valid[s:e, None, :]
            prev = (ps[None, :, None] == endp[s:e, None, :]).any(axis=2)
            nxt = (pe[None, :, None] == startp[s:e, None, :]).any(axis=2)
            hours_left = np.maximum((free[s:e, None] - dur[None, :]) / 60.0, 0.5)
            look = nrem[s:e, None] / hours_left
            Eb = np.where(v & (E[s:e, None, :] <= ms[None, :, None]), E[s:e, None, :], -np.inf).max(axis=2)
            Sa = np.where(v & (S[s:e, None, :] >= me[None, :, None]), S[s:e, None, :], np.inf).min(axis=2)
            gap_b = np.where(np.isfinite(Eb), (ms[None, :] - Eb) / 1440.0, 0.0)
            gap_a = np.where(np.isfinite(Sa), (Sa - me[None, :]) / 1440.0, 0.0)
            return np.stack([prev, nxt, look, gap_b, gap_a], axis=2).astype(float)

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
        use_mask: bool = True, n_remaining: int = 0, later: bool = False, stops_left: int = 0,
    ) -> tuple[int, int] | None:
        """Sample an interval [start, end) in minutes; None if no pair is available.

        ``n_remaining``: tours still to schedule after this one; ``later``: not the
        first tour scheduled; ``stops_left``: stop purposes still to place (timetable terms).
        """
        self._sync_offsets()
        avail = self.mask(tw, min_duration) if use_mask else np.ones(len(self.ps), dtype=bool)
        if not avail.any():
            return None
        dyn = self.dyn_fn([tw_state(tw, n_remaining, later, stops_left)])
        j = int(self.clogit.sample(z[None, :], rng, avail[None, :], dyn)[0])
        if j < 0:
            return None
        return self.realise(j, tw if use_mask else TimeWindow(), rng, min_duration)

    def realise(self, j: int, tw: TimeWindow, rng: np.random.Generator, min_duration: int = 1) -> tuple[int, int] | None:
        """Minutes for pair ``j``: start uniform over its feasible minutes, then end uniform given the start."""
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
