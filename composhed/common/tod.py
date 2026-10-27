"""Joint tour start/end period choice with time-window availability (DAY-2/4, ASIM-4/6).

A ``TourTOD`` model is a ``ConditionalLogit`` over all (start period, end period)
pairs of a ``PeriodGrid`` with end ≥ start. Alternative attributes are:

* generic (sparse): start-period, end-period and implied-duration-bin dummies,
  i.e. the constants / shape terms of DaySim and ActivitySim TOD models;
* shape (dense, interacted with person/tour features ``z``): start, end, duration
  and duration² (fractions of the day).

Availability comes from ``TimeWindow.period_availability_mask``. Chosen periods
are converted to minutes by ``IntraPeriodSampler`` restricted to the window that
made the pair available, so placed tours never overlap and keep ``buffer``
minutes of home between them.

``IndependentTiming`` is the AB-2 ablation (CompoShed-style KDE start plus an
independent log-normal duration) and ``layout_rescale`` the AB-3 ablation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde

from composhed.common.choice import ConditionalLogit
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import IntraPeriodSampler, PeriodGrid
from composhed.common.timewindow import TimeWindow

DUR_BIN_EDGES = [0, 60, 120, 180, 240, 360, 480, 600, 720, 1440]


class TourTOD:
    def __init__(self, grid: PeriodGrid, buffer: int = 10, l2: float = 1e-3, maxiter: int = 300) -> None:
        self.grid = grid
        self.buffer = buffer
        self.ps, self.pe = grid.pairs()
        self.start_lo, self.start_hi = grid.lo(self.ps), grid.hi(self.ps)
        self.end_lo, self.end_hi = grid.lo(self.pe), grid.hi(self.pe)
        mid_s = (self.start_lo + self.start_hi) / 2.0
        mid_e = (self.end_lo + self.end_hi) / 2.0
        dur = np.maximum(mid_e - mid_s, 0.0)
        dbin = np.clip(np.searchsorted(DUR_BIN_EDGES, dur, side="right") - 1, 0, len(DUR_BIN_EDGES) - 2)
        J, n = len(self.ps), grid.n
        nb = len(DUR_BIN_EDGES) - 1
        rows = np.repeat(np.arange(J), 3)
        # Drop the first dummy of each block for identification.
        cols = np.stack([self.ps - 1, n - 1 + self.pe - 1, 2 * (n - 1) + dbin - 1], axis=1).ravel()
        keep = np.stack([self.ps > 0, self.pe > 0, dbin > 0], axis=1).ravel()
        F = sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(J, 2 * (n - 1) + nb - 1))
        G = np.column_stack([mid_s / 1440, mid_e / 1440, dur / 1440, (dur / 1440) ** 2])
        self.clogit = ConditionalLogit(F, G, l2=l2, maxiter=maxiter)
        self.dbin = dbin
        self.n_dur_bins = nb
        self.alt_index = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(self.ps, self.pe))}
        self.intra = IntraPeriodSampler(grid)
        self.intra_end = IntraPeriodSampler(grid)
        # Calibration constants on start-period, end-period and duration-bin marginals.
        self.start_offset = np.zeros(grid.n)
        self.end_offset = np.zeros(grid.n)
        self.dur_offset = np.zeros(nb)

    # ---- alternatives --------------------------------------------------------
    def alt_of(self, start: int, end: int) -> int:
        return self.alt_index[(int(self.grid.period_of(start)), int(self.grid.end_period_of(end)))]

    def mask(self, tw: TimeWindow, min_duration: int = 1) -> np.ndarray:
        return tw.period_availability_mask(
            self.start_lo, self.start_hi, self.end_lo, self.end_hi, self.buffer, min_duration
        )

    def _sync_offsets(self) -> None:
        self.clogit.offset = self.start_offset[self.ps] + self.end_offset[self.pe] + self.dur_offset[self.dbin]

    # ---- estimation ------------------------------------------------------------
    def fit(self, Z: np.ndarray, starts: np.ndarray, ends: np.ndarray, masks: np.ndarray, key: str) -> "TourTOD":
        y = np.array([self.alt_of(s, e) for s, e in zip(starts, ends)], dtype=int)
        self.clogit.fit(Z, y, masks)
        self.intra.fit({key: list(starts)})
        self.intra_end.fit({key: list(ends)}, end=True)
        self.key = key
        return self

    def validate(self, Z: np.ndarray, starts: np.ndarray, ends: np.ndarray, masks: np.ndarray) -> dict:
        y = np.array([self.alt_of(s, e) for s, e in zip(starts, ends)], dtype=int)
        m = self.clogit.validate(Z, y, masks)
        P = self.clogit.predict_proba(Z, masks) if len(y) else np.zeros((0, len(self.ps)))
        m["mean_start_obs"] = float(np.mean(starts)) if len(y) else float("nan")
        m["mean_start_pred"] = float((P @ ((self.start_lo + self.start_hi) / 2)).mean()) if len(y) else float("nan")
        m["forced_available"] = getattr(self.clogit, "forced_available_", 0)
        return m

    # ---- generation ------------------------------------------------------------
    def sample(
        self, z: np.ndarray, tw: TimeWindow, rng: np.random.Generator, min_duration: int = 1,
        use_mask: bool = True,
    ) -> tuple[int, int] | None:
        """Sample a tour interval [start, end) in minutes; None if no pair is available."""
        self._sync_offsets()
        avail = self.mask(tw, min_duration) if use_mask else np.ones(len(self.ps), dtype=bool)
        if not avail.any():
            return None
        j = int(self.clogit.sample(z[None, :], rng, avail[None, :])[0])
        if j < 0:
            return None
        return self.realise(j, tw if use_mask else TimeWindow(), rng, min_duration)

    def realise(self, j: int, tw: TimeWindow, rng: np.random.Generator, min_duration: int = 1) -> tuple[int, int] | None:
        sl, sh, el, eh = (int(v[j]) for v in (self.start_lo, self.start_hi, self.end_lo, self.end_hi))
        for a, b in tw.usable_windows(self.buffer):
            s_min, e_max = max(sl, a), min(eh, b)
            if s_min < sh and e_max > el and e_max - s_min >= min_duration:
                s = self.intra.sample(self.key, self.ps[j], s_min, min(sh, e_max - min_duration + 1), rng)
                e = self.intra_end.sample(self.key, self.pe[j], max(el, s + min_duration - 1), e_max, rng, end=True)
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
