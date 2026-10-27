"""SR-11/12: time-period grids and within-period minute resolution."""

import numpy as np

DAY = 1440


class PeriodGrid:
    """Contiguous periods covering [0, 1440) defined by their edges.

    ``PeriodGrid.uniform(30)`` gives DaySim's 48 half-hour periods;
    ``PeriodGrid.with_early_bin(60)`` gives ActivitySim's hourly grid from 05:00
    with one "early" period covering 00:00–05:00.
    """

    def __init__(self, edges: list[int]) -> None:
        edges = [int(e) for e in edges]
        if edges[0] != 0 or edges[-1] != DAY or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("edges must increase from 0 to 1440")
        self.edges = np.array(edges, dtype=int)
        self.n = len(edges) - 1

    @classmethod
    def uniform(cls, resolution: int) -> "PeriodGrid":
        if DAY % resolution:
            raise ValueError("resolution must divide 1440")
        return cls(list(range(0, DAY + 1, resolution)))

    @classmethod
    def with_early_bin(cls, resolution: int = 60, early_end: int = 300) -> "PeriodGrid":
        return cls([0] + list(range(early_end, DAY + 1, resolution)))

    def lo(self, p) -> np.ndarray:
        return self.edges[np.asarray(p)]

    def hi(self, p) -> np.ndarray:
        return self.edges[np.asarray(p) + 1]

    def width(self, p) -> np.ndarray:
        return self.hi(p) - self.lo(p)

    def period_of(self, minute) -> np.ndarray:
        """Period containing a start minute (0 <= minute < 1440)."""
        return np.clip(np.searchsorted(self.edges, np.asarray(minute), side="right") - 1, 0, self.n - 1)

    def end_period_of(self, minute) -> np.ndarray:
        """Period containing an end minute (0 < minute <= 1440): an end at a period edge belongs to the earlier period."""
        return np.clip(np.searchsorted(self.edges, np.asarray(minute), side="left") - 1, 0, self.n - 1)

    def midpoints(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """All (start period, end period) pairs with start <= end."""
        ps, pe = np.triu_indices(self.n)
        return ps.astype(int), pe.astype(int)


class IntraPeriodSampler:
    """Empirical within-period minute distribution (SR-12), stored in the bundle.

    For every key (e.g. ``"work:start"``) and period it keeps a histogram of observed
    minutes inside that period. Sparse cells (< ``min_count``) fall back to the key's
    pooled distribution of relative offsets, then to uniform. Sampling is always
    restricted to a feasible [lo, hi) range.
    """

    def __init__(self, grid: PeriodGrid, min_count: int = 20, n_rel_bins: int = 30) -> None:
        self.grid = grid
        self.min_count = min_count
        self.n_rel_bins = n_rel_bins
        self.cells: dict[tuple[str, int], np.ndarray] = {}
        self.pooled: dict[str, np.ndarray] = {}

    def fit(self, observations: dict[str, list[float]], end: bool = False) -> "IntraPeriodSampler":
        """observations: key -> observed minutes; ``end`` uses end-period convention."""
        for key, minutes in observations.items():
            m = np.asarray(minutes, dtype=float)
            if not len(m):
                continue
            p = self.grid.end_period_of(m) if end else self.grid.period_of(m)
            lo = self.grid.lo(p)
            w = self.grid.width(p)
            rel = np.clip((m - lo - (1 if end else 0)) / w, 0, 0.999999)
            self.pooled[key] = np.bincount((rel * self.n_rel_bins).astype(int), minlength=self.n_rel_bins).astype(float)
            for period in np.unique(p):
                sel = m[p == period]
                if len(sel) >= self.min_count:
                    lo_p = int(self.grid.lo(period))
                    off = (sel - lo_p - (1 if end else 0)).astype(int)
                    hist = np.bincount(np.clip(off, 0, int(self.grid.width(period)) - 1),
                                       minlength=int(self.grid.width(period))).astype(float)
                    self.cells[(key, int(period))] = hist
        return self

    def sample(self, key: str, period: int, lo: int, hi: int, rng: np.random.Generator, end: bool = False) -> int:
        """Sample a minute in period ``period`` restricted to [lo, hi) (start) or (lo, hi] (end)."""
        p_lo = int(self.grid.lo(period))
        width = int(self.grid.width(period))
        shift = 1 if end else 0
        minutes = p_lo + shift + np.arange(width)
        ok = (minutes >= lo + shift) & (minutes < hi + shift)
        if not ok.any():
            raise ValueError("empty feasible range in period")
        hist = self.cells.get((key, int(period)))
        if hist is None and key in self.pooled:
            rel_bins = (np.arange(width) / width * self.n_rel_bins).astype(int)
            hist = self.pooled[key][rel_bins] / np.maximum(np.bincount(rel_bins, minlength=self.n_rel_bins)[rel_bins], 1)
        w = np.where(ok, hist if hist is not None else 1.0, 0.0)
        if w.sum() <= 0:
            w = ok.astype(float)
        return int(rng.choice(minutes, p=w / w.sum()))
