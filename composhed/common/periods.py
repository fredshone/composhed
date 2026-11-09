"""SR-11/12: time-period grids. Minutes inside a chosen period are drawn uniformly (``TourTOD.realise``)."""

import numpy as np

DAY = 1440


class PeriodGrid:
    """Contiguous periods covering [0, 1440) defined by their edges.

    ``PeriodGrid.uniform(30)`` gives DaySim's 48 half-hour periods;
    ``PeriodGrid.with_early_bin(60, 360, 330)`` gives ActivitySim's 19 hourly
    periods 5–23, the first holding every time before 06:00 and written at 05:30.
    ``points`` overrides where each period is written (default: its midpoint).
    """

    def __init__(self, edges: list[int], points: list[int] | None = None) -> None:
        edges = [int(e) for e in edges]
        if edges[0] != 0 or edges[-1] != DAY or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("edges must increase from 0 to 1440")
        self.edges = np.array(edges, dtype=int)
        self.n = len(edges) - 1
        self._points = None if points is None else np.array(points, dtype=int)

    @classmethod
    def uniform(cls, resolution: int) -> "PeriodGrid":
        if DAY % resolution:
            raise ValueError("resolution must divide 1440")
        return cls(list(range(0, DAY + 1, resolution)))

    @classmethod
    def with_early_bin(cls, resolution: int = 60, early_end: int = 300, early_point: int | None = None) -> "PeriodGrid":
        edges = [0] + list(range(early_end, DAY + 1, resolution))
        if early_point is None:
            return cls(edges)
        mids = [(a + b) // 2 for a, b in zip(edges[1:-1], edges[2:])]
        return cls(edges, [early_point] + mids)

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

    def points(self) -> np.ndarray:
        """One whole minute per period (its midpoint unless overridden): where a period-precision model writes its
        times."""
        if self._points is not None:
            return self._points.copy()
        return (self.edges[:-1] + self.edges[1:]) // 2

    def midpoints(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """All (start period, end period) pairs with start <= end."""
        ps, pe = np.triu_indices(self.n)
        return ps.astype(int), pe.astype(int)
