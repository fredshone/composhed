"""SR-13/14: minute-level time window (ActivitySim timetable / DaySim time windows)."""

import numpy as np

DAY = 1440


class TimeWindow:
    """Tracks reserved minutes of a day.

    Out-of-home tours are reserved as half-open minute intervals, so a tour may
    start in the minute another ends. Availability queries can keep a ``buffer``
    of home time next to existing tours and at the start and end of the day; the
    DaySim and ActivitySim variants use none.
    """

    def __init__(self, day: int = DAY) -> None:
        self.day = day
        self.busy = np.zeros(day, dtype=bool)
        self.points: list[int] = []  # zero-length tours: no later tour may span them
        self.tours: list[tuple[int, int]] = []  # every reserved [start, end), in reservation order

    def copy(self) -> "TimeWindow":
        tw = TimeWindow(self.day)
        tw.busy = self.busy.copy()
        tw.points = list(self.points)
        tw.tours = list(self.tours)
        return tw

    def is_free(self, start: int, end: int) -> bool:
        start, end = max(0, int(start)), min(self.day, int(end))
        return end > start and not self.busy[start:end].any()

    def reserve(self, start: int, end: int) -> None:
        """Reserve [start, end); a zero-length tour reserves no minutes but is remembered as a point."""
        start, end = int(start), int(end)
        if end == start:
            self.points.append(start)
            self.tours.append((start, end))
            return
        if not self.is_free(start, end):
            raise ValueError(f"interval [{start}, {end}) is not free")
        self.busy[start:end] = True
        self.tours.append((start, end))

    def remaining(self) -> int:
        return int((~self.busy).sum())

    def free_windows(self) -> list[tuple[int, int]]:
        """Maximal free intervals as (start, end)."""
        free = np.concatenate([[False], ~self.busy, [False]]).astype(np.int8)
        d = np.diff(free)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        return list(zip(starts.tolist(), ends.tolist()))

    def busy_intervals(self) -> list[tuple[int, int]]:
        """Reserved (out-of-home) intervals as (start, end)."""
        busy = np.concatenate([[False], self.busy, [False]]).astype(np.int8)
        d = np.diff(busy)
        return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))

    def usable_windows(self, buffer: int = 0) -> list[tuple[int, int]]:
        """Free windows shrunk by ``buffer`` next to reserved tours and at the day edges.

        A tour placed inside a usable window leaves >= buffer minutes of home
        between it and any other tour and at either end of the day.
        """
        out = []
        for a, b in self.free_windows():
            a2, b2 = a + buffer, b - buffer
            if b2 > a2:
                out.append((a2, b2))
        return out

    def largest_window(self, buffer: int = 0) -> tuple[int, int] | None:
        wins = self.usable_windows(buffer)
        return max(wins, key=lambda w: w[1] - w[0]) if wins else None

    def window_containing(self, minute: float, buffer: int = 0) -> tuple[int, int] | None:
        for a, b in self.usable_windows(buffer):
            if a <= minute < b:
                return a, b
        return None

    def availability_mask(
        self, starts: np.ndarray, ends: np.ndarray, buffer: int = 0
    ) -> np.ndarray:
        """Minute-level alternatives: [start, end) fits inside a usable window."""
        starts, ends = np.asarray(starts), np.asarray(ends)
        mask = np.zeros(len(starts), dtype=bool)
        for a, b in self.usable_windows(buffer):
            mask |= (starts >= a) & (ends <= b) & (ends > starts)
        return mask

    def period_availability_mask(
        self,
        start_lo: np.ndarray,
        start_hi: np.ndarray,
        end_lo: np.ndarray,
        end_hi: np.ndarray,
        buffer: int = 0,
        min_duration: int = 1,
    ) -> np.ndarray:
        """Period-pair alternatives: some start in [start_lo, start_hi) and end in
        (end_lo, end_hi] exist inside one usable window with end - start >= min_duration.
        """
        mask = np.zeros(len(start_lo), dtype=bool)
        for a, b in self.usable_windows(buffer):
            s_min = np.maximum(start_lo, a)
            e_max = np.minimum(end_hi, b)
            mask |= (s_min < start_hi) & (e_max > end_lo) & (e_max - s_min >= min_duration)
        return mask
