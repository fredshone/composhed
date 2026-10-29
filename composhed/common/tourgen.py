"""Tour placement and half-tour stop filling shared by the DaySim and ActivitySim variants.

``place_tours`` schedules tours one at a time in the given (priority) order:

* ``timing="tod"`` — joint start/end period choice restricted by the
  ``TimeWindow`` (DAY-2/4, ASIM-4/6);
* ``timing="independent"`` — AB-2, KDE start + independent duration, resampled
  until it fits the window.

A tour whose choice set is empty is dropped and counted (DAY-9, ASIM-11); a pair
that cannot be realised in minutes is redrawn up to ``max_resample`` times first.
With ``consistency="rescale"`` (AB-3) no availability is used: every tour is
sampled freely and overlaps are removed afterwards by ``layout_rescale``.

``prior`` tours (already placed, e.g. ActivitySim's mandatory tours before the
non-mandatory frequency choice) occupy the timetable from the start and take part
in the ``rescale`` layout; they are returned with the new tours.
"""

from dataclasses import dataclass, field

import numpy as np

from composhed.common.report import GenerationReport
from composhed.common.schedule import from_episodes
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import layout_rescale


@dataclass
class PlannedTour:
    purpose: str
    group: str
    z: np.ndarray
    start: float | None = None
    end: float | None = None
    outbound: list[tuple[str, float]] = field(default_factory=list)
    inbound: list[tuple[str, float]] = field(default_factory=list)
    n_remaining: int = 0  # tours still to schedule after this one (ActivitySim look-ahead term)

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


def place_tours(
    tours: list[PlannedTour],
    tod: dict,
    indep: dict,
    x_label: np.ndarray,
    rng: np.random.Generator,
    report: GenerationReport,
    buffer: int,
    timing: str = "tod",
    consistency: str = "constraint",
    max_resample: int = 50,
    prior: list[PlannedTour] = (),
) -> list[PlannedTour]:
    """Assign [start, end) to each tour; returns the prior and newly placed tours, sorted by start."""
    tw = TimeWindow()
    constrained = consistency == "constraint"
    placed: list[PlannedTour] = list(prior)
    if constrained:
        for t in prior:
            tw.reserve(int(t.start), int(t.end))
    for t in tours:
        interval = None
        if timing == "independent":
            interval, used = indep[t.group].sample(x_label, tw, rng, buffer, max_resample, use_mask=constrained)
            report.resamples += used if interval is not None else max_resample
        else:
            model = tod[t.group]
            if constrained and not model.mask(tw).any():
                interval = None
            else:
                for attempt in range(max_resample + 1):
                    interval = model.sample(t.z, tw, rng, use_mask=constrained, n_remaining=t.n_remaining)
                    if interval is not None:
                        break
                    report.resamples += 1
        if interval is None:
            report.tours_dropped += 1
            continue
        t.start, t.end = float(interval[0]), float(interval[1])
        if constrained:
            tw.reserve(int(t.start), int(t.end))
        placed.append(t)
    if not constrained and len(placed) > len(prior):
        new, changed = layout_rescale([(t.start, t.end) for t in placed], buffer)
        placed.sort(key=lambda t: t.start)
        for t, (s, e) in zip(placed, new):
            t.start, t.end = s, e
        if changed:
            report.rescales += 1
    return sorted(placed, key=lambda t: t.start)


def fit_stop_durations(
    durations: list[float], available: float, truncate_ok: bool, report: GenerationReport
) -> list[float]:
    """Rescale stop durations that overflow the tour (AB-3 path, counted)."""
    total = sum(durations)
    if total <= available or not durations:
        return durations
    report.rescales += 1
    return [d * available / total for d in durations]


def tours_to_rows(tours: list[PlannedTour], report: GenerationReport) -> list[dict]:
    """Lay out each tour as outbound stops → primary → inbound stops; gaps are home."""
    episodes = []
    for t in tours:
        cursor = t.start
        for p, d in t.outbound:
            episodes.append((p, cursor, cursor + d))
            cursor += d
        primary_end = t.end - sum(d for _, d in t.inbound)
        episodes.append((t.purpose, cursor, primary_end))
        cursor = primary_end
        for p, d in t.inbound:
            episodes.append((p, cursor, cursor + d))
            cursor += d
    return from_episodes(episodes, report)
