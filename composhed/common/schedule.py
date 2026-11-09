"""Shared schedule finaliser and invariant checker."""

import numpy as np

from composhed.common.report import GenerationReport
from composhed.data import ACTIVITY_TYPES

DAY = 1440


class InvalidSchedule(ValueError):
    pass


def check_rows(rows: list[dict]) -> None:
    """Raise InvalidSchedule unless rows satisfy I1, I2, I4 and I5.

    Neighbouring episodes of one type (I3) are allowed: they are generated, as in
    the observed data, and left for the evaluation to score. Zero-length episodes
    are allowed too: a source model can give a stop, an activity or a home stay
    between tours no time at its own precision.
    """
    if not rows:
        raise InvalidSchedule("empty schedule")
    if sum(r["duration"] for r in rows) != DAY:
        raise InvalidSchedule("I1: durations do not sum to 1440")
    if rows[0]["act"] != "home" or rows[-1]["act"] != "home":
        raise InvalidSchedule("I2: schedule must start and end at home")
    cursor = 0
    for i, r in enumerate(rows):
        if not all(isinstance(r[k], int) for k in ("start", "end", "duration")):
            raise InvalidSchedule("I4: non-integer times")
        if r["start"] != cursor or r["end"] - r["start"] != r["duration"] or r["duration"] < 0:
            raise InvalidSchedule("I4: non-contiguous times")
        if r["act"] not in ACTIVITY_TYPES:
            raise InvalidSchedule(f"I5: unknown activity {r['act']!r}")
        cursor = r["end"]
    if cursor != DAY:
        raise InvalidSchedule("I4: schedule does not end at 1440")


def finalise(seq: list[tuple[str, float]], report: GenerationReport | None = None) -> list[dict]:
    """Convert an (act, duration) sequence summing to 1440 into checked integer rows.

    Durations may be fractional: cumulative boundaries are rounded, so the total is
    preserved exactly. Every episode is kept, including those of zero minutes and
    neighbouring episodes of one type. A sequence whose total is not 1440 is an
    error, not something to repair here.
    """
    total = float(sum(d for _, d in seq))
    if abs(total - DAY) > 1e-6:
        raise InvalidSchedule(f"sequence totals {total:.3f}, not {DAY}")
    bounds = np.rint(np.cumsum([0.0] + [float(d) for _, d in seq])).astype(int)
    bounds[-1] = DAY
    episodes = [(a, int(bounds[i]), int(bounds[i + 1])) for i, (a, _) in enumerate(seq)]
    rows = [{"act": a, "start": s, "end": e, "duration": e - s} for a, s, e in episodes]
    check_rows(rows)
    return rows


def from_tours(
    tours: list[list[tuple[str, float, float]]], report: GenerationReport | None = None
) -> list[dict]:
    """Build a day from non-overlapping tours, each a list of (act, start, end) episodes.

    Home fills the time before, between and after the tours; a home stay is
    written between every two tours, even when it lasts zero minutes.
    """
    seq: list[tuple[str, float]] = []
    cursor = 0.0
    for episodes in sorted(tours, key=lambda t: (t[0][1], t[-1][2])):
        s0 = episodes[0][1]
        if s0 < cursor - 1e-6:
            raise InvalidSchedule("overlapping tours")
        seq.append(("home", max(s0 - cursor, 0.0)))
        for act, s, e in episodes:
            seq.append((act, e - s))
        cursor = episodes[-1][2]
    seq.append(("home", DAY - cursor))
    return finalise(seq, report)


def home_day() -> list[dict]:
    return [{"act": "home", "start": 0, "end": DAY, "duration": DAY}]
