"""Shared schedule finaliser and invariant checker (I1–I5) that counts repairs."""

import numpy as np

from composhed.data import ACTIVITY_TYPES
from composhed.common.report import GenerationReport

DAY = 1440


class InvalidSchedule(ValueError):
    pass


def check_rows(rows: list[dict]) -> None:
    """Raise InvalidSchedule unless rows satisfy I1–I5."""
    if not rows:
        raise InvalidSchedule("empty schedule")
    if sum(r["duration"] for r in rows) != DAY:
        raise InvalidSchedule("I1: durations do not sum to 1440")
    if rows[0]["act"] != "home" or rows[-1]["act"] != "home":
        raise InvalidSchedule("I2: schedule must start and end at home")
    cursor = 0
    for i, r in enumerate(rows):
        if i and rows[i - 1]["act"] == r["act"]:
            raise InvalidSchedule("I3: consecutive identical activities")
        if not all(isinstance(r[k], int) for k in ("start", "end", "duration")):
            raise InvalidSchedule("I4: non-integer times")
        if r["start"] != cursor or r["end"] - r["start"] != r["duration"] or r["duration"] <= 0:
            raise InvalidSchedule("I4: non-contiguous times")
        if r["act"] not in ACTIVITY_TYPES:
            raise InvalidSchedule(f"I5: unknown activity {r['act']!r}")
        cursor = r["end"]
    if cursor != DAY:
        raise InvalidSchedule("I4: schedule does not end at 1440")


def finalise(seq: list[tuple[str, float]], report: GenerationReport | None = None) -> list[dict]:
    """Convert an (act, duration) sequence summing to 1440 into checked integer rows.

    Durations may be fractional: cumulative boundaries are rounded, so the total is
    preserved exactly. Zero-length episodes are dropped and consecutive identical
    activities merged; each merge is counted in ``report``. A sequence whose total
    is not 1440 is an error, not something to repair here.
    """
    total = float(sum(d for _, d in seq))
    if abs(total - DAY) > 1e-6:
        raise InvalidSchedule(f"sequence totals {total:.3f}, not {DAY}")
    bounds = np.rint(np.cumsum([0.0] + [float(d) for _, d in seq])).astype(int)
    bounds[-1] = DAY
    episodes = [(a, int(bounds[i]), int(bounds[i + 1])) for i, (a, _) in enumerate(seq)]
    episodes = [e for e in episodes if e[2] > e[1]]

    merged: list[list] = []
    n_merges = 0
    for act, s, e in episodes:
        if merged and merged[-1][0] == act:
            merged[-1][2] = e
            n_merges += 1
        else:
            merged.append([act, s, e])
    if report is not None and n_merges:
        report.merges += n_merges
        report.persons_merged += 1

    rows = [{"act": a, "start": s, "end": e, "duration": e - s} for a, s, e in merged]
    check_rows(rows)
    return rows


def from_episodes(
    episodes: list[tuple[str, float, float]], report: GenerationReport | None = None
) -> list[dict]:
    """Build a day from non-overlapping (act, start, end) out-of-home episodes; gaps are home."""
    seq: list[tuple[str, float]] = []
    cursor = 0.0
    for act, s, e in sorted(episodes, key=lambda x: x[1]):
        if s < cursor - 1e-6:
            raise InvalidSchedule("overlapping episodes")
        if s > cursor:
            seq.append(("home", s - cursor))
        seq.append((act, e - s))
        cursor = e
    seq.append(("home", DAY - cursor))
    if seq[0][0] != "home":
        seq.insert(0, ("home", 0.0))
    return finalise(seq, report)


def home_day() -> list[dict]:
    return [{"act": "home", "start": 0, "end": DAY, "duration": DAY}]
