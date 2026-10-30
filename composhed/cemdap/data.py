"""CEMDAP training records (SPECIFICATION §5.4).

Each person-day is converted with ``extract_day_structure(collapse_mandatory_tours=True)``
into the CEMDAP pattern: an optional single mandatory block with outbound and
inbound stops, separate non-mandatory tours before / after it, and home stays.
Non-mandatory tours are ordered lists of stops (no primary activity).

The *generation order* used by ``chain.CemdapChain`` is reproduced here so that
the residual-time covariates seen in training match those at generation:

* mandatory day, before side (budget = start − 1): outbound stops, then each
  pre-tour's stops followed by the home stay after it; the first home stay is
  the residual.
* mandatory day, after side (budget = 1440 − end − 1): inbound stops, then for
  each post-tour the home stay before it and its stops; the last home stay is
  the residual.
* non-mandatory day: first home stay (departure), then (budget = 1440 − departure − 1) each
  tour's stops followed by the home stay after it (except the last); the last
  home stay is the residual.
"""

from dataclasses import dataclass, field

from composhed.common.tours import (
    DayStructure,
    ExtractionCounters,
    extract_day_structure,
)
from composhed.data import LABEL_COLS

POSITIONS = ["outbound", "inbound", "pre_tour", "post_tour", "nm_tour"]
HOME_SIDES = ["before", "after", "nm"]
MAX_POSITION_COUNT = 3
MAX_NM_TOURS = 4
MAX_TOUR_STOPS = 4
DAY = 1440
MAND_START_ORIGIN = 180  # CEMDAP measures work start from 03:00


@dataclass
class CemdapRecord:
    pid: object
    labels: dict
    mand_type: str | None = None
    mand_start: int = 0
    mand_duration: int = 0
    n_out: int = 0
    n_in: int = 0
    n_pre: int = 0
    n_post: int = 0
    n_nm_tours: int = 0
    tour_sizes: list[tuple[str, int, int]] = field(default_factory=list)  # (position, tour index, n stops)
    stops: list[dict] = field(default_factory=list)  # purpose/duration/position/index/residual
    home_stays: list[dict] = field(default_factory=list)  # between-tour stays: side/duration/residual
    first_departure: int | None = None


def _stop(purpose, duration, position, index, residual) -> dict:
    return {"purpose": purpose, "duration": int(duration), "position": position,
            "index": int(index), "residual": float(residual)}


def build_record(pid, labels: dict, day: DayStructure) -> CemdapRecord:
    rec = CemdapRecord(pid=pid, labels=labels)
    mand = [i for i, t in enumerate(day.tours) if t.mandatory]
    if mand:
        m = mand[0]
        mt = day.tours[m]
        rec.mand_type = mt.purpose
        rec.mand_start = mt.primary_start
        rec.mand_duration = mt.primary_duration
        rec.n_out, rec.n_in = len(mt.outbound), len(mt.inbound)
        pre, post = day.tours[:m], day.tours[m + 1:]
        rec.n_pre, rec.n_post = len(pre), len(post)
        mand_start = mt.primary_start
        # Before side: outbound stops, then pre tours each followed by their home stay.
        resid = mand_start - 1
        for k, s in enumerate(mt.outbound):
            rec.stops.append(_stop(s.act, s.duration, "outbound", k, resid))
            resid -= s.duration
        for ti, t in enumerate(pre):
            rec.tour_sizes.append(("pre_tour", ti, len(t.stops) + 1))
            for k, s in enumerate(_tour_stops(t)):
                rec.stops.append(_stop(s[0], s[1], "pre_tour", k, resid))
                resid -= s[1]
            h = day.home_stays[ti + 1]
            rec.home_stays.append({"side": "before", "duration": int(h), "residual": float(resid)})
            resid -= h
        # After side: inbound stops, then each post tour preceded by its home stay.
        resid = DAY - mand_start - mt.primary_duration - 1
        for k, s in enumerate(mt.inbound):
            rec.stops.append(_stop(s.act, s.duration, "inbound", k, resid))
            resid -= s.duration
        for ti, t in enumerate(post):
            h = day.home_stays[m + 1 + ti]
            rec.home_stays.append({"side": "after", "duration": int(h), "residual": float(resid)})
            resid -= h
            rec.tour_sizes.append(("post_tour", ti, len(t.stops) + 1))
            for k, s in enumerate(_tour_stops(t)):
                rec.stops.append(_stop(s[0], s[1], "post_tour", k, resid))
                resid -= s[1]
    else:
        rec.n_nm_tours = len(day.tours)
        if day.tours:
            rec.first_departure = int(day.home_stays[0])
        resid = DAY - 1 - (day.home_stays[0] if day.tours else 0)
        for ti, t in enumerate(day.tours):
            rec.tour_sizes.append(("nm_tour", ti, len(t.stops) + 1))
            for k, s in enumerate(_tour_stops(t)):
                rec.stops.append(_stop(s[0], s[1], "nm_tour", k, resid))
                resid -= s[1]
            if ti < len(day.tours) - 1:
                h = day.home_stays[ti + 1]
                rec.home_stays.append({"side": "nm", "duration": int(h), "residual": float(resid)})
                resid -= h
    return rec


def _tour_stops(tour) -> list[tuple[str, int]]:
    """All activities of a non-mandatory tour in time order."""
    acts = [(s.act, s.start, s.duration) for s in tour.outbound + tour.inbound]
    acts.append((tour.purpose, tour.primary_start, tour.primary_duration))
    return [(a, d) for a, _, d in sorted(acts, key=lambda x: x[1])]


def build_records(attr_df, sched_by_pid: dict, pids: set | None = None) -> tuple[list[CemdapRecord], ExtractionCounters]:
    counters = ExtractionCounters()
    records = []
    for row in attr_df.iter_rows(named=True):
        pid = row["pid"]
        if pid not in sched_by_pid or (pids is not None and pid not in pids):
            continue
        day = extract_day_structure(sched_by_pid[pid], collapse_mandatory_tours=True, counters=counters)
        records.append(build_record(pid, {k: row[k] for k in LABEL_COLS}, day))
    return records, counters
