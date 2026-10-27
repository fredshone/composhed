"""DaySim training records.

Days are extracted with ``extract_day_structure(collapse_mandatory_tours=False)``:
second work / education tours are kept (DAY-1). Tours are ordered by the fixed
DaySim priority (DAY-3): work → education → escort → medical → shop → visit →
other; ties keep time order.

Day pattern (DAY-1): (#work tours ≤ 2, #education tours ≤ 2, #non-mandatory
tours by primary purpose with ≤ 4 in total, stops-on-mandatory flag). Excess
non-mandatory tours are removed from the pattern count only (largest purpose
first) and logged.
"""

from dataclasses import dataclass, field

from composhed.common.tours import ExtractionCounters, Tour, extract_day_structure
from composhed.data import LABEL_COLS

PRIORITY = ["work", "education", "escort", "medical", "shop", "visit", "other"]
NM_PURPOSES = ["escort", "medical", "shop", "visit", "other"]
TOUR_PURPOSES = PRIORITY
MAX_MAND = 2
MAX_NM_TOURS = 4


def tod_group(purpose: str) -> str:
    return purpose if purpose in ("work", "education") else "nonmandatory"


def pattern_vector(tours: list[Tour], counters: dict | None = None) -> tuple:
    n_work = min(sum(t.purpose == "work" for t in tours), MAX_MAND)
    n_edu = min(sum(t.purpose == "education" for t in tours), MAX_MAND)
    nm = [sum(t.purpose == p for t in tours) for p in NM_PURPOSES]
    while sum(nm) > MAX_NM_TOURS:
        nm[nm.index(max(nm))] -= 1
        if counters is not None:
            counters["nm_tours_capped"] = counters.get("nm_tours_capped", 0) + 1
    stops_on_mand = int(any(t.mandatory and t.stops for t in tours))
    return (n_work, n_edu, *nm, stops_on_mand)


def pattern_label(vec: tuple) -> str:
    w, e, *nm, s = vec
    return f"W{w}E{e}|" + "".join(f"{p[:2]}{n}" for p, n in zip(NM_PURPOSES, nm)) + f"|S{s}"


def pattern_tours(vec: tuple) -> list[str]:
    """Tour purposes implied by a pattern, in priority order."""
    w, e, *nm, _ = vec
    return ["work"] * w + ["education"] * e + [p for p, n in zip(NM_PURPOSES, nm) for _ in range(n)]


def priority_order(tours: list[Tour]) -> list[Tour]:
    return sorted(tours, key=lambda t: (PRIORITY.index(t.purpose), t.start))


@dataclass
class DaysimRecord:
    pid: object
    labels: dict
    pattern: tuple
    tours: list[Tour] = field(default_factory=list)  # priority order


def build_records(attr_df, sched_by_pid: dict, pids: set | None = None):
    counters = ExtractionCounters()
    caps: dict = {}
    records = []
    for row in attr_df.iter_rows(named=True):
        pid = row["pid"]
        if pid not in sched_by_pid or (pids is not None and pid not in pids):
            continue
        day = extract_day_structure(sched_by_pid[pid], collapse_mandatory_tours=False, counters=counters)
        tours = priority_order(day.tours)
        records.append(DaysimRecord(pid, {k: row[k] for k in LABEL_COLS}, pattern_vector(tours, caps), tours))
    return records, counters, caps
