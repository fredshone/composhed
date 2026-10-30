"""DaySim training records.

Days are extracted with ``extract_day_structure(collapse_mandatory_tours=False)``:
second work / education tours are kept. Tours are ordered by the fixed DaySim
priority (DAY-3): work → school → escort → personal business → shop → meal →
social/recreation, mapped to NTS as work → education → escort → other → shop →
visit → leisure (no meal purpose; visit and leisure both fall under DaySim's
social/recreation and are kept apart); ties keep time order.

Day pattern (DAY-1, ``IndividualPersonDayPatternModel``): for each of the 7 tour
purposes whether the day has a tour of it, and for each of the 5 stop purposes
(escort, other, shop, visit, leisure) whether the day has a stop of it. Work and
education stops do not exist here: work or education inside a tour is merged
into the work block (docs/tour_rules.md). As DaySim, patterns have at most 3
tour purposes, at most 4 stop purposes, at most 5 of both together, and stops
only with a tour. As DaySim, work tours are available only to workers
(``employment = employed``) and education tours only to students (``employment =
student`` or a school-age child, SCHOOL / DRIVING). Observed days outside these
rules are left out of the day-pattern estimation (counted); their tours are
still used by the later models.

Exact number of tours (DAY-1b, ``PersonExactNumberOfToursModel``): 1, 2 or 3+
tours of each purpose in the pattern; tours beyond 3 of a purpose are dropped
from the record (counted).
"""

from dataclasses import dataclass, field
from itertools import combinations

from composhed.activitysim.persontype import person_type
from composhed.common.tours import ExtractionCounters, Tour, extract_day_structure
from composhed.data import LABEL_COLS

PRIORITY = ["work", "education", "escort", "other", "shop", "visit", "leisure"]
TOUR_PURPOSES = PRIORITY
NM_PURPOSES = PRIORITY[2:]
STOP_PURPOSES = ["escort", "other", "shop", "visit", "leisure"]
MAX_TOUR_PURPOSES = 3
MAX_STOP_PURPOSES = 4
MAX_TOUR_AND_STOP_PURPOSES = 5
MAX_TOURS_PER_PURPOSE = 3


def tod_group(purpose: str) -> str:
    return purpose if purpose in ("work", "education") else "nonmandatory"


def feasible(tours: tuple, stops: tuple) -> bool:
    nt, ns = sum(tours), sum(stops)
    return (nt <= MAX_TOUR_PURPOSES and ns <= MAX_STOP_PURPOSES and nt + ns <= MAX_TOUR_AND_STOP_PURPOSES
            and (ns == 0 or nt > 0))


def enumerate_patterns() -> list[tuple]:
    """All feasible (tour flags + stop flags) vectors, as DaySim's ``InitializeDayPatterns``."""
    out = []
    for nt in range(MAX_TOUR_PURPOSES + 1):
        for tp in combinations(range(len(TOUR_PURPOSES)), nt):
            tours = tuple(int(i in tp) for i in range(len(TOUR_PURPOSES)))
            for ns in range(MAX_STOP_PURPOSES + 1):
                for sp in combinations(range(len(STOP_PURPOSES)), ns):
                    stops = tuple(int(i in sp) for i in range(len(STOP_PURPOSES)))
                    if feasible(tours, stops):
                        out.append(tours + stops)
    return out


def pattern_label(vec: tuple) -> str:
    tours, stops = vec[:len(TOUR_PURPOSES)], vec[len(TOUR_PURPOSES):]
    return ("T" + "".join(p[:2] for p, f in zip(TOUR_PURPOSES, tours) if f)
            + "|S" + "".join(p[:2] for p, f in zip(STOP_PURPOSES, stops) if f))


def observed_pattern(tours: list[Tour]) -> tuple:
    tp = {t.purpose for t in tours}
    sp = {s.act for t in tours for s in t.stops}
    return tuple(int(p in tp) for p in TOUR_PURPOSES) + tuple(int(p in sp) for p in STOP_PURPOSES)


def is_worker(labels: dict) -> bool:
    return labels.get("employment") == "employed"


def is_student(labels: dict, ptype: str) -> bool:
    return labels.get("employment") == "student" or ptype in ("SCHOOL", "DRIVING")


def available(vec: tuple, worker: bool, student: bool) -> bool:
    """DaySim's availability: work tours only for workers, school tours only for students."""
    return (worker or not vec[0]) and (student or not vec[1])


def pattern_tours(vec: tuple, counts: dict) -> list[str]:
    """Tour purposes of a pattern in priority order, ``counts[purpose]`` of each."""
    return [p for p, f in zip(TOUR_PURPOSES, vec[:len(TOUR_PURPOSES)]) if f for _ in range(counts.get(p, 1))]


def priority_order(tours: list[Tour]) -> list[Tour]:
    return sorted(tours, key=lambda t: (PRIORITY.index(t.purpose), t.start))


@dataclass
class DaysimRecord:
    pid: object
    labels: dict
    ptype: str
    pattern: tuple            # observed flags
    counts: dict              # tours per purpose, capped at MAX_TOURS_PER_PURPOSE
    tours: list[Tour] = field(default_factory=list)  # priority order
    in_choice_set: bool = True  # feasible and available: used to estimate the day-pattern model


def build_records(attr_df, sched_by_pid: dict, pids: set | None = None):
    counters = ExtractionCounters()
    caps: dict = {}
    records = []
    for row in attr_df.iter_rows(named=True):
        pid = row["pid"]
        if pid not in sched_by_pid or (pids is not None and pid not in pids):
            continue
        day = extract_day_structure(sched_by_pid[pid], collapse_mandatory_tours=False, counters=counters)
        tours, seen = [], {}
        for t in priority_order(day.tours):
            seen[t.purpose] = seen.get(t.purpose, 0) + 1
            if seen[t.purpose] <= MAX_TOURS_PER_PURPOSE:
                tours.append(t)
            else:
                caps["tours_dropped"] = caps.get("tours_dropped", 0) + 1
        vec = observed_pattern(tours)
        labels = {k: row[k] for k in LABEL_COLS}
        ptype = person_type(row["age"], row["employment"])
        ok = feasible(vec[:len(TOUR_PURPOSES)], vec[len(TOUR_PURPOSES):])
        if not ok:
            caps["patterns_infeasible"] = caps.get("patterns_infeasible", 0) + 1
        elif not available(vec, is_worker(labels), is_student(labels, ptype)):
            caps["patterns_unavailable"] = caps.get("patterns_unavailable", 0) + 1
            ok = False
        counts = {p: sum(t.purpose == p for t in tours) for p in TOUR_PURPOSES}
        records.append(DaysimRecord(pid, labels, ptype, vec, counts, tours, ok))
    return records, counters, caps
