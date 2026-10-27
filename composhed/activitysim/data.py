"""ActivitySim training records (person-day; household interactions out of scope, §7.3).

* CDAP (ASIM-2): M if the day has a mandatory tour, N if it has only
  non-mandatory tours, H otherwise.
* Mandatory tour frequency (ASIM-3): work1 / work2 / school1 / school2 /
  work_and_school from the number of work and education tours (2+ → 2).
* Non-mandatory tour frequency (ASIM-5): tours per ActivitySim purpose
  (escort, shopping, othmaint ← medical, social ← visit, othdiscr ← other),
  at most ``MAX_NM_TOURS`` in total (excess removed from the largest purpose,
  logged). No extension step.
* Tours are scheduled mandatory first (work, then school) and then in the
  fixed ActivitySim purpose order.
"""

from dataclasses import dataclass, field

from composhed.activitysim.persontype import person_type
from composhed.common.tours import ExtractionCounters, Tour, extract_day_structure
from composhed.data import LABEL_COLS

ASIM_PURPOSE = {"work": "work", "education": "school", "escort": "escort", "shop": "shopping",
                "medical": "othmaint", "visit": "social", "other": "othdiscr"}
NTS_ACT = {v: k for k, v in ASIM_PURPOSE.items()}
NM_ORDER = ["escort", "shopping", "othmaint", "social", "othdiscr"]
TOUR_ORDER = ["work", "school"] + NM_ORDER
MTF_ALTS = ["work1", "work2", "school1", "school2", "work_and_school"]
MAX_NM_TOURS = 4
MAX_STOPS = 3


def mtf_label(n_work: int, n_school: int) -> str | None:
    if n_work and n_school:
        return "work_and_school"
    if n_work:
        return "work2" if n_work >= 2 else "work1"
    if n_school:
        return "school2" if n_school >= 2 else "school1"
    return None


def mtf_tours(label: str) -> list[str]:
    return {"work1": ["work"], "work2": ["work", "work"], "school1": ["school"], "school2": ["school", "school"],
            "work_and_school": ["work", "school"]}[label]


def nm_vector(purposes: list[str], caps: dict | None = None) -> tuple:
    v = [purposes.count(p) for p in NM_ORDER]
    while sum(v) > MAX_NM_TOURS:
        v[v.index(max(v))] -= 1
        if caps is not None:
            caps["nm_tours_capped"] = caps.get("nm_tours_capped", 0) + 1
    return tuple(v)


def nm_label(vec: tuple) -> str:
    return "".join(f"{p[:2]}{n}" for p, n in zip(NM_ORDER, vec))


def nm_tours(vec: tuple) -> list[str]:
    return [p for p, n in zip(NM_ORDER, vec) for _ in range(n)]


@dataclass
class AsimTour:
    purpose: str  # ActivitySim purpose
    start: int
    end: int
    outbound: list[tuple[str, int, int]] = field(default_factory=list)  # (act, start, duration)
    inbound: list[tuple[str, int, int]] = field(default_factory=list)
    primary_start: int = 0
    primary_end: int = 0


@dataclass
class AsimRecord:
    pid: object
    labels: dict
    ptype: str
    cdap: str
    mtf: str | None
    nm: tuple
    tours: list[AsimTour]  # scheduling order


def _asim_tour(t: Tour) -> AsimTour:
    return AsimTour(
        purpose=ASIM_PURPOSE[t.purpose], start=t.start, end=t.end,
        outbound=[(s.act, s.start, s.duration) for s in t.outbound],
        inbound=[(s.act, s.start, s.duration) for s in t.inbound],
        primary_start=t.start + sum(s.duration for s in t.outbound),
        primary_end=t.end - sum(s.duration for s in t.inbound),
    )


def build_records(attr_df, sched_by_pid: dict, pids: set | None = None, unmapped=None):
    counters = ExtractionCounters()
    caps: dict = {}
    records = []
    for row in attr_df.iter_rows(named=True):
        pid = row["pid"]
        if pid not in sched_by_pid or (pids is not None and pid not in pids):
            continue
        day = extract_day_structure(sched_by_pid[pid], collapse_mandatory_tours=False, counters=counters)
        tours = sorted((_asim_tour(t) for t in day.tours), key=lambda t: (TOUR_ORDER.index(t.purpose), t.start))
        n_work = sum(t.purpose == "work" for t in tours)
        n_school = sum(t.purpose == "school" for t in tours)
        nm = nm_vector([t.purpose for t in tours if t.purpose in NM_ORDER], caps)
        cdap = "M" if n_work + n_school else ("N" if tours else "H")
        records.append(AsimRecord(pid, {k: row[k] for k in LABEL_COLS},
                                  person_type(row["age"], row["employment"], unmapped), cdap,
                                  mtf_label(n_work, n_school), nm, tours))
    return records, counters, caps
