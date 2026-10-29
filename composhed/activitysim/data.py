"""ActivitySim training records (person-day; household interactions out of scope, §7.3).

* CDAP (ASIM-2): M if the day has a mandatory tour, N if it has only
  non-mandatory tours, H otherwise.
* Mandatory tour frequency (ASIM-3): work1 / work2 / school1 / school2 /
  work_and_school from the number of work and education tours (2+ → 2).
* Non-mandatory tour frequency (ASIM-5): tours per ActivitySim purpose
  (escort, shopping, othmaint ← other, othdiscr ← leisure, social ← visit).
  As prototype_mtc, the frequency choice covers up to ``NM_CAPS`` tours per
  purpose (escort 2, others 1); the extension step adds up to ``MAX_EXTRA`` more
  tours of a chosen purpose (``nm_extra``).
* Tours beyond the caps (2 per mandatory purpose; the non-mandatory cap plus
  ``MAX_EXTRA``) are dropped from the record, latest first, so every model is
  estimated on the tours the frequency models can generate.
* Tours are scheduled mandatory first (work then school; school then work for
  non-workers on work_and_school days, as ActivitySim's ``process_mandatory_tours``)
  and then in ActivitySim's purpose order (alternatives-file column order:
  escort, shopping, othmaint, othdiscr, [eatout,] social).
"""

from dataclasses import dataclass, field

from composhed.activitysim.persontype import person_type
from composhed.common.tours import ExtractionCounters, Tour, extract_day_structure
from composhed.data import LABEL_COLS

ASIM_PURPOSE = {"work": "work", "education": "school", "escort": "escort", "shop": "shopping",
                "other": "othmaint", "leisure": "othdiscr", "visit": "social"}
NTS_ACT = {v: k for k, v in ASIM_PURPOSE.items()}
NM_ORDER = ["escort", "shopping", "othmaint", "othdiscr", "social"]
NM_ABBREV = {"escort": "es", "shopping": "sh", "othmaint": "om", "othdiscr": "od", "social": "so"}
WORKER_TYPES = {"FTW"}
MTF_ALTS = ["work1", "work2", "school1", "school2", "work_and_school"]
NM_CAPS = {"escort": 2, "shopping": 1, "othmaint": 1, "othdiscr": 1, "social": 1}
MAX_EXTRA = 2
MAX_STOPS = 3


def mtf_label(n_work: int, n_school: int) -> str | None:
    if n_work and n_school:
        return "work_and_school"
    if n_work:
        return "work2" if n_work >= 2 else "work1"
    if n_school:
        return "school2" if n_school >= 2 else "school1"
    return None


def mtf_tours(label: str, worker: bool = True) -> list[str]:
    """Mandatory tours in scheduling order; non-workers schedule school before work."""
    if label == "work_and_school":
        return ["work", "school"] if worker else ["school", "work"]
    return {"work1": ["work"], "work2": ["work", "work"], "school1": ["school"], "school2": ["school", "school"]}[label]


def tour_order(ptype: str) -> list[str]:
    mand = ["work", "school"] if ptype in WORKER_TYPES else ["school", "work"]
    return mand + NM_ORDER


def nm_vector(purposes: list[str], caps: dict | None = None) -> tuple[tuple, tuple]:
    """(frequency-model vector capped at ``NM_CAPS``, extension tours per purpose capped at ``MAX_EXTRA``)."""
    base, extra = [], []
    for p in NM_ORDER:
        n = purposes.count(p)
        base.append(min(n, NM_CAPS[p]))
        extra.append(min(n - base[-1], MAX_EXTRA))
        if caps is not None and n > base[-1] + extra[-1]:
            caps["nm_tours_capped"] = caps.get("nm_tours_capped", 0) + n - base[-1] - extra[-1]
    return tuple(base), tuple(extra)


def nm_label(vec: tuple) -> str:
    return "".join(f"{NM_ABBREV[p]}{n}" for p, n in zip(NM_ORDER, vec))


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
    nm_extra: tuple = (0, 0, 0, 0, 0)


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
        ptype = person_type(row["age"], row["employment"], unmapped)
        tours = [_asim_tour(t) for t in day.tours]
        n_work = sum(t.purpose == "work" for t in tours)
        n_school = sum(t.purpose == "school" for t in tours)
        nm, extra = nm_vector([t.purpose for t in tours if t.purpose in NM_ORDER], caps)
        keep = {"work": min(n_work, 2), "school": min(n_school, 2),
                **{p: n + x for p, n, x in zip(NM_ORDER, nm, extra)}}
        tours = _drop_excess(tours, keep, caps)
        order = tour_order(ptype)
        tours.sort(key=lambda t: (order.index(t.purpose), t.start))
        cdap = "M" if n_work + n_school else ("N" if tours else "H")
        records.append(AsimRecord(pid, {k: row[k] for k in LABEL_COLS}, ptype, cdap,
                                  mtf_label(n_work, n_school), nm, tours, extra))
    return records, counters, caps


def _drop_excess(tours: list[AsimTour], keep: dict, caps: dict) -> list[AsimTour]:
    """Keep the earliest ``keep[purpose]`` tours of each purpose; count the rest."""
    out, seen = [], {}
    for t in sorted(tours, key=lambda t: t.start):
        seen[t.purpose] = seen.get(t.purpose, 0) + 1
        if seen[t.purpose] <= keep[t.purpose]:
            out.append(t)
        else:
            caps["tours_dropped"] = caps.get("tours_dropped", 0) + 1
    return out
