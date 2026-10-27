"""SR-7/8/9: home-based tour and stop structure of an observed day.

Rules (documented in docs/tour_rules.md):

* A tour is a maximal run of non-home activities between two home episodes.
  Zero-length rows are dropped first.
* Primary activity (SR-8), ``primary_rule="priority"``: work > education > the
  longest discretionary activity (ties: earliest). ``"longest"`` picks the longest
  activity of any type.
* Fragmented mandatory activity (SR-9), as CompoShed ``build_training_dataset``:
  all mandatory episodes in a tour are merged into one block that starts at the
  first mandatory start, lasts the *sum* of the mandatory episode durations and is
  typed by the first mandatory episode. Discretionary activities before the first
  mandatory start are outbound stops; all later ones (including those inside the
  block) are inbound stops.
* Multiple mandatory tours: ``collapse_mandatory_tours=True`` (CEMDAP) folds the
  mandatory time of later mandatory tours into the first block and turns their
  stops into a separate non-mandatory tour; ``False`` (DaySim/ActivitySim) keeps
  them as tours of their own.
"""

from dataclasses import dataclass, field

from composhed.data import DISC_ACTS, MANDATORY_ACTS

PRIORITY = {"work": 0, "education": 1}


@dataclass
class Stop:
    act: str
    start: int
    duration: int


@dataclass
class Tour:
    purpose: str
    primary_start: int
    primary_duration: int
    outbound: list[Stop] = field(default_factory=list)
    inbound: list[Stop] = field(default_factory=list)
    start: int = 0
    n_mandatory_episodes: int = 0

    @property
    def mandatory(self) -> bool:
        return self.purpose in MANDATORY_ACTS

    @property
    def duration(self) -> int:
        return (
            self.primary_duration
            + sum(s.duration for s in self.outbound)
            + sum(s.duration for s in self.inbound)
        )

    @property
    def end(self) -> int:
        return self.start + self.duration

    @property
    def stops(self) -> list[Stop]:
        return self.outbound + self.inbound


@dataclass
class DayStructure:
    tours: list[Tour]
    home_stays: list[int]  # len(tours) + 1: before, between, after

    @property
    def mandatory_tours(self) -> list[Tour]:
        return [t for t in self.tours if t.mandatory]

    def total(self) -> int:
        return sum(self.home_stays) + sum(t.duration for t in self.tours)


@dataclass
class ExtractionCounters:
    persons: int = 0
    zero_rows_dropped: int = 0
    fragmented_tours: int = 0
    persons_fragmented: int = 0
    mixed_mandatory_tours: int = 0
    multi_mandatory_days: int = 0
    collapsed_tours: int = 0
    tours_per_day: dict[int, int] = field(default_factory=dict)
    mandatory_tours_per_day: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def log(self, prefix: str = "  ") -> None:
        print(
            f"{prefix}tour extraction: persons={self.persons:,} fragmented mandatory tours="
            f"{self.fragmented_tours} (persons {self.persons_fragmented}) mixed work+edu tours="
            f"{self.mixed_mandatory_tours} multi-mandatory days={self.multi_mandatory_days} "
            f"collapsed tours={self.collapsed_tours} zero-length rows dropped={self.zero_rows_dropped}"
        )
        print(f"{prefix}tours/day {dict(sorted(self.tours_per_day.items()))}  "
              f"mandatory tours/day {dict(sorted(self.mandatory_tours_per_day.items()))}")


def _split_tours(activities: list[dict]) -> tuple[list[list[dict]], list[int]]:
    segments: list[list[dict]] = []
    homes: list[int] = []
    current: list[dict] = []
    home_acc = 0
    for a in activities:
        if a["act"] == "home":
            if current:
                segments.append(current)
                homes.append(home_acc)
                current, home_acc = [], 0
            home_acc += int(a["duration"])
        else:
            current.append(a)
    if current:
        segments.append(current)
        homes.append(home_acc)
        home_acc = 0
    homes.append(home_acc)
    return segments, homes


def _make_tour(seg: list[dict], primary_rule: str, counters: ExtractionCounters) -> Tour:
    mand = [a for a in seg if a["act"] in MANDATORY_ACTS]
    if mand and primary_rule == "priority":
        first = mand[0]
        if len(mand) > 1:
            counters.fragmented_tours += 1
        if len({a["act"] for a in mand}) > 1:
            counters.mixed_mandatory_tours += 1
        m_start = int(first["start"])
        outbound = [Stop(a["act"], int(a["start"]), int(a["duration"]))
                    for a in seg if a["act"] in DISC_ACTS and a["start"] < m_start]
        inbound = [Stop(a["act"], int(a["start"]), int(a["duration"]))
                   for a in seg if a["act"] in DISC_ACTS and a["start"] >= m_start]
        return Tour(
            purpose=first["act"],
            primary_start=m_start,
            primary_duration=sum(int(a["duration"]) for a in mand),
            outbound=outbound,
            inbound=inbound,
            start=int(seg[0]["start"]),
            n_mandatory_episodes=len(mand),
        )
    # Longest activity (ties: earliest) is primary; everything else is a stop.
    best = max(range(len(seg)), key=lambda i: (int(seg[i]["duration"]), -i))
    p = seg[best]
    return Tour(
        purpose=p["act"],
        primary_start=int(p["start"]),
        primary_duration=int(p["duration"]),
        outbound=[Stop(a["act"], int(a["start"]), int(a["duration"])) for a in seg[:best]],
        inbound=[Stop(a["act"], int(a["start"]), int(a["duration"])) for a in seg[best + 1:]],
        start=int(seg[0]["start"]),
        n_mandatory_episodes=sum(a["act"] in MANDATORY_ACTS for a in seg),
    )


def _collapse(tours: list[Tour], homes: list[int], counters: ExtractionCounters) -> tuple[list[Tour], list[int]]:
    """Fold later mandatory tours into the first mandatory block (CEMDAP mode)."""
    first = next(i for i, t in enumerate(tours) if t.mandatory)
    new_tours: list[Tour] = []
    new_homes: list[int] = [homes[0]]
    for i, t in enumerate(tours):
        if i <= first or not t.mandatory:
            new_tours.append(t)
            new_homes.append(homes[i + 1])
            continue
        counters.collapsed_tours += 1
        tours[first].primary_duration += t.primary_duration
        stops = t.stops
        if stops:
            best = max(range(len(stops)), key=lambda j: (stops[j].duration, -j))
            p = stops[best]
            new_tours.append(Tour(p.act, p.start, p.duration, stops[:best], stops[best + 1:], start=t.start))
            new_homes.append(homes[i + 1])
        else:
            # The tour vanishes: merge the home stays either side of it.
            new_homes[-1] += homes[i + 1]
    return new_tours, new_homes


def extract_day_structure(
    activities: list[dict],
    collapse_mandatory_tours: bool = False,
    primary_rule: str = "priority",
    counters: ExtractionCounters | None = None,
) -> DayStructure:
    """Convert one person's sorted activity rows into a DayStructure.

    Time is conserved: ``DayStructure.total()`` equals the schedule length.
    """
    counters = counters if counters is not None else ExtractionCounters()
    counters.persons += 1
    acts = [a for a in activities if int(a["duration"]) > 0]
    counters.zero_rows_dropped += len(activities) - len(acts)
    segments, homes = _split_tours(acts)
    n_frag_before = counters.fragmented_tours
    tours = [_make_tour(seg, primary_rule, counters) for seg in segments]
    if counters.fragmented_tours > n_frag_before:
        counters.persons_fragmented += 1
    n_mand = sum(t.mandatory for t in tours)
    counters.mandatory_tours_per_day[n_mand] = counters.mandatory_tours_per_day.get(n_mand, 0) + 1
    if n_mand > 1:
        counters.multi_mandatory_days += 1
        if collapse_mandatory_tours:
            tours, homes = _collapse(tours, homes, counters)
    n = len(tours)
    counters.tours_per_day[n] = counters.tours_per_day.get(n, 0) + 1
    return DayStructure(tours=tours, home_stays=homes)


def group_schedules(sched_df) -> dict:
    """Group a polars schedules frame (sorted by pid, start) into {pid: [row dicts]}."""
    out: dict = {}
    for row in sched_df.iter_rows(named=True):
        out.setdefault(row["pid"], []).append(row)
    return out
