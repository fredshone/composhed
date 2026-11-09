# How days are split into tours

The CEMDAP, DaySim and ActivitySim variants all read their training data
through one function, `composhed.common.tours.extract_day_structure`. These
rules are defined there and nowhere else. The compositional baseline uses its
own, similar rules in `composhed/data.py` (see [compositional.md](compositional.md)).

## Tours

A **tour** is a trip that leaves home and comes back: an unbroken run of
non-home activities between two home stays. A day is therefore:

```
home, tour 1, home, tour 2, ..., tour n, home
```

* Zero-length rows are dropped first (NTS 2023 has none).
* Time always adds up: home stays plus tours make exactly 1,440 minutes. This
  was checked for all 91,706 NTS 2023 days.
* Each tour has one **main activity**. The stops before it are **outbound**
  and the stops after it are **inbound**; each of these is a half-tour.

## Choosing the main activity

Work comes first, then education, then the longest discretionary activity
(ties go to the earliest). An alternative rule, "longest activity of any type",
is available for sensitivity checks.

The CEMDAP variant treats non-mandatory tours as plain lists of stops, so for
CEMDAP the main activity only matters on work and education tours.

## Work or education interrupted by other activities

Sometimes a tour contains several work or education episodes, as in
`home → work → shop → work → home`. NTS 2023 has 240 such tours, 12 of them
mixing work and education. **No episode is merged.** Each variant reads them
the way its source represents them:

* **DaySim and ActivitySim:** the main activity is the longest episode of the
  tour's highest-priority mandatory type (work, then education; ties go to the
  earliest). Every other episode, including the other work or education
  episodes, is an outbound or inbound **stop**. Both sources have work and
  school stop purposes (DaySim's `IntermediateStopGenerationModel`,
  ActivitySim's trip purposes). The example above becomes a work tour with
  main activity the longer work episode and two stops (shop and work) on one
  side. NTS 2023 has 244 such work or education stops. The sources would
  instead read most of these as tours from work (at-work subtours), which
  these variants do not model.
* **CEMDAP:** CEMDAP's worker pattern has a **work-based period**, from the
  start of work to its end, with **work-based tours** inside it. The main
  activity is that period: it starts at the first mandatory episode, ends at
  the end of the last, and takes the higher-priority type (a tour with both
  work and education is a work tour). The mandatory episodes are its **work
  stays**, and each run of activities between two of them is a work-based
  tour (228 on NTS 2023). Stops before the first episode are outbound
  (commute) stops, stops after the last are inbound.

Every training run logs these counts and stores them in `validation.json` under
`info.extraction`.

## More than one work or school tour in a day

461 NTS 2023 days have more than one work / education tour: 444 have two and
17 have three or more. The variants treat them differently:

| Variant | Rule |
|---|---|
| CEMDAP | One work-based period per day, as CEMDAP's worker pattern has one commute each way. Later work / school time is added to the last work stay of the first period. Later tours' stops (their work-based tours included) become a separate non-mandatory tour after it, with the longest stop as its main activity. A later tour with no stops disappears, and the home stays either side of it are joined. (479 tours are folded in this way.) |
| DaySim | Kept as second (and third) work or school tours, because DaySim includes these in its choices. |
| ActivitySim | Kept as second work or school tours (ActivitySim's work2 / school2). The 14 days with three or more keep the earliest two of each type; the rest are dropped from the record (logged), so every model is estimated on tours the frequency models can generate. On work_and_school days, non-workers' school tour is scheduled first, as in ActivitySim. |

## Caps

To keep choice sets manageable, some counts are capped. For CEMDAP and DaySim
a capped item is only removed from the *count* model; the activities
themselves still train the type, duration and timing models. The ActivitySim
variant drops tours beyond its caps from the training record altogether
(latest first; 129 tours on NTS 2023), so its scheduling models never see
tours its frequency models cannot produce.

| What is counted | Cap | Share of NTS within the cap |
|---|---|---|
| Stops per half-tour (DaySim, ActivitySim) | 3 | 99.92% of half-tours |
| Stops or tours per CEMDAP position | 3 | 99.98% |
| Non-mandatory tours per day | 4 | 99.84% of days |
| Work or education tours of one type (DaySim, ActivitySim) | 2 | 99.93% |
| Stops per CEMDAP non-mandatory tour | 4 | 99.59% of tours |

## Travel time and the home stay between tours

There are no locations, so moving between activities takes no time, and no
variant adds a minimum home stay:

* **CEMDAP:** home stays between tours are drawn from CEMDAP's home-stay
  duration model, bounded only by the time left, so they can be as short as
  zero. The day may start or end with a zero-length home stay too.
* **DaySim:** a tour may start in the minute the previous one ends, as
  DaySim's time windows allow; the home stay between them then lasts zero
  minutes.
* **ActivitySim:** as its timetable allows, a tour may start in the period
  the previous one ends. Both are written at that period's midpoint, so the
  home stay between them lasts zero minutes.

A zero-length home stay between two tours is still written, so the output
shows that the person went home.
