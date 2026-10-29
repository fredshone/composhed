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

* Zero-length rows are dropped first (16 home rows in NTS 2023).
* Time always adds up: home stays plus tours make exactly 1,440 minutes. This
  was checked for all 56,734 NTS days.
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
`home → work → shop → work → home`. Following the compositional baseline:

* all the work / education episodes in the tour become **one block**. It
  starts at the first one, lasts their **combined** duration, and takes the
  type of the first one (a tour with both work and education becomes a single
  block);
* discretionary activities before the block are **outbound** stops;
* everything after the block's start is an **inbound** stop. That includes
  activities squeezed between work episodes, such as the shop above, so there
  are no tours from work.

NTS 2023 has 275 such tours (one each for 275 people) and 11 tours that mix work and education.
In 47 of them a stop inside the block ends before the block's (combined) end,
so the trip departures are out of order. The ActivitySim variant leaves these
47 tours out of its trip departure table (counted in `validation.json` under
`info.trip_departure_tours_skipped`).
Every training run logs these counts and stores them in `validation.json` under
`info.extraction`.

## More than one work or school tour in a day

406 NTS days have more than one work / education tour: 392 have two and 14 have
three or more. The variants treat them differently:

| Variant | Rule |
|---|---|
| CEMDAP | One mandatory activity per day, as in the compositional baseline. Later work / school time is added to the first block. Later tours' stops become a separate non-mandatory tour after it, with the longest stop as its main activity. A later tour with no stops disappears, and the home stays either side of it are joined. (421 tours are folded in this way.) |
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

## Travel time and the 10-minute home stay

There are no locations, so moving between activities takes no time. The only
cost of going home is a minimum home stay of 10 minutes between tours,
borrowed from the compositional baseline (`HOME_INTER`):

* **CEMDAP:** every home stay between tours is drawn with a lower bound of 10
  minutes.
* **DaySim and ActivitySim:** every scheduled tour keeps a 10-minute buffer
  free on either side. The day also starts and ends with at least 1 minute at
  home.

To check whether 10 minutes is reasonable, CEMDAP's `validation.json` records
the 5th percentile of observed home stays between tours
(`info.between_tour_home_p05`).
