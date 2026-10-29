"""DaySim tour time-of-day utility specs (DAY-2/4), after DaySim's ``WorkTourTimeModel.cs``.

The alternatives are (arrival, departure) period pairs **at the primary
destination** on DaySim's half-hour grid; stops are timed afterwards by the trip
time model. As in DaySim:

* arrival-period, departure-period and duration constants are grouped by period
  midpoint (``IsLeftExclusiveBetween`` ranges of WorkTourTimeModel, terms 11–40);
  mandatory and non-mandatory tours use different duration groups;
* arrival "shift" (arrival period index in half hours) and duration "shift"
  (periods between arrival and departure, in half hours) are interacted with
  person type, tour purpose, income, the four priority flags (first / later of
  2+ tours, same or different purpose), stop purposes in the pattern and escort
  stops (terms 41–70, 139–158);
* time-band interactions: high income and escort / shopping tours with early
  arrival and late departure, full-time workers with under 9 hours, escort /
  shopping with under 1 hour and 1–2 hours, priority flags with under 8 / 4
  hours (terms 71–84, 169–180).

Terms that need data NTS lacks are left out (mode impedances, transit, subtours,
the meal purpose). Saturday and Sunday shift effects are added because NTS
covers every day of the week and DaySim models a weekday. Work, education and
other tours have separate coefficients, as DaySim's three tour time models.
"""

import numpy as np

from composhed.activitysim.persontype import PERSON_TYPES
from composhed.common.periods import PeriodGrid
from composhed.common.tod import TodSpec

SHIFT_PURPOSES = ["escort", "other", "shop", "visit"]  # leisure (social/recreation) is the reference
PRIORITY_FLAGS = ["high_priority_same", "low_priority_same", "high_priority_different", "low_priority_different"]
Z_NAMES = (["const"] + [f"ptype_{p}" for p in PERSON_TYPES] + [f"purpose_{p}" for p in SHIFT_PURPOSES]
           + ["income_low", "income_high"] + PRIORITY_FLAGS + ["n_stop_purposes", "escort_stops", "saturday", "sunday"])

H = 60
ARR_GROUPS = [("lt6", 0, 6 * H), ("6_7", 6 * H, 7 * H), ("7_8", 7 * H, 8 * H), ("8_9", 8 * H, 9 * H),
              ("9_10", 9 * H, 10 * H), ("10_13", 10 * H, 13 * H), ("13_16", 13 * H, 16 * H),
              ("16_19", 16 * H, 19 * H), ("19_22", 19 * H, 22 * H), ("22_24", 22 * H, 24 * H + 1)]
DEP_GROUPS = [("lt7", 0, 7 * H), ("7_10", 7 * H, 10 * H), ("10_13", 10 * H, 13 * H), ("13_15", 13 * H, 15 * H),
              ("15_16", 15 * H, 16 * H), ("16_17", 16 * H, 17 * H), ("17_18", 17 * H, 18 * H),
              ("18_19", 18 * H, 19 * H), ("19_21", 19 * H, 21 * H), ("21_24", 21 * H, 24 * H + 1)]
DUR_MAND = [("0_3", 0, 3), ("3_5", 3, 5), ("5_7", 5, 7), ("7_9", 7, 9), ("9_10", 9, 10), ("10_11", 10, 11),
            ("11_12", 11, 12), ("12_14", 12, 14), ("14_18", 14, 18), ("18_plus", 18, 99)]
DUR_NM = [("0_1", 0, 1), ("1_2", 1, 2), ("2_3", 2, 3), ("3_5", 3, 5), ("5_7", 5, 7), ("7_9", 7, 9),
          ("9_12", 9, 12), ("12_14", 12, 14), ("14_18", 14, 18), ("18_plus", 18, 99)]
REF = {"arr": "8_9", "dep": "17_18", "dur_mand": "9_10", "dur_nm": "1_2"}


def _group(x: np.ndarray, groups) -> np.ndarray:
    out = np.full(len(x), -1)
    for k, (_, a, b) in enumerate(groups):
        out[(x >= a) & (x < b)] = k
    return out


def _spec(grid: PeriodGrid, pa, pd, mandatory: bool, purposes: bool) -> TodSpec:
    mid_a, mid_d = grid.midpoints()[pa], grid.midpoints()[pd]
    half_hours = 48.0 / grid.n  # DaySim: shift * (48 / SMALL_DAY_PERIOD_TOTAL_TRIP_TIMES)
    arr_shift = pa * half_hours
    dur_shift = (pd - pa) * half_hours
    dur_h = (mid_d - mid_a) / 60.0
    dur_groups = DUR_MAND if mandatory else DUR_NM
    a_idx, d_idx, u_idx = _group(mid_a, ARR_GROUPS), _group(mid_d, DEP_GROUPS), _group(dur_h, dur_groups)
    g = {"arr_shift": arr_shift, "dur_shift": dur_shift,
         "early_arrive": (mid_a < 7 * H), "late_depart": (mid_d >= 21 * H),
         "dur_lt1": dur_h < 1, "dur_1_2": (dur_h >= 1) & (dur_h < 2), "dur_lt4": dur_h < 4, "dur_lt8": dur_h < 8,
         "dur_lt9": dur_h < 9}
    g |= {f"arr_{n}": a_idx == k for k, (n, _, _) in enumerate(ARR_GROUPS)}
    g |= {f"dep_{n}": d_idx == k for k, (n, _, _) in enumerate(DEP_GROUPS)}
    g |= {f"dur_{n}": u_idx == k for k, (n, _, _) in enumerate(dur_groups)}
    g = {k: np.asarray(v, dtype=float) for k, v in g.items()}
    dur_ref = REF["dur_mand"] if mandatory else REF["dur_nm"]
    terms = ([("const", f"arr_{n}") for n, _, _ in ARR_GROUPS if n != REF["arr"]]
             + [("const", f"dep_{n}") for n, _, _ in DEP_GROUPS if n != REF["dep"]]
             + [("const", f"dur_{n}") for n, _, _ in dur_groups if n != dur_ref])
    shifters = ([f"ptype_{p}" for p in PERSON_TYPES] + (([f"purpose_{p}" for p in SHIFT_PURPOSES]) if purposes else [])
                + ["income_low", "income_high"] + PRIORITY_FLAGS
                + ["n_stop_purposes", "escort_stops", "saturday", "sunday"])
    terms += [(z, s) for z in shifters for s in ("arr_shift", "dur_shift")]
    terms += [("income_high", "early_arrive"), ("income_high", "late_depart"), ("ptype_FTW", "dur_lt9"),
              ("high_priority_same", "dur_lt8"), ("low_priority_same", "dur_lt8"),
              ("high_priority_different", "dur_lt8"), ("low_priority_different", "dur_lt4")]
    if purposes:
        for p in ("escort", "shop"):
            terms += [(f"purpose_{p}", "early_arrive"), (f"purpose_{p}", "late_depart"),
                      (f"purpose_{p}", "dur_lt1"), (f"purpose_{p}", "dur_1_2")]
    return TodSpec(Z_NAMES, g, terms, u_idx, len(dur_groups))


def work_spec(grid, pa, pd):
    return _spec(grid, pa, pd, mandatory=True, purposes=False)


def education_spec(grid, pa, pd):
    return _spec(grid, pa, pd, mandatory=True, purposes=False)


def nonmandatory_spec(grid, pa, pd):
    return _spec(grid, pa, pd, mandatory=False, purposes=True)


SPECS = {"work": work_spec, "education": education_spec, "nonmandatory": nonmandatory_spec}


def priority_flags(rank: int, n_tours: int, same_idx: int, same_count: int) -> dict:
    """DaySim's four priority flags for the tour scheduled ``rank``-th of ``n_tours``."""
    hps = same_count > 1 and same_idx == 0
    lps = same_idx > 0
    return {"high_priority_same": float(hps), "low_priority_same": float(lps),
            "high_priority_different": float(rank == 0 and n_tours > 1 and not hps),
            "low_priority_different": float(rank > 0 and not lps)}
