"""ActivitySim tour scheduling utility specs (ASIM-4/6), after prototype_mtc v1.6.0.

The terms are those of ``tour_scheduling_work.csv``, ``tour_scheduling_school.csv``
and ``tour_scheduling_nonmandatory.csv``, row by row, with the coefficients the
``*_coefficients.csv`` files constrain to 0 left out (the reference constants).
Times are in hours, as ActivitySim's ``start`` / ``end`` / ``duration`` columns:
``start`` is the clock hour of the departure period, ``end`` that of the arrival
period, ``duration = end - start``. On the hourly grid the first period is
05:00 and holds every earlier time, as ActivitySim's first period does.

Rows whose variables NTS lacks are left out: mode-choice logsum and auto times,
CBD destination, household composition, joint tours, eat-out tours and
part-time workers (NTS has none). Saturday and Sunday shift effects are added,
because NTS covers every day and ActivitySim models a weekday. The timetable
rows (previous tour ends / begins, adjacent windows, tours still to schedule per
free period) are ``TourTOD``'s ``"asim"`` dynamic terms.
"""

import numpy as np

from composhed.activitysim.persontype import PERSON_TYPES
from composhed.common.periods import PeriodGrid
from composhed.common.tod import TodSpec

NM_PURPOSES = ["escort", "shopping", "othmaint", "othdiscr", "social"]
Z_NAMES = (["const", "is_escort", "is_not_escort"] + [f"ptype_{p}" for p in PERSON_TYPES]
           + ["income", "high_income", "rural", "saturday", "sunday", "first_of_2plus", "subsequent",
              "work_and_school_worker", "work_and_school_student"]
           + [f"purpose_{p}" for p in NM_PURPOSES]
           + ["num_mand", "num_escort", "num_non_escort", "first_of_2plus_same", "subsequent_same"])

DEP_GROUPS = [("early", 0, 6), ("6", 6, 7), ("7", 7, 8), ("8", 8, 9), ("9", 9, 10), ("10_12", 10, 13),
              ("13_15", 13, 16), ("16_18", 16, 19), ("19_21", 19, 22), ("late", 22, 99)]
ARR_GROUPS = [("early", 0, 7), ("7_9", 7, 10), ("10_12", 10, 13), ("13_14", 13, 15), ("15", 15, 16),
              ("16", 16, 17), ("17", 17, 18), ("18", 18, 19), ("19_21", 19, 22), ("late", 22, 99)]
DUR_MAND = [("0_2", 0, 3), ("3_4", 3, 5), ("5_6", 5, 7), ("7_8", 7, 9), ("9", 9, 10), ("10", 10, 11),
            ("11", 11, 12), ("12_13", 12, 14), ("14_plus", 14, 99)]
DUR_NM = [("0_1", 0, 2), ("2_3", 2, 4), ("4_5", 4, 6), ("6_7", 6, 8), ("8_10", 8, 11), ("11_13", 11, 14),
          ("14_plus", 14, 99)]
# Constants the coefficient files constrain to 0.
FIXED = {
    "work": {"dep_7", "arr_early", "arr_15", "dur_10"},
    "school": {"dep_7", "arr_15", "dur_7_8"},
    "is_not_escort": {"dep_16_18", "arr_19_21", "dur_0_1"},
    "is_escort": {"dep_13_15", "dur_0_1"} | {f"arr_{n}" for n, _, _ in ARR_GROUPS if n not in ("19_21", "late")},
}


def clock_hours(grid: PeriodGrid, p: np.ndarray) -> np.ndarray:
    """Clock hour of each period's start; a wide first (early) period counts as the hour before period 1."""
    lo = grid.lo(p) / 60.0
    if grid.n > 1 and grid.width(0) > grid.width(1):
        lo = np.where(np.asarray(p) == 0, grid.lo(1) / 60.0 - 1.0, lo)
    return lo


def _group(h: np.ndarray, groups) -> np.ndarray:
    hf = np.floor(h + 1e-9)
    out = np.full(len(h), -1)
    for k, (_, a, b) in enumerate(groups):
        out[(hf >= a) & (hf < b)] = k
    return out


def _dummies(prefix: str, idx: np.ndarray, groups) -> dict:
    return {f"{prefix}_{name}": (idx == k).astype(float) for k, (name, _, _) in enumerate(groups)}


def _base(grid: PeriodGrid, ps: np.ndarray, pe: np.ndarray, dur_groups) -> tuple[dict, np.ndarray]:
    S, E = clock_hours(grid, ps), clock_hours(grid, pe)
    D = E - S
    dur_idx = _group(D, dur_groups)
    g = {"start": S, "duration": D}
    g |= _dummies("dep", _group(S, DEP_GROUPS), DEP_GROUPS)
    g |= _dummies("arr", _group(E, ARR_GROUPS), ARR_GROUPS)
    g |= _dummies("dur", dur_idx, dur_groups)
    hs, he = np.floor(S + 1e-9), np.floor(E + 1e-9)
    g |= {"dep_lt6": (hs < 6), "dep_lt7": (hs < 7), "dep_lt8": (hs < 8), "dep_10_12": (hs > 9) & (hs < 13),
          "arr_gt22": (he > 22), "dur_lt2": (D < 2), "dur_lt6": (D < 6), "dur_lt8": (D < 8), "dur_lt9": (D < 9)}
    return {k: np.asarray(v, dtype=float) for k, v in g.items()}, dur_idx


def _constants(z: str, dur_groups, fixed: set) -> list[tuple[str, str]]:
    names = ([f"dep_{n}" for n, _, _ in DEP_GROUPS] + [f"arr_{n}" for n, _, _ in ARR_GROUPS]
             + [f"dur_{n}" for n, _, _ in dur_groups])
    return [(z, g) for g in names if g not in fixed]


def _shifts(names: list[str], what=("start", "duration")) -> list[tuple[str, str]]:
    return [(n, w) for n in names for w in what]


def work_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    """``tour_scheduling_work.csv``: ``ptype == 4`` (non-working adult) duration, ``ptype == 3`` (university
    student) departure, income × departure, first of 2+ and second tour shifts, high-income and rural early /
    late interactions, full-time worker under 9 hours and 10–12 departure, under-8-hour terms."""
    g, dbin = _base(grid, ps, pe, DUR_MAND)
    terms = (_constants("const", DUR_MAND, FIXED["work"])
             + [("ptype_NONWORK", "duration"), ("ptype_UNIV", "start"), ("income", "start")]
             + _shifts(["first_of_2plus", "subsequent", "saturday", "sunday"])
             + [("high_income", "dep_lt6"), ("high_income", "arr_gt22"), ("rural", "dep_lt6"), ("rural", "arr_gt22"),
                ("ptype_FTW", "dur_lt9"), ("ptype_FTW", "dep_10_12"),
                ("first_of_2plus", "dur_lt8"), ("subsequent", "dur_lt8"),
                ("work_and_school_worker", "dur_lt8"), ("work_and_school_student", "dur_lt8")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_MAND))


def school_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    """``tour_scheduling_school.csv``: full-time worker departure and duration, non-worker departure, university
    student departure and duration, ``ptype == 7`` duration, first of 2+ and subsequent shifts, high-income
    early / late interactions, under-6-hour terms."""
    g, dbin = _base(grid, ps, pe, DUR_MAND)
    terms = (_constants("const", DUR_MAND, FIXED["school"])
             + _shifts(["ptype_FTW", "ptype_UNIV"]) + [("ptype_NONWORK", "start"), ("ptype_SCHOOL", "duration")]
             + [("first_of_2plus", "start"), ("first_of_2plus", "duration"), ("subsequent", "duration")]
             + _shifts(["saturday", "sunday"])
             + [("high_income", "dep_lt6"), ("high_income", "arr_gt22"),
                ("first_of_2plus", "dur_lt6"), ("subsequent", "dur_lt6"),
                ("work_and_school_worker", "dur_lt6"), ("work_and_school_student", "dur_lt6")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_MAND))


def nonmandatory_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    """``tour_scheduling_nonmandatory.csv``: separate escort and non-escort constants; shopping, maintenance and
    visit departure and duration shifts; ``ptype == 6`` and ``ptype == 7`` departure and duration shifts; numbers
    of mandatory, escort and other tours × departure; first of 2+ / subsequent tours of one purpose; the
    early-departure, late-arrival and short-duration interactions."""
    g, dbin = _base(grid, ps, pe, DUR_NM)
    terms = (_constants("is_not_escort", DUR_NM, FIXED["is_not_escort"])
             + _constants("is_escort", DUR_NM, FIXED["is_escort"])
             + _shifts([f"purpose_{p}" for p in ("shopping", "othmaint", "social")])
             + _shifts(["ptype_DRIVING", "ptype_SCHOOL"])
             + [("num_mand", "start"), ("num_escort", "start"), ("num_non_escort", "start"),
                ("first_of_2plus_same", "start"), ("subsequent_same", "duration")]
             + _shifts(["saturday", "sunday"])
             + [("purpose_othmaint", "dep_lt7"), ("purpose_shopping", "dep_lt8"), ("purpose_shopping", "arr_gt22"),
                ("ptype_SCHOOL", "arr_gt22"), ("ptype_UNIV", "arr_gt22"),
                ("purpose_shopping", "dur_lt2"), ("purpose_othdiscr", "dur_lt2")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_NM))


SPECS = {"work": work_spec, "school": school_spec, "nonmandatory": nonmandatory_spec}
