"""ActivitySim tour scheduling utility specs (ASIM-4/6), after prototype_mtc v1.6.0.

The terms follow ``tour_scheduling_work.csv``, ``tour_scheduling_school.csv`` and
``tour_scheduling_nonmandatory.csv``. Times are in hours, as ActivitySim's
``start`` / ``end`` / ``duration`` columns: ``start`` is the clock hour of the
departure period, ``end`` that of the arrival period, ``duration = end - start``.
The early period (00:00–05:00 on the hourly grid) counts as the hour before the
first regular period, as ActivitySim maps earlier times into its first period.

Each spec has:

* grouped departure, arrival and duration constants (MTC's groups; one reference
  group per block). Non-mandatory tours have separate escort and non-escort sets;
* linear shift effects (``start``, ``duration``) interacted with person type,
  income, tour position and purpose;
* time-band interactions (early departure, late arrival, short duration, ...).

Terms that need data NTS does not have are left out: mode-choice logsum and
auto times, CBD destination, household composition, joint tours. Day of week is
added (Saturday and Sunday shift effects), because NTS covers every day and
ActivitySim models a weekday.
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
# Reference groups (constrained to 0), as in the MTC coefficient files where they exist.
REF = {"dep": "7", "arr": "15", "dur_mand": "10", "dur_nm": "0_1"}


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
          "arr_gt22": (he > 22), "arr_19_21": (he > 18) & (he < 22),
          "dur_lt2": (D < 2), "dur_lt6": (D < 6), "dur_lt8": (D < 8), "dur_lt9": (D < 9)}
    return {k: np.asarray(v, dtype=float) for k, v in g.items()}, dur_idx


def _constants(z: str, dur_groups, dur_ref: str) -> list[tuple[str, str]]:
    return ([(z, f"dep_{n}") for n, _, _ in DEP_GROUPS if n != REF["dep"]]
            + [(z, f"arr_{n}") for n, _, _ in ARR_GROUPS if n != REF["arr"]]
            + [(z, f"dur_{n}") for n, _, _ in dur_groups if n != dur_ref])


def _shifts(names: list[str], what=("start", "duration")) -> list[tuple[str, str]]:
    return [(n, w) for n in names for w in what]


def work_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    g, dbin = _base(grid, ps, pe, DUR_MAND)
    terms = (_constants("const", DUR_MAND, REF["dur_mand"])
             + _shifts([f"ptype_{p}" for p in PERSON_TYPES if p != "FTW"])
             + [("income", "start")]
             + _shifts(["first_of_2plus", "subsequent", "saturday", "sunday"])
             + [("high_income", "dep_lt6"), ("high_income", "arr_gt22"), ("rural", "dep_lt6"), ("rural", "arr_gt22"),
                ("ptype_FTW", "dur_lt9"), ("ptype_FTW", "dep_10_12"),
                ("first_of_2plus", "dur_lt8"), ("subsequent", "dur_lt8"),
                ("work_and_school_worker", "dur_lt8"), ("work_and_school_student", "dur_lt8")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_MAND))


def school_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    g, dbin = _base(grid, ps, pe, DUR_MAND)
    terms = (_constants("const", DUR_MAND, REF["dur_mand"])
             + _shifts([f"ptype_{p}" for p in PERSON_TYPES if p != "SCHOOL"])
             + [("first_of_2plus", "start"), ("first_of_2plus", "duration"), ("subsequent", "duration")]
             + _shifts(["saturday", "sunday"])
             + [("high_income", "dep_lt6"), ("high_income", "arr_gt22"),
                ("first_of_2plus", "dur_lt6"), ("subsequent", "dur_lt6"),
                ("work_and_school_worker", "dur_lt6"), ("work_and_school_student", "dur_lt6")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_MAND))


def nonmandatory_spec(grid: PeriodGrid, ps, pe) -> TodSpec:
    g, dbin = _base(grid, ps, pe, DUR_NM)
    terms = (_constants("is_not_escort", DUR_NM, REF["dur_nm"]) + _constants("is_escort", DUR_NM, REF["dur_nm"])
             + _shifts([f"purpose_{p}" for p in ("shopping", "othmaint", "social")])
             + _shifts([f"ptype_{p}" for p in PERSON_TYPES if p != "FTW"])
             + [("num_mand", "start"), ("num_escort", "start"), ("num_non_escort", "start"),
                ("first_of_2plus_same", "start"), ("subsequent_same", "duration")]
             + _shifts(["saturday", "sunday"])
             + [("purpose_othmaint", "dep_lt7"), ("purpose_shopping", "dep_lt8"), ("purpose_shopping", "arr_gt22"),
                ("ptype_SCHOOL", "arr_gt22"), ("ptype_UNIV", "arr_gt22"),
                ("purpose_shopping", "dur_lt2"), ("purpose_othdiscr", "dur_lt2")])
    return TodSpec(Z_NAMES, g, terms, dbin, len(DUR_NM))


SPECS = {"work": work_spec, "school": school_spec, "nonmandatory": nonmandatory_spec}
