"""ActivitySim (``prototype_mtc``) spec variables built from the NTS labels.

Only variables with an NTS counterpart are defined; spec rows that use any
other variable are dropped (``Spec.restrict``). The counterparts:

| ActivitySim variable | From NTS | Approximation |
|---|---|---|
| ``ptype`` | ``person_type`` (MTC codes; no part-time workers, code 2) | see ``persontype`` |
| ``age`` | midpoint of the ``age`` band | thresholds fall inside bands |
| ``sex``, ``female`` | ``sex`` | — |
| ``income_in_thousands`` and the income dummies | ``hh_income`` band rank → 10, 30, 45, 75, 150 | MTC's 20 / 50 / 100 k$ thresholds mapped onto the NTS bands |
| ``auto_ownership``, ``num_workers``, ``num_drivers`` | ``veh_per_adult``: 0 → 0, 0-1 → 0.5, 1+ → 1.5 cars, against 1 worker / driver | per adult, not per household |
| ``home_is_urban`` | ``hh_zone == urban`` | — |
| ``auPkTotal``, ``auOpRetail`` (auto accessibility) | ``wt_travel_time_to_cities``: 3 − band rank (shorter travel time = more accessible) | one index for both |
| joint-tour counts, ``is_joint``, at-work subtours | 0 / False | none are modelled |

Day of week is the one variable ActivitySim does not have: ``saturday`` and
``sunday`` dummies, used only by the rows ``models`` adds.
"""

import numpy as np

from composhed.activitysim.persontype import person_type

MTC_PTYPE = {"FTW": 1, "UNIV": 3, "NONWORK": 4, "RETIRED": 5, "DRIVING": 6, "SCHOOL": 7, "PRESCHOOL": 8}
MTC_SEGMENT = {"FTW": "PTYPE_FULL", "UNIV": "PTYPE_UNIVERSITY", "NONWORK": "PTYPE_NONWORK", "RETIRED": "PTYPE_RETIRED",
               "DRIVING": "PTYPE_DRIVING", "SCHOOL": "PTYPE_SCHOOL", "PRESCHOOL": "PTYPE_PRESCHOOL"}
INCOME_K = [10.0, 30.0, 45.0, 75.0, 150.0]
CARS = {"0": 0.0, "0-1": 0.5, "1+": 1.5}

PERSON_VARS = {"ptype", "age", "sex", "female", "income_in_thousands", "medium_low_income", "medium_high_income",
               "high_income", "auto_ownership", "num_workers", "num_drivers", "no_cars", "car_sufficiency",
               "more_cars_than_workers", "home_is_urban", "auPkTotal", "auOpRetail", "num_hh_joint_tours",
               "num_hh_joint_shop_tours", "num_hh_joint_maint_tours", "num_hh_joint_eatout_tours",
               "num_hh_joint_othdiscr_tours", "num_hh_joint_social_tours", "is_joint", "num_atwork_subtours",
               "saturday", "sunday"}


def _lower(v) -> float:
    try:
        return float(str(v).split("-")[0].rstrip("+"))
    except ValueError:
        return float("inf")


def band_ranks(values) -> dict:
    """Rank of each band label by its lower bound (``unknown`` and other labels last)."""
    levels = sorted(set(map(str, values)), key=lambda v: (_lower(v), v))
    return {v: i for i, v in enumerate(levels)}


def _age_mid(band: str) -> float:
    try:
        a, b = str(band).split("-")
        return (float(a) + float(b)) / 2.0
    except ValueError:
        return np.nan


class PersonVars:
    """Builds ActivitySim person variables from NTS label dicts (band ranks fixed at training time)."""

    def __init__(self, label_rows: list[dict]) -> None:
        self.income_rank = band_ranks(r["hh_income"] for r in label_rows)
        self.access_rank = band_ranks(r["wt_travel_time_to_cities"] for r in label_rows)
        self.n_access = sum(1 for v in self.access_rank if _lower(v) != float("inf"))

    def __call__(self, rows: list[dict]) -> dict:
        n = len(rows)

        def col(values, dtype=float):
            return np.asarray(values, dtype=dtype).reshape(n, 1)

        ptypes = [person_type(r["age"], r["employment"]) for r in rows]
        inc = [INCOME_K[min(self.income_rank.get(str(r["hh_income"]), 2), len(INCOME_K) - 1)] for r in rows]
        cars = [CARS.get(str(r["veh_per_adult"]), 1.0) for r in rows]
        acc = [max(self.n_access - 1 - self.access_rank.get(str(r["wt_travel_time_to_cities"]), 0), 0) for r in rows]
        env = {
            "ptype": col([MTC_PTYPE[p] for p in ptypes]),
            "age": col([_age_mid(r["age"]) for r in rows]),
            "sex": col([2 if r["sex"] == "female" else 1 for r in rows]),
            "female": col([r["sex"] == "female" for r in rows], bool),
            "income_in_thousands": col(inc),
            "auto_ownership": col(cars),
            "num_workers": np.ones((n, 1)),
            "num_drivers": np.ones((n, 1)),
            "home_is_urban": col([r["hh_zone"] == "urban" for r in rows], bool),
            "auPkTotal": col(acc),
            "auOpRetail": col(acc),
            "saturday": col([r["day"] == "saturday" for r in rows], bool),
            "sunday": col([r["day"] == "sunday" for r in rows], bool),
            "is_joint": np.zeros((n, 1), dtype=bool),
            "num_atwork_subtours": np.zeros((n, 1)),
        }
        i = env["income_in_thousands"]
        env |= {"medium_low_income": (i > 20) & (i <= 50), "medium_high_income": (i > 50) & (i <= 100),
                "high_income": i > 100, "no_cars": env["auto_ownership"] == 0,
                "car_sufficiency": env["auto_ownership"] - env["num_workers"],
                "more_cars_than_workers": env["auto_ownership"] >= env["num_workers"]}
        for k in ("num_hh_joint_tours", "num_hh_joint_shop_tours", "num_hh_joint_maint_tours",
                  "num_hh_joint_eatout_tours", "num_hh_joint_othdiscr_tours", "num_hh_joint_social_tours"):
            env[k] = np.zeros((n, 1))
        env["_ptype_names"] = ptypes
        return env


def subset(env: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1 else
                [v[i] for i in idx] if isinstance(v, list) else v) for k, v in env.items()}
