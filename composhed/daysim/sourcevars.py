"""DaySim person variables built from the NTS labels.

DaySim's day-pattern and exact-number-of-tours models use the same 28 person
terms for every purpose (``IndividualPersonDayPatternModel`` terms 100p+1 to
100p+28). Those with an NTS counterpart are kept:

| DaySim term | From NTS | Approximation |
|---|---|---|
| person type flags (8) | ``person_type``: retired, non-working adult, university student, driving-age student, child 5–15 (SCHOOL), child under 5 (PRESCHOOL); full-time worker is the reference; no part-time workers | see ``activitysim.persontype`` |
| income 0–25k, 25–45k, 75k+ (45–75k reference) | ``hh_income`` band rank 0, 1, 4 (ranks 2–3 reference) | bands mapped by position |
| cars per driver | ``veh_per_adult``: 0, 0.5, 1 | per adult |
| female adult (terms 16–18 split by children in the household) | female and 18 or over | no household children in NTS: one term |
| age 18–25, 26–35, 51–65 | age bands 18-27, 27-35, 48-62 | band edges |
| mixed-use density | ``hh_zone == urban`` | one density proxy |
| purpose logsum | ``wt_travel_time_to_cities``: 3 − band rank | one accessibility index for every purpose |

Dropped (no NTS counterpart): only adult, only worker, male with children,
works at home, intersection density, transit pass, paid parking, diary type.
Day of week (``saturday``, ``sunday``) is the one addition.
"""

import numpy as np

from composhed.activitysim.sourcevars import band_ranks

PTYPE_FLAGS = {"RETIRED": "retired", "NONWORK": "nonworking_adult", "UNIV": "university_student",
               "DRIVING": "driving_age_student", "SCHOOL": "child_5_15", "PRESCHOOL": "child_under_5"}
PERSON_TERMS = (list(PTYPE_FLAGS.values()) + ["income_0_25k", "income_25_45k", "income_75k_plus", "cars_per_driver",
                                             "female_adult", "age_18_25", "age_26_35", "age_51_65", "mixed_density",
                                             "logsum"])
CARS = {"0": 0.0, "0-1": 0.5, "1+": 1.0}
ADULT_AGES = {"18-27", "27-35", "35-41", "41-48", "48-55", "55-62", "62-71", "71-118"}


class DaysimVars:
    def __init__(self, label_rows: list[dict]) -> None:
        self.income_rank = band_ranks(r["hh_income"] for r in label_rows)
        self.access_rank = band_ranks(r["wt_travel_time_to_cities"] for r in label_rows)
        self.n_access = len([v for v in self.access_rank if v[0].isdigit()])

    def person(self, labels: dict, ptype: str) -> dict:
        inc = self.income_rank.get(str(labels["hh_income"]), 2)
        age = str(labels["age"])
        v = {f: float(ptype == p) for p, f in PTYPE_FLAGS.items()}
        v |= {"income_0_25k": float(inc == 0), "income_25_45k": float(inc == 1), "income_75k_plus": float(inc >= 4),
              "cars_per_driver": CARS.get(str(labels["veh_per_adult"]), 0.5),
              "female_adult": float(labels["sex"] == "female" and age in ADULT_AGES),
              "age_18_25": float(age == "18-27"), "age_26_35": float(age == "27-35"),
              "age_51_65": float(age in ("48-55", "55-62")),
              "mixed_density": float(labels["hh_zone"] == "urban"),
              "logsum": float(max(self.n_access - 1 - self.access_rank.get(str(labels["wt_travel_time_to_cities"]), 0),
                                  0)),
              "adult_male": float(labels["sex"] == "male" and age in ADULT_AGES),
              "saturday": float(labels["day"] == "saturday"), "sunday": float(labels["day"] == "sunday")}
        return v

    def vector(self, labels: dict, ptype: str, names: list[str]) -> np.ndarray:
        v = self.person(labels, ptype)
        return np.array([v[k] for k in names])
