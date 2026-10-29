"""ASIM-1: ActivitySim person types from NTS ``age`` × ``employment``.

As in ActivitySim's ``annotate_persons``, children are typed by age and adults
by employment / student status. Rules, applied in order (first match wins):

| # | Condition | Person type |
|---|---|---|
| 1 | age 0-11, employment = void (not in school) | PRESCHOOL |
| 2 | age 0-11 | SCHOOL (non-driving-age student; the 0-11 bin also holds 5-year-olds) |
| 3 | age 11-18 | DRIVING (the 11-18 bin mixes non-driving 11-15 and driving 16-17 year-olds) |
| 4 | employment = employed | FTW (worker: NTS ``employment`` no longer separates full / part time, so PTW is merged in) |
| 5 | employment = retired | RETIRED (self-reported; the 62-71 bin straddles ActivitySim's 65+ rule) |
| 6 | employment = student | UNIV (university student) |
| 7 | employment ∈ {unemployed, void, unknown} | NONWORK (non-working adult) |
| 8 | anything else | NONWORK, counted as *unmapped* |
"""

from collections import Counter

PERSON_TYPES = ["FTW", "UNIV", "NONWORK", "RETIRED", "DRIVING", "SCHOOL", "PRESCHOOL"]


def person_type(age: str, employment: str, unmapped: Counter | None = None) -> str:
    age, employment = str(age), str(employment)
    if age == "0-11":
        return "PRESCHOOL" if employment == "void" else "SCHOOL"
    if age == "11-18":
        return "DRIVING"
    if employment == "employed":
        return "FTW"
    if employment == "retired":
        return "RETIRED"
    if employment == "student":
        return "UNIV"
    if employment in ("unemployed", "void", "unknown"):
        return "NONWORK"
    if unmapped is not None:
        unmapped[(age, employment)] += 1
    return "NONWORK"
