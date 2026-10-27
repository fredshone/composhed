"""ASIM-1: ActivitySim person types from NTS ``age`` × ``employment``.

Rules, applied in order (first match wins):

| # | Condition | Person type |
|---|---|---|
| 1 | age ≤10 | CHILD (pre-school and non-driving child cannot be separated: one ≤10 bin) |
| 2 | employment = ft-employed | FTW (full-time worker) |
| 3 | employment = pt-employed | PTW (part-time worker) |
| 4 | employment = retired | RETIRED |
| 5 | employment = student, age ≥21 | UNIV (university student) |
| 6 | employment ∈ {student, unknown}, age 10-21 | DRIVING_STUDENT (school-age; the 10-21 bin mixes driving and non-driving ages) |
| 7 | employment ∈ {unemployed, other, unknown}, age ≥21 | NONWORK (non-working adult) |
| 8 | anything else (e.g. unemployed / other aged 10-21) | NONWORK, counted as *unmapped* |
"""

from collections import Counter

PERSON_TYPES = ["FTW", "PTW", "UNIV", "NONWORK", "RETIRED", "DRIVING_STUDENT", "CHILD"]
_ADULT_AGES = {"21-32", "32-40", "40-48", "48-54", "54-61", "61-68", "68-75", ">75"}


def person_type(age: str, employment: str, unmapped: Counter | None = None) -> str:
    age, employment = str(age), str(employment)
    if age == "≤10":
        return "CHILD"
    if employment == "ft-employed":
        return "FTW"
    if employment == "pt-employed":
        return "PTW"
    if employment == "retired":
        return "RETIRED"
    adult = age in _ADULT_AGES
    if employment == "student" and adult:
        return "UNIV"
    if employment in ("student", "unknown") and age == "10-21":
        return "DRIVING_STUDENT"
    if employment in ("unemployed", "other", "unknown") and adult:
        return "NONWORK"
    if unmapped is not None:
        unmapped[(age, employment)] += 1
    return "NONWORK"
