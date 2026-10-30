"""Golden-output regression test for the existing compositional and MDCEV variants (N4 / A6).

Trains and generates both existing variants on the synthetic fixture from
``test_integration`` with a fixed seed and compares SHA-256 hashes of the
generated CSVs against values captured on 2026-10-29, after the fixture moved
to the foundata NTS 2023 schema (leisure replaces medical, veh_per_adult
replaces vehicles); the compositional schedules hash was updated on
2026-10-30 when merging of neighbouring same-type episodes was removed. The hashes are machine-specific (BLAS, optimiser builds); regenerate them with ``python -m tests.test_regression_existing``
on a clean checkout if the environment changes.
"""

import hashlib
import os
import random

import numpy as np

from tests.test_integration import (
    ATTR_FIELDNAMES,
    SCHED_FIELDNAMES,
    _make_synthetic_data,
    _write_csv,
)

GOLDEN = {
    "compositional_attributes": "02eb8c5e639c95b96bc4452009481ea83d4131247743f605b26f741dfc3564d2",
    "compositional_schedules": "3f9af1ad08fd98100c5a1f584deef49de89ba461b02fcf4f71f734cd4d0944f4",
    "mdcev_attributes": "02eb8c5e639c95b96bc4452009481ea83d4131247743f605b26f741dfc3564d2",
    "mdcev_schedules": "77f38aa1fd6919a50dcbc818c3b55e9408fbe2ec82708466899bb1b1de75cec1",
}


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _run(tmp: str) -> dict[str, str]:
    from composhed.generate import generate
    from composhed.generate_mdcev import generate as generate_mdcev
    from composhed.train import train
    from composhed.train_mdcev import train as train_mdcev

    attr_rows, sched_rows = _make_synthetic_data()
    attr_path = os.path.join(tmp, "attributes.csv")
    sched_path = os.path.join(tmp, "schedules.csv")
    _write_csv(attr_path, attr_rows, ATTR_FIELDNAMES)
    _write_csv(sched_path, sched_rows, SCHED_FIELDNAMES)

    out = {}
    models = os.path.join(tmp, "models")
    _seed(0)
    train(attr_path, sched_path, models)
    _seed(0)
    a, s = os.path.join(tmp, "comp_a.csv"), os.path.join(tmp, "comp_s.csv")
    generate(attr_path, os.path.join(models, "composhed_models.pkl"), a, s)
    out["compositional_attributes"], out["compositional_schedules"] = _sha(a), _sha(s)

    _seed(0)
    train_mdcev(attr_path, sched_path, models)
    _seed(0)
    a, s = os.path.join(tmp, "mdcev_a.csv"), os.path.join(tmp, "mdcev_s.csv")
    generate_mdcev(attr_path, os.path.join(models, "mdcev_models.pkl"), a, s)
    out["mdcev_attributes"], out["mdcev_schedules"] = _sha(a), _sha(s)
    return out


def test_existing_variants_bit_identical(tmp_path):
    hashes = _run(str(tmp_path))
    for key, expected in GOLDEN.items():
        assert hashes[key] == expected, f"{key}: output changed ({hashes[key]})"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for k, v in _run(d).items():
            print(f'    "{k}": "{v}",')
