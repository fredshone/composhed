"""Shared fixture and invariant checks for the source-aligned variant tests (A2a, A2d)."""

import os

import polars as pl

from composhed.data import ACTIVITY_TYPES
from tests.test_integration import ATTR_FIELDNAMES, SCHED_FIELDNAMES, _make_synthetic_data, _write_csv

# Small calibration settings keep the tests fast.
FAST = dict(calib_iters=3, calib_sample=300)


def write_fixture(tmp_path) -> tuple[str, str, int]:
    """The test_integration fixture (H, W, WD, E, ED, D) written to CSV."""
    attr_rows, sched_rows = _make_synthetic_data()
    attr_path = os.path.join(tmp_path, "attributes.csv")
    sched_path = os.path.join(tmp_path, "schedules.csv")
    _write_csv(attr_path, attr_rows, ATTR_FIELDNAMES)
    _write_csv(sched_path, sched_rows, SCHED_FIELDNAMES)
    return attr_path, sched_path, len(attr_rows)


def assert_invariants(sched_path: str, n_persons: int) -> pl.DataFrame:
    """I1–I5 for every person, and every input person present (I7)."""
    df = pl.read_csv(sched_path)
    assert df["pid"].n_unique() == n_persons
    for pid, rows in df.group_by("pid", maintain_order=True):
        rows = rows.sort("start")
        acts = rows["act"].to_list()
        starts, ends, durs = rows["start"].to_list(), rows["end"].to_list(), rows["duration"].to_list()
        assert sum(durs) == 1440, f"pid={pid}: I1"
        assert acts[0] == "home" and acts[-1] == "home", f"pid={pid}: I2"
        assert all(a != b for a, b in zip(acts, acts[1:])), f"pid={pid}: I3"
        assert starts[0] == 0 and ends[-1] == 1440, f"pid={pid}: I4"
        assert all(e == s for e, s in zip(ends, starts[1:])), f"pid={pid}: I4 contiguity"
        assert all(e - s == d > 0 for s, e, d in zip(starts, ends, durs)), f"pid={pid}: I4 durations"
        assert set(acts) <= set(ACTIVITY_TYPES), f"pid={pid}: I5"
    return df


def run_variant(train, generate, tmp_path, train_kwargs=None, seed=7):
    """Train → generate twice with the same seed; check invariants and byte-identical output (SR-20)."""
    attr, sched, n = write_fixture(tmp_path)
    models_dir = os.path.join(tmp_path, "models")
    bundle = train(attr, sched, models_dir, seed=0, **{**FAST, **(train_kwargs or {})})
    outs = []
    for k in range(2):
        a, s = os.path.join(tmp_path, f"a{k}.csv"), os.path.join(tmp_path, f"s{k}.csv")
        report = generate(attr, bundle, a, s, seed=seed)
        outs.append((open(a, "rb").read(), open(s, "rb").read()))
    assert outs[0] == outs[1], "generation is not reproducible for a fixed seed"
    df = assert_invariants(os.path.join(tmp_path, "s0.csv"), n)
    assert os.path.exists(os.path.join(tmp_path, "generation_report.json"))
    assert os.path.exists(os.path.join(models_dir, "validation.json"))
    assert report.n_persons == n and report.fallbacks / n < 0.01
    return bundle, df, report
