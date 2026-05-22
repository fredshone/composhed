"""Integration test: train → generate pipeline end-to-end with synthetic data.

Builds a minimal in-memory dataset covering all 6 DAP types (H, W, WD, E, ED, D)
and all discretionary activity types, trains all 6 sub-models, generates
schedules, and asserts the hard schedule invariants:
  1. sum(duration) == 1440  (24-hour budget)
  2. First activity == "home"
  3. Last activity == "home"
  4. No two consecutive activities have the same type
"""

import csv
import os

import numpy as np
import polars as pl

from composhed.generate import generate
from composhed.train import train

ATTR_FIELDNAMES = [
    "pid", "age", "hh_income", "sex", "employment",
    "day", "hh_zone", "access_egress_distance", "vehicles",
]
SCHED_FIELDNAMES = ["pid", "act", "start", "end", "duration"]


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_data():
    """Build ~52 synthetic persons covering all 6 DAP types.

    DAP breakdown:
      H   — pids  0– 4  (5 persons)
      W   — pids  5–12  (8 persons, work mandatory)
      WD  — pids 13–27  (15 persons, work + discretionary)
      E   — pids 28–32  (5 persons, education mandatory only)
      ED  — pids 33–37  (5 persons, education + discretionary)
      D   — pids 38–51  (14 persons, discretionary only)
    """
    attr_rows = []
    sched_rows = []

    def add(pid, age, hh_income, sex, employment, day, hh_zone, dist, vehicles, activities):
        attr_rows.append(dict(
            pid=pid, age=age, hh_income=hh_income, sex=sex, employment=employment,
            day=day, hh_zone=hh_zone, access_egress_distance=dist, vehicles=vehicles,
        ))
        for act, start, end, dur in activities:
            sched_rows.append(dict(pid=pid, act=act, start=start, end=end, duration=dur))

    # ------------------------------------------------------------------
    # H: home all day (pids 0–4)
    # ------------------------------------------------------------------
    for pid in range(5):
        s = "M" if pid % 2 == 0 else "F"
        add(pid, "30-44", 3, s, "employed", "Mon", "zone1", 1.0, 1,
            [("home", 0, 1440, 1440)])

    # ------------------------------------------------------------------
    # W: work mandatory only (pids 5–12)
    # ------------------------------------------------------------------
    w_specs = [
        ("30-44", 3, "M", "employed",   "Mon", "zone1", 1.0, 1, 480, 300),
        ("18-29", 2, "F", "employed",   "Tue", "zone2", 2.0, 0, 540, 360),
        ("45-59", 4, "M", "employed",   "Wed", "zone1", 1.5, 2, 420, 300),
        ("30-44", 3, "F", "employed",   "Thu", "zone2", 1.0, 1, 480, 360),
        ("18-29", 2, "M", "employed",   "Fri", "zone1", 2.0, 0, 540, 300),
        ("45-59", 4, "F", "employed",   "Mon", "zone2", 1.5, 1, 480, 300),
        ("60+",   4, "M", "retired",    "Tue", "zone1", 1.0, 1, 540, 360),
        ("45-59", 3, "F", "employed",   "Wed", "zone2", 2.0, 2, 480, 300),
    ]
    for i, (age, inc, sex, emp, day, zone, dist, veh, ws_t, md) in enumerate(w_specs):
        pid = 5 + i
        we = ws_t + md
        add(pid, age, inc, sex, emp, day, zone, dist, veh, [
            ("home", 0, ws_t, ws_t),
            ("work", ws_t, we, md),
            ("home", we, 1440, 1440 - we),
        ])

    # ------------------------------------------------------------------
    # WD: work + discretionary (pids 13–27)
    # ------------------------------------------------------------------
    wd_data = [
        ("30-44", 3, "M", "employed",   "Mon", "zone1", 1.0, 1, [
            ("home", 0, 360, 360), ("escort", 360, 420, 60),
            ("work", 420, 720, 300), ("home", 720, 1440, 720)]),
        ("18-29", 2, "F", "employed",   "Tue", "zone2", 2.0, 0, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("shop", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("45-59", 4, "M", "employed",   "Wed", "zone1", 1.5, 2, [
            ("home", 0, 360, 360), ("medical", 360, 420, 60),
            ("work", 420, 780, 360), ("visit", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("30-44", 3, "F", "employed",   "Thu", "zone2", 1.0, 1, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("other", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("30-44", 3, "M", "employed",   "Fri", "zone1", 2.0, 1, [
            ("home", 0, 420, 420), ("work", 420, 720, 300),
            ("visit", 720, 780, 60), ("shop", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("18-29", 2, "F", "employed",   "Mon", "zone2", 1.0, 0, [
            ("home", 0, 360, 360), ("escort", 360, 420, 60),
            ("work", 420, 720, 300), ("shop", 720, 780, 60), ("home", 780, 1440, 660)]),
        ("45-59", 4, "M", "employed",   "Tue", "zone1", 1.5, 2, [
            ("home", 0, 480, 480), ("work", 480, 840, 360),
            ("medical", 840, 900, 60), ("other", 900, 960, 60), ("home", 960, 1440, 480)]),
        ("30-44", 3, "F", "employed",   "Wed", "zone2", 1.0, 1, [
            ("home", 0, 360, 360), ("shop", 360, 420, 60), ("work", 420, 720, 300),
            ("visit", 720, 780, 60), ("escort", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("18-29", 2, "M", "employed",   "Thu", "zone1", 2.0, 0, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("other", 780, 840, 60), ("shop", 840, 900, 60), ("home", 900, 1440, 540)]),
        ("45-59", 4, "F", "employed",   "Fri", "zone2", 1.5, 2, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("visit", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("30-44", 3, "M", "employed",   "Mon", "zone1", 1.0, 1, [
            ("home", 0, 360, 360), ("medical", 360, 420, 60), ("work", 420, 720, 300),
            ("shop", 720, 780, 60), ("other", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("18-29", 2, "F", "employed",   "Tue", "zone2", 2.0, 0, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("escort", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("45-59", 4, "M", "employed",   "Wed", "zone1", 1.5, 2, [
            ("home", 0, 420, 420), ("work", 420, 720, 300),
            ("other", 720, 780, 60), ("visit", 780, 840, 60),
            ("medical", 840, 900, 60), ("home", 900, 1440, 540)]),
        ("30-44", 3, "F", "employed",   "Thu", "zone2", 1.0, 1, [
            ("home", 0, 360, 360), ("other", 360, 420, 60),
            ("work", 420, 720, 300), ("home", 720, 1440, 720)]),
        ("30-44", 3, "M", "employed",   "Fri", "zone1", 2.0, 1, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("shop", 780, 840, 60), ("medical", 840, 900, 60), ("home", 900, 1440, 540)]),
    ]
    for i, (age, inc, sex, emp, day, zone, dist, veh, acts) in enumerate(wd_data):
        add(13 + i, age, inc, sex, emp, day, zone, dist, veh, acts)

    # ------------------------------------------------------------------
    # E: education mandatory only (pids 28–32)
    # ------------------------------------------------------------------
    e_specs = [
        ("18-29", 2, "M", "student", "Mon", "zone1", 1.0, 0, 480, 300),
        ("18-29", 2, "F", "student", "Tue", "zone2", 2.0, 0, 540, 300),
        ("18-29", 3, "M", "student", "Wed", "zone1", 1.5, 0, 480, 300),
        ("18-29", 2, "F", "student", "Thu", "zone2", 1.0, 0, 540, 300),
        ("18-29", 3, "M", "student", "Fri", "zone1", 2.0, 0, 480, 300),
    ]
    for i, (age, inc, sex, emp, day, zone, dist, veh, ws_t, md) in enumerate(e_specs):
        pid = 28 + i
        we = ws_t + md
        add(pid, age, inc, sex, emp, day, zone, dist, veh, [
            ("home",      0,    ws_t, ws_t),
            ("education", ws_t, we,   md),
            ("home",      we,   1440, 1440 - we),
        ])

    # ------------------------------------------------------------------
    # ED: education + discretionary (pids 33–37)
    # ------------------------------------------------------------------
    ed_data = [
        ("18-29", 2, "M", "student", "Mon", "zone1", 1.0, 0, [
            ("home", 0, 480, 480), ("education", 480, 780, 300),
            ("shop", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("18-29", 2, "F", "student", "Tue", "zone2", 2.0, 0, [
            ("home", 0, 540, 540), ("education", 540, 840, 300),
            ("visit", 840, 900, 60), ("home", 900, 1440, 540)]),
        ("18-29", 3, "M", "student", "Wed", "zone1", 1.5, 0, [
            ("home", 0, 480, 480), ("education", 480, 780, 300),
            ("other", 780, 840, 60), ("home", 840, 1440, 600)]),
        ("18-29", 2, "F", "student", "Thu", "zone2", 1.0, 0, [
            ("home", 0, 360, 360), ("escort", 360, 420, 60),
            ("education", 420, 720, 300), ("home", 720, 1440, 720)]),
        ("18-29", 3, "M", "student", "Fri", "zone1", 2.0, 0, [
            ("home", 0, 480, 480), ("education", 480, 780, 300),
            ("medical", 780, 840, 60), ("home", 840, 1440, 600)]),
    ]
    for i, (age, inc, sex, emp, day, zone, dist, veh, acts) in enumerate(ed_data):
        add(33 + i, age, inc, sex, emp, day, zone, dist, veh, acts)

    # ------------------------------------------------------------------
    # D: discretionary only (pids 38–51)
    # ------------------------------------------------------------------
    d_data = [
        ("60+",   4, "M", "retired",     "Mon", "zone1", 1.0, 1, [
            ("home", 0, 480, 480), ("shop",    480, 540, 60), ("home", 540, 1440, 900)]),
        ("45-59", 3, "F", "not working", "Tue", "zone2", 1.5, 0, [
            ("home", 0, 540, 540), ("visit",   540, 600, 60), ("home", 600, 1440, 840)]),
        ("18-29", 4, "M", "employed",    "Wed", "zone1", 2.0, 0, [
            ("home", 0, 480, 480), ("escort",  480, 540, 60),
            ("other", 540, 600, 60), ("home", 600, 1440, 840)]),
        ("60+",   4, "F", "retired",     "Thu", "zone2", 1.0, 1, [
            ("home", 0, 480, 480), ("medical", 480, 540, 60), ("home", 540, 1440, 900)]),
        ("45-59", 3, "M", "not working", "Fri", "zone1", 1.5, 0, [
            ("home", 0, 540, 540), ("shop",    540, 600, 60),
            ("visit", 600, 660, 60), ("home", 660, 1440, 780)]),
        ("30-44", 3, "F", "employed",    "Mon", "zone2", 2.0, 1, [
            ("home", 0, 600, 600), ("other",   600, 660, 60),
            ("shop", 660, 720, 60), ("visit", 720, 780, 60), ("home", 780, 1440, 660)]),
        ("60+",   4, "M", "retired",     "Tue", "zone1", 1.0, 1, [
            ("home", 0, 540, 540), ("escort",  540, 600, 60), ("home", 600, 1440, 840)]),
        ("45-59", 3, "F", "not working", "Wed", "zone2", 1.5, 0, [
            ("home", 0, 480, 480), ("visit",   480, 540, 60),
            ("medical", 540, 600, 60), ("home", 600, 1440, 840)]),
        ("45-59", 2, "M", "employed",    "Thu", "zone1", 2.0, 1, [
            ("home", 0, 540, 540), ("shop",    540, 600, 60), ("home", 600, 1440, 840)]),
        ("60+",   4, "F", "retired",     "Fri", "zone2", 1.0, 1, [
            ("home", 0, 600, 600), ("other",   600, 660, 60), ("home", 660, 1440, 780)]),
        ("60+",   3, "M", "retired",     "Mon", "zone1", 1.5, 0, [
            ("home", 0, 480, 480), ("medical", 480, 540, 60), ("home", 540, 1440, 900)]),
        ("45-59", 2, "F", "not working", "Tue", "zone2", 2.0, 0, [
            ("home", 0, 540, 540), ("escort",  540, 600, 60), ("home", 600, 1440, 840)]),
        ("30-44", 4, "M", "employed",    "Wed", "zone1", 1.0, 1, [
            ("home", 0, 480, 480), ("shop",    480, 540, 60),
            ("visit", 540, 600, 60), ("home", 600, 1440, 840)]),
        ("60+",   3, "F", "retired",     "Thu", "zone2", 1.5, 1, [
            ("home", 0, 600, 600), ("other",   600, 660, 60), ("home", 660, 1440, 780)]),
    ]
    for i, (age, inc, sex, emp, day, zone, dist, veh, acts) in enumerate(d_data):
        add(38 + i, age, inc, sex, emp, day, zone, dist, veh, acts)

    return attr_rows, sched_rows


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_train_generate_pipeline(tmp_path):
    """Full train → generate pipeline with synthetic in-memory data."""
    attr_rows, sched_rows = _make_synthetic_data()
    n_persons = len(attr_rows)

    attr_path = str(tmp_path / "attributes.csv")
    sched_path = str(tmp_path / "schedules.csv")
    _write_csv(attr_path, attr_rows, ATTR_FIELDNAMES)
    _write_csv(sched_path, sched_rows, SCHED_FIELDNAMES)

    # --- Train all 6 sub-models ---
    models_dir = str(tmp_path / "models")
    train(attr_path, sched_path, models_dir)

    models_path = os.path.join(models_dir, "composhed_models.pkl")
    assert os.path.exists(models_path), "Model bundle not written"

    # --- Generate schedules ---
    out_attr = str(tmp_path / "out_attr.csv")
    out_sched = str(tmp_path / "out_sched.csv")
    np.random.seed(42)
    generate(attr_path, models_path, out_attr, out_sched)

    assert os.path.exists(out_attr), "Output attributes not written"
    assert os.path.exists(out_sched), "Output schedules not written"

    sched_df = pl.read_csv(out_sched)
    assert len(sched_df) > 0, "Output schedules empty"

    pids = sched_df["pid"].unique().to_list()
    assert len(pids) == n_persons, f"Expected {n_persons} pids in output, got {len(pids)}"

    # --- Per-person schedule invariants ---
    for pid in sorted(pids):
        rows = sched_df.filter(pl.col("pid") == pid).sort("start")
        acts = rows["act"].to_list()
        durs = rows["duration"].to_list()

        assert sum(durs) == 1440, f"pid={pid}: sum(duration)={sum(durs)}"
        assert acts[0] == "home", f"pid={pid}: first act={acts[0]!r}"
        assert acts[-1] == "home", f"pid={pid}: last act={acts[-1]!r}"

        for j in range(len(acts) - 1):
            assert acts[j] != acts[j + 1], (
                f"pid={pid}: consecutive '{acts[j]}' at positions {j},{j+1}"
            )
