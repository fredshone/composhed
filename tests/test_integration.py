"""Integration test: train → generate pipeline end-to-end with synthetic data.

Builds a minimal in-memory dataset covering all 4 DAP types (H, W, WD, D)
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

ATTR_FIELDNAMES = ["pid", "gender", "age_group", "car_access", "work_status", "income"]
SCHED_FIELDNAMES = ["pid", "act", "start", "end", "duration"]


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_data():
    """Build ~42 synthetic persons covering all 4 DAP types.

    DAP breakdown:
      H  — pids  0– 4  (5 persons)
      W  — pids  5–16  (12 persons, work & education, various work_status)
      WD — pids 17–31  (15 persons, 1–3 disc activities, before/after work mix)
      D  — pids 32–41  (10 persons, 1–3 disc activities)

    All 5 discretionary types (escort, medical, other, shop, visit) appear
    across slots 1, 2, and 3+ to allow each sub-model to fit.
    """
    attr_rows = []
    sched_rows = []

    def add(pid, gender, age_group, car_access, work_status, income, activities):
        attr_rows.append(
            dict(
                pid=pid,
                gender=gender,
                age_group=age_group,
                car_access=car_access,
                work_status=work_status,
                income=income,
            )
        )
        for act, start, end, dur in activities:
            sched_rows.append(dict(pid=pid, act=act, start=start, end=end, duration=dur))

    # ------------------------------------------------------------------
    # H: home all day (pids 0–4)
    # ------------------------------------------------------------------
    for pid in range(5):
        g = "M" if pid % 2 == 0 else "F"
        add(pid, g, "30-44", "yes", "employed", 3, [("home", 0, 1440, 1440)])

    # ------------------------------------------------------------------
    # W: mandatory only (pids 5–16)
    # Variety in work_status so KDE fits per work_status category.
    # ------------------------------------------------------------------
    w_specs = [
        ("M", "30-44", "yes", "employed", 3, "work",      480, 300),
        ("F", "18-29", "no",  "employed", 2, "work",      540, 360),
        ("M", "45-59", "yes", "employed", 4, "work",      420, 300),
        ("F", "30-44", "yes", "employed", 3, "work",      480, 360),
        ("M", "18-29", "no",  "student",  2, "work",      540, 300),
        ("F", "45-59", "yes", "employed", 3, "work",      480, 300),
        ("M", "30-44", "yes", "student",  3, "education", 480, 300),
        ("F", "18-29", "no",  "student",  2, "education", 540, 300),
        ("M", "45-59", "yes", "employed", 4, "work",      420, 300),
        ("F", "30-44", "yes", "employed", 3, "work",      480, 300),
        ("M", "60+",   "no",  "retired",  4, "work",      540, 360),
        ("F", "45-59", "yes", "employed", 3, "work",      480, 300),
    ]
    for i, (g, ag, ca, ws, inc, act_type, ws_t, md) in enumerate(w_specs):
        pid = 5 + i
        we = ws_t + md
        add(pid, g, ag, ca, ws, inc, [
            ("home",    0,    ws_t, ws_t),
            (act_type,  ws_t, we,   md),
            ("home",    we,   1440, 1440 - we),
        ])

    # ------------------------------------------------------------------
    # WD: mandatory + discretionary (pids 17–31)
    # Mix of before-work and after-work disc activities.
    # Covers all 5 disc types and slots 1, 2, 3+.
    # ------------------------------------------------------------------
    wd_data = [
        # 1 disc before work
        ("M", "30-44", "yes", "employed", 3, [
            ("home", 0, 360, 360), ("escort", 360, 420, 60),
            ("work", 420, 720, 300), ("home", 720, 1440, 720)]),
        # 1 disc after work
        ("F", "18-29", "no",  "employed", 2, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("shop", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 1 before + 1 after (slot 1, 2) — md=360 to vary rem_budget
        ("M", "45-59", "yes", "employed", 4, [
            ("home", 0, 360, 360), ("medical", 360, 420, 60),
            ("work", 420, 780, 360), ("visit", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 1 disc after work
        ("F", "30-44", "yes", "employed", 3, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("other", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 2 disc after work (slots 1, 2)
        ("M", "30-44", "yes", "employed", 3, [
            ("home", 0, 420, 420), ("work", 420, 720, 300),
            ("visit", 720, 780, 60), ("shop", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 1 before + 1 after
        ("F", "18-29", "no",  "employed", 2, [
            ("home", 0, 360, 360), ("escort", 360, 420, 60),
            ("work", 420, 720, 300), ("shop", 720, 780, 60), ("home", 780, 1440, 660)]),
        # 2 disc after work (slots 1, 2) — md=360 to vary rem_budget
        ("M", "45-59", "yes", "employed", 4, [
            ("home", 0, 480, 480), ("work", 480, 840, 360),
            ("medical", 840, 900, 60), ("other", 900, 960, 60), ("home", 960, 1440, 480)]),
        # 1 before + 2 after (slots 1, 2, 3+)
        ("F", "30-44", "yes", "employed", 3, [
            ("home", 0, 360, 360), ("shop", 360, 420, 60), ("work", 420, 720, 300),
            ("visit", 720, 780, 60), ("escort", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 2 disc after work
        ("M", "18-29", "no",  "employed", 2, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("other", 780, 840, 60), ("shop", 840, 900, 60), ("home", 900, 1440, 540)]),
        # 1 disc after work
        ("F", "45-59", "yes", "employed", 4, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("visit", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 1 before + 2 after (slots 1, 2, 3+)
        ("M", "30-44", "yes", "employed", 3, [
            ("home", 0, 360, 360), ("medical", 360, 420, 60), ("work", 420, 720, 300),
            ("shop", 720, 780, 60), ("other", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 1 disc after work
        ("F", "18-29", "no",  "employed", 2, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("escort", 780, 840, 60), ("home", 840, 1440, 600)]),
        # 3 disc after work (slots 1, 2, 3+)
        ("M", "45-59", "yes", "employed", 4, [
            ("home", 0, 420, 420), ("work", 420, 720, 300),
            ("other", 720, 780, 60), ("visit", 780, 840, 60),
            ("medical", 840, 900, 60), ("home", 900, 1440, 540)]),
        # 1 disc before work
        ("F", "30-44", "yes", "employed", 3, [
            ("home", 0, 360, 360), ("other", 360, 420, 60),
            ("work", 420, 720, 300), ("home", 720, 1440, 720)]),
        # 2 disc after work
        ("M", "30-44", "yes", "employed", 3, [
            ("home", 0, 480, 480), ("work", 480, 780, 300),
            ("shop", 780, 840, 60), ("medical", 840, 900, 60), ("home", 900, 1440, 540)]),
    ]
    for i, (g, ag, ca, ws, inc, acts) in enumerate(wd_data):
        add(17 + i, g, ag, ca, ws, inc, acts)

    # ------------------------------------------------------------------
    # D: discretionary only (pids 32–41)
    # Mix of work_status (retired / not working / employed) so all three
    # KDEs fit per work_status category (≥3 records each).
    # The three "employed" D persons have distinct age/income/car to break
    # the implicit-constant collinearity that arises when WD is always
    # employed and D is always retired/not-working.
    # ------------------------------------------------------------------
    d_data = [
        # retired (4 persons → KDE ≥ 3)
        ("M", "60+",   "no",  "retired",     4, [
            ("home", 0, 480, 480), ("shop",   480, 540, 60), ("home", 540, 1440, 900)]),
        ("F", "45-59", "yes", "not working", 3, [
            ("home", 0, 540, 540), ("visit",  540, 600, 60), ("home", 600, 1440, 840)]),
        # employed – distinct age/income/car to avoid implicit constant
        ("M", "18-29", "no",  "employed",    4, [
            ("home", 0, 480, 480), ("escort", 480, 540, 60),
            ("other", 540, 600, 60), ("home", 600, 1440, 840)]),
        ("F", "60+",   "no",  "retired",     4, [
            ("home", 0, 480, 480), ("medical",480, 540, 60), ("home", 540, 1440, 900)]),
        ("M", "45-59", "yes", "not working", 3, [
            ("home", 0, 540, 540), ("shop",   540, 600, 60),
            ("visit", 600, 660, 60), ("home", 660, 1440, 780)]),
        # employed – varied departure time (600) to break KDE singularity
        ("F", "30-44", "yes", "employed",    3, [
            ("home", 0, 600, 600), ("other",  600, 660, 60),
            ("shop", 660, 720, 60), ("visit", 720, 780, 60), ("home", 780, 1440, 660)]),
        ("M", "60+",   "no",  "retired",     4, [
            ("home", 0, 540, 540), ("escort", 540, 600, 60), ("home", 600, 1440, 840)]),
        ("F", "45-59", "yes", "not working", 3, [
            ("home", 0, 480, 480), ("visit",  480, 540, 60),
            ("medical", 540, 600, 60), ("home", 600, 1440, 840)]),
        # employed – varied departure time (540) and income=2
        ("M", "45-59", "yes", "employed",    2, [
            ("home", 0, 540, 540), ("shop",   540, 600, 60), ("home", 600, 1440, 840)]),
        ("F", "60+",   "no",  "retired",     4, [
            ("home", 0, 600, 600), ("other",  600, 660, 60), ("home", 660, 1440, 780)]),
    ]
    for i, (g, ag, ca, ws, inc, acts) in enumerate(d_data):
        add(32 + i, g, ag, ca, ws, inc, acts)

    return attr_rows, sched_rows


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_train_generate_pipeline(tmp_path):
    """Full train → generate pipeline with synthetic in-memory data."""
    attr_rows, sched_rows = _make_synthetic_data()

    attr_path = str(tmp_path / "attributes.csv")
    sched_path = str(tmp_path / "schedules.csv")
    _write_csv(attr_path, attr_rows, ATTR_FIELDNAMES)
    _write_csv(sched_path, sched_rows, SCHED_FIELDNAMES)

    # --- Train all 6 sub-models ---
    models_dir = str(tmp_path / "models")
    train(attr_path, sched_path, models_dir)

    models_path = os.path.join(models_dir, "composhed_models.pkl")
    assert os.path.exists(models_path), "Model bundle not written"

    # --- Generate schedules for all 42 persons ---
    out_attr = str(tmp_path / "out_attr.csv")
    out_sched = str(tmp_path / "out_sched.csv")
    np.random.seed(42)
    generate(attr_path, models_path, out_attr, out_sched)

    assert os.path.exists(out_attr), "Output attributes not written"
    assert os.path.exists(out_sched), "Output schedules not written"

    sched_df = pl.read_csv(out_sched)
    assert len(sched_df) > 0, "Output schedules empty"

    pids = sched_df["pid"].unique().to_list()
    assert len(pids) == 42, f"Expected 42 pids in output, got {len(pids)}"

    # --- Per-person schedule invariants ---
    for pid in sorted(pids):
        rows = sched_df.filter(pl.col("pid") == pid).sort("start")
        acts = rows["act"].to_list()
        durs = rows["duration"].to_list()

        # 1. 24-hour budget
        assert sum(durs) == 1440, f"pid={pid}: sum(duration)={sum(durs)}"

        # 2. Starts with home
        assert acts[0] == "home", f"pid={pid}: first act={acts[0]!r}"

        # 3. Ends with home
        assert acts[-1] == "home", f"pid={pid}: last act={acts[-1]!r}"

        # 4. No two consecutive identical activities
        for j in range(len(acts) - 1):
            assert acts[j] != acts[j + 1], (
                f"pid={pid}: consecutive '{acts[j]}' at positions {j},{j+1}"
            )
