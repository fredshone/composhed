import csv
import sys

import pytest

from composhed.evaluate import evaluate

ATTR_FIELDNAMES = ["pid", "gender", "age_group", "car_access", "work_status", "income"]
SCHED_FIELDNAMES = ["pid", "act", "start", "end", "duration"]


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_minimal_data(pid_offset: int = 0):
    """Ten persons: home-only and simple home/work/home, all summing to 1440 min."""
    attr_rows = []
    sched_rows = []
    for i in range(10):
        pid = i + pid_offset
        ws = "employed" if i < 7 else "retired"
        gender = "M" if i % 2 == 0 else "F"
        attr_rows.append(dict(pid=pid, gender=gender, age_group="30-44",
                               car_access="yes", work_status=ws, income=3))
        if i < 5:
            sched_rows.append(dict(pid=pid, act="home", start=0, end=1440, duration=1440))
        else:
            sched_rows.extend([
                dict(pid=pid, act="home",  start=0,   end=480,  duration=480),
                dict(pid=pid, act="work",  start=480, end=780,  duration=300),
                dict(pid=pid, act="home",  start=780, end=1440, duration=660),
            ])
    return attr_rows, sched_rows


def _write_dataset(tmp_path, name, pid_offset=0):
    attr_rows, sched_rows = _make_minimal_data(pid_offset)
    attr_path = str(tmp_path / f"{name}_attrs.csv")
    sched_path = str(tmp_path / f"{name}_schedules.csv")
    _write_csv(attr_path, attr_rows, ATTR_FIELDNAMES)
    _write_csv(sched_path, sched_rows, SCHED_FIELDNAMES)
    return attr_path, sched_path


def test_evaluate_basic(tmp_path, capsys):
    target_attr, target_sched = _write_dataset(tmp_path, "target")
    model_attr, model_sched = _write_dataset(tmp_path, "model")

    evaluate(
        target_schedules=target_sched,
        modelled_schedules=[model_sched],
    )
    out = capsys.readouterr().out
    assert "ranking" in out.lower()
    assert "model(s) vs target" in out


def test_evaluate_with_attributes(tmp_path):
    target_attr, target_sched = _write_dataset(tmp_path, "target")
    model_attr, model_sched = _write_dataset(tmp_path, "model")

    evaluate(
        target_schedules=target_sched,
        modelled_schedules=[model_sched],
        target_attributes=target_attr,
        modelled_attributes=[model_attr],
    )


def test_evaluate_multiple_models(tmp_path, capsys):
    target_attr, target_sched = _write_dataset(tmp_path, "target")
    _, model_a = _write_dataset(tmp_path, "model_a")
    _, model_b = _write_dataset(tmp_path, "model_b")

    evaluate(
        target_schedules=target_sched,
        modelled_schedules=[model_a, model_b],
    )
    out = capsys.readouterr().out
    assert "model_a" in out
    assert "model_b" in out
    assert "2 model(s)" in out


def test_evaluate_output_dir(tmp_path):
    target_attr, target_sched = _write_dataset(tmp_path, "target")
    _, model_sched = _write_dataset(tmp_path, "model")
    out_dir = str(tmp_path / "results")

    evaluate(
        target_schedules=target_sched,
        modelled_schedules=[model_sched],
        output_dir=out_dir,
    )
    import os
    written = list(os.walk(out_dir))
    assert written, "No files written to output directory"
    all_files = [f for _, _, files in written for f in files]
    assert any(f.endswith(".csv") for f in all_files), f"No CSV files found: {all_files}"


def test_evaluate_mismatched_attributes(tmp_path):
    target_attr, target_sched = _write_dataset(tmp_path, "target")
    _, model_a = _write_dataset(tmp_path, "model_a")
    _, model_b = _write_dataset(tmp_path, "model_b")
    model_attr, _ = _write_dataset(tmp_path, "model_attrs")

    # Patch sys.argv to simulate CLI call
    sys.argv = [
        "evaluate",
        "--target-schedules", target_sched,
        "--modelled-schedules", model_a, model_b,
        "--modelled-attributes", model_attr,  # 1 attr for 2 schedules
    ]
    from composhed.evaluate import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
