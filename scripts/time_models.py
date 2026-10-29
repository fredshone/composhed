#!/usr/bin/env python3
"""Clean training and generation times, one model at a time.

The stability runs share the machine between many jobs, so their times measure
the load as much as the model. This script trains and then generates each
requested variant (and ablation) strictly one after another, with the same
commands as ``run_stability.py``, and records per stage:

* wall-clock seconds;
* CPU seconds (user + system) and peak resident memory of that process alone;
* the 1-minute load average just before the stage, so busy runs can be spotted.

Writes ``<out-dir>/timings.csv`` (one row per variant × ablation × seed × stage,
appended across invocations) and ``<out-dir>/machine.json``. No evaluation is run.

    uv run python scripts/time_models.py --attributes A.parquet --schedules S.parquet \\
        --variants compositional,mdcev,cemdap,daysim,activitysim --no-calibrate
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_stability import (
    ABLATION_VARIANTS,
    ABLATIONS,
    ALL_VARIANTS,
    _generate_cmd,
    _parse_runs,
    _train_cmd,
)

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


def _timed(cmd: list[str], log_path: Path, env: dict) -> dict:
    """Run one command and return its own wall time, CPU time and peak memory."""
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    with open(log_path, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return {
        "returncode": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "peak_rss_gb": usage.ru_maxrss / 1024 ** 2,  # Linux reports KiB
        "load_before": load,
    }


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True))
@click.option("--schedules", required=True, type=click.Path(exists=True))
@click.option("--seeds", default="0", show_default=True, help="Comma-separated seeds (repeats)")
@click.option("--out-dir", default="timing_runs", show_default=True)
@click.option("--variants", default=",".join(ALL_VARIANTS), show_default=True)
@click.option("--ablations", default="", show_default=True, help="<variant>:<preset>, as in run_stability.py")
@click.option("--calibrate/--no-calibrate", default=True, show_default=True)
@click.option("--mdcev-max-records", default=None, type=int)
@click.option("--threads", default=1, show_default=True, type=int,
              help="BLAS/OpenMP threads per process (the stability runs use 1)")
@click.option("--max-load", default=1.0, show_default=True, type=float,
              help="Warn when the 1-minute load average is above this before a stage")
@click.option("--generate/--no-generate", default=True, show_default=True, help="Also time generation")
def main(attributes, schedules, seeds, out_dir, variants, ablations, calibrate, mdcev_max_records, threads,
         max_load, generate) -> None:
    runs = _parse_runs(variants, ablations, False, False)
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{v: str(threads) for v in THREAD_VARS}}
    (base / "machine.json").write_text(json.dumps({
        "host": platform.node(), "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
        "python": platform.python_version(), "threads_per_process": threads,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    out_csv = base / "timings.csv"

    for seed in [int(s) for s in seeds.split(",")]:
        for variant, ablation in runs:
            name = f"{variant}__{ablation}" if ablation else variant
            extra = ABLATIONS.get(ablation, [])
            if not calibrate and variant in ABLATION_VARIANTS["no-calibrate"] and "--no-calibrate" not in extra:
                extra = extra + ["--no-calibrate"]
            run_dir = base / name / f"seed_{seed}"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / "run.log"
            log.write_text("")

            stages = []
            train_cmd, model_pkl = _train_cmd(variant, attributes, schedules, models_dir, seed, extra,
                                              mdcev_max_records)
            stages.append(("train", train_cmd))
            if generate:
                stages.append(("generate", _generate_cmd(variant, attributes, model_pkl, run_dir, seed)))
            for stage, cmd in stages:
                if os.getloadavg()[0] > max_load:
                    click.echo(f"  warning: load average {os.getloadavg()[0]:.1f} > {max_load}; "
                               "times will be inflated", err=True)
                click.echo(f"[seed={seed}, {name}] {stage}...", nl=False)
                row = {"variant": variant, "ablation": ablation or "none", "seed": seed, "stage": stage,
                       **_timed(cmd, log, env)}
                click.echo(f" rc={row['returncode']} wall={row['wall_s']:.0f}s cpu={row['cpu_s']:.0f}s "
                           f"rss={row['peak_rss_gb']:.1f}GB")
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                if row["returncode"] != 0:
                    break

    click.echo(f"\nTimings in {out_csv}")


if __name__ == "__main__":
    main()
