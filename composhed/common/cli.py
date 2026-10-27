"""Shared CLI options (SR-4), seeding (I6, SR-20) and output writing."""

import random

import click
import numpy as np
import polars as pl

from composhed.data import LABEL_COLS
from composhed.generate import _write_csv_with_index


def make_rng(seed: int | None) -> np.random.Generator:
    """Seed numpy and random as the existing mains do, and return the Generator used for sampling."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return np.random.default_rng(seed)


def train_options(f):
    for opt in reversed([
        click.option("--attributes", required=True, type=click.Path(exists=True)),
        click.option("--schedules", required=True, type=click.Path(exists=True)),
        click.option("--output-dir", default="models", show_default=True),
        click.option("--seed", default=None, type=int, help="Random seed"),
        click.option("--holdout-fraction", default=0.0, show_default=True, type=float,
                     help="Share of pids held out for component validation (SR-18)"),
        click.option("--split-seed", default=0, show_default=True, type=int),
        click.option("--calibrate/--no-calibrate", default=True, show_default=True,
                     help="Run the calibration stage (SR-15, AB-1)"),
        click.option("--calib-iters", default=8, show_default=True, type=int),
        click.option("--calib-sample", default=20000, show_default=True, type=int,
                     help="Persons simulated per calibration iteration"),
        click.option("--independent-timing", is_flag=True, default=False,
                     help="AB-2: CompoShed-style KDE start + independent log-normal duration"),
        click.option("--consistency", type=click.Choice(["constraint", "rescale"]), default="constraint",
                     show_default=True, help="AB-3: feasibility by construction or by rescaling"),
        click.option("--max-resample", default=50, show_default=True, type=int),
    ]):
        f = opt(f)
    return f


def generate_options(f):
    for opt in reversed([
        click.option("--attributes", required=True, type=click.Path(exists=True)),
        click.option("--models", required=True, type=click.Path(exists=True)),
        click.option("--out-attributes", default="synthetic_attributes.csv", show_default=True),
        click.option("--out-schedules", default="synthetic_schedules.csv", show_default=True),
        click.option("--seed", default=None, type=int, help="Random seed"),
        click.option("--report", default=None, help="generation_report.json path "
                     "(default: next to --out-schedules)"),
    ]):
        f = opt(f)
    return f


def write_outputs(attr_rows: list[dict], sched_rows: list[dict], out_attributes: str, out_schedules: str) -> None:
    print(f"Writing {out_attributes} ...")
    _write_csv_with_index(pl.DataFrame(attr_rows).select(["pid"] + LABEL_COLS), out_attributes)
    print(f"Writing {out_schedules} ...")
    _write_csv_with_index(
        pl.DataFrame(sched_rows).select(["pid", "act", "start", "end", "duration"]), out_schedules
    )


def default_report_path(out_schedules: str, report: str | None) -> str:
    if report:
        return report
    import os

    return os.path.join(os.path.dirname(os.path.abspath(out_schedules)), "generation_report.json")
