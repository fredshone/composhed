"""Train the DaySim-aligned variant and save ``daysim_models.pkl``."""

import time

import click

from composhed.common.bundle import save_bundle
from composhed.common.cli import make_rng, train_options
from composhed.common.report import ValidationReport
from composhed.common.runner import encode_labels, load_training_data, output_paths
from composhed.daysim.data import build_records
from composhed.daysim.models import DaysimModels

VARIANT = "daysim"


def train(
    attributes: str,
    schedules: str,
    output_dir: str,
    seed: int | None = None,
    holdout_fraction: float = 0.0,
    split_seed: int = 0,
    independent_timing: bool = False,
    consistency: str = "constraint",
    max_resample: int = 50,
    time_resolution: int = 30,
) -> str:
    t0 = time.time()
    print("Loading data...")
    attr_df, sched_by_pid, train_pids, hold_pids = load_training_data(
        attributes, schedules, holdout_fraction, split_seed)
    print("Extracting DaySim day structures (multiple mandatory tours kept)...")
    records, counters, caps = build_records(attr_df, sched_by_pid, train_pids)
    counters.log()
    if caps:
        print(f"  tours beyond 3 per purpose dropped / days outside DaySim's pattern choice set: {caps}")
    X, feature_names = encode_labels([r.labels for r in records])
    hold_records = build_records(attr_df, sched_by_pid, hold_pids)[0] if hold_pids else []
    X_hold = encode_labels([r.labels for r in hold_records], feature_names)[0] if hold_records else None

    print(f"\nFitting DaySim components (time resolution {time_resolution} min)...")
    models = DaysimModels(time_resolution=time_resolution).fit(records, X, feature_names)

    print("\nValidation (in-sample)")
    report = ValidationReport()
    hold = models.validate(hold_records, X_hold) if hold_records else {}
    for name, metrics in models.validate(records, X).items():
        report.add(name, metrics, hold.get(name))
    report.info = {"extraction": counters.as_dict(), "pattern_caps": caps, "n_train": len(records),
                   "n_holdout": len(hold_records), "n_patterns": len(models.pattern_vectors),
                   "pattern_days_left_out": models.pattern_left_out, "n_params": models.n_params(),
                   "spec_coefficients": {"stop_generation": models.stopgen.coef_table(),
                                         **{f"exact {p}": m.coef_table() for p, m in models.exact.items()
                                            if not isinstance(m, int)}},
                   "tod_coefficients": {g: t.coef_table() for g, t in models.tod.items() if t is not None}}

    timing = "independent" if independent_timing else "tod"
    config = {"timing": timing, "consistency": consistency, "max_resample": max_resample,
              "time_resolution": time_resolution, "home_inter": models.home_inter,
              "holdout_fraction": holdout_fraction, "split_seed": split_seed, "seed": seed}
    path, val_path = output_paths(output_dir, VARIANT)
    save_bundle(output_dir, VARIANT, {"daysim": models}, feature_names, config)
    report.write(val_path)
    print(f"\nModels saved to {path} ({time.time() - t0:.1f}s); validation written to {val_path}")
    return path


@click.command()
@train_options
@click.option("--time-resolution", type=click.Choice(["15", "30", "60"]), default="30", show_default=True,
              help="DAY-10 / AB-4: TOD period length in minutes")
def main(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, independent_timing, consistency,
         max_resample, time_resolution) -> None:
    make_rng(seed)
    train(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, independent_timing, consistency,
          max_resample, int(time_resolution))


if __name__ == "__main__":
    main()
