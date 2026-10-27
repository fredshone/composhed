"""Train the DaySim-aligned variant and save ``daysim_models.pkl``."""

import time

import click

from composhed.common.bundle import save_bundle
from composhed.common.calibration import location_accessors, offset_accessors
from composhed.common.cli import make_rng, train_options
from composhed.common.report import ValidationReport
from composhed.common.runner import encode_labels, load_training_data, output_paths, run_calibration, targets_from_specs
from composhed.daysim.chain import DaysimChain, array_accessors, record_trace, trace_stats
from composhed.daysim.data import build_records
from composhed.daysim.models import DIRECTIONS, GROUPS, DaysimModels

VARIANT = "daysim"


def _calibration_specs(m: DaysimModels, timing: str) -> dict:
    specs = {"pattern": ("share", *offset_accessors(m.pattern))}
    for k in DIRECTIONS:
        specs[f"stops_{k}"] = ("share", *offset_accessors(m.nstops[k]))
    for g in GROUPS:
        tod = m.tod.get(g)
        if tod is None:
            continue
        if timing == "tod":
            # Start, end and duration constants act on the same choice: damp their joint step.
            specs[f"tod_start_{g}"] = ("share", *array_accessors(tod.start_offset), 1 / 3)
            specs[f"tod_end_{g}"] = ("share", *array_accessors(tod.end_offset), 1 / 3)
            specs[f"tod_duration_{g}"] = ("share", *array_accessors(tod.dur_offset), 1 / 3)
        else:
            specs[f"tour_duration_mean_{g}"] = ("mean", *location_accessors(m.indep[g].dur_))
    return specs


def train(
    attributes: str,
    schedules: str,
    output_dir: str,
    seed: int | None = None,
    holdout_fraction: float = 0.0,
    split_seed: int = 0,
    calibrate: bool = True,
    calib_iters: int = 8,
    calib_sample: int = 20000,
    independent_timing: bool = False,
    consistency: str = "constraint",
    max_resample: int = 50,
    time_resolution: int = 30,
    min_support: int = 30,
) -> str:
    t0 = time.time()
    print("Loading data...")
    attr_df, sched_by_pid, train_pids, hold_pids = load_training_data(
        attributes, schedules, holdout_fraction, split_seed)
    print("Extracting DaySim day structures (multiple mandatory tours kept)...")
    records, counters, caps = build_records(attr_df, sched_by_pid, train_pids)
    counters.log()
    if caps:
        print(f"  non-mandatory tours removed from pattern counts (cap 4): {caps}")
    X, feature_names = encode_labels([r.labels for r in records])
    hold_records = build_records(attr_df, sched_by_pid, hold_pids)[0] if hold_pids else []
    X_hold = encode_labels([r.labels for r in hold_records], feature_names)[0] if hold_records else None

    print(f"\nFitting DaySim components (time resolution {time_resolution} min)...")
    models = DaysimModels(time_resolution=time_resolution, min_support=min_support).fit(records, X, feature_names)

    print("\nValidation (in-sample)")
    report = ValidationReport()
    hold = models.validate(hold_records, X_hold) if hold_records else {}
    for name, metrics in models.validate(records, X).items():
        report.add(name, metrics, hold.get(name))
    report.info = {"extraction": counters.as_dict(), "pattern_caps": caps, "n_train": len(records),
                   "n_holdout": len(hold_records), "n_patterns": len(models.pattern_vectors),
                   "pattern_persons_mapped": models.pattern_merged}

    timing = "independent" if independent_timing else "tod"
    chain = DaysimChain(models, timing=timing, consistency=consistency, max_resample=max_resample)
    observed = trace_stats([record_trace(r, models) for r in records], models, timing)
    calib = None
    if calibrate:
        print("\nCalibration (targets from training data)")
        targets = targets_from_specs(observed, _calibration_specs(models, timing))
        attr_rows = [{"pid": r.pid, **r.labels} for r in records]
        calib = run_calibration(chain, targets, lambda tr: trace_stats(tr, models, timing), X, attr_rows,
                                calib_sample, calib_iters, seed)
        report.info["calibration"] = calib.to_dict()

    config = {"timing": timing, "consistency": consistency, "max_resample": max_resample,
              "time_resolution": time_resolution, "min_support": min_support, "home_inter": models.home_inter,
              "calibrate": calibrate, "calib_iters": calib_iters, "calib_sample": calib_sample,
              "holdout_fraction": holdout_fraction, "split_seed": split_seed, "seed": seed}
    path, val_path = output_paths(output_dir, VARIANT)
    save_bundle(output_dir, VARIANT, {"daysim": models}, feature_names, config,
                calibration={"targets": observed, "report": calib.to_dict() if calib else None})
    report.write(val_path)
    print(f"\nModels saved to {path} ({time.time() - t0:.1f}s); validation written to {val_path}")
    return path


@click.command()
@train_options
@click.option("--time-resolution", type=click.Choice(["15", "30", "60"]), default="30", show_default=True,
              help="DAY-10 / AB-4: TOD period length in minutes")
@click.option("--min-support", default=30, show_default=True, type=int, help="DAY-1 pattern support threshold")
def main(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
         calib_sample, independent_timing, consistency, max_resample, time_resolution, min_support) -> None:
    make_rng(seed)
    train(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
          calib_sample, independent_timing, consistency, max_resample, int(time_resolution), min_support)


if __name__ == "__main__":
    main()
