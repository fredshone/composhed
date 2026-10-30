"""Train the ActivitySim-aligned variant and save ``activitysim_models.pkl``."""

import time
from collections import Counter

import click

from composhed.activitysim.data import build_records
from composhed.activitysim.models import AsimModels
from composhed.common.bundle import save_bundle
from composhed.common.cli import make_rng, train_options
from composhed.common.report import ValidationReport
from composhed.common.runner import encode_labels, load_training_data, output_paths

VARIANT = "activitysim"


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
    time_resolution: int = 60,
    segmentation: bool = True,
) -> str:
    t0 = time.time()
    print("Loading data...")
    attr_df, sched_by_pid, train_pids, hold_pids = load_training_data(
        attributes, schedules, holdout_fraction, split_seed)
    print("Extracting ActivitySim tours (multiple mandatory tours kept)...")
    unmapped: Counter = Counter()
    records, counters, caps = build_records(attr_df, sched_by_pid, train_pids, unmapped)
    counters.log()
    ptypes = Counter(r.ptype for r in records)
    print(f"  person types: {dict(ptypes)}")
    if unmapped:
        print(f"  unmapped age × employment (→ NONWORK): {dict(unmapped)}")
    if caps:
        print(f"  tours beyond the frequency caps (2 per mandatory purpose; non-mandatory cap + 2 extension), "
              f"dropped: {caps}")
    X, feature_names = encode_labels([r.labels for r in records])
    hold_records = build_records(attr_df, sched_by_pid, hold_pids)[0] if hold_pids else []
    X_hold = encode_labels([r.labels for r in hold_records], feature_names)[0] if hold_records else None

    print(f"\nFitting ActivitySim components (time resolution {time_resolution} min, "
          f"segmentation {'on' if segmentation else 'off'})...")
    models = AsimModels(time_resolution=time_resolution, segmented=segmentation).fit(records, X, feature_names)

    print("\nValidation (in-sample)")
    report = ValidationReport()
    hold = models.validate(hold_records, X_hold) if hold_records else {}
    for name, metrics in models.validate(records, X).items():
        report.add(name, metrics, hold.get(name))
    report.info = {"extraction": counters.as_dict(), "person_types": dict(ptypes),
                   "unmapped": {f"{a}|{e}": n for (a, e), n in unmapped.items()}, "nm_caps": caps,
                   "n_train": len(records), "n_holdout": len(hold_records),
                   "nm_alternatives": len(models.nm_vectors), "n_params": models.n_params(),
                   "spec_coefficients": {"cdap": models.cdap.coef_table(), "mtf": models.mtf.coef_table(),
                                         **{f"nmtf {s}": m.coef_table() for s, m in models.nmtf.items()},
                                         **{f"stop_frequency {s}": m.coef_table()
                                            for s, m in models.stopfreq.items()}},
                   "nm_extension_lookup": models.nm_extension.as_dict(),
                   "tod_coefficients": {g: t.coef_table() for g, t in models.tod.items() if t is not None},
                   "trip_purpose_lookup": models.trip_purpose.as_dict(),
                   "trip_departure_tours_skipped": models.departure_records_skipped}

    timing = "independent" if independent_timing else "tod"
    config = {"timing": timing, "consistency": consistency, "max_resample": max_resample,
              "time_resolution": time_resolution, "segmentation": segmentation,
              "home_inter": models.home_inter, "holdout_fraction": holdout_fraction,
              "split_seed": split_seed, "seed": seed}
    path, val_path = output_paths(output_dir, VARIANT)
    save_bundle(output_dir, VARIANT, {"activitysim": models}, feature_names, config)
    report.write(val_path)
    print(f"\nModels saved to {path} ({time.time() - t0:.1f}s); validation written to {val_path}")
    return path


@click.command()
@train_options
@click.option("--time-resolution", type=click.Choice(["15", "30", "60"]), default="60", show_default=True,
              help="ASIM-10 / AB-4: tdd period length in minutes after the 00:00–05:00 early bin")
@click.option("--segmentation/--no-segmentation", default=True, show_default=True,
              help="AB-5: one non-mandatory tour frequency coefficient file per person type (as prototype_mtc), "
                   "or the full-time worker file for everyone, and person-type terms in scheduling")
def main(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, independent_timing, consistency,
         max_resample, time_resolution, segmentation) -> None:
    make_rng(seed)
    train(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, independent_timing, consistency,
          max_resample, int(time_resolution), segmentation)


if __name__ == "__main__":
    main()
