import argparse
import sys
from pathlib import Path

import pandas as pd
from acteval import Evaluator

_ATTR_COLS = ["pid", "gender", "age_group", "car_access", "work_status", "income"]


def _load_schedules(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "hid" in df.columns:
        df = df.drop(columns=["hid"])
    return df[["pid", "act", "start", "end", "duration"]].sort_values(["pid", "start"]).reset_index(drop=True)


def _load_attributes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["car_access"] = df["car_access"].replace("unknown", "yes")
    return df[_ATTR_COLS].reset_index(drop=True)


def _unique_names(paths: list[str]) -> list[str]:
    stems = [Path(p).stem for p in paths]
    names = []
    seen: dict[str, int] = {}
    for stem in stems:
        if stem not in seen:
            seen[stem] = 0
            names.append(stem)
        else:
            seen[stem] += 1
            names.append(f"{stem}_{seen[stem] + 1}")
    return names


def evaluate(
    target_schedules: str,
    modelled_schedules: list[str],
    target_attributes: str | None = None,
    modelled_attributes: list[str] | None = None,
    split_on: list[str] | None = None,
    output_dir: str | None = None,
) -> None:
    target_sched_pd = _load_schedules(target_schedules)
    target_attrs_pd = _load_attributes(target_attributes) if target_attributes else None

    names = _unique_names(modelled_schedules)
    synthetic = {
        name: _load_schedules(path)
        for name, path in zip(names, modelled_schedules)
    }
    attributes = None
    if modelled_attributes:
        attributes = {
            name: _load_attributes(path)
            for name, path in zip(names, modelled_attributes)
        }

    # acteval requires split_on whenever target_attributes is provided
    effective_split_on = split_on
    if target_attrs_pd is not None and effective_split_on is None:
        effective_split_on = [c for c in target_attrs_pd.columns if c != "pid"]

    evaluator = Evaluator(
        target=target_sched_pd,
        target_attributes=target_attrs_pd,
        split_on=effective_split_on,
    )
    result = evaluator.compare(synthetic=synthetic, attributes=attributes)

    n = len(names)
    print(f"\n=== Evaluation: {n} model(s) vs target ===\n")
    print("Domain summary:")
    print(result.summary().to_string())
    print("\nModel ranking (mean distance, lower is better):")
    print(result.rank_models().to_string())

    if output_dir:
        result.save(output_dir)
        print(f"\nResults saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate modelled schedules against a target using acteval."
    )
    parser.add_argument("--target-schedules", required=True, help="Target schedules CSV")
    parser.add_argument("--target-attributes", default=None, help="Target attributes CSV (optional)")
    parser.add_argument("--modelled-schedules", nargs="+", required=True, metavar="PATH",
                        help="One or more modelled schedules CSVs")
    parser.add_argument("--modelled-attributes", nargs="*", default=None, metavar="PATH",
                        help="Modelled attributes CSVs (must match --modelled-schedules count if given)")
    parser.add_argument("--split-on", nargs="*", default=None, metavar="COL",
                        help="Attribute columns to split results on")
    parser.add_argument("--output-dir", default=None, help="Directory to save result CSVs")
    args = parser.parse_args()

    if args.modelled_attributes is not None and len(args.modelled_attributes) != len(args.modelled_schedules):
        print(
            f"Error: --modelled-attributes has {len(args.modelled_attributes)} path(s) "
            f"but --modelled-schedules has {len(args.modelled_schedules)}. Counts must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluate(
        target_schedules=args.target_schedules,
        modelled_schedules=args.modelled_schedules,
        target_attributes=args.target_attributes,
        modelled_attributes=args.modelled_attributes,
        split_on=args.split_on,
        output_dir=args.output_dir,
    )
