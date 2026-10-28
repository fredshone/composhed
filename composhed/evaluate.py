import argparse
import sys
from pathlib import Path

import pandas as pd
from acteval import Evaluator

from composhed.data import LABEL_COLS


def _load_schedules(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "hid" in df.columns:
        df = df.drop(columns=["hid"])
    if "duration" not in df.columns:
        df["duration"] = df["end"] - df["start"]
    return df[["pid", "act", "start", "end", "duration"]].sort_values(["pid", "start"]).reset_index(drop=True)


def _load_attributes(path: str) -> pd.DataFrame:
    """Load attributes as strings with nulls filled as "unknown" (as data.load_attributes).

    Uses LABEL_COLS when the file has them; otherwise keeps every non-index column
    so that older attribute layouts still load.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    cols = LABEL_COLS if all(c in df.columns for c in LABEL_COLS) else [c for c in df.columns if c != "pid"]
    df[cols] = df[cols].astype("string").fillna("unknown").astype(str)
    return df[["pid"] + cols].reset_index(drop=True)


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


def results_to_frame(result) -> pd.DataFrame:
    """Flatten an acteval EvalResult into tidy rows (model, domain, metric, value).

    Rows come from the combined feature distances (one per domain × feature), the
    per-domain summary (metric "domain_mean") and the creativity descriptions
    (novelty, diversity), which are the Table 9 quantities.
    """
    rows: list[dict] = []
    dist = result.groups.combined.distances
    models = [c for c in dist.columns if c != "unit"]
    for (domain, feature), row in dist.iterrows():
        for m in models:
            rows.append({"model": m, "domain": domain, "metric": feature, "value": float(row[m])})
    summary = result.summary()
    for domain, row in summary.iterrows():
        for m in models:
            rows.append({"model": m, "domain": domain, "metric": "domain_mean", "value": float(row[m])})
    desc = result.groups.combined.descriptions
    # acteval >= 0.3 names the reference column "target" (was "observed").
    ref = "target" if "target" in desc.columns else "observed"
    for (domain, feature), row in desc.iterrows():
        if domain != "creativity":
            continue
        for m in models:
            rows.append({"model": m, "domain": domain, "metric": feature, "value": float(row[m])})
        rows.append({"model": "observed", "domain": domain, "metric": feature, "value": float(row[ref])})
    return pd.DataFrame(rows, columns=["model", "domain", "metric", "value"])


def evaluate_to_frame(
    target_schedules: str,
    modelled_schedules: list[str],
    names: list[str] | None = None,
    target_attributes: str | None = None,
    modelled_attributes: list[str] | None = None,
    split_on: list[str] | None = None,
    split_order: int | None = None,
) -> pd.DataFrame:
    """Compare modelled schedules against a target and return tidy metric rows.

    With attributes, acteval scores every slice of ``split_on`` labels (default
    LABEL_COLS) at ``split_order`` (acteval default 1) and the combined rows are
    the slice-weighted averages. Without attributes the population is unsliced.
    """
    target = _load_schedules(target_schedules)
    names = names or _unique_names(modelled_schedules)
    synthetic = {name: _load_schedules(path) for name, path in zip(names, modelled_schedules)}
    attributes = None
    target_attrs = None
    if target_attributes is not None:
        target_attrs = _load_attributes(target_attributes)
        target_attrs = target_attrs[target_attrs["pid"].isin(target["pid"].unique())].reset_index(drop=True)
        attributes = {name: _load_attributes(path) for name, path in zip(names, modelled_attributes)}
        split_on = split_on or [c for c in target_attrs.columns if c != "pid"]
    else:
        split_on = None
    evaluator = Evaluator(target=target, target_attributes=target_attrs, split_on=split_on,
                          split_order=split_order)
    result = evaluator.compare(synthetic=synthetic, attributes=attributes)
    return results_to_frame(result)


def evaluate(
    target_schedules: str,
    modelled_schedules: list[str],
    target_attributes: str | None = None,
    modelled_attributes: list[str] | None = None,
    split_on: list[str] | None = None,
    output_dir: str | None = None,
    split_order: int | None = None,
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
        split_order=split_order,
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
    parser.add_argument("--split-order", type=int, default=None,
                        help="Number of split-on labels combined per slice (acteval default 1)")
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
        split_order=args.split_order,
    )
