#!/usr/bin/env python3
"""EV-2: aggregate stability-run metrics into one comparison table.

Reads every ``<runs-dir>/*/seed_*/metrics.csv`` (written by run_stability.py),
adds repair shares from ``generation_report.json`` (EV-3) and prints mean ± std
per variant × ablation. Writes ``<runs-dir>/summary.csv`` (long form) and
``<runs-dir>/summary_table.csv`` (the printed table).

ActVAE (or any other external model) rows can be appended with ``--external``:
a CSV with columns ``variant, metric, value`` (one row per metric; repeat rows
for multiple seeds), where ``metric`` uses the column names of the table.
"""

import json
from pathlib import Path

import click
import pandas as pd

# (column, domain, metric) — the Table 9 set plus the main per-feature distances.
COLUMNS = [
    ("participation", "participations", "domain_mean"),
    ("transition", "transitions", "domain_mean"),
    ("timing", "timing", "domain_mean"),
    ("creativity", "creativity", "domain_mean"),
    ("feasibility", "feasibility", "domain_mean"),
    ("part. rate EMD", "participations", "participation rate"),
    ("2-gram EMD", "transitions", "2-gram"),
    ("start EMD", "timing", "start times"),
    ("duration EMD", "timing", "durations"),
    ("novel", "creativity", "novelty"),
    ("unique", "creativity", "diversity"),
]
REPAIR = ["fallback_share", "rescale_share", "repair_share", "merge_share"]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*/seed_*/metrics.csv")):
        df = pd.read_csv(path)
        df = df[df["model"] != "observed"]
        rec = {"variant": df["variant"].iloc[0], "ablation": df["ablation"].iloc[0], "seed": int(df["seed"].iloc[0])}
        for col, domain, metric in COLUMNS:
            sel = df[(df["domain"] == domain) & (df["metric"] == metric)]
            rec[col] = float(sel["value"].iloc[0]) if len(sel) else float("nan")
        rep = path.parent / "generation_report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            for k in REPAIR:
                rec[k] = r.get(k, float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows)


def load_external(path: str) -> pd.DataFrame:
    ext = pd.read_csv(path)
    out = []
    for variant, g in ext.groupby("variant"):
        n = int(g.groupby("metric").size().max())
        for i in range(n):
            rec = {"variant": variant, "ablation": "external", "seed": i}
            for metric, mg in g.groupby("metric"):
                if i < len(mg):
                    rec[metric] = float(mg["value"].iloc[i])
            out.append(rec)
    return pd.DataFrame(out)


@click.command()
@click.option("--runs-dir", default="stability_runs", show_default=True, type=click.Path(exists=True))
@click.option("--external", default=None, type=click.Path(exists=True),
              help="CSV of external results (e.g. ActVAE): variant, metric, value")
def main(runs_dir: str, external: str | None) -> None:
    base = Path(runs_dir)
    df = load_runs(base)
    if external:
        df = pd.concat([df, load_external(external)], ignore_index=True)
    if df.empty:
        raise click.ClickException(f"no metrics.csv found under {runs_dir}")
    df.to_csv(base / "summary.csv", index=False)
    value_cols = [c for c, _, _ in COLUMNS] + [c for c in REPAIR if c in df.columns]
    g = df.groupby(["variant", "ablation"], sort=False)
    mean, std, n = g[value_cols].mean(), g[value_cols].std(), g.size()
    table = mean.copy().astype(object)
    for c in value_cols:
        table[c] = [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else (f"{m:.4f}" if pd.notna(m) else "–")
                    for m, s in zip(mean[c], std[c])]
    table.insert(0, "seeds", n)
    table.to_csv(base / "summary_table.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("Lower is better for all distance columns; novel / unique are rates.")
        print(table.to_string())
    print(f"\nWritten {base / 'summary.csv'} and {base / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
