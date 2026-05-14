#!/usr/bin/env python3
"""Run 5-seed stability check for CompSched (compositional and MDCEV variants)."""

import subprocess
import sys
import time
from pathlib import Path

import click


def _run_cmd(cmd: list[str], log_path: Path, append: bool = False) -> tuple[int, float]:
    t0 = time.time()
    mode = "a" if append else "w"
    with open(log_path, mode) as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return result.returncode, time.time() - t0


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True), help="Path to attributes CSV")
@click.option("--schedules", required=True, type=click.Path(exists=True), help="Path to schedules CSV")
@click.option("--seeds", default="0,1,2,3,4", show_default=True, help="Comma-separated seed list")
@click.option("--out-dir", default="stability_runs", show_default=True, help="Output root directory")
@click.option("--mdcev-max-records", default=5000, show_default=True, type=int)
@click.option("--skip-mdcev", is_flag=True, default=False, help="Run compositional variant only")
def main(
    attributes: str,
    schedules: str,
    seeds: str,
    out_dir: str,
    mdcev_max_records: int,
    skip_mdcev: bool,
) -> None:
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    base = Path(out_dir)
    variants = ["compositional"] + ([] if skip_mdcev else ["mdcev"])
    summary: list[tuple[int, str, str, float]] = []

    for seed in seed_list:
        for variant in variants:
            run_dir = base / f"seed_{seed}" / variant
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / "run.log"

            click.echo(f"[seed={seed}, variant={variant}] training...", nl=False)

            if variant == "compositional":
                train_cmd = [
                    sys.executable, "-m", "composhed.train",
                    "--attributes", attributes,
                    "--schedules", schedules,
                    "--output-dir", str(models_dir),
                    "--seed", str(seed),
                ]
                model_pkl = models_dir / "composhed_models.pkl"
            else:
                train_cmd = [
                    sys.executable, "-m", "composhed.train_mdcev",
                    "--attributes", attributes,
                    "--schedules", schedules,
                    "--output-dir", str(models_dir),
                    "--seed", str(seed),
                    "--max-records", str(mdcev_max_records),
                ]
                model_pkl = models_dir / "mdcev_models.pkl"

            rc, elapsed = _run_cmd(train_cmd, log)
            train_status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            click.echo(f" {train_status} ({elapsed:.0f}s)")

            if rc != 0:
                summary.append((seed, variant, f"train={train_status}", elapsed))
                continue

            click.echo(f"[seed={seed}, variant={variant}] generating...", nl=False)

            if variant == "compositional":
                gen_cmd = [
                    sys.executable, "-m", "composhed.generate",
                    "--attributes", attributes,
                    "--models", str(model_pkl),
                    "--out-attributes", str(run_dir / "synthetic_attributes.csv"),
                    "--out-schedules", str(run_dir / "synthetic_schedules.csv"),
                    "--seed", str(seed),
                ]
            else:
                gen_cmd = [
                    sys.executable, "-m", "composhed.generate_mdcev",
                    "--attributes", attributes,
                    "--models", str(model_pkl),
                    "--out-attributes", str(run_dir / "synthetic_mdcev_attributes.csv"),
                    "--out-schedules", str(run_dir / "synthetic_mdcev_schedules.csv"),
                    "--seed", str(seed),
                ]

            rc2, elapsed2 = _run_cmd(gen_cmd, log, append=True)
            gen_status = "OK" if rc2 == 0 else f"FAIL(rc={rc2})"
            click.echo(f" {gen_status} ({elapsed2:.0f}s)")
            summary.append((seed, variant, f"train={train_status} gen={gen_status}", elapsed + elapsed2))

    click.echo("\n=== Stability run summary ===")
    for seed, variant, status, elapsed in summary:
        click.echo(f"  seed={seed:2d}  {variant:<14}  {status}  ({elapsed:.0f}s total)")


if __name__ == "__main__":
    main()
