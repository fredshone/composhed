#!/usr/bin/env python3
"""Multi-seed stability runs for every CompoShed variant, with acteval evaluation (EV-1, EV-2).

Layout: ``<out-dir>/<variant>[__<ablation>]/seed_<n>/`` containing ``models/``,
``synthetic_attributes.csv``, ``synthetic_schedules.csv``, ``run.log``,
``generation_report.json`` (new variants), ``validation.json`` (new variants)
and ``metrics.csv`` (one tidy row per acteval metric). Summarise with
``scripts/summarise_runs.py``.

Every run generates one schedule for every person in ``--attributes`` and is
evaluated against the full ``--schedules`` (PLAN §0.0).
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import click

ALL_VARIANTS = ["compositional", "mdcev", "cemdap", "daysim", "activitysim"]
NEW_VARIANTS = {"cemdap", "daysim", "activitysim"}

# Ablation presets (SPECIFICATION §8): name -> extra train arguments.
ABLATIONS = {
    "no-calibrate": ["--no-calibrate"],  # AB-1
    "independent-timing": ["--independent-timing"],  # AB-2
    "rescale": ["--consistency", "rescale"],  # AB-3
    "res15": ["--time-resolution", "15"],  # AB-4
    "res30": ["--time-resolution", "30"],
    "res60": ["--time-resolution", "60"],
    "no-segmentation": ["--no-segmentation"],  # AB-5
}
ABLATION_VARIANTS = {
    "no-calibrate": NEW_VARIANTS | {"mdcev"},
    "independent-timing": NEW_VARIANTS,
    "rescale": NEW_VARIANTS,
    "res15": {"daysim", "activitysim"},
    "res30": {"daysim", "activitysim"},
    "res60": {"daysim", "activitysim"},
    "no-segmentation": {"activitysim"},
}


def _run_cmd(cmd: list[str], log_path: Path, append: bool = False) -> tuple[int, float]:
    t0 = time.time()
    with open(log_path, "a" if append else "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return result.returncode, time.time() - t0


def _train_cmd(variant: str, attributes: str, schedules: str, models_dir: Path, seed: int,
               extra: list[str], mdcev_max_records: int | None) -> tuple[list[str], Path]:
    base = ["--attributes", attributes, "--schedules", schedules, "--output-dir", str(models_dir),
            "--seed", str(seed)]
    if variant == "compositional":
        return [sys.executable, "-m", "composhed.train", *base], models_dir / "composhed_models.pkl"
    if variant == "mdcev":
        cmd = [sys.executable, "-m", "composhed.train_mdcev", *base, "--free-gamma", *extra]
        if mdcev_max_records is not None:
            cmd += ["--max-records", str(mdcev_max_records)]
        return cmd, models_dir / "mdcev_models.pkl"
    return [sys.executable, "-m", f"composhed.{variant}.train", *base, *extra], models_dir / f"{variant}_models.pkl"


def _generate_cmd(variant: str, attributes: str, model_pkl: Path, run_dir: Path, seed: int) -> list[str]:
    module = {"compositional": "composhed.generate", "mdcev": "composhed.generate_mdcev"}.get(
        variant, f"composhed.{variant}.generate")
    return [sys.executable, "-m", module, "--attributes", attributes, "--models", str(model_pkl),
            "--out-attributes", str(run_dir / "synthetic_attributes.csv"),
            "--out-schedules", str(run_dir / "synthetic_schedules.csv"), "--seed", str(seed)]


def _evaluate(run_dir: Path, target: str, variant: str, ablation: str, seed: int,
              target_attributes: str | None = None, split_order: int | None = None) -> None:
    from composhed.evaluate import evaluate_to_frame

    sliced = split_order is not None
    df = evaluate_to_frame(
        target, [str(run_dir / "synthetic_schedules.csv")], names=[variant],
        target_attributes=target_attributes if sliced else None,
        modelled_attributes=[str(run_dir / "synthetic_attributes.csv")] if sliced else None,
        split_order=split_order,
    )
    df.insert(0, "seed", seed)
    df.insert(0, "ablation", ablation or "none")
    df.insert(0, "variant", variant)
    df.to_csv(run_dir / "metrics.csv", index=False)


def _parse_runs(variants: str, ablations: str, skip_compositional: bool, skip_mdcev: bool) -> list[tuple[str, str]]:
    chosen = [v.strip() for v in variants.split(",") if v.strip()]
    for v in chosen:
        if v not in ALL_VARIANTS:
            raise click.BadParameter(f"unknown variant {v!r}; choose from {ALL_VARIANTS}")
    chosen = [v for v in chosen if not (v == "compositional" and skip_compositional)
              and not (v == "mdcev" and skip_mdcev)]
    runs = [(v, "") for v in chosen]
    for item in [a.strip() for a in ablations.split(",") if a.strip()]:
        if ":" not in item:
            raise click.BadParameter(f"ablation {item!r} must be <variant>:<preset>")
        v, name = item.split(":", 1)
        if name not in ABLATIONS:
            raise click.BadParameter(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
        if v not in ABLATION_VARIANTS[name]:
            raise click.BadParameter(f"ablation {name!r} does not apply to {v!r}")
        runs.append((v, name))
    return runs


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True), help="Attributes CSV")
@click.option("--schedules", required=True, type=click.Path(exists=True), help="Training schedules CSV")
@click.option("--target-schedules", default=None, type=click.Path(exists=True),
              help="Schedules to evaluate against (default: --schedules)")
@click.option("--seeds", default="0,1,2,3,4", show_default=True, help="Comma-separated seed list")
@click.option("--out-dir", default="stability_runs", show_default=True, help="Output root directory")
@click.option("--variants", default=",".join(ALL_VARIANTS), show_default=True,
              help="Comma-separated variants to run")
@click.option("--ablations", default="", show_default=True,
              help="Comma-separated <variant>:<preset> ablation runs, e.g. cemdap:no-calibrate,daysim:res15. "
                   f"Presets: {', '.join(sorted(ABLATIONS))}")
@click.option("--mdcev-max-records", default=None, type=int)
@click.option("--skip-compositional", is_flag=True, default=False, help="Alias: drop compositional from --variants")
@click.option("--skip-mdcev", is_flag=True, default=False, help="Alias: drop mdcev from --variants")
@click.option("--evaluate/--no-evaluate", default=True, show_default=True, help="Run acteval per run")
@click.option("--split-order", default=None, type=int,
              help="Evaluate per label slice (all LABEL_COLS) at this acteval split order; default unsliced")
@click.option("--calibrate/--no-calibrate", default=True, show_default=True,
              help="Calibrate every variant that has a calibration stage (--no-calibrate: AB-1 for all)")
@click.option("--skip-existing", is_flag=True, default=False,
              help="Skip runs whose metrics.csv already exists")
def main(attributes, schedules, target_schedules, seeds, out_dir, variants, ablations, mdcev_max_records,
         skip_compositional, skip_mdcev, evaluate, split_order, calibrate, skip_existing) -> None:
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    runs = _parse_runs(variants, ablations, skip_compositional, skip_mdcev)
    target = target_schedules or schedules
    base = Path(out_dir)
    summary: list[tuple] = []

    for variant, ablation in runs:
        name = f"{variant}__{ablation}" if ablation else variant
        extra = ABLATIONS.get(ablation, [])
        if not calibrate and variant in ABLATION_VARIANTS["no-calibrate"] and "--no-calibrate" not in extra:
            extra = extra + ["--no-calibrate"]
        for seed in seed_list:
            run_dir = base / name / f"seed_{seed}"
            if skip_existing and (run_dir / "metrics.csv").exists():
                summary.append((seed, name, "skipped (exists)", 0.0))
                continue
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / "run.log"

            click.echo(f"[seed={seed}, {name}] training...", nl=False)
            train_cmd, model_pkl = _train_cmd(variant, attributes, schedules, models_dir, seed, extra,
                                              mdcev_max_records)
            rc, t_train = _run_cmd(train_cmd, log)
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            click.echo(f" {status} ({t_train:.0f}s)")
            if rc != 0:
                summary.append((seed, name, f"train={status}", t_train))
                continue
            if (models_dir / "validation.json").exists():
                shutil.copy(models_dir / "validation.json", run_dir / "validation.json")

            click.echo(f"[seed={seed}, {name}] generating...", nl=False)
            rc2, t_gen = _run_cmd(_generate_cmd(variant, attributes, model_pkl, run_dir, seed), log, append=True)
            gen_status = "OK" if rc2 == 0 else f"FAIL(rc={rc2})"
            click.echo(f" {gen_status} ({t_gen:.0f}s)")
            status_str = f"train={status} gen={gen_status}"

            if rc2 == 0 and evaluate:
                click.echo(f"[seed={seed}, {name}] evaluating...", nl=False)
                t0 = time.time()
                try:
                    _evaluate(run_dir, target, variant, ablation, seed, attributes, split_order)
                    ev = "OK"
                except Exception as exc:  # noqa: BLE001 — keep going with other runs
                    ev = f"FAIL({exc})"
                click.echo(f" {ev} ({time.time() - t0:.0f}s)")
                status_str += f" eval={ev}"
            summary.append((seed, name, status_str, t_train + t_gen))

    click.echo("\n=== Stability run summary ===")
    for seed, name, status, elapsed in summary:
        click.echo(f"  seed={seed:2d}  {name:<34}  {status}  ({elapsed:.0f}s total)")
    click.echo(f"\nSummarise with: python scripts/summarise_runs.py --runs-dir {out_dir}")


if __name__ == "__main__":
    main()
