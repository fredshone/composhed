"""Generate schedules with the DaySim-aligned variant."""

import click

from composhed.common.bundle import load_bundle
from composhed.common.cli import (
    default_report_path,
    generate_options,
    make_rng,
    write_outputs,
)
from composhed.common.report import GenerationReport
from composhed.common.runner import generate_population
from composhed.daysim.chain import DaysimChain

VARIANT = "daysim"


def generate(attributes: str, models: str, out_attributes: str, out_schedules: str, seed: int | None = None,
             report_path: str | None = None) -> GenerationReport:
    rng = make_rng(seed)
    bundle = load_bundle(models, VARIANT)
    cfg = bundle["config"]
    chain = DaysimChain(bundle["models"]["daysim"], timing=cfg["timing"], consistency=cfg["consistency"],
                        max_resample=cfg["max_resample"])
    report = GenerationReport(variant=VARIANT)
    attr_rows, sched_rows = generate_population(chain, attributes, bundle["feature_names"], rng, report)
    write_outputs(attr_rows, sched_rows, out_attributes, out_schedules)
    report.print_summary()
    report.write(default_report_path(out_schedules, report_path))
    return report


@click.command()
@generate_options
def main(attributes, models, out_attributes, out_schedules, seed, report) -> None:
    generate(attributes, models, out_attributes, out_schedules, seed, report)


if __name__ == "__main__":
    main()
