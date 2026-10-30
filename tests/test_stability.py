"""Harness tests: run selection / ablation parsing and the summary script (EV-1, EV-2)."""

import json

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from run_stability import _parse_runs
from scripts.summarise_runs import main as summarise


def test_parse_runs():
    runs = _parse_runs("compositional,mdcev,cemdap", "cemdap:rescale,daysim:res15", True, False)
    assert runs == [("mdcev", ""), ("cemdap", ""), ("cemdap", "rescale"), ("daysim", "res15")]
    with pytest.raises(click.BadParameter):
        _parse_runs("cemdap", "cemdap:no-segmentation", False, False)
    with pytest.raises(click.BadParameter):
        _parse_runs("nope", "", False, False)


def test_summarise(tmp_path):
    for variant, ablation, seed, v in [("cemdap", "none", 0, 0.1), ("cemdap", "none", 1, 0.2),
                                       ("daysim", "res15", 0, 0.3)]:
        d = tmp_path / (variant if ablation == "none" else f"{variant}__{ablation}") / f"seed_{seed}"
        d.mkdir(parents=True)
        pd.DataFrame([dict(variant=variant, ablation=ablation, seed=seed, model=variant,
                           domain="participations", metric="domain_mean", value=v)]).to_csv(d / "metrics.csv",
                                                                                             index=False)
        (d / "generation_report.json").write_text(json.dumps({"fallback_share": 0.0, "repair_share": 0.01}))
    ext = tmp_path / "actvae.csv"
    pd.DataFrame([dict(variant="ActVAE", metric="participation", value=0.07)]).to_csv(ext, index=False)
    res = CliRunner().invoke(summarise, ["--runs-dir", str(tmp_path), "--external", str(ext)])
    assert res.exit_code == 0, res.output
    assert "0.1500 ± 0.0707" in res.output and "ActVAE" in res.output
    assert (tmp_path / "summary.csv").exists()
