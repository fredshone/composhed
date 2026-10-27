"""Validation (SR-19) and generation (I7, §6.4, EV-3) reporting."""

import json
import math
from dataclasses import asdict, dataclass, field


def _clean(obj):
    """Make a metrics structure JSON-safe (NaN → None, numpy scalars → Python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2, sort_keys=False)


def print_validation(step: str, metrics: dict, indent: str = "  ") -> None:
    """Print one component's validation summary in the style of train._print_validation."""
    n = metrics.get("n", "")
    n_str = f"n={n:,}" if isinstance(n, int) else ""
    print(f"{indent}{step:<40}{n_str}")
    parts = []
    for key, label, fmt in (
        ("accuracy", "accuracy", "{:.3f}"),
        ("prsquared", "pseudo-R²", "{:.3f}"),
        ("log_r2", "log-R²", "{:.3f}"),
        ("mae", "MAE", "{:.2f}"),
        ("mae_minutes", "MAE", "{:.0f} min"),
        ("mean_actual", "mean obs", "{:.1f}"),
        ("mean_pred", "mean pred", "{:.1f}"),
        ("distr", "distr", "{}"),
    ):
        v = metrics.get(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        parts.append(f"{label} {fmt.format(v)}")
    if parts:
        print(f"{indent}  " + "   ".join(parts))
    shares = metrics.get("shares")
    if shares:
        row = "   ".join(f"{k}: {v['obs']:.3f}/{v['pred']:.3f}" for k, v in list(shares.items())[:8])
        print(f"{indent}  shares obs/pred  {row}")
    for sub, m in metrics.get("groups", {}).items():
        print_validation(str(sub), m, indent + "  ")


class ValidationReport:
    """Collects per-component validation metrics; written to validation.json."""

    def __init__(self) -> None:
        self.components: dict[str, dict] = {}
        self.holdout: dict[str, dict] = {}
        self.info: dict = {}

    def add(self, name: str, metrics: dict, holdout: dict | None = None) -> None:
        self.components[name] = metrics
        print_validation(name, metrics)
        if holdout is not None:
            self.holdout[name] = holdout
            print_validation(f"{name} [holdout]", holdout)

    def to_dict(self) -> dict:
        out = {"components": self.components, "info": self.info}
        if self.holdout:
            out["holdout"] = self.holdout
        return out

    def write(self, path: str) -> None:
        write_json(path, self.to_dict())


@dataclass
class GenerationReport:
    """Counts every non-standard event during generation (I7, §6.4, EV-3).

    ``persons_repaired`` counts persons with a fallback or at least one rescale, so
    feasibility can be reported net of repair (EV-3). Merges of consecutive
    identical activities in finalisation are counted separately (``merges``,
    ``persons_merged``).
    """

    variant: str = ""
    n_persons: int = 0
    fallbacks: int = 0
    resamples: int = 0
    rescales: int = 0
    persons_rescaled: int = 0
    tours_dropped: int = 0
    stops_dropped: int = 0
    merges: int = 0
    persons_merged: int = 0
    persons_repaired: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    extra: dict[str, int] = field(default_factory=dict)

    def count(self, key: str, n: int = 1) -> None:
        if hasattr(self, key) and isinstance(getattr(self, key), int):
            setattr(self, key, getattr(self, key) + n)
        else:
            self.extra[key] = self.extra.get(key, 0) + n

    def fallback(self, reason: str) -> None:
        self.fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def shares(self) -> dict[str, float]:
        n = max(self.n_persons, 1)
        return {
            "fallback_share": self.fallbacks / n,
            "rescale_share": self.persons_rescaled / n,
            "merge_share": self.persons_merged / n,
            "repair_share": self.persons_repaired / n,
        }

    def to_dict(self) -> dict:
        return {**asdict(self), **self.shares()}

    def write(self, path: str) -> None:
        write_json(path, self.to_dict())

    def print_summary(self) -> None:
        s = self.shares()
        print(
            f"Generation report [{self.variant}]: persons={self.n_persons:,}  "
            f"fallbacks={self.fallbacks} ({s['fallback_share']:.2%})  resamples={self.resamples}  "
            f"rescales={self.rescales} (persons {s['rescale_share']:.2%})  tours_dropped={self.tours_dropped}  "
            f"stops_dropped={self.stops_dropped}  merges={self.merges}  repaired={s['repair_share']:.2%}"
        )
        if self.fallback_reasons:
            print(f"  fallback reasons: {self.fallback_reasons}")
        if self.extra:
            print(f"  other events: {self.extra}")
