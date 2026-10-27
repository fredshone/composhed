"""SR-15/16/17: one shared calibration routine for choice constants and duration locations.

Calibration follows ActivitySim practice: simulate the full model chain on (a
sample of) training persons, compare simulated outcomes with targets computed
from the training data, and adjust constants by a damped log-ratio step

    offset_k += clip(lr · (log target_k − log simulated_k), ±max_step)

For choice models the offset is an alternative-specific constant (ASC); for
duration models it is the location (log-scale intercept) of the AFT / log-normal
model, so ``log(target mean / simulated mean)`` is the scale adjustment. Iteration
stops when every target is within tolerance or after ``max_iter`` iterations.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Target:
    """One calibration target group.

    ``values`` maps outcome keys to target shares (``kind="share"``) or a single
    ``{"mean": value}`` (``kind="mean"``). ``get`` / ``set`` read and write the
    model offsets as a dict keyed like ``values``. ``lr_scale`` damps targets whose
    constants act on the same choice (e.g. start, end and duration constants of
    one time-of-day model), so their joint step is not over-corrected.
    """

    name: str
    values: dict
    kind: str
    get: Callable[[], dict]
    set: Callable[[dict], None]
    lr_scale: float = 1.0


@dataclass
class TargetReport:
    name: str
    kind: str
    target: dict
    uncalibrated: dict
    calibrated: dict = field(default_factory=dict)
    max_error: float = float("nan")
    tolerance: float = float("nan")
    within_tolerance: bool = False
    history: list = field(default_factory=list)


@dataclass
class CalibrationReport:
    iterations: int = 0
    best_iteration: int = 0
    converged: bool = False
    share_tol: float = 0.0
    mean_rel_tol: float = 0.0
    targets: dict[str, TargetReport] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "best_iteration": self.best_iteration,
            "converged": self.converged,
            "share_tol": self.share_tol,
            "mean_rel_tol": self.mean_rel_tol,
            "targets": {k: v.__dict__ for k, v in self.targets.items()},
        }

    def print(self) -> None:
        print(f"  calibration: iterations={self.iterations} best={self.best_iteration} converged={self.converged} "
              f"(share tol {self.share_tol}, mean rel tol {self.mean_rel_tol})")
        for name, r in self.targets.items():
            flag = "ok " if r.within_tolerance else "OUT"
            tol_s = f" (tol {r.tolerance:.3f})" if r.kind == "share" else ""
            if r.kind == "mean":
                print(f"    [{flag}] {name:<34} target {r.target['mean']:8.1f}  "
                      f"uncal {r.uncalibrated.get('mean', float('nan')):8.1f}  cal {r.calibrated.get('mean', float('nan')):8.1f}")
            else:
                print(f"    [{flag}] {name:<34} max |share error| uncal "
                      f"{_err(r.target, r.uncalibrated, 'share'):.4f} → cal {r.max_error:.4f}{tol_s}")


def _err(target: dict, sim: dict, kind: str) -> float:
    if kind == "mean":
        t, s = target["mean"], sim.get("mean", 0.0)
        return abs(s - t) / max(abs(t), 1e-9)
    return max((abs(sim.get(k, 0.0) - v) for k, v in target.items() if not str(k).startswith("__")), default=0.0)


def _tol(target: dict, sim: dict, kind: str, share_tol: float, mean_rel_tol: float) -> float:
    """Share tolerance is max(share_tol, 2 standard errors of the simulated share).

    Small segments (e.g. a person type with ~100 simulated persons) cannot be
    matched to 0.01 by any constant because of simulation noise; ``share_dict``
    records the simulated count under ``"__n__"``.
    """
    if kind == "mean":
        return mean_rel_tol
    n = sim.get("__n__")
    if not n:
        return share_tol
    p = max((v for k, v in target.items() if not str(k).startswith("__")), default=0.5)
    return max(share_tol, 2.0 * float(np.sqrt(p * (1 - p) / n)))


def calibrate(
    targets: list[Target],
    simulate: Callable[[], dict[str, dict]],
    max_iter: int = 8,
    lr: float = 0.8,
    max_step: float = 1.0,
    share_tol: float = 0.01,
    mean_rel_tol: float = 0.02,
) -> CalibrationReport:
    """Iteratively adjust offsets until simulated outcomes match targets.

    ``simulate()`` runs the model chain with the current offsets and returns
    ``{target name: {key: simulated value}}``.
    """
    report = CalibrationReport(share_tol=share_tol, mean_rel_tol=mean_rel_tol)
    sim = simulate()
    for t in targets:
        report.targets[t.name] = TargetReport(t.name, t.kind, dict(t.values), dict(sim.get(t.name, {})))

    def record(sim_: dict) -> None:
        for t in targets:
            report.targets[t.name].history.append(round(_err(t.values, sim_.get(t.name, {}), t.kind), 5))

    record(sim)

    def tol(t: Target, s: dict) -> float:
        return _tol(t.values, s, t.kind, share_tol, mean_rel_tol)

    def within(t: Target, s: dict) -> bool:
        return _err(t.values, s, t.kind) <= tol(t, s)

    def score(s_: dict) -> float:
        return float(np.mean([_err(t.values, s_.get(t.name, {}), t.kind) / tol(t, s_.get(t.name, {}))
                              for t in targets]))

    # Keep the best iterate: with simulation noise the last iterate is not always the best.
    best = (score(sim), [t.get() for t in targets], sim, 0)
    for it in range(1, max_iter + 1):
        if all(within(t, sim.get(t.name, {})) for t in targets):
            break
        for t in targets:
            s = sim.get(t.name, {})
            if within(t, s):
                continue
            off = t.get()
            for k, target_v in t.values.items():
                if str(k).startswith("__") or k not in off:
                    continue
                sim_v = s.get(k, 0.0)
                if target_v <= 0:
                    step = -max_step if sim_v > 0 else 0.0
                elif sim_v <= 0:
                    step = max_step
                else:
                    step = float(np.clip(lr * (np.log(target_v) - np.log(sim_v)), -max_step, max_step))
                off[k] = off[k] + step * t.lr_scale
            t.set(off)
        sim = simulate()
        record(sim)
        report.iterations = it
        sc = score(sim)
        if sc < best[0]:
            best = (sc, [t.get() for t in targets], sim, it)

    _, offsets, sim, report.best_iteration = best
    for t, off in zip(targets, offsets):
        t.set(off)
    report.converged = all(within(t, sim.get(t.name, {})) for t in targets)
    for t in targets:
        r = report.targets[t.name]
        r.calibrated = dict(sim.get(t.name, {}))
        r.max_error = _err(t.values, r.calibrated, t.kind)
        r.tolerance = tol(t, r.calibrated)
        r.within_tolerance = within(t, r.calibrated)
    return report


def share_dict(values, keys) -> dict:
    """Shares of each key among ``values`` (keys absent from values get 0), plus the count as ``"__n__"``."""
    values = list(values)
    n = max(len(values), 1)
    out = {k: 0.0 for k in keys}
    for v in values:
        if v in out:
            out[v] += 1.0 / n
    out["__n__"] = len(values)
    return out


def offset_accessors(model, keys: list | None = None, attr: str = "offset"):
    """get/set closures exposing ``model.offset`` (indexed by model.classes_) as a dict."""
    classes = list(keys if keys is not None else model.classes_)

    def get() -> dict:
        arr = getattr(model, attr)
        return {c: float(arr[i]) for i, c in enumerate(classes)}

    def set_(d: dict) -> None:
        arr = getattr(model, attr)
        for i, c in enumerate(classes):
            arr[i] = d[c]

    return get, set_


def location_accessors(model):
    """get/set closures for a duration model's location offset (kind="mean")."""

    def get() -> dict:
        return {"mean": float(model.loc_offset)}

    def set_(d: dict) -> None:
        model.loc_offset = d["mean"]

    return get, set_
