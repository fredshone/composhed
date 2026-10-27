"""SR-18: optional pid-level holdout split (component validation only, see PLAN §0.0)."""

import numpy as np


def holdout_split(pids: list, fraction: float, seed: int | None) -> tuple[set, set]:
    """Split pids into (train, holdout). fraction=0 puts every pid in train."""
    pids = sorted(set(pids), key=str)
    if fraction <= 0:
        return set(pids), set()
    rng = np.random.default_rng(seed)
    n_hold = int(round(len(pids) * fraction))
    idx = rng.permutation(len(pids))[:n_hold]
    holdout = {pids[i] for i in idx}
    return set(pids) - holdout, holdout
