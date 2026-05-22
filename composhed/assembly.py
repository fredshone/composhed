"""Step 6 — Rule-based 24-hour schedule assembly."""

import numpy as np

_MDCEV_DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]


def assemble_schedule(
    dap: str,
    mandatory_duration: float,
    mandatory_type: str,
    disc_activities: list[tuple[str, int]],
    work_start: float | None,
    first_departure: float | None,
    before_work_flags: list[bool],
) -> list[dict]:
    """Assemble a valid 24-hour schedule.

    Returns list of dicts: [{act, start, end, duration}, ...]
    Sum of durations == 1440, starts and ends with 'home'.
    """
    md = int(round(mandatory_duration))
    disc = [(a, int(round(d))) for a, d in disc_activities]

    # ---- H: home all day ------------------------------------------------
    if dap == "H":
        return _to_rows([("home", 1440)])

    # ---- Budget enforcement: ensure ≥60 min home ------------------------
    disc = _enforce_budget(md if dap in ("W", "WD", "E", "ED") else 0, disc)

    # ---- W/E: home → mandatory → home -----------------------------------
    if dap in ("W", "E"):
        ws = int(np.clip(round(work_start or 480), 0, 1440 - md - 60))
        home_eve = 1440 - ws - md
        if home_eve < 30:
            ws = max(0, 1440 - md - 30)
            home_eve = 1440 - ws - md
        seq = [("home", ws), (mandatory_type, md), ("home", home_eve)]
        return _to_rows(seq)

    # ---- WD/ED: home → pre-mandatory → mandatory → post-mandatory → home
    if dap in ("WD", "ED"):
        ws = int(np.clip(round(work_start or 480), 30, 1440 - md - 30))
        pre, post = _split_by_flags(disc, before_work_flags)

        # Ensure pre-work activities fit before work_start
        pre_total = sum(d for _, d in pre)
        avail_pre = ws - 30  # at least 30 min home_morning
        if pre_total > avail_pre:
            if avail_pre > 0:
                pre = _scale_activities(pre, avail_pre)
            else:
                post = pre + post
                pre = []
            pre_total = sum(d for _, d in pre)

        # Ensure post-work activities fit after work_end
        work_end = ws + md
        avail_post = 1440 - work_end - 30  # at least 30 min home_evening
        post_total = sum(d for _, d in post)
        if post_total > avail_post:
            if avail_post > 0:
                post = _scale_activities(post, avail_post)
            else:
                post = []
            post_total = sum(d for _, d in post)

        home_morn = ws - pre_total
        home_eve = 1440 - work_end - post_total

        seq = (
            [("home", home_morn)]
            + pre
            + [(mandatory_type, md)]
            + post
            + [("home", home_eve)]
        )
        return _to_rows(seq)

    # ---- D: home → disc → home ------------------------------------------
    if dap == "D":
        if not disc:
            return _to_rows([("home", 1440)])
        fd = int(np.clip(round(first_departure or 480), 0, 1380))
        disc_total = sum(d for _, d in disc)
        home_eve = 1440 - fd - disc_total
        if home_eve < 30:
            avail = 1440 - fd - 30
            if avail > 0 and disc_total > 0:
                disc = _scale_activities(disc, avail)
                disc_total = sum(d for _, d in disc)
            home_eve = 1440 - fd - disc_total
        seq = [("home", fd)] + disc + [("home", home_eve)]
        return _to_rows(seq)

    raise ValueError(f"Unknown DAP: {dap}")


def assemble_mdcev_schedule(
    alloc: dict[str, float],
    dap: str,
    mandatory_type: str,
    x_label: np.ndarray,
    employment: str,
    anchor_model,
    episode_model,
) -> list[dict]:
    """Assemble a 1440-min schedule from MDCEV allocations without rescaling.

    Durations for non-home activities are taken verbatim from alloc (rounded to
    integers). Home time fills the remainder, split around the anchor start time.
    Sum is guaranteed to equal 1440.
    """
    if dap == "H":
        return _to_rows([("home", 1440)])

    # Round non-home durations; home gets the exact remainder so sum == 1440
    mandatory = int(round(alloc.get(mandatory_type, 0.0)))
    disc_durs: dict[str, int] = {
        atype: int(round(alloc.get(atype, 0.0)))
        for atype in _MDCEV_DISC_TYPES
        if alloc.get(atype, 0.0) > 1.0
    }
    home_total = 1440 - mandatory - sum(disc_durs.values())
    home_total = max(0, home_total)

    # Build flat episode list for disc activities
    disc_episodes: list[tuple[str, int]] = []
    for atype, dur in disc_durs.items():
        disc_episodes.extend(_split_duration(atype, dur, episode_model))

    # ---- W / E: home → mandatory → home ------------------------------------
    if dap in ("W", "E"):
        ws = int(np.clip(
            round(anchor_model.sample_work_start(employment)),
            0, max(0, home_total - 30),
        ))
        seq = [("home", ws), (mandatory_type, mandatory), ("home", home_total - ws)]
        return _to_rows(seq)

    # ---- WD / ED: home → pre → mandatory → post → home --------------------
    if dap in ("WD", "ED"):
        # Sample work start conservatively, then refine given pre-work placement
        ws_raw = int(np.clip(
            round(anchor_model.sample_work_start(employment)),
            30, max(30, home_total - 30),
        ))
        before_flags = anchor_model.sample_before_work_flags(
            x_label, disc_episodes, float(ws_raw)
        )
        pre, post = _split_by_flags(disc_episodes, before_flags)
        pre_total = sum(d for _, d in pre)

        # Re-clip ws to ensure home_morning ≥ 30 and home_evening ≥ 30
        ws_lo = pre_total + 30
        ws_hi = home_total + pre_total - 30
        if ws_lo > ws_hi:
            ws = pre_total + max(0, home_total) // 2
        else:
            ws = int(np.clip(ws_raw, ws_lo, ws_hi))

        # If pre no longer fits, move entirely to post
        if ws - pre_total < 30:
            post = pre + post
            pre = []
            pre_total = 0

        home_morn = ws - pre_total
        home_eve = home_total - home_morn
        seq = (
            [("home", home_morn)]
            + pre
            + [(mandatory_type, mandatory)]
            + post
            + [("home", home_eve)]
        )
        return _to_rows(seq)

    # ---- D: home → disc episodes → home ------------------------------------
    if dap == "D":
        fd = int(np.clip(
            round(anchor_model.sample_first_departure(employment)),
            0, max(0, home_total - 30),
        ))
        seq = [("home", fd)] + disc_episodes + [("home", home_total - fd)]
        return _to_rows(seq)

    raise ValueError(f"Unknown DAP for MDCEV assembly: {dap}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_duration(
    atype: str, total: int, episode_model
) -> list[tuple[str, int]]:
    """Split total minutes into n episodes; last episode absorbs rounding."""
    n = episode_model.sample(atype, float(total))
    per = total // n
    last = total - per * (n - 1)
    return [(atype, per)] * (n - 1) + [(atype, last)]


def _enforce_budget(
    mandatory_dur: int, disc: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """Proportionally rescale disc activities if total non-home time > 1380."""
    total_nonhome = mandatory_dur + sum(d for _, d in disc)
    if total_nonhome > 1380 and total_nonhome > 0:
        disc_total = sum(d for _, d in disc)
        allowed_disc = max(0, 1380 - mandatory_dur)
        if disc_total > 0:
            disc = _scale_activities(disc, allowed_disc)
    return disc


def _scale_activities(
    activities: list[tuple[str, int]], budget: int
) -> list[tuple[str, int]]:
    """Proportionally scale activity durations to fit within budget."""
    total = sum(d for _, d in activities)
    if total <= 0:
        return activities
    scale = budget / total
    scaled = [(a, max(10, int(d * scale))) for a, d in activities]
    # Fix rounding: adjust last item
    diff = budget - sum(d for _, d in scaled)
    if scaled:
        last_a, last_d = scaled[-1]
        scaled[-1] = (last_a, max(10, last_d + diff))
    return scaled


def _split_by_flags(
    disc: list[tuple[str, int]], flags: list[bool]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split disc activities into pre-work and post-work by flags."""
    if len(flags) < len(disc):
        flags = flags + [False] * (len(disc) - len(flags))
    pre = [(a, d) for (a, d), f in zip(disc, flags) if f]
    post = [(a, d) for (a, d), f in zip(disc, flags) if not f]
    return pre, post


def _to_rows(seq: list[tuple[str, int]]) -> list[dict]:
    """Convert (act, dur) sequence to row dicts, fix rounding, assert 1440."""
    # Remove zero-duration activities
    seq = [(a, d) for a, d in seq if d > 0]

    # Merge consecutive same-type activities (e.g. two "visit" slots → one)
    merged: list[tuple[str, int]] = []
    for act, dur in seq:
        if merged and merged[-1][0] == act:
            merged[-1] = (act, merged[-1][1] + dur)
        else:
            merged.append((act, dur))
    seq = merged

    # Fix rounding so sum == 1440
    total = sum(d for _, d in seq)
    if total != 1440:
        # Find last home and adjust
        for i in reversed(range(len(seq))):
            if seq[i][0] == "home":
                new_dur = seq[i][1] + (1440 - total)
                if new_dur >= 0:
                    seq[i] = ("home", new_dur)
                    break
        else:
            # Fallback: adjust last activity
            seq[-1] = (seq[-1][0], seq[-1][1] + (1440 - total))

    # Remove any newly zeroed entries
    seq = [(a, d) for a, d in seq if d > 0]

    # Build rows
    rows = []
    cursor = 0
    for act, dur in seq:
        rows.append({"act": act, "start": cursor, "end": cursor + dur, "duration": dur})
        cursor += dur

    return rows
