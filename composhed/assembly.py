"""Step 6 — Rule-based 24-hour schedule assembly."""

import numpy as np


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
    disc = _enforce_budget(md if dap in ("W", "WD") else 0, disc)

    # ---- W: home → work → home ------------------------------------------
    if dap == "W":
        ws = int(np.clip(round(work_start or 480), 0, 1440 - md - 60))
        home_eve = 1440 - ws - md
        if home_eve < 30:
            ws = max(0, 1440 - md - 30)
            home_eve = 1440 - ws - md
        seq = [("home", ws), (mandatory_type, md), ("home", home_eve)]
        return _to_rows(seq)

    # ---- WD: home → pre-work → work → post-work → home -----------------
    if dap == "WD":
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
