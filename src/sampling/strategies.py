"""
Frame sampling strategies — the core experimental variable of the revised
(Jetson-first) study. For a given clip and frame budget N, each strategy
returns a list of frame indices to feed to the VLM.

    uniform_sampling          baseline, no crash-awareness
    impact_centered_sampling  concentrate frames around the moment of impact
    phase_based_sampling      allocate frames across pre-crash / conflict /
                               collision / aftermath phases

All three are compared at multiple frame budgets (e.g. 4 / 8 / 16) against
accuracy, temporal-order sensitivity, latency, and peak memory on-device.
"""

from __future__ import annotations

import numpy as np


def uniform_sampling(total_frames: int, n: int) -> list[int]:
    if n >= total_frames:
        return list(range(total_frames))
    return list(np.linspace(0, total_frames - 1, num=n, dtype=int))


def impact_centered_sampling(total_frames: int, n: int, impact_frame: int) -> list[int]:
    """
    Concentrate samples around `impact_frame`, falling back to uniform
    coverage of the rest of the clip for any remaining budget.

    Simple version: draw from a triangular density centered on impact_frame,
    then snap to unique integer indices. Refine once we see how CrashSight's
    per-clip impact annotations (if any) are actually structured.
    """
    if n >= total_frames:
        return list(range(total_frames))

    half_window = max(1, total_frames // (n + 1))
    lo = max(0, impact_frame - half_window)
    hi = min(total_frames - 1, impact_frame + half_window)

    # Half the budget densely around impact, half spread across the full clip
    n_dense = max(1, n // 2)
    n_context = n - n_dense

    dense = np.linspace(lo, hi, num=n_dense, dtype=int)
    context = np.linspace(0, total_frames - 1, num=max(n_context, 1), dtype=int)

    idxs = sorted(set(dense.tolist()) | set(context.tolist()))
    return idxs[:n] if len(idxs) > n else idxs


def phase_based_sampling(
    total_frames: int,
    n: int,
    phase_bounds: dict[str, tuple[int, int]],
    phase_order: tuple[str, ...] = ("pre_crash", "conflict", "collision", "aftermath"),
) -> list[int]:
    """
    Allocate the frame budget across annotated phases, roughly proportional
    to phase duration but guaranteeing at least one frame per phase when
    budget allows. Requires `phase_bounds` from CrashSightDataset.

    phase_bounds: {"pre_crash": (start_idx, end_idx), ...}
    """
    phases = [p for p in phase_order if p in phase_bounds]
    if not phases:
        return uniform_sampling(total_frames, n)

    durations = {p: max(1, phase_bounds[p][1] - phase_bounds[p][0]) for p in phases}
    total_duration = sum(durations.values())

    # Proportional allocation with a floor of 1 frame/phase where budget allows
    alloc = {p: max(1, round(n * durations[p] / total_duration)) for p in phases}
    # Trim/pad to exactly match n
    diff = n - sum(alloc.values())
    for p in list(alloc)[: abs(diff)] if diff != 0 else []:
        alloc[p] += 1 if diff > 0 else -1

    idxs: list[int] = []
    for p in phases:
        start, end = phase_bounds[p]
        k = max(1, alloc.get(p, 1))
        idxs.extend(np.linspace(start, end, num=k, dtype=int).tolist())

    return sorted(set(idxs))[:n]


STRATEGIES = {
    "uniform": uniform_sampling,
    "impact_centered": impact_centered_sampling,
    "phase_based": phase_based_sampling,
}
