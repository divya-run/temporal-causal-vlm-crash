"""
Metrics for both the diagnostic phase (reproduce CrashSight's failure) and
the main frame-budget x sampling-strategy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    clip_id: str
    tier: str
    predicted_letter: str | None
    correct_letter: str
    latency_ms: float
    peak_memory_mb: float | None = None
    # filled in when running the temporal-order probe:
    shuffled_predicted_letter: str | None = None
    reversed_predicted_letter: str | None = None


def accuracy(results: list[RunResult], tier: str | None = None) -> float:
    subset = [r for r in results if tier is None or r.tier == tier]
    if not subset:
        return float("nan")
    correct = sum(1 for r in subset if r.predicted_letter == r.correct_letter)
    return correct / len(subset)


def temporal_order_sensitivity(results: list[RunResult]) -> dict:
    """
    RiskCueBench-style probe: how much does the answer change when frame
    order is shuffled or reversed? A LOW delta before intervention indicates
    pattern-matching on salient frames rather than genuine temporal
    reasoning; success after intervention means this delta goes UP.

    Returns fraction of items whose predicted answer changed under each
    perturbation, plus accuracy under each condition for reference.
    """
    n = len(results)
    if n == 0:
        return {}

    shuffled_flip_rate = sum(
        1 for r in results
        if r.shuffled_predicted_letter is not None
        and r.shuffled_predicted_letter != r.predicted_letter
    ) / n

    reversed_flip_rate = sum(
        1 for r in results
        if r.reversed_predicted_letter is not None
        and r.reversed_predicted_letter != r.predicted_letter
    ) / n

    return {
        "n_items": n,
        "shuffled_flip_rate": shuffled_flip_rate,
        "reversed_flip_rate": reversed_flip_rate,
        "original_accuracy": accuracy(results),
        "shuffled_accuracy": sum(
            1 for r in results if r.shuffled_predicted_letter == r.correct_letter
        ) / n,
        "reversed_accuracy": sum(
            1 for r in results if r.reversed_predicted_letter == r.correct_letter
        ) / n,
    }


def latency_memory_summary(results: list[RunResult]) -> dict:
    latencies = [r.latency_ms for r in results]
    mems = [r.peak_memory_mb for r in results if r.peak_memory_mb is not None]
    return {
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else float("nan"),
        "fps": 1000.0 / (sum(latencies) / len(latencies)) if latencies else float("nan"),
        "mean_peak_memory_mb": sum(mems) / len(mems) if mems else None,
    }
