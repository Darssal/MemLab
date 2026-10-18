from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def mean_std_ci95(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * (std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci95)}
