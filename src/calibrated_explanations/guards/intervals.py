"""Interval utilities for perturbation guards."""

from typing import List, Tuple


def union_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Union a list of intervals."""
    if not intervals:
        return []

    # Sort by start
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def in_intervals(value: float, intervals: List[Tuple[float, float]]) -> bool:
    """Check if value is in any of the intervals."""
    return any(low <= value <= high for low, high in intervals)
