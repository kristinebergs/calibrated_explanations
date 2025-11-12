"""Interval utilities for guards.

Provides a small helper to union/merge overlapping 1D intervals.
"""

from typing import List, Tuple


def union_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge a list of (low, high) intervals into a minimal list of disjoint intervals.

    Intervals that touch (e.g., [0,1] and [1,2]) are considered mergeable.
    Returns an empty list if input is empty.
    """
    if not intervals:
        return []

    # Sort by low endpoint
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_low, cur_high = sorted_intervals[0]

    for low, high in sorted_intervals[1:]:
        if low <= cur_high + 1e-12:
            # Overlapping or touching — extend current
            cur_high = max(cur_high, high)
        else:
            merged.append((cur_low, cur_high))
            cur_low, cur_high = low, high

    merged.append((cur_low, cur_high))
    return merged


def in_intervals(value: float, intervals: List[Tuple[float, float]]) -> bool:
    """Check if value is in any of the intervals."""
    return any(low <= value <= high for low, high in intervals)
