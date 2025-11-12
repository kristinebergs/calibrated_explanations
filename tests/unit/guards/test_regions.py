import numpy as np
from calibrated_explanations.guards import ConformalRegionOracle
from calibrated_explanations.guards.intervals import union_intervals


def test_intervals_clip_to_global_bounds():
    # Create synthetic data with known global bounds [0, 1] for both features
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.9, 0.1],
    ])
    y = np.zeros(len(X), dtype=int)

    # Use a single cluster so the computed interval around the center can be large
    guard = ConformalRegionOracle(alpha=0.5, n_clusters=1)
    guard.fit(X, y)

    # Choose an instance near the upper edge so unclipped high would exceed 1.0
    x_inst = np.array([0.95, 0.95])
    ctx = 0
    intervals = guard.intervals(x_inst, ctx)

    # Intervals are lists of (rel_low, rel_high) tuples per feature
    assert len(intervals) == 2

    for j, feats in enumerate(intervals):
        for rel_low, rel_high in feats:
            # After clipping, the absolute bounds must lie within [0, 1]
            abs_low = x_inst[j] + rel_low
            abs_high = x_inst[j] + rel_high
            assert abs_low >= 0 - 1e-12
            assert abs_high <= 1 + 1e-12


def test_union_intervals_overlapping():
    # Overlapping and touching intervals should be merged
    intervals = [(0.0, 0.5), (0.4, 1.0), (1.0, 1.5), (2.0, 2.5)]
    merged = union_intervals(intervals)
    assert merged == [(0.0, 1.5), (2.0, 2.5)]


def test_union_intervals_empty():
    assert union_intervals([]) == []
