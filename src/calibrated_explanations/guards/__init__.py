"""Backward compatibility module for guards.

This module provides imports from the new location for code that uses the old
calibrated_explanations.guards location.

All guard implementations have been moved to:
    calibrated_explanations.core.explain.guards

This module is deprecated and should not be used in new code.
"""

import warnings

from calibrated_explanations.core.explain.guards.interval_learner_adapter import (
    IntervalLearnerAdapter,
)
from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle

# Issue deprecation warning
warnings.warn(
    "importing from calibrated_explanations.guards is deprecated. "
    "Use calibrated_explanations.core.explain.guards instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConformalRegionOracle", "IntervalLearnerAdapter"]
