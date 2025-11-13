"""Guards submodule for calibrated explanations.

This module provides the ConformalRegionOracle for filtering out-of-distribution
perturbations during explanation generation. The oracle uses conformal prediction
with confidence modulation to provide finite-sample coverage guarantees.
"""

from .regions import ConformalRegionOracle

__all__ = ["ConformalRegionOracle"]
