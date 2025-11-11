"""Guards submodule for calibrated explanations."""

from typing import Protocol, Any, Optional

from .regions import ConformalRegionOracle


class BaseGuard(Protocol):
    """Protocol for perturbation guards."""

    def fit(self, X, y):
        """Fit the guard on training data."""
        ...

    def label_context(self, x, **kwargs):
        """Get label context for instance x."""
        ...

    def intervals(self, x, label_ctx):
        """Get admissible intervals for each feature."""
        ...

    def accept(self, x_prime, label_ctx):
        """Check if perturbed instance is acceptable."""
        ...


__all__ = ["BaseGuard", "ConformalRegionOracle"]
