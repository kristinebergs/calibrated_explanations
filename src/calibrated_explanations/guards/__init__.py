"""Guards submodule for calibrated explanations."""

from typing import Protocol

from .regions import ConformalRegionOracle


class BaseGuard(Protocol):
    """Protocol for perturbation guards."""

    def fit(self, xs, ys):
        """Fit the guard on training data."""
        raise NotImplementedError()

    def label_context(self, x_instance, **kwargs):
        """Get label context for instance x."""
        raise NotImplementedError()

    def intervals(self, x_instance, label_ctx):
        """Get admissible intervals for each feature."""
        raise NotImplementedError()

    def accept(self, x_prime, label_ctx):
        """Check if perturbed instance is acceptable."""
        raise NotImplementedError()


__all__ = ["BaseGuard", "ConformalRegionOracle"]
