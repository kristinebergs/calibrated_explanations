"""Guard plugin protocol and registry.

Guards are explanation-related plugins that filter perturbations and feature
candidates during explanation generation. They implement the guard contract
and are discovered/resolved through the plugin registry.

Aligns with ADR-006 (plugin trust model) and ADR-001 (package boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
import numpy as np

from .types import PluginMeta


@dataclass(frozen=True)
class GuardContext:
    """Immutable context for guard plugins.

    Provides read-only access to explainer configuration and calibration
    state needed for perturbation filtering decisions.
    """

    task: str  # "classification" | "regression"
    mode: str  # "factual" | "alternative" | "fast"
    learner: Any  # read-only ref to fitted learner
    x_cal: np.ndarray  # calibration features (read-only)
    y_cal: np.ndarray  # calibration targets (read-only)
    interval_learner: Any  # fitted interval calibrator (read-only)
    feature_names: Sequence[str]
    categorical_features: Sequence[int]
    num_features: int
    metadata: Mapping[str, Any]  # arbitrary plugin metadata


class GuardPlugin(Protocol):
    """Protocol for guard plugins.

    Guards filter perturbations and candidates during explanation generation
    to enforce domain constraints or conformal safety properties.

    A guard plugin must be initialized once per explainer instance and can
    then be called multiple times during explanation generation.

    Attributes
    ----------
    plugin_meta : PluginMeta
        Metadata describing the plugin's capabilities and requirements.
    """

    plugin_meta: PluginMeta

    def supports_mode(self, mode: str, *, task: str) -> bool:
        """Check if this guard supports the given explanation mode and task.

        Parameters
        ----------
        mode : str
            Explanation mode ("factual", "alternative", "fast", etc.)
        task : str
            Explainer task ("classification" or "regression")

        Returns
        -------
        bool
            True if the guard can operate in this mode/task combination.

        Examples
        --------
        >>> plugin.supports_mode("factual", task="classification")
        True
        >>> plugin.supports_mode("fast", task="regression")
        False
        """
        ...

    def initialize(self, context: GuardContext) -> None:
        """Initialize the guard with explainer context.

        Called once per CalibratedExplainer instance after calibration setup
        is complete. The plugin should use the context to fit any internal
        state (e.g., conformal regions, bounds, etc.).

        The context is frozen and read-only; plugins must not mutate it.

        Parameters
        ----------
        context : GuardContext
            Immutable context with calibration data and metadata.

        Raises
        ------
        ConfigurationError
            If initialization fails (e.g., cannot fit the guard with provided data).

        Examples
        --------
        >>> context = GuardContext(...)
        >>> plugin.initialize(context)
        """
        ...

    def filter_perturbations(
        self,
        perturbed_x: np.ndarray,
        perturbed_feature: np.ndarray,
        x_orig: np.ndarray,
        prediction: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter perturbed instances based on guard logic.

        Called during explanation generation to filter out perturbations
        that violate guard constraints. Typically invoked by explanation
        plugins after perturbation generation but before weight computation.

        Parameters
        ----------
        perturbed_x : ndarray of shape (n_perturbed, n_features)
            Perturbed instance matrix.
        perturbed_feature : ndarray of shape (n_perturbed, ...)
            Metadata about perturbed features (e.g., which feature was perturbed,
            which instance it originated from).
        x_orig : ndarray of shape (n_instances, n_features)
            Original instances being explained.
        prediction : Mapping[str, Any]
            Calibrated predictions for x_orig containing keys:
            - "predict": ndarray of shape (n_instances,) – point estimates
            - "low": ndarray of shape (n_instances,) – lower interval bounds
            - "high": ndarray of shape (n_instances,) – upper interval bounds

        Returns
        -------
        perturbed_x_filtered : ndarray
            Filtered perturbed instances (subset of perturbed_x).
        perturbed_feature_filtered : ndarray
            Corresponding metadata rows (subset of perturbed_feature).

        Notes
        -----
        The guard should return indices/mask of accepted perturbations.
        If no perturbations are rejected, return (perturbed_x, perturbed_feature).

        Examples
        --------
        >>> perturbed_x_filtered, perturbed_feature_filtered = plugin.filter_perturbations(
        ...     perturbed_x, perturbed_feature, x_orig, prediction
        ... )
        """
        ...

    def filter_candidates(
        self,
        feature_index: int,
        candidates: np.ndarray,
        x_orig: np.ndarray,
        calibrated_pred: Optional[Tuple[float, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Filter candidate feature values based on guard logic.

        Called during candidate enumeration to restrict the search space
        to values that the guard deems acceptable for the given feature
        in the context of the original instance.

        Parameters
        ----------
        feature_index : int
            Index of the feature being filtered.
        candidates : ndarray of shape (n_candidates,)
            Candidate values for the feature.
        x_orig : ndarray of shape (1, n_features) or (n_instances, n_features)
            Original instance(s) being explained.
        calibrated_pred : Tuple[float, Tuple[float, float]], optional
            Calibrated prediction for x_orig: (predict, (low, high)).

        Returns
        -------
        filtered_candidates : ndarray
            Subset of candidates that satisfy guard constraints.

        Examples
        --------
        >>> filtered = plugin.filter_candidates(feature_index, candidates, x_orig)
        """
        ...

    def accept_batch(
        self,
        x_batch: np.ndarray,
        calibrated_predictions: Optional[Sequence[Optional[Tuple]]] = None,
    ) -> np.ndarray:
        """Batch acceptance check.

        Determines which instances in a batch are accepted by the guard.

        Parameters
        ----------
        x_batch : ndarray of shape (n_samples, n_features)
            Batch of instances to check.
        calibrated_predictions : Sequence of Tuples, optional
            Per-instance calibrated predictions: [(predict, (low, high)), ...].
            May be None if predictions are not available.

        Returns
        -------
        mask : ndarray of shape (n_samples,), dtype=bool
            True for accepted instances, False for rejected.

        Examples
        --------
        >>> mask = plugin.accept_batch(x_batch, calibrated_predictions)
        >>> accepted_instances = x_batch[mask]
        """
        ...


__all__ = ["GuardPlugin", "GuardContext"]