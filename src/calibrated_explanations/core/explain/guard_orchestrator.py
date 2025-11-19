"""Guard orchestrator for managing guard plugin lifecycle.

This module provides GuardOrchestrator, which manages the lifecycle of a GuardPlugin
and provides a unified interface for perturbation and candidate filtering during
explanation generation.

Part of Phase 2: Move & Refactor GuardOrchestrator (ADR-001, ADR-006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ...plugins.guards import GuardContext, GuardPlugin

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


class GuardOrchestrator:
    """Orchestrator for guard plugin lifecycle and filtering operations.

    This class manages a GuardPlugin instance, handles initialization with
    GuardContext, and provides delegation methods for filtering operations.
    It acts as the bridge between the explanation pipeline and guard plugins.

    The orchestrator can operate with or without a guard plugin:
    - With plugin: Delegates filtering to the plugin
    - Without plugin: Passthrough behavior (no filtering)
    """

    def __init__(
        self,
        explainer: CalibratedExplainer,
        guard_plugin: Optional[GuardPlugin] = None,
    ) -> None:
        """Initialize the guard orchestrator.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer instance this orchestrator serves.
        guard_plugin : GuardPlugin, optional
            The guard plugin to use for filtering. If None, no filtering occurs.
        """
        self.explainer = explainer
        self._guard_plugin = guard_plugin
        self._initialized = False

    def set_plugin(self, plugin: Optional[GuardPlugin]) -> None:
        """Set or replace the guard plugin.

        Parameters
        ----------
        plugin : GuardPlugin or None
            The new guard plugin to use. If None, disables guarding.
        """
        self._guard_plugin = plugin
        self._initialized = False  # Require re-initialization

    def get_plugin(self) -> Optional[GuardPlugin]:
        """Get the current guard plugin.

        Returns
        -------
        GuardPlugin or None
            The current guard plugin, or None if no plugin is set.
        """
        return self._guard_plugin

    def initialize(self, context: GuardContext) -> None:
        """Initialize the guard plugin with the provided context.

        Parameters
        ----------
        context : GuardContext
            The frozen context containing explainer state for guard initialization.

        Raises
        ------
        RuntimeError
            If no guard plugin is set.
        """
        if self._guard_plugin is None:
            return  # No plugin configured, skip initialization
        
        self._guard_plugin.initialize(context)
        self._initialized = True

    def filter_perturbations(
        self,
        perturbed_x: np.ndarray,
        perturbed_feature: np.ndarray,
        x_orig: np.ndarray,
        prediction: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter perturbed instances using the guard plugin.

        If no guard plugin is set, returns the input arrays unchanged.

        Parameters
        ----------
        perturbed_x : np.ndarray
            The perturbed feature matrices.
        perturbed_feature : np.ndarray
            The perturbed feature indices/metadata.
        x_orig : np.ndarray
            The original instances being explained.
        prediction : Mapping[str, Any]
            The prediction results including calibrated intervals.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The filtered perturbed_x and perturbed_feature arrays.
        """
        if self._guard_plugin is None:
            return perturbed_x, perturbed_feature

        return self._guard_plugin.filter_perturbations(
            perturbed_x, perturbed_feature, x_orig, prediction
        )

    def filter_candidates(
        self,
        feature_index: int,
        candidates: np.ndarray,
        x_orig: np.ndarray,
        calibrated_pred: Optional[Tuple[float, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Filter candidate values for a feature using the guard plugin.

        If no guard plugin is set, returns the input candidates unchanged.

        Parameters
        ----------
        feature_index : int
            The index of the feature being filtered.
        candidates : np.ndarray
            The candidate values to filter.
        x_orig : np.ndarray
            The original instance.
        calibrated_pred : Tuple[float, Tuple[float, float]], optional
            The calibrated prediction for the instance.

        Returns
        -------
        np.ndarray
            The filtered candidate values.
        """
        if self._guard_plugin is None:
            return candidates

        return self._guard_plugin.filter_candidates(
            feature_index, candidates, x_orig, calibrated_pred
        )

    def accept_batch(
        self,
        x_batch: np.ndarray,
        calibrated_predictions: Optional[list[Optional[Tuple[float, Tuple[float, float]]]]] = None,
    ) -> np.ndarray:
        """Check acceptance for a batch of instances using the guard plugin.

        If no guard plugin is set, accepts all instances.

        Parameters
        ----------
        x_batch : np.ndarray
            The batch of instances to check.
        calibrated_predictions : list of tuples, optional
            The calibrated predictions for each instance.

        Returns
        -------
        np.ndarray
            Boolean mask indicating which instances are accepted.
        """
        if self._guard_plugin is None:
            return np.ones(len(x_batch), dtype=bool)

        return self._guard_plugin.accept_batch(x_batch, calibrated_predictions)

    def get_guard(self) -> Any:
        """Get the underlying guard instance from the plugin.

        Returns
        -------
        Any
            The guard instance (e.g., ConformalRegionOracle) or None.
        """
        if self._guard_plugin is None:
            return None
        
        if hasattr(self._guard_plugin, "get_guard"):
            return self._guard_plugin.get_guard()
        return None
        """Get filtering metrics from the guard plugin.

        Returns
        -------
        Dict[str, int]
            Dictionary of metric names to values. Empty dict if no plugin.
        """
        if self._guard_plugin is None:
            return {}

        # GuardPlugin doesn't have a get_metrics method in the protocol,
        # but our implementation does. This is a bit of a hack, but works
        # for the built-in plugin.
        if hasattr(self._guard_plugin, "metrics"):
            return dict(self._guard_plugin.metrics)  # type: ignore
        return {}


__all__ = ["GuardOrchestrator"]
