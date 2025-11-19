"""Guard orchestration within the explanation pipeline.

This module provides GuardOrchestrator which manages guard plugin lifecycle
and perturbation filtering during explanation generation.

Part of Phase 2: Move & Refactor GuardOrchestrator (ADR-001).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from calibrated_explanations.plugins.guards import GuardContext, GuardPlugin

logger = logging.getLogger(__name__)


class GuardOrchestrator:
    """Orchestrate guard plugin usage during explanation generation.

    This class owns the guard plugin instance and provides thin delegation
    APIs for filtering perturbations and candidates during explanation.

    The orchestrator is initialized by ExplanationOrchestrator with a guard
    plugin instance and frozen context. All guard decisions flow through
    this orchestrator, and any guard failures are surfaced immediately
    (log + raise) to avoid silently continuing with unfiltered data.
    """

    def __init__(self, explainer: Any, guard_plugin: Optional[GuardPlugin] = None) -> None:
        """Initialize with explainer reference and optional guard plugin.

        Parameters
        ----------
        explainer : CalibratedExplainer
            Parent explainer instance (used to access metadata).
        guard_plugin : GuardPlugin, optional
            Guard plugin instance. If None, guarding is disabled.
        """
        self.explainer = explainer
        self._guard_plugin = guard_plugin
        self._context: Optional[GuardContext] = None
        self.metrics: Dict[str, int] = {
            "accept_calls": 0,
            "accept_rejections": 0,
            "filtered_perturbations": 0,
            "filtered_candidates": 0,
        }

    def initialize(self, context: GuardContext) -> None:
        """Initialize guard with frozen context.

        Parameters
        ----------
        context : GuardContext
            Immutable context for guard plugin initialization.
        """
        self._context = context
        if self._guard_plugin is not None:
            try:
                self._guard_plugin.initialize(context)
                logger.debug("Guard plugin initialized successfully")
            except Exception as exc:  # pragma: no cover
                logger.error("Guard plugin initialization failed: %s", exc)
                raise

    def set_plugin(self, plugin: Optional[GuardPlugin]) -> None:
        """Replace the guard plugin instance.

        Parameters
        ----------
        plugin : GuardPlugin, optional
            New guard plugin instance, or None to disable guarding.
        """
        self._guard_plugin = plugin
        if plugin is not None and self._context is not None:
            try:
                plugin.initialize(self._context)
            except Exception as exc:  # pragma: no cover
                logger.error("Guard plugin initialization failed: %s", exc)
                raise

    def get_plugin(self) -> Optional[GuardPlugin]:
        """Return the active guard plugin or None."""
        return self._guard_plugin

    def filter_perturbations(
        self,
        perturbed_x: np.ndarray,
        perturbed_feature: np.ndarray,
        x_orig: np.ndarray,
        prediction: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter perturbed instances using guard plugin.

        Delegates to the guard plugin if one is configured, otherwise
        returns all perturbations unfiltered.

        Parameters
        ----------
        perturbed_x : ndarray
            Perturbed instances matrix.
        perturbed_feature : ndarray
            Metadata about perturbed features.
        x_orig : ndarray
            Original instances.
        prediction : Mapping
            Calibrated predictions with "predict", "low", "high" keys.

        Returns
        -------
        perturbed_x_filtered : ndarray
            Filtered perturbed instances.
        perturbed_feature_filtered : ndarray
            Corresponding metadata rows.
        """
        if self._guard_plugin is None or len(perturbed_x) == 0:
            return perturbed_x, perturbed_feature

        try:
            filtered_x, filtered_feature = self._guard_plugin.filter_perturbations(
                perturbed_x, perturbed_feature, x_orig, prediction
            )
            n_rejected = len(perturbed_x) - len(filtered_x)
            self.metrics["filtered_perturbations"] += n_rejected
            return filtered_x, filtered_feature
        except Exception as exc:  # pragma: no cover
            logger.error("Guard perturbation filtering failed: %s", exc)
            raise

    def filter_candidates(
        self,
        feature_index: int,
        candidates: np.ndarray,
        x_orig: np.ndarray,
        calibrated_pred: Optional[Tuple[float, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Filter candidate values using guard plugin.

        Parameters
        ----------
        feature_index : int
            Index of feature being filtered.
        candidates : ndarray
            Candidate values to filter.
        x_orig : ndarray
            Original instance(s).
        calibrated_pred : Tuple, optional
            Calibrated prediction (predict, (low, high)).

        Returns
        -------
        filtered_candidates : ndarray
            Filtered candidate values.
        """
        if self._guard_plugin is None:
            return candidates

        try:
            filtered = self._guard_plugin.filter_candidates(
                feature_index, candidates, x_orig, calibrated_pred
            )
            n_rejected = len(candidates) - len(filtered)
            self.metrics["filtered_candidates"] += n_rejected
            return filtered
        except Exception as exc:  # pragma: no cover
            logger.error("Guard candidate filtering failed: %s", exc)
            raise

    def accept_batch(
        self,
        x_batch: np.ndarray,
        calibrated_predictions: Optional[Sequence[Optional[Tuple]]] = None,
    ) -> np.ndarray:
        """Batch acceptance check using guard plugin.

        Parameters
        ----------
        x_batch : ndarray
            Batch of instances to check.
        calibrated_predictions : Sequence, optional
            Per-instance calibrated predictions.

        Returns
        -------
        mask : ndarray of bool
            True for accepted instances.
        """
        if self._guard_plugin is None:
            return np.ones(len(x_batch), dtype=bool)

        try:
            mask = self._guard_plugin.accept_batch(x_batch, calibrated_predictions)
            n_rejected = int(np.count_nonzero(~mask))
            self.metrics["accept_calls"] += len(mask)
            self.metrics["accept_rejections"] += n_rejected
            return mask
        except Exception as exc:  # pragma: no cover
            logger.error("Guard accept_batch failed: %s", exc)
            raise

    def get_metrics(self) -> Dict[str, int]:
        """Return guard filtering metrics."""
        return dict(self.metrics)

    def accept(
        self,
        x_instance: np.ndarray,
        calibrated_prediction: Optional[Tuple[float, Tuple[float, float]]] = None,
    ) -> bool:
        """Backward-compatible single-instance accept helper."""
        if self._guard_plugin is None:
            return True

        batch = np.asarray(x_instance)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        elif batch.ndim == 0:
            batch = batch.reshape(1, 1)
        predictions: Optional[Sequence[Optional[Tuple]]] = None
        if calibrated_prediction is not None:
            predictions = [calibrated_prediction]

        result = self.accept_batch(batch, predictions)
        return bool(result[0]) if len(result) else True

    def fit_guard(self, guard_params: Optional[Mapping[str, Any]] = None) -> None:
        """Re-fit the guard plugin with updated metadata parameters."""
        if self._guard_plugin is None:
            logger.debug("fit_guard called without an active guard plugin; ignoring")
            return
        if self._context is None:
            logger.debug("fit_guard called before guard context is available; ignoring")
            return

        metadata = dict(self._context.metadata)
        if guard_params is not None:
            try:
                metadata["guard_params"] = dict(guard_params)
            except Exception:  # pragma: no cover - defensive
                metadata["guard_params"] = guard_params

        refreshed_context = replace(self._context, metadata=metadata)
        self.initialize(refreshed_context)

    def get_guard(self) -> Any:
        """Return the underlying guard instance when exposed by the plugin."""
        if self._guard_plugin is None:
            return None

        getter = getattr(self._guard_plugin, "get_guard", None)
        if callable(getter):
            return getter()
        return getattr(self._guard_plugin, "_guard", None)

    def set_guard(self, guard: Any) -> None:
        """Assign or replace the guard instance when supported by the plugin."""
        if self._guard_plugin is None:
            logger.debug("set_guard called without an active guard plugin; ignoring")
            return

        setter = getattr(self._guard_plugin, "set_guard", None)
        if callable(setter):
            setter(guard)
            return
        if hasattr(self._guard_plugin, "_guard"):
            setattr(self._guard_plugin, "_guard", guard)


__all__ = ["GuardOrchestrator"]
