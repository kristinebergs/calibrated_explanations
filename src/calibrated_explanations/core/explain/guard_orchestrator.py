"""Guard orchestrator plugin for conformal region-based filtering.

This module provides GuardOrchestratorPlugin, a GuardPlugin implementation that
wraps ConformalRegionOracle to provide perturbation and candidate filtering
during explanation generation.

Part of Phase 2: Plugin-first Guards (ADR-001, ADR-006).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...guards.regions import ConformalRegionOracle
from ...guards.interval_learner_adapter import IntervalLearnerAdapter
from ...plugins.guards import GuardContext, GuardPlugin
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class GuardOrchestratorPlugin(GuardPlugin):
    """Plugin implementation of guard orchestrator for conformal region filtering.

    This plugin wraps ConformalRegionOracle to provide perturbation and candidate
    filtering during explanation generation. It implements the GuardPlugin protocol
    and integrates with the plugin manager.

    The plugin maintains the same filtering logic as the original GuardOrchestrator
    but operates within the plugin architecture.
    """

    plugin_meta = {
        "name": "core.guard.conformal_regions",
        "version": "1.0.0",
        "description": "Conformal region-based perturbation filtering",
        "provider": "calibrated_explanations",
        "capabilities": ["perturbation_filtering", "candidate_filtering"],
        "modes": ["factual", "alternative", "fast"],
        "tasks": ["classification", "regression"],
        "dependencies": [],
        "schema_version": 1,
        "trust": True,  # Built-in plugin, trusted by default
    }

    def __init__(self) -> None:
        self._guard: Optional[ConformalRegionOracle] = None
        self._guard_params: Dict[str, Any] | None = None
        self._enforcement: bool = False
        self._context: Optional[GuardContext] = None
        self.metrics: Dict[str, int] = {
            "accept_calls": 0,
            "accept_rejections": 0,
            "filtered_perturbations": 0,
            "filtered_candidates": 0,
        }

    def supports_mode(self, mode: str, *, task: str) -> bool:
        """Check if this guard supports the given explanation mode and task."""
        return (
            mode in self.plugin_meta["modes"] and
            task in self.plugin_meta["tasks"]
        )

    def initialize(self, context: GuardContext) -> None:
        """Initialize the guard with explainer context.

        Creates and fits the ConformalRegionOracle using calibration data
        from the provided context.
        """
        self._context = context

        # Extract guard parameters from context metadata
        guard_params = context.metadata.get("guard_params", {})
        params_copy = dict(guard_params) if guard_params else {}

        self._enforcement = bool(params_copy.pop("enforcement", False))  # Default to False for plugin
        if not params_copy:
            msg = "No guard_params provided; guard disabled."
            logger.info(msg)
            return

        self._guard_params = params_copy

        try:
            # Wrap the interval learner to support uq_interval parameter
            wrapped_learner = IntervalLearnerAdapter(context.interval_learner)

            guard = ConformalRegionOracle(**self._guard_params, enforcement=self._enforcement)

            # Fit using context calibration data and wrapped interval learner
            guard.fit(
                context.x_cal,
                context.y_cal,
                interval_learner=wrapped_learner,
            )

            self._guard = guard
            logger.info("Guard fitted successfully: %s", self._guard_params)

        except Exception as exc:  # pylint: disable=broad-except
            self._guard = None
            msg = f"Failed to fit guard: {exc}. Guard disabled."
            if self._enforcement:
                raise ConfigurationError(msg) from exc
            logger.warning(msg)

    def filter_perturbations(
        self,
        perturbed_x: np.ndarray,
        perturbed_feature: np.ndarray,
        x_orig: np.ndarray,
        prediction: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter perturbed instances using the guard's accept method."""
        if self._guard is None or len(perturbed_x) == 0:
            if self._guard is None and self._enforcement:
                raise RuntimeError("No guard present for filtering perturbations")
            return perturbed_x, perturbed_feature

        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted for perturbation filtering"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning(msg)
            return perturbed_x, perturbed_feature

        try:  # pylint: disable=broad-except
            pred_vals = prediction.get("predict", np.array([]))
            lows = prediction.get("low", np.array([]))
            highs = prediction.get("high", np.array([]))

            # Build calibrated predictions per instance
            cal_preds: list[Optional[Tuple[float, Tuple[float, float]]]] = []
            n_instances = len(x_orig)
            for i in range(n_instances):
                if i < len(pred_vals) and i < len(lows) and i < len(highs):
                    cal_preds.append((pred_vals[i], (lows[i], highs[i])))
                else:
                    cal_preds.append(None)

            # For each perturbed row, fetch associated instance calibrated pred
            calibrated_pred_list = []
            for idx in range(len(perturbed_x)):
                instance_idx = 0
                if len(perturbed_feature) > idx and len(perturbed_feature[idx]) > 1:
                    try:
                        instance_idx = int(perturbed_feature[idx, 1])
                    except (ValueError, TypeError):
                        instance_idx = 0
                calibrated_pred_list.append(
                    cal_preds[instance_idx] if instance_idx < len(cal_preds) else None
                )

            mask = self._guard.accept_batch(perturbed_x, calibrated_pred_list)
            filtered_x, filtered_feat = perturbed_x[mask], perturbed_feature[mask]
            self.metrics["filtered_perturbations"] += int(np.count_nonzero(~mask))
            return filtered_x, filtered_feat

        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Guard filtering failed; returning unfiltered perturbations: {exc}"
            if self._enforcement:
                raise
            logger.debug(msg)
            return perturbed_x, perturbed_feature

    def filter_candidates(
        self,
        feature_index: int,
        candidates: np.ndarray,
        x_orig: np.ndarray,
        calibrated_pred: Optional[Tuple[float, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Filter candidate values for a single feature using guard intervals."""
        if self._guard is None or x_orig is None:
            if self._guard is None and self._enforcement:
                raise RuntimeError("No guard present for filtering candidates")
            return candidates

        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted for candidate filtering"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning(msg)
            return candidates

        try:
            intervals = self._guard.intervals(x_orig, calibrated_pred)
            if feature_index < len(intervals) and intervals[feature_index]:
                mask = np.zeros(len(candidates), dtype=bool)
                for low, high in intervals[feature_index]:
                    mask |= (candidates >= low) & (candidates <= high)
                filtered = candidates[mask]
                self.metrics["filtered_candidates"] += int(np.count_nonzero(~mask))
                return filtered
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"filter_candidates failed: {exc}"
            if self._enforcement:
                raise
            logger.debug(msg)
            return candidates
        return candidates

    def get_guard(self) -> Optional[ConformalRegionOracle]:
        """Return the fitted guard or None if disabled."""
        return self._guard

    def set_guard(self, guard: Optional[ConformalRegionOracle]) -> None:
        """Assign or replace the guard instance.

        If None is provided, disable guarding. If a guard is provided it must
        be fitted (or the call will be accepted but be ineffective).
        """
        if guard is not None and getattr(guard, "_fitted", False) is False:
            msg = "Provided guard is not fitted; cannot set guard."
            if self._enforcement:
                raise ValueError(msg)
            logger.warning(msg)
            # Still set the guard even if unfitted, as per comment
        self._guard = guard

    def accept_batch(
        self,
        x_batch: np.ndarray,
        calibrated_predictions: Optional[Sequence[Optional[Tuple[float, Tuple[float, float]]]]] = None,
    ) -> np.ndarray:
        """Batch acceptance check for guard constraints."""
        if self._guard is None:
            msg = f"No guard present: accepting all by default for batch of size {len(x_batch)}"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.debug(msg)
            return np.ones(len(x_batch), dtype=bool)

        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted for batch accept"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning("%s; defaulting to accept_all", msg)
            return np.ones(len(x_batch), dtype=bool)

        try:
            mask = self._guard.accept_batch(np.asarray(x_batch), calibrated_predictions)
            self.metrics["accept_calls"] += len(mask)
            self.metrics["accept_rejections"] += int(np.count_nonzero(~mask))
            return mask
        except Exception as exc:  # pylint: disable=broad-except # pragma: no cover - defensive
            msg = f"Guard accept_batch() failed; defaulting to accept_all. Reason: {exc}"
            if self._enforcement:
                raise
            logger.warning(msg)
            return np.ones(len(x_batch), dtype=bool)


__all__ = ["GuardOrchestratorPlugin"]