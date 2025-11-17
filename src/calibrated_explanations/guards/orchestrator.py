"""Guard orchestrator wrapper for ConformalRegionOracle.

This module provides a lightweight GuardOrchestrator that centralizes guard
lifecycle and runtime operations so that the rest of the codebase can keep
thin delegators to it. It wraps the existing ConformalRegionOracle in
`guards.regions` and exposes accept/intervals/filter helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .regions import ConformalRegionOracle

logger = logging.getLogger(__name__)


class GuardOrchestrator:
    """Simple orchestrator that owns the guard instance and provides thin
    runtime APIs for acceptance checks and interval queries.

    It is intentionally lightweight: heavy computation stays within
    ConformalRegionOracle.
    """

    def __init__(self, explainer: Any):
        self._explainer = explainer
        self._guard: Optional[ConformalRegionOracle] = None
        self._guard_params: Dict[str, Any] | None = None
        self._enforcement: bool = False
        self.metrics: Dict[str, int] = {
            "accept_calls": 0,
            "accept_rejections": 0,
            "filtered_perturbations": 0,
            "filtered_candidates": 0,
        }

    def initialize_chains(self) -> None:
        """Initialize plugin/chains used by the orchestrator.

        Currently no-op; present for API compatibility with other orchestrators.
        """
        return None

    def fit_guard(self, guard_params: Dict[str, Any]) -> None:
        """Create and fit guard using the provided parameters.

        If the ConformalRegionOracle cannot be imported or fitting fails, the
        guard will remain None and a warning will be emitted.
        """
        params_copy = dict(guard_params) if guard_params else {}
        self._enforcement = bool(params_copy.pop("enforcement", True))
        if not params_copy:
            msg = "No guard_params provided; guard disabled."
            if self._enforcement:
                raise ValueError(msg)
            logger.info(msg)
            return
        self._guard_params = params_copy
        try:
            # Wrap the interval learner to support uq_interval parameter
            from .interval_learner_adapter import IntervalLearnerAdapter
            
            interval_learner = self._explainer.interval_learner
            wrapped_learner = IntervalLearnerAdapter(interval_learner)
            
            guard = ConformalRegionOracle(**self._guard_params, enforcement=self._enforcement)
            # Fit using explainer calibration data and wrapped interval_learner
            guard.fit(
                self._explainer.x_cal,
                self._explainer.y_cal,
                interval_learner=wrapped_learner,
            )
            self._guard = guard
            logger.info("Guard fitted successfully: %s", self._guard_params)
        except Exception as exc:  # pylint: disable=broad-except
            self._guard = None
            msg = f"Failed to fit guard: {exc}. Guard disabled."
            if self._enforcement:
                raise RuntimeError(msg) from exc
            logger.warning(msg)

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
            return
        self._guard = guard

    def get_guard(self) -> Optional[ConformalRegionOracle]:
        """Return the current guard instance or None."""
        return self._guard

    def accept(self, x_new: Any, calibrated_prediction: Optional[tuple] = None) -> bool:
        """Single-instance acceptance check.

        Returns True when guard is absent or the guard accepts the point.
        """
        if self._guard is None:
            msg = "No guard present: accepting by default"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.debug(msg)
            return True
        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning("%s; defaulting to accept=True", msg)
            return True
        if calibrated_prediction is None and self._enforcement:
            raise RuntimeError("Calibrated prediction required for guard enforcement")
        try:
            accepted = bool(self._guard.accept(x_new, calibrated_prediction))
            self.metrics["accept_calls"] += 1
            if not accepted:
                self.metrics["accept_rejections"] += 1
            return accepted
        except Exception as exc:  # pylint: disable=broad-except # pragma: no cover - defensive
            msg = f"Guard accept() failed; defaulting to accept=True. Reason: {exc}"
            if self._enforcement:
                raise
            logger.warning(msg)
            return True

    def accept_batch(
        self,
        x_new_batch: Sequence[Any],
        calibrated_predictions: Optional[Sequence[Optional[tuple]]] = None,
    ) -> np.ndarray:
        """Batch acceptance vector.

        calibrated_predictions may be provided (one per row). If not, we pass None
        items to the guard's accept_batch.
        """
        if self._guard is None:
            msg = f"No guard present: accepting all by default for batch of size {len(x_new_batch)}"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.debug(msg)
            return np.ones(len(x_new_batch), dtype=bool)
        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted for batch accept"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning("%s; defaulting to accept_all", msg)
            return np.ones(len(x_new_batch), dtype=bool)
        try:
            mask = self._guard.accept_batch(np.asarray(x_new_batch), calibrated_predictions)
            self.metrics["accept_calls"] += len(mask)
            self.metrics["accept_rejections"] += int(np.count_nonzero(~mask))
            return mask
        except Exception as exc:  # pylint: disable=broad-except # pragma: no cover - defensive
            msg = f"Guard accept_batch() failed; defaulting to accept_all. Reason: {exc}"
            if self._enforcement:
                raise
            logger.warning(msg)
            return np.ones(len(x_new_batch), dtype=bool)

    def intervals(
        self, x_orig: Any, calibrated_prediction: Optional[tuple] = None
    ) -> List[List[tuple]]:
        """Return feature-wise allowed intervals around x_orig.

        If guard is not present, return empty intervals (no filtering).
        """
        if self._guard is None:
            msg = "No guard present for intervals lookup"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.debug(msg)
            return [[] for _ in range(self._explainer.num_features)]
        if getattr(self._guard, "_fitted", False) is False:
            msg = "Guard present but not fitted for intervals"
            if self._enforcement:
                raise RuntimeError(msg)
            logger.warning(msg)
            return [[] for _ in range(self._explainer.num_features)]
        return self._guard.intervals(x_orig, calibrated_prediction)

    def filter_perturbations(
        self,
        perturbed_x: np.ndarray,
        perturbed_feature: np.ndarray,
        x_orig: np.ndarray,
        prediction: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter perturbed instances using the guard's accept method.

        Expects perturbed_feature rows to carry an origin instance index under
        column 1, similar to the current helper code.
        """
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
            cal_preds: List[Optional[tuple]] = []
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
        x_orig: np.ndarray = None,
        calibrated_pred: tuple = None,
    ) -> np.ndarray:
        """Filter candidate values for a single feature using guard intervals.

        Returns the (possibly filtered) candidates array.
        """
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


__all__ = ["GuardOrchestrator"]
