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
        if not guard_params:
            logger.info("No guard_params provided; guard disabled.")
            return
        self._guard_params = dict(guard_params)
        try:
            guard = ConformalRegionOracle(**self._guard_params)
            # Fit using explainer calibration data and interval_learner
            guard.fit(
                self._explainer.x_cal,
                self._explainer.y_cal,
                interval_learner=self._explainer.interval_learner,
            )
            self._guard = guard
            logger.info("Guard fitted successfully: %s", self._guard_params)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to fit guard: %s. Guard disabled.", exc)
            self._guard = None

    def set_guard(self, guard: Optional[ConformalRegionOracle]) -> None:
        """Assign or replace the guard instance.

        If None is provided, disable guarding. If a guard is provided it must
        be fitted (or the call will be accepted but be ineffective).
        """
        self._guard = guard

    def get_guard(self) -> Optional[ConformalRegionOracle]:
        """Return the current guard instance or None."""
        return self._guard

    def accept(self, x_new: Any, calibrated_prediction: Optional[tuple] = None) -> bool:
        """Single-instance acceptance check.

        Returns True when guard is absent or the guard accepts the point.
        """
        if self._guard is None:
            logger.debug("No guard present: accepting by default")
            return True
        try:
            return bool(self._guard.accept(x_new, calibrated_prediction))
        except Exception as exc:  # pylint: disable=broad-except # pragma: no cover - defensive
            logger.warning("Guard accept() failed; defaulting to accept=True. Reason: %s", exc)
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
            logger.debug("No guard present: accepting all by default for batch of size %d", len(x_new_batch))
            return np.ones(len(x_new_batch), dtype=bool)
        try:
            return self._guard.accept_batch(np.asarray(x_new_batch), calibrated_predictions)
        except Exception as exc:  # pylint: disable=broad-except # pragma: no cover - defensive
            logger.warning("Guard accept_batch() failed; defaulting to accept_all. Reason: %s", exc)
            return np.ones(len(x_new_batch), dtype=bool)

    def intervals(
        self, x_orig: Any, calibrated_prediction: Optional[tuple] = None
    ) -> List[List[tuple]]:
        """Return feature-wise allowed intervals around x_orig.

        If guard is not present, return empty intervals (no filtering).
        """
        if self._guard is None:
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
            return perturbed_x[mask], perturbed_feature[mask]
        except Exception:  # pylint: disable=broad-except
            logger.debug("Guard filtering failed; returning unfiltered perturbations")
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
            return candidates
        try:
            intervals = self._guard.intervals(x_orig, calibrated_pred)
            if feature_index < len(intervals) and intervals[feature_index]:
                mask = np.zeros(len(candidates), dtype=bool)
                for low, high in intervals[feature_index]:
                    mask |= (candidates >= low) & (candidates <= high)
                return candidates[mask]
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("filter_candidates failed: %s", exc)
            return candidates
        return candidates


__all__ = ["GuardOrchestrator"]
