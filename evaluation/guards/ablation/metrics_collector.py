"""
Metrics collection and normalization for ablation experiments.

This module defines standard metrics for evaluating guarded explanations
and provides utilities for collecting, aggregating, and normalizing results
across different parameter configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np

from calibrated_explanations import WrapCalibratedExplainer

logger = logging.getLogger(__name__)


@dataclass
class AblationMetrics:
    """Standard metrics for a single ablation trial."""

    # Basic trial info
    trial_id: int
    seed: int
    alpha: float
    distance: str
    n_clusters: int

    # Coverage metrics
    test_set_acceptance_rate: float
    test_set_rejection_rate: float

    # Explanation quality metrics
    factual_explanation_runtime: float
    alternative_explanations_runtime: float
    num_factual_explanations_valid: int
    num_factual_explanations_total: int
    factual_validity_rate: float

    # Metadata
    task_type: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates metrics from ablation trials.

    Provides methods to compute standard metrics from explainer outputs
    and to aggregate results across seeds/configurations.
    """

    @staticmethod
    def _get_guard_acceptance_mask(
        explainer: WrapCalibratedExplainer,
        X_test: np.ndarray,
    ) -> np.ndarray | None:
        """Try to compute guard acceptances for a batch of instances."""
        calibrated = getattr(explainer, "explainer", None)
        if calibrated is None:
            return None

        plugin_manager = getattr(calibrated, "_plugin_manager", None)
        if plugin_manager is None:
            return None

        guard_orchestrator = getattr(plugin_manager, "guard_orchestrator", None)
        if guard_orchestrator is None:
            return None

        guard_plugin = guard_orchestrator.get_plugin()
        if guard_plugin is None:
            return None

        try:
            prediction = calibrated.predict(X_test, uq_interval=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Guard coverage: prediction failed: %s", exc)
            return None

        if not (
            isinstance(prediction, tuple)
            and len(prediction) == 2
            and isinstance(prediction[1], tuple)
            and len(prediction[1]) == 2
        ):
            return None

        preds, interval = prediction
        low, high = interval
        preds_arr = np.asarray(preds)
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)

        if low_arr.ndim > 1:
            low_arr = low_arr[:, -1]
        if high_arr.ndim > 1:
            high_arr = high_arr[:, -1]

        if preds_arr.size == 0 or low_arr.size == 0 or high_arr.size == 0:
            return None

        if len(low_arr) < preds_arr.size or len(high_arr) < preds_arr.size:
            logger.debug(
                "Guard coverage: interval lengths mismatch (pred=%s, low=%s, high=%s)",
                preds_arr.shape,
                low_arr.shape,
                high_arr.shape,
            )
            return None

        calibrated_preds: list[tuple[Any, Tuple[float, float]]] = []
        for i in range(preds_arr.size):
            pred_value = preds_arr[i].item() if isinstance(preds_arr[i], np.generic) else preds_arr[i]
            calibrated_preds.append(
                (
                    pred_value,
                    (float(low_arr[i]), float(high_arr[i])),
                )
            )

        try:
            mask = guard_plugin.accept_batch(np.asarray(X_test), calibrated_preds)
            return np.asarray(mask, dtype=bool)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Guard coverage: accept_batch failed: %s", exc)
            return None

    @staticmethod
    def compute_coverage_metrics(
        explainer: WrapCalibratedExplainer,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float,
        max_instances: int = 100,
        threshold: float | None = None,
    ) -> Dict[str, float]:
        """
        Compute test-set coverage metrics for the guard.

        Parameters
        ----------
        explainer : WrapCalibratedExplainer
            The guarded explainer instance.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test labels.
        alpha : float
            Miscalibration level (coverage target = 1 - alpha).
        max_instances : int, optional
            Limit number of instances evaluated (for speed).
        threshold : float, optional
            Threshold for probabilistic regression.

        Returns
        -------
        dict
            Keys:
            - "acceptance_rate": empirical acceptance rate
            - "rejection_rate": empirical rejection rate
            - "coverage": estimate of true positive coverage
        """
        # Limit to max_instances for speed
        n_eval = min(max_instances, len(X_test))
        X_eval = X_test[:n_eval]

        mask = MetricsCollector._get_guard_acceptance_mask(explainer, X_eval)
        if mask is not None:
            acceptance_rate = float(np.mean(mask)) if mask.size else 0.0
            rejection_rate = 1.0 - acceptance_rate
            return {
                "acceptance_rate": acceptance_rate,
                "rejection_rate": rejection_rate,
                "coverage": acceptance_rate,
            }

        # Fallback to factual explanations when guard metadata is unavailable
        acceptances = 0
        rejections = 0

        try:
            if threshold is not None:
                factual_batch = explainer.explain_factual(X_eval, threshold=threshold)
            else:
                factual_batch = explainer.explain_factual(X_eval)
            acceptances = len(factual_batch)
        except Exception:
            rejections = len(X_eval)

        total = acceptances + rejections
        acceptance_rate = acceptances / total if total > 0 else 0.0
        rejection_rate = 1.0 - acceptance_rate

        return {
            "acceptance_rate": acceptance_rate,
            "rejection_rate": rejection_rate,
            "coverage": acceptance_rate,  # Placeholder for true coverage estimate
        }

    @staticmethod
    def compute_explanation_quality_metrics(
        explainer: WrapCalibratedExplainer,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_instances: int = 50,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """
        Compute explanation quality metrics.

        Parameters
        ----------
        explainer : WrapCalibratedExplainer
            The guarded explainer instance.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test labels.
        max_instances : int, optional
            Limit number of instances evaluated (for speed).
        threshold : float, optional
            Threshold for probabilistic regression.

        Returns
        -------
        dict
            Keys:
            - "factual_validity_rate": fraction of valid factual explanations
            - "factual_explanation_runtime": time for all explanations (seconds)
            - "alternative_explanations_runtime": time for alternatives (seconds)
        """
        import time

        # Limit to max_instances for speed
        n_eval = min(max_instances, len(X_test))
        X_eval = X_test[:n_eval]

        valid_count = 0
        total_count = 0

        # Time factual explanations
        start = time.perf_counter()
        try:
            if threshold is not None:
                factual_batch = explainer.explain_factual(X_eval, threshold=threshold)
            else:
                factual_batch = explainer.explain_factual(X_eval)
            
            for i, explanation in enumerate(factual_batch):
                # Simple heuristic: explanation is valid if it's not empty/None
                if explanation is not None:
                    valid_count += 1
                total_count += 1
        except Exception:
            total_count = n_eval
            valid_count = 0
            print("Warning: Exception occurred while exploring factual explanations")

        factual_runtime = time.perf_counter() - start

        # Time alternative explanations
        start = time.perf_counter()
        try:
            if threshold is not None:
                alt_batch = explainer.explore_alternatives(X_eval, threshold=threshold)
            else:
                alt_batch = explainer.explore_alternatives(X_eval)
            # Count as successful if returned non-empty
            alt_runtime = 0.0
        except Exception:
            alt_runtime = 0.0
            print("Warning: Exception occurred while exploring alternative explanations")

        alt_runtime = time.perf_counter() - start

        validity_rate = valid_count / total_count if total_count > 0 else 0.0

        return {
            "factual_validity_rate": validity_rate,
            "num_factual_valid": valid_count,
            "num_factual_total": total_count,
            "factual_explanation_runtime": factual_runtime,
            "alternative_explanations_runtime": alt_runtime,
        }

    @staticmethod
    def aggregate_metrics(
        results_list: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple seeds/runs.

        Parameters
        ----------
        results_list : list[dict]
            List of result dictionaries (one per seed).

        Returns
        -------
        dict
            Aggregated statistics (mean, std, min, max) for each metric.
        """
        if not results_list:
            return {}

        # Extract metric names from first result
        metric_names = [k for k in results_list[0].keys() if k not in {"trial_id", "seed", "alpha", "distance", "n_clusters", "task_type"}]

        aggregated = {}
        for metric_name in metric_names:
            values = [r[metric_name] for r in results_list if metric_name in r]
            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }

        return aggregated
