"""Adapter to provide uq_interval interface for standard interval learners.

This module adapts interval learners (like VennAbers, IntervalRegressor) from the
calibrated_explanations library to support the uq_interval parameter required by
the augmented-space ConformalRegionOracle.
"""

import numpy as np


class IntervalLearnerAdapter:
    """Adapter providing uq_interval interface for standard calibration learners.

    Wraps VennAbers, IntervalRegressor, or similar learners to provide a consistent
    predict() method that supports the uq_interval=True parameter.
    """

    def __init__(self, interval_learner):
        """Initialize adapter wrapping the interval learner.

        Parameters
        ----------
        interval_learner : object
            An interval learner with either:
            - predict_proba(x, output_interval=True) for classification
            - predict(x, output_interval=True) for regression
        """
        self._learner = interval_learner

    def predict(self, x_arr, uq_interval=False):
        """Predict with optional uncertainty quantification.

        Parameters
        ----------
        x_arr : array-like
            Input samples.
        uq_interval : bool, default=False
            If True, return (predictions, (lower, upper)) tuple.
            If False, return legacy format [(lower, upper), ...] for backward compatibility.

        Returns
        -------
        predictions or list
            If uq_interval=True: Tuple of (preds_array, (lower_array, upper_array))
            If uq_interval=False: List of (lower, upper) tuples
        """
        # Try to detect the learner type and call appropriate method
        learner_type = str(type(self._learner))
        if "VennAbers" in learner_type:
            # Classification: VennAbers
            return self._predict_classification(x_arr, uq_interval)
        elif "IntervalRegressor" in learner_type:
            # Regression: IntervalRegressor
            return self._predict_regression(x_arr, uq_interval)
        else:
            raise TypeError(
                f"Unsupported interval learner type: {type(self._learner)}. "
                "Expected VennAbers (classification) or IntervalRegressor (regression)."
            )

    def _predict_classification(self, x_arr, uq_interval):
        """Adapt VennAbers.predict_proba() to uq_interval interface."""
        try:
            result = self._learner.predict_proba(x_arr, output_interval=True)
            # VennAbers returns: (probs, low, high, classes) or (probs, low, high)
            if len(result) == 4:
                probs, low, high, _ = result
            else:
                probs, low, high = result

            # Extract positive class probability (for binary) or use as-is
            if probs.ndim == 2 and probs.shape[1] == 2:
                # Binary classification: use positive class probability
                preds = probs[:, 1]
                low_values = low
                high_values = high
            else:
                # Multiclass or other: use first column
                preds = probs[:, 0] if probs.ndim == 2 else probs
                low_values = low
                high_values = high

            if uq_interval:
                # Convert arrays if needed
                if not isinstance(low_values, np.ndarray):
                    low_values = np.asarray(low_values)
                if not isinstance(high_values, np.ndarray):
                    high_values = np.asarray(high_values)
                if not isinstance(preds, np.ndarray):
                    preds = np.asarray(preds)
                return preds, (low_values, high_values)
            else:
                # Legacy format: list of (lower, upper) tuples
                return list(zip(low_values, high_values))
        except Exception as exc:
            raise ValueError(
                f"VennAbers.predict_proba(x, output_interval=True) failed: {exc}"
            ) from exc

    def _predict_regression(self, x_arr, uq_interval):
        """Adapt IntervalRegressor.predict() to uq_interval interface."""
        try:
            # IntervalRegressor from crepes uses predict_interval
            result = self._learner.predict_interval(x_arr)

            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                preds, intervals = result
                if isinstance(intervals, tuple) and len(intervals) == 2:
                    lower, upper = intervals
                else:
                    # Assume intervals is array-like with (n, 2) shape
                    lower, upper = intervals[:, 0], intervals[:, 1]
            else:
                # If just predictions, assume default intervals [0, 1]
                preds = result
                lower = np.zeros_like(preds)
                upper = np.ones_like(preds)

            preds = np.asarray(preds)
            lower = np.asarray(lower)
            upper = np.asarray(upper)

            if uq_interval:
                return preds, (lower, upper)
            else:
                # Legacy format: list of (lower, upper) tuples
                return list(zip(lower, upper))
        except Exception as exc:
            raise ValueError(
                f"IntervalRegressor.predict_interval(x) failed: {exc}"
            ) from exc

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped learner."""
        return getattr(self._learner, name)
