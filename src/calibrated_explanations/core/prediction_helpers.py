"""Phase 1A prediction helper delegators.

This module introduces *thin* wrapper functions around existing private
methods of ``CalibratedExplainer``. It is an intermediate, mechanical step
that allows future extractions without touching behavior now. Tests will
exercise these wrappers to lock in semantics before moving logic bodies.
"""

from __future__ import annotations

import warnings as _warnings
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np

if TYPE_CHECKING:
    from ..explanations import CalibratedExplanations

from ..utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)
from ..utils import assert_threshold, safe_isinstance
from .explain._computation import explain_predict_step

# Local typing protocol to avoid importing CalibratedExplainer and creating cycles.
# Captures just the members used by these helpers.
ThresholdLike = Union[
    float,
    Tuple[float, float],
    Sequence[Tuple[float, float]],
    np.ndarray,
]


def _n_samples(x: Any) -> int:
    """Return the number of samples represented by ``x``."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _normalize_conditional_bins(bins: Any, *, n_samples: int) -> np.ndarray:
    """Normalize Mondrian category labels to one label per sample."""
    arr = np.asarray(bins)
    if arr.ndim == 0:
        arr = np.full(n_samples, arr.item())
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    if len(arr) != n_samples:
        raise DataShapeError(
            "The length of Mondrian bins must match the number of samples.",
            details={"bins_length": int(len(arr)), "n_samples": n_samples},
        )
    return arr


def _apply_conditional_categorizer(mc: Any, x: Any) -> Any:
    """Apply a Mondrian categorizer or callable without importing crepes eagerly."""
    apply = getattr(mc, "apply", None)
    if callable(apply):
        return apply(x)
    return mc(x)


def _sorted_labels(values: np.ndarray) -> list[Any]:
    """Return deterministic labels for messages/details."""
    labels = np.unique(values).tolist()
    return sorted(labels, key=lambda item: str(item))


def resolve_conditional_bins(
    x: Any,
    bins: Any = None,
    *,
    mc: Any = None,
    calibration_bins: Any = None,
) -> np.ndarray | None:
    """Resolve and validate Mondrian bins per ADR-039 D2/D3.

    Parameters
    ----------
    x : array-like
        Instances for the public inference call.
    bins : array-like, optional
        Explicit per-instance Mondrian category labels.
    mc : callable, optional
        Stored Mondrian categorizer used to derive category labels.
    calibration_bins : array-like, optional
        Category labels used during calibration.

    Returns
    -------
    ndarray or None
        Normalized per-instance category labels, or ``None`` for global
        calibration.

    Raises
    ------
    ValidationError
        If required bins are omitted or labels are outside the calibration
        vocabulary.
    ConfigurationError
        If conditional inputs conflict with the active calibration state.
    DataShapeError
        If the category label count does not match the number of samples.
    """
    n_samples = _n_samples(x)
    has_calibration_bins = calibration_bins is not None

    if mc is not None:
        if bins is not None:
            raise ConfigurationError(
                "A configured Mondrian categorizer derives bins automatically; "
                "remove bins= or recalibrate without mc.",
                details={"conflict": "bins and mc"},
            )
        bins = _apply_conditional_categorizer(mc, x)
    elif has_calibration_bins:
        if bins is None:
            raise ValidationError(
                "This explainer was calibrated with Mondrian bins; pass bins= "
                "with one category label per instance. Use reuse_conditional=True "
                "to recalibrate with a stored categorizer, or call calibrate() "
                "without bins/mc to return to global calibration.",
                details={
                    "n_instances": n_samples,
                    "requirement": "bins required for bins-calibrated inference",
                },
            )
    elif bins is not None:
        raise ConfigurationError(
            "This explainer was not calibrated with Mondrian bins, so conditional "
            "output is unavailable; calibrate with bins= or mc= first.",
            details={"requirement": "conditional calibration required before bins inference"},
        )
    else:
        return None

    resolved = _normalize_conditional_bins(bins, n_samples=n_samples)
    if has_calibration_bins:
        known = np.asarray(calibration_bins).reshape(-1)
        unknown = np.setdiff1d(np.unique(resolved), np.unique(known))
        if len(unknown) > 0:
            raise ValidationError(
                "Mondrian bins contain labels that were not seen during calibration.",
                details={
                    "unknown_labels": _sorted_labels(unknown),
                    "known_labels": _sorted_labels(known),
                },
            )
    return resolved


class _ExplainerProtocol(Protocol):
    """Structural subset of ``CalibratedExplainer`` used by helper functions."""

    num_features: int
    mode: str
    x_cal: np.ndarray
    interval_learner: Any

    @property
    def prediction_orchestrator(self) -> Any:
        """Return the prediction orchestrator."""
        ...

    def is_mondrian(self) -> bool:
        """Return True when a Mondrian (per-bin) calibration is active."""
        ...

    def is_multiclass(self) -> bool:
        """Return True when the underlying task involves more than two classes."""
        ...

    def is_fast(self) -> bool:
        """Return True when the specialized fast explainer path is available."""
        ...

    def rule_boundaries(self, x: np.ndarray, x_perturbed: np.ndarray) -> Any:
        """Return rule boundary metadata for categorical perturbations."""
        ...


# NOTE: We intentionally avoid importing CalibratedExplainer for type-only usage to
# prevent cyclical import complexity during the gradual split.


def validate_and_prepare_input(explainer: _ExplainerProtocol, x: Any) -> np.ndarray:
    """Validate and prepare input data (extracted logic).

    Mechanical move from ``CalibratedExplainer._validate_and_prepare_input``.
    """
    if safe_isinstance(x, "pandas.core.frame.DataFrame"):
        x = x.values  # pragma: no cover - passthrough
    if len(x.shape) == 1:  # noqa: PLR2004
        x = x.reshape(1, -1)
    if x.shape[1] != explainer.num_features:
        raise DataShapeError("Number of features must match calibration data")
    return cast(np.ndarray, np.asarray(x))


def initialize_explanation(
    explainer: _ExplainerProtocol,
    x: np.ndarray,
    low_high_percentiles: Tuple[int, int],
    threshold: Optional[ThresholdLike],
    bins: Optional[np.ndarray],
    features_to_ignore: Optional[Sequence[int]],
) -> CalibratedExplanations:
    """Initialize explanation object (extracted logic)."""
    from ..explanations import CalibratedExplanations  # pylint: disable=import-outside-toplevel

    is_mondrian = getattr(explainer, "is_mondrian", False)
    if callable(is_mondrian):
        is_mondrian = is_mondrian()
    if is_mondrian:
        if bins is None:
            raise ValidationError("Bins required for Mondrian explanations")
        if len(bins) != len(x):  # pragma: no cover - defensive
            raise DataShapeError("The length of bins must match the number of added instances.")
    explanation = CalibratedExplanations(
        explainer,
        x,
        threshold,
        bins,
        features_to_ignore,
        condition_source=getattr(explainer, "condition_source", "prediction"),
    )
    if threshold is not None:
        if "regression" not in explainer.mode:
            raise ValidationError(
                "The threshold parameter is only supported for mode='regression'."
            )
        if isinstance(threshold, (list, np.ndarray)) and isinstance(threshold[0], tuple):
            _warnings.warn(
                "Having a list of interval thresholds (i.e. a list of tuples) is likely going to be very slow. Consider using a single interval threshold for all instances.",
                stacklevel=2,
            )
        assert_threshold(threshold, x)
    elif "regression" in explainer.mode:
        explanation.low_high_percentiles = low_high_percentiles
    return explanation


def predict_internal(
    explainer: _ExplainerProtocol,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    low_high_percentiles: Tuple[int, int] = (5, 95),
    classes: Optional[Sequence[int]] = None,
    bins: Optional[np.ndarray] = None,
    feature: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the internal prediction logic (mechanically moved)."""
    orchestrator = explainer.prediction_orchestrator
    return orchestrator.predict_internal(
        x,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        classes=classes,
        bins=bins,
        feature=feature,
    )


__all__ = [
    "validate_and_prepare_input",
    "resolve_conditional_bins",
    "initialize_explanation",
    "predict_internal",
    "explain_predict_step",
    "format_regression_prediction",
    "format_classification_prediction",
    "handle_uncalibrated_regression_prediction",
    "handle_uncalibrated_classification_prediction",
]


def format_regression_prediction(
    predict: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Format regression predictions with optional thresholds and intervals.

    Parameters
    ----------
    predict : np.ndarray
        The predicted values.
    low : np.ndarray
        Lower bounds of prediction intervals.
    high : np.ndarray
        Upper bounds of prediction intervals.
    threshold : float, int, array-like, optional
        Threshold for probabilistic regression. If provided, returns probabilities.
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Formatted predictions with optional intervals.
    """
    if threshold is None:
        return (predict, (low, high)) if uq_interval else predict

    # Thresholded prediction - convert to probability labels
    def get_label(prob_val, thresh):
        if np.isscalar(thresh):
            return f"y_hat <= {thresh}" if prob_val >= 0.5 else f"y_hat > {thresh}"
        if isinstance(thresh, tuple):
            return (
                f"{thresh[0]} < y_hat <= {thresh[1]}"
                if prob_val >= 0.5
                else f"y_hat <= {thresh[0]} || y_hat > {thresh[1]}"
            )
        return "Error in format_regression_prediction.get_label()"

    if np.isscalar(threshold) or isinstance(threshold, tuple):
        new_classes = [get_label(predict[i], threshold) for i in range(len(predict))]
    else:
        new_classes = [get_label(predict[i], threshold[i]) for i in range(len(predict))]

    return (new_classes, (low, high)) if uq_interval else new_classes


def format_classification_prediction(
    predict: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    new_classes: Optional[np.ndarray],
    is_multiclass_val: bool,
    original_class_values: Optional[np.ndarray] = None,
    label_map: Optional[dict] = None,
    class_labels: Optional[np.ndarray] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Format classification predictions with optional class label mapping and intervals.

    Parameters
    ----------
    predict : np.ndarray
        The predicted probabilities.
    low : np.ndarray
        Lower bounds of prediction intervals.
    high : np.ndarray
        Upper bounds of prediction intervals.
    new_classes : np.ndarray or None
        Predicted class indices or None.
    is_multiclass_val : bool
        Whether this is a multiclass problem.
    original_class_values : np.ndarray, optional
        Original class labels in encoded-index order for dtype-preserving decoding.
    label_map : dict, optional
        Mapping from numeric class indices to labels.
    class_labels : array-like, optional
        Human-readable class labels.
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Formatted predictions with optional intervals.
    """
    if new_classes is None:
        new_classes = (predict >= 0.5).astype(int)

    # When class_labels or label_map are provided we may need to map numeric
    # indices to human-readable labels. Be defensive: new_classes can already
    # contain labels (strings) or numeric indices. Also allow dict-style
    # mappings.
    if original_class_values is not None:
        arr_nc = np.asarray(new_classes)
        if np.issubdtype(arr_nc.dtype, np.integer):
            new_classes = np.asarray(original_class_values)[arr_nc.astype(int, copy=False)]
        else:
            new_classes = np.asarray(new_classes)
    elif label_map is not None or class_labels is not None:
        # Prefer explicit mapping function when label_map provided
        if label_map is not None:
            inverse_map = {
                int(value): key
                for key, value in label_map.items()
                if isinstance(value, (int, np.integer))
            }
            mapped = []
            for cls in new_classes:
                try:
                    mapped.append(inverse_map.get(int(cls), cls))
                except (TypeError, ValueError):
                    mapped.append(cls)
            new_classes = np.array(mapped)
        else:
            # class_labels may be a sequence (list/ndarray) or a mapping.
            if isinstance(class_labels, dict):
                mapped = [class_labels.get(c, class_labels.get(int(c), c)) for c in new_classes]
                new_classes = np.array(mapped)
            else:
                # sequence-like: map only when new_classes are integer indices
                try:
                    arr_nc = np.asarray(new_classes)
                    if np.issubdtype(arr_nc.dtype, np.integer):
                        mapped = [class_labels[int(c)] for c in arr_nc]
                        new_classes = np.array(mapped)
                    else:
                        # Assume new_classes are already label values; coerce to ndarray
                        new_classes = np.asarray(new_classes)
                except (
                    Exception
                ):  # ADR002_ALLOW: tolerate unexpected label containers.  # pragma: no cover
                    new_classes = np.asarray(new_classes)

    return (new_classes, (low, high)) if uq_interval else new_classes


def handle_uncalibrated_regression_prediction(
    learner,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Handle uncalibrated regression prediction.

    Parameters
    ----------
    learner : object
        The underlying predictive learner.
    x : np.ndarray
        Input data.
    threshold : float, int, array-like, optional
        Threshold for regression tasks (not allowed for uncalibrated).
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Uncalibrated predictions.

    Raises
    ------
    ValidationError
        If threshold is provided.
    """
    if threshold is not None:
        raise ValidationError(
            "A thresholded prediction is not possible for uncalibrated predictions."
        )

    predict = learner.predict(x)
    return (predict, (predict, predict)) if uq_interval else predict


def handle_uncalibrated_classification_prediction(
    learner,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Handle uncalibrated classification prediction.

    Parameters
    ----------
    learner : object
        The underlying predictive learner.
    x : np.ndarray
        Input data.
    threshold : float, int, array-like, optional
        Threshold (not allowed for classification).
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Uncalibrated class predictions (not probabilities).

    Raises
    ------
    ValidationError
        If threshold is provided.
    """
    if threshold is not None:
        raise ValidationError("A thresholded prediction is not possible for uncalibrated learners.")

    # Use learner.predict() to get class predictions, not probabilities
    predictions = learner.predict(x)
    if uq_interval:
        # For intervals, use the same predictions as bounds (no uncertainty)
        return predictions, (predictions, predictions)
    return predictions
