"""Validation helpers shared across the core package.

These utilities centralize defensive argument checks while preserving
existing behavior, ensuring future refactors can rely on a consistent
error vocabulary.

ADR-002 compliance: Validation functions use the exception taxonomy
(ValidationError, DataShapeError, NotFittedError, ConfigurationError, etc.)
and accept optional details payloads.
"""

from __future__ import annotations

import numbers
import sys
from collections.abc import Sequence
from typing import Any, Literal, Type, cast

import numpy as np
import numpy.typing as npt

from ..utils.exceptions import (
    CalibratedError,
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
    ValidationError,
)


def validate_not_none(value: Any, name: str) -> None:
    """Raise ``ValidationError`` when ``value`` is ``None``."""
    if value is None:
        raise ValidationError(f"Argument '{name}' must not be None.")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    """Ensure that ``value`` is an instance of ``expected_type``."""
    if not isinstance(value, expected_type):
        raise DataShapeError(
            f"Argument '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}."
        )


def validate_non_empty(value: Any, name: str) -> None:
    """Ensure that length-aware inputs are not empty."""
    if hasattr(value, "__len__") and len(value) == 0:
        raise ValidationError(f"Argument '{name}' must not be empty.")


def normalize_mode(value: Any, *, param: str = "mode") -> str:
    """Return a canonical explainer mode literal.

    Parameters
    ----------
    value : Any
        Candidate mode value.
    param : str, default="mode"
        Parameter name used in the structured error payload.

    Returns
    -------
    str
        Lower-case canonical mode (``"classification"`` or ``"regression"``).

    Raises
    ------
    ValidationError
        If ``value`` is not a supported mode string.
    """
    if not isinstance(value, str):
        raise ValidationError(
            "mode must be either 'classification' or 'regression'.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )
    normalized = value.lower()
    if normalized not in {"classification", "regression"}:
        raise ValidationError(
            "mode must be either 'classification' or 'regression'.",
            details={"param": param, "value": value, "normalized": normalized},
        )
    return normalized


def validate_bool_parameter(value: Any, *, param: str) -> bool:
    """Require a real boolean instead of truthy/falsy coercion."""
    if type(value) is not bool:  # noqa: E721 - reject bool-like subclasses/coercions explicitly
        raise ValidationError(
            f"{param} must be a boolean.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )
    return value


def validate_seed_parameter(value: Any, *, param: str = "seed") -> int:
    """Require an integer random seed."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(
            "seed must be an integer.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )
    return int(value)


def validate_sample_percentiles(value: Any, *, param: str = "sample_percentiles") -> list[float]:
    """Validate percentile checkpoints used by explanation summaries."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            "sample_percentiles must be a non-empty sequence of numeric values between 0 and 100.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )
    percentiles = list(value)
    if not percentiles:
        raise ValidationError(
            "sample_percentiles must be a non-empty sequence of numeric values between 0 and 100.",
            details={"param": param, "value": percentiles, "requirement": "non-empty"},
        )

    normalized: list[float] = []
    for percentile in percentiles:
        if isinstance(percentile, bool) or not isinstance(percentile, numbers.Real):
            raise ValidationError(
                "sample_percentiles must contain only numeric values.",
                details={
                    "param": param,
                    "value": percentiles,
                    "invalid_value": percentile,
                    "actual_type": type(percentile).__name__,
                },
            )
        numeric = float(percentile)
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 100.0:
            raise ValidationError(
                "sample_percentiles values must be between 0 and 100.",
                details={
                    "param": param,
                    "value": percentiles,
                    "invalid_value": numeric,
                    "allowed_range": [0.0, 100.0],
                },
            )
        normalized.append(numeric)

    if normalized != sorted(normalized):
        raise ValidationError(
            "sample_percentiles must be sorted in ascending order.",
            details={"param": param, "value": percentiles},
        )
    return normalized


def validate_low_high_percentiles(
    value: Any,
    *,
    param: str = "low_high_percentiles",
) -> tuple[float, float]:
    """Validate regression interval percentile bounds."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            "low_high_percentiles must be a sequence of exactly two numeric percentiles.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )

    percentiles = list(value)
    if len(percentiles) != 2:
        raise ValidationError(
            "low_high_percentiles must contain exactly two values.",
            details={"param": param, "value": percentiles, "actual_length": len(percentiles)},
        )
    if any(
        isinstance(percentile, bool) or not isinstance(percentile, numbers.Real)
        for percentile in percentiles
    ):
        raise ValidationError(
            "low_high_percentiles must contain only numeric values.",
            details={
                "param": param,
                "value": percentiles,
                "value_types": [type(percentile).__name__ for percentile in percentiles],
            },
        )

    low = float(percentiles[0])
    high = float(percentiles[1])
    if low > high:
        raise ValidationError(
            "The low percentile must be smaller than (or equal to) the high percentile.",
            details={"param": param, "low": low, "high": high},
        )
    if low == -np.inf and high == np.inf:
        raise ValidationError(
            "The percentiles cannot both be infinite.",
            details={"param": param, "low": low, "high": high},
        )
    if low != -np.inf and not (0.0 < low <= 50.0):
        raise ValidationError(
            "The lower percentile must be between 0 and 50, or -np.inf for one-sided intervals.",
            details={"param": param, "low": low},
        )
    if high != np.inf and not (50.0 <= high < 100.0):
        raise ValidationError(
            "The upper percentile must be between 50 and 100, or np.inf for one-sided intervals.",
            details={"param": param, "high": high},
        )
    return (low, high)


def validate_features_to_ignore(
    value: Any,
    *,
    n_features: int,
    param: str = "features_to_ignore",
) -> list[int]:
    """Validate a feature-index ignore list against the calibrated feature count."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            "features_to_ignore must be a sequence of feature indices.",
            details={"param": param, "value": value, "actual_type": type(value).__name__},
        )

    indices = list(value)
    normalized: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise ValidationError(
                "features_to_ignore must contain only integer feature indices.",
                details={
                    "param": param,
                    "value": indices,
                    "invalid_value": index,
                    "actual_type": type(index).__name__,
                },
            )
        normalized.append(int(index))

    invalid = [index for index in normalized if index < 0 or index >= n_features]
    if invalid:
        raise ValidationError(
            "features_to_ignore contains indices outside the calibrated feature range.",
            details={
                "param": param,
                "value": indices,
                "invalid_indices": invalid,
                "n_features": int(n_features),
            },
        )
    return normalized


def validate_fast_tuning_parameters(
    *,
    noise_type: Any,
    scale_factor: Any,
    severity: Any,
) -> tuple[str, int, float]:
    """Validate FAST perturbation controls before plugin construction."""
    if not isinstance(noise_type, str) or noise_type not in {"uniform", "gaussian"}:
        raise ValidationError(
            "noise_type must be either 'uniform' or 'gaussian'.",
            details={
                "param": "noise_type",
                "value": noise_type,
                "allowed_values": ["uniform", "gaussian"],
            },
        )
    if isinstance(scale_factor, bool) or not isinstance(scale_factor, numbers.Integral):
        raise ValidationError(
            "scale_factor must be a positive integer.",
            details={
                "param": "scale_factor",
                "value": scale_factor,
                "actual_type": type(scale_factor).__name__,
            },
        )
    if int(scale_factor) <= 0:
        raise ValidationError(
            "scale_factor must be a positive integer.",
            details={"param": "scale_factor", "value": int(scale_factor)},
        )
    if isinstance(severity, bool) or not isinstance(severity, numbers.Real):
        raise ValidationError(
            "severity must be a non-negative number.",
            details={
                "param": "severity",
                "value": severity,
                "actual_type": type(severity).__name__,
            },
        )
    severity_value = float(severity)
    if not np.isfinite(severity_value) or severity_value < 0.0:
        raise ValidationError(
            "severity must be a non-negative number.",
            details={"param": "severity", "value": severity_value},
        )
    return noise_type, int(scale_factor), severity_value


def validate_explainer_init_kwargs(
    kwargs: dict[str, Any],
    *,
    mode: Any,
    n_features: int,
) -> tuple[str, dict[str, Any]]:
    """Validate constructor/calibration kwargs at the public boundary."""
    normalized_mode = normalize_mode(mode)
    validated = dict(kwargs)

    if "suppress_crepes_errors" in validated:
        validated["suppress_crepes_errors"] = validate_bool_parameter(
            validated["suppress_crepes_errors"], param="suppress_crepes_errors"
        )
    if "fast" in validated:
        validated["fast"] = validate_bool_parameter(validated["fast"], param="fast")
    if "seed" in validated:
        validated["seed"] = validate_seed_parameter(validated["seed"])
    if "sample_percentiles" in validated:
        validated["sample_percentiles"] = validate_sample_percentiles(
            validated["sample_percentiles"]
        )
    if "features_to_ignore" in validated:
        validated["features_to_ignore"] = validate_features_to_ignore(
            validated["features_to_ignore"],
            n_features=n_features,
        )
    if {"noise_type", "scale_factor", "severity"} & validated.keys():
        noise_type, scale_factor, severity = validate_fast_tuning_parameters(
            noise_type=validated.get("noise_type", "uniform"),
            scale_factor=validated.get("scale_factor", 5),
            severity=validated.get("severity", 1),
        )
        validated["noise_type"] = noise_type
        validated["scale_factor"] = scale_factor
        validated["severity"] = severity

    return normalized_mode, validated


def validate_classification_calibration_targets(y: Any, *, learner: Any | None = None) -> None:
    """Ensure classification calibration data spans at least two known classes."""
    y_arr = _as_1d_array(y)
    unique_classes = np.unique(y_arr)
    details: dict[str, Any] = {
        "param": "y_cal",
        "unique_class_count": int(unique_classes.shape[0]),
        "unique_classes": np.asarray(unique_classes).tolist(),
    }
    if unique_classes.shape[0] < 2:
        details["requirement"] = "at least two classes in calibration data"
        if learner is not None and hasattr(learner, "classes_"):
            learner_classes = np.asarray(learner.classes_)
            details["model_class_count"] = int(learner_classes.shape[0])
            details["model_classes"] = learner_classes.tolist()

        raise ValidationError(
            "Classification calibration requires at least two unique target classes in y_cal.",
            details=details,
        )

    if learner is not None and hasattr(learner, "classes_"):
        learner_classes = np.asarray(learner.classes_).reshape(-1)
        if learner_classes.size == 0:
            return
        unknown_classes = np.setdiff1d(unique_classes, learner_classes)
        if unknown_classes.size > 0:
            details.update(
                {
                    "requirement": "calibration labels must be a subset of learner.classes_",
                    "model_class_count": int(learner_classes.shape[0]),
                    "model_classes": learner_classes.tolist(),
                    "unknown_classes": np.asarray(unknown_classes).tolist(),
                }
            )
            raise ValidationError(
                "Classification calibration labels must be a subset of the fitted learner classes.",
                details=details,
            )


def validate_inputs(
    x: Any,
    y: Any | None = None,
    task: Literal["auto", "classification", "regression"] = "auto",
    allow_nan: bool = False,
    require_y: bool = False,
    n_features: int | None = None,
    class_labels: Any | None = None,
    check_finite: bool = True,
) -> None:
    """Validate input features and target for downstream operations.

    This function provides the primary validation entry point per ADR-002,
    accepting feature matrix x and optional target y with comprehensive
    shape, dtype, and value checks.

    Parameters
    ----------
    x : array-like
        Feature matrix of shape (n_samples, n_features). Can be a NumPy array,
        pandas DataFrame, or similar.
    y : array-like, optional
        Target vector of shape (n_samples,). If provided, length must match x.
        Default is None.
    task : {"auto", "classification", "regression"}, default="auto"
        Task type. When "auto", inferred from model capabilities or y dtype.
    allow_nan : bool, default=False
        If False, raises ValidationError when x or y contain NaN values.
    require_y : bool, default=False
        If True, raises ValidationError when y is None.
    n_features : int, optional
        Expected number of features in x. If provided and mismatch occurs,
        raises DataShapeError.
    class_labels : array-like, optional
        Class labels for classification tasks. Stored for later use.
    check_finite : bool, default=True
        If True, checks that x and y contain only finite values (except NaN
        when allow_nan=True).

    Raises
    ------
    ValidationError
        When y is required but None, or when values contain NaN/inf unexpectedly.
    DataShapeError
        When x is not 2D, feature count mismatches, or y length mismatches.

    Examples
    --------
    >>> from calibrated_explanations.core.validation import validate_inputs
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([0, 1])
    >>> validate_inputs(x, y, task="classification", require_y=True, n_features=2)
    # Passes silently if valid
    """
    validate_not_none(x, "x")
    x_arr = _as_2d_array(x)
    if x_arr.ndim != 2:
        raise DataShapeError(
            "Argument 'x' must be 2D (n_samples, n_features).",
            details={"param": "x", "ndim": x_arr.ndim, "expected": 2},
        )
    n_samples = x_arr.shape[0]
    if n_features is not None and x_arr.shape[1] != n_features:
        raise DataShapeError(
            f"Argument 'x' must have {n_features} features, got {x_arr.shape[1]}.",
            details={
                "param": "x",
                "expected_features": n_features,
                "actual_features": x_arr.shape[1],
            },
        )

    if require_y and y is None:
        raise ValidationError(
            "Argument 'y' must be provided when require_y=True.",
            details={"param": "y", "requirement": "required", "task": task},
        )

    if y is not None:
        y_arr = _as_1d_array(y)
        if y_arr.shape[0] != n_samples:
            raise DataShapeError(
                f"Length of 'y' ({y_arr.shape[0]}) does not match number of samples in x ({n_samples}).",
                details={
                    "param": "y",
                    "y_length": y_arr.shape[0],
                    "x_samples": n_samples,
                },
            )
        if (
            check_finite
            and not allow_nan
            and np.issubdtype(y_arr.dtype, np.number)
            and not np.isfinite(y_arr).all()
        ):
            raise ValidationError(
                "Argument 'y' contains NaN or infinite values.",
                details={"param": "y", "check": "finitude", "allow_nan": allow_nan},
            )

    if check_finite and not allow_nan and not np.isfinite(x_arr).all():
        raise ValidationError(
            "Argument 'x' contains NaN or infinite values.",
            details={"param": "x", "check": "finitude", "allow_nan": allow_nan},
        )

    # Store class_labels if provided (for metadata tracking)
    if class_labels is not None:
        validate_not_none(class_labels, "class_labels")

    # Reserve task inference for future behavior without changing runtime output.
    _ = infer_task(x, y, None) if task == "auto" else task


def infer_task(
    x: Any = None, y: Any = None, model: Any = None
) -> Literal["classification", "regression"]:
    """Infer the task type using model capabilities or target dtype.

    Priority is given to model capabilities (``predict_proba`` implies
    classification). When a model is unavailable, heuristics based on the
    target dtype are used. Regression is the safe fallback.
    """
    if model is not None:
        if hasattr(model, "predict_proba"):
            return "classification"
        return "regression"
    if y is not None:
        y_arr = _as_1d_array(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            return "regression"
        return "classification"
    return "regression"


def _as_2d_array(x: Any) -> npt.NDArray[np.generic]:
    """Return ``x`` coerced to a 2D ``ndarray``."""
    if hasattr(x, "values") and hasattr(x, "shape"):
        try:
            return cast(npt.NDArray[np.generic], np.asarray(x.values))
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            # pragma: no cover - fallback
            return cast(npt.NDArray[np.generic], np.asarray(x))
    return cast(npt.NDArray[np.generic], np.asarray(x))


def _as_1d_array(y: Any) -> npt.NDArray[np.generic]:
    """Return ``y`` coerced to a flattened 1D ``ndarray``."""
    if hasattr(y, "values") and not isinstance(y, np.ndarray):
        y = y.values
    arr = cast(npt.NDArray[np.generic], np.asarray(y))
    return cast(npt.NDArray[np.generic], arr.reshape(-1))


def validate_inputs_matrix(
    x: Any,
    y: Any | None = None,
    *,
    task: Literal["auto", "classification", "regression"] = "auto",
    allow_nan: bool = False,
    require_y: bool = False,
    n_features: int | None = None,
    check_finite: bool = True,
) -> None:
    """Validate a feature/target matrix pair for downstream operations.

    - Ensure ``x`` is 2D and matches the expected feature count when provided.
    - Confirm that ``y`` has the same number of samples when supplied.
    - Guard against NaN or infinite values unless explicitly allowed.
    - Reject empty calibration-style inputs when ``require_y=True`` so public
      boundaries fail with CE exceptions before deeper indexing code runs.
    """
    validate_not_none(x, "x")
    x_arr = _as_2d_array(x)
    if x_arr.ndim != 2:
        raise DataShapeError("Argument 'x' must be 2D (n_samples, n_features).")
    n_samples = x_arr.shape[0]
    if require_y and n_samples == 0:
        raise DataShapeError(
            "Argument 'x' must contain at least one sample when calibration targets are required.",
            details={
                "param": "x",
                "require_y": True,
                "x_samples": 0,
                "requirement": "non-empty calibration data",
            },
        )
    if n_features is not None and x_arr.shape[1] != n_features:
        raise DataShapeError(f"Argument 'x' must have {n_features} features, got {x_arr.shape[1]}.")

    if require_y and y is None:
        raise ValidationError("Argument 'y' must be provided when require_y=True.")
    if y is not None:
        y_arr = _as_1d_array(y)
        if y_arr.shape[0] != n_samples:
            raise DataShapeError(
                f"Length of 'y' ({y_arr.shape[0]}) does not match number of samples in x ({n_samples})."
            )
        if (
            check_finite
            and not allow_nan
            and np.issubdtype(y_arr.dtype, np.number)
            and not np.isfinite(y_arr).all()
        ):
            raise ValidationError("Argument 'y' contains NaN or infinite values.")

    if check_finite and not allow_nan and not np.isfinite(x_arr).all():
        raise ValidationError("Argument 'x' contains NaN or infinite values.")

    # Reserve task inference for future behavior without changing runtime output.
    _ = infer_task(x, y, None) if task == "auto" else task


def validate_model(model: Any) -> None:
    """Validate minimal model protocol requirements."""
    validate_not_none(model, "model")
    if not hasattr(model, "predict"):
        raise ModelNotSupportedError("Model must implement a 'predict' method.")


def validate_fit_state(obj: Any, *, require: bool = True) -> None:
    """Validate fit state flags before executing stateful operations."""
    if not require:
        return
    if hasattr(obj, "fitted") and not obj.fitted:
        raise NotFittedError("Operation requires a fitted estimator/explainer.")


def validate(
    condition: bool,
    exc_cls: Type[CalibratedError],
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    """Conditional validation helper for common patterns.

    Raises an exception when a condition is False, enabling concise guard clauses.

    Parameters
    ----------
    condition : bool
        Condition to check. If False, raises exc_cls.
    exc_cls : Type[CalibratedError]
        Exception class to raise when condition is False.
    message : str
        Error message.
    details : dict, optional
        Structured error details to attach to the exception.

    Raises
    ------
    exc_cls
        If condition is False.

    Examples
    --------
    >>> from calibrated_explanations.core.validation import validate
    >>> from calibrated_explanations.utils.exceptions import ValidationError
    >>> validate(len(x) > 0, ValidationError, "x must not be empty", details={"param": "x"})
    """
    if not condition:
        raise exc_cls(message, details=details)


__all__ = [
    "normalize_mode",
    "validate_bool_parameter",
    "validate_explainer_init_kwargs",
    "validate_features_to_ignore",
    "validate_low_high_percentiles",
    "validate_inputs",
    "validate_not_none",
    "validate_type",
    "validate_non_empty",
    "validate_inputs_matrix",
    "validate_model",
    "validate_sample_percentiles",
    "validate_seed_parameter",
    "validate_fast_tuning_parameters",
    "validate_fit_state",
    "infer_task",
    "validate",
]
