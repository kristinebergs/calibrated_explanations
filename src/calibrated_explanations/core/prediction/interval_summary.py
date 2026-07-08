"""Interval summary selection for probabilistic predictions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ...utils.exceptions import ValidationError


class IntervalSummary(Enum):
    """Describe how probabilistic intervals are summarized into point estimates.

    - REGULARIZED_MEAN: regularized Venn-Abers mean (default; legacy behavior).
    - MEAN: arithmetic mean of the interval bounds.
    - LOWER: lower interval bound.
    - UPPER: upper interval bound.
    """

    REGULARIZED_MEAN = "regularized_mean"
    MEAN = "mean"
    LOWER = "lower"
    UPPER = "upper"


def coerce_interval_summary(value: Any) -> IntervalSummary:
    """Return a validated IntervalSummary.

    ``None`` is treated as "caller omitted the parameter" and resolves to the
    ``REGULARIZED_MEAN`` default. Any other value that is not an
    ``IntervalSummary`` member or one of its string values raises
    ``ValidationError`` rather than silently falling back to the default.

    Raises
    ------
    ValidationError
        If ``value`` is not ``None``, an ``IntervalSummary`` member, or a
        matching string value.
    """
    if value is None:
        return IntervalSummary.REGULARIZED_MEAN
    try:
        return IntervalSummary(value)
    except (ValueError, TypeError) as exc:
        valid_values = [member.value for member in IntervalSummary]
        raise ValidationError(
            f"Invalid interval_summary value: {value!r}. Expected one of "
            f"{valid_values} or an IntervalSummary member.",
            details={
                "param": "interval_summary",
                "value": repr(value),
                "valid_values": valid_values,
            },
        ) from exc


__all__ = ["IntervalSummary", "coerce_interval_summary"]
