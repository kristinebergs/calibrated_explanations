"""Tests for IntervalSummary value coercion (bug-list Finding 4)."""

from __future__ import annotations

import pytest

from calibrated_explanations.core.prediction.interval_summary import (
    IntervalSummary,
    coerce_interval_summary,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_should_return_member_unchanged() -> None:
    assert coerce_interval_summary(IntervalSummary.MEAN) is IntervalSummary.MEAN


def test_should_coerce_valid_string_value() -> None:
    assert coerce_interval_summary("mean") is IntervalSummary.MEAN
    assert coerce_interval_summary("regularized_mean") is IntervalSummary.REGULARIZED_MEAN


def test_should_default_to_regularized_mean_when_omitted() -> None:
    assert coerce_interval_summary(None) is IntervalSummary.REGULARIZED_MEAN


def test_should_raise_validation_error_for_boolean_values() -> None:
    with pytest.raises(ValidationError):
        coerce_interval_summary(True)
    with pytest.raises(ValidationError):
        coerce_interval_summary(False)


def test_should_raise_validation_error_for_unknown_string() -> None:
    with pytest.raises(ValidationError):
        coerce_interval_summary("mena")


def test_should_raise_validation_error_for_unrecognized_type() -> None:
    with pytest.raises(ValidationError):
        coerce_interval_summary(object())
