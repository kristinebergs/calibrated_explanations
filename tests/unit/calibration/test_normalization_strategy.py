"""Tests for multiclass interval normalization strategy coercion."""

from __future__ import annotations

import pytest

from calibrated_explanations.calibration.normalization_strategy import (
    NormalizationStrategy,
    coerce_normalization_strategy,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_should_return_strategy_member_unchanged() -> None:
    assert (
        coerce_normalization_strategy(NormalizationStrategy.SIMPLEX)
        is NormalizationStrategy.SIMPLEX
    )


def test_should_coerce_valid_strings_case_insensitively() -> None:
    assert coerce_normalization_strategy("SCALE") is NormalizationStrategy.SCALE
    assert coerce_normalization_strategy("coherence") is NormalizationStrategy.COHERENCE


def test_should_raise_validation_error_for_boolean_values() -> None:
    with pytest.raises(ValidationError):
        coerce_normalization_strategy(True)
    with pytest.raises(ValidationError):
        coerce_normalization_strategy(False)


def test_should_raise_validation_error_for_unknown_string() -> None:
    with pytest.raises(ValidationError):
        coerce_normalization_strategy("not-a-strategy")


def test_should_raise_validation_error_for_unrecognized_type() -> None:
    with pytest.raises(ValidationError):
        coerce_normalization_strategy(object())


def test_should_raise_validation_error_for_none() -> None:
    with pytest.raises(ValidationError):
        coerce_normalization_strategy(None)
