"""Tests for multiclass interval normalization strategy coercion."""

from __future__ import annotations


from calibrated_explanations.calibration.normalization_strategy import (
    NormalizationStrategy,
    coerce_normalization_strategy,
)


def test_should_return_strategy_member_unchanged() -> None:
    assert (
        coerce_normalization_strategy(NormalizationStrategy.SIMPLEX)
        is NormalizationStrategy.SIMPLEX
    )


def test_should_fall_back_to_scale_for_boolean_values() -> None:
    assert coerce_normalization_strategy(True) is NormalizationStrategy.SCALE
    assert coerce_normalization_strategy(False) is NormalizationStrategy.SCALE


def test_should_coerce_strings_and_fallback_for_unknown_values() -> None:
    assert coerce_normalization_strategy("SCALE") is NormalizationStrategy.SCALE
    assert coerce_normalization_strategy("coherence") is NormalizationStrategy.COHERENCE
    assert coerce_normalization_strategy("not-a-strategy") is NormalizationStrategy.SCALE
    assert coerce_normalization_strategy(object()) is NormalizationStrategy.SCALE
