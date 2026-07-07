"""Capability contract tests for conjunctive multi-feature explanation rules.

Requirements verified:
  CE-REQ-EXPL-CONJ-API-001    — add_conjunctions callable without exception
                                 (CE-CAP-EXPL-CONJ-001)
  CE-REQ-EXPL-CONJ-RETURN-001 — return type and collection cardinality contract
                                 (CE-CAP-EXPL-CONJ-001)
  CE-REQ-EXPL-CONJ-RULE-001   — multi-feature conjunction rules produced when
                                 max_rule_size >= 2 (CE-CAP-EXPL-CONJ-001)
  CE-REQ-EXPL-CONJ-PARAM-001  — max_rule_size=1 suppresses multi-feature rules
                                 (CE-CAP-EXPL-CONJ-001)

TIF interface used: CE-TIF-EXPL-CONJ-001
TIF executable: development/capabilities/verification/tif/tif_conjunction.py

Tests call run_conjunction_tif_scenario() and assert on the returned
ConjunctionObservation against acceptance criteria stated in the requirement files.
TIF observes; tests assert.

These tests do not assert that conjunctions produce better explanations than
single-feature rules. See development/capabilities/requirements/ for the full
assumption boundary.
"""

from __future__ import annotations

import pytest

# tif_conjunction is importable because tests/capabilities/conftest.py adds
# development/capabilities/verification/tif/ to sys.path.
from tif_conjunction import run_conjunction_tif_scenario


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-CONJ-API-001 — API availability (callable without exception)
# ---------------------------------------------------------------------------


def test_should_not_raise_when_factual_collection_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-API-001: add_conjunctions on factual collection is callable.

    Acceptance criterion:
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=2,
        n_top_features=5,
    )

    assert not obs.exception_raised, (
        f"CE-REQ-EXPL-CONJ-API-001: add_conjunctions on factual collection raised "
        f"{obs.exception_type}"
    )


def test_should_not_raise_when_alternative_collection_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-API-001: add_conjunctions on alternative collection is callable.

    Acceptance criterion:
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="alternative",
        object_level="collection",
        max_rule_size=2,
        n_top_features=5,
    )

    assert not obs.exception_raised, (
        f"CE-REQ-EXPL-CONJ-API-001: add_conjunctions on alternative collection raised "
        f"{obs.exception_type}"
    )


def test_should_not_raise_when_individual_factual_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-API-001: add_conjunctions on individual FactualExplanation.

    Acceptance criterion:
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="individual",
        max_rule_size=2,
        n_top_features=5,
    )

    assert not obs.exception_raised, (
        f"CE-REQ-EXPL-CONJ-API-001: add_conjunctions on individual FactualExplanation raised "
        f"{obs.exception_type}"
    )


def test_should_not_raise_when_individual_alternative_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-API-001: add_conjunctions on individual AlternativeExplanation.

    Acceptance criterion:
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="alternative",
        object_level="individual",
        max_rule_size=2,
        n_top_features=5,
    )

    assert not obs.exception_raised, (
        f"CE-REQ-EXPL-CONJ-API-001: add_conjunctions on individual AlternativeExplanation raised "
        f"{obs.exception_type}"
    )


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-CONJ-RETURN-001 — Return type and cardinality contract
# ---------------------------------------------------------------------------


def test_should_preserve_cardinality_when_factual_collection_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-RETURN-001: factual collection cardinality preserved.

    Acceptance criteria:
    - observation.result_is_none is False.
    - observation.result_len == observation.n_instances.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=2,
        n_top_features=5,
    )

    assert (
        not obs.result_is_none
    ), "CE-REQ-EXPL-CONJ-RETURN-001: add_conjunctions on factual collection must return non-None"
    assert obs.result_len == obs.n_instances, (
        f"CE-REQ-EXPL-CONJ-RETURN-001: len(result)={obs.result_len} != "
        f"n_instances={obs.n_instances}"
    )


def test_should_preserve_cardinality_when_alternative_collection_add_conjunctions():
    """Verify CE-REQ-EXPL-CONJ-RETURN-001: alternative collection cardinality preserved.

    Acceptance criteria:
    - observation.result_is_none is False.
    - observation.result_len == observation.n_instances.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="alternative",
        object_level="collection",
        max_rule_size=2,
        n_top_features=5,
    )

    assert not obs.result_is_none, "CE-REQ-EXPL-CONJ-RETURN-001: add_conjunctions on alternative collection must return non-None"
    assert obs.result_len == obs.n_instances, (
        f"CE-REQ-EXPL-CONJ-RETURN-001: len(result)={obs.result_len} != "
        f"n_instances={obs.n_instances}"
    )


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-CONJ-RULE-001 — Multi-feature conjunction rule semantics
# ---------------------------------------------------------------------------


def test_should_produce_conjunctive_rules_when_max_rule_size_two():
    """Verify CE-REQ-EXPL-CONJ-RULE-001: multi-feature rules produced when max_rule_size=2.

    Fixture has n_informative=3 and n_features=4, sufficient for conjunction generation.

    Acceptance criterion:
    - observation.any_has_conjunctive_rules is True.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=2,
        n_top_features=5,
    )

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-CONJ-RULE-001: unexpected exception {obs.exception_type}"
    assert obs.any_has_conjunctive_rules, (
        "CE-REQ-EXPL-CONJ-RULE-001: expected at least one item to have "
        "has_conjunctive_rules == True when max_rule_size=2 and n_informative=3, "
        "but no conjunctive rules were produced. This may indicate a regression "
        "in conjunction generation behavior."
    )


def test_should_produce_conjunctive_rules_when_max_rule_size_three():
    """Verify CE-REQ-EXPL-CONJ-RULE-001: multi-feature rules produced when max_rule_size=3.

    max_rule_size >= 2 covers the minimum case (2) and larger values (3).
    Fixture has n_informative=3 and n_features=4, sufficient for 3-feature conjunction rules.

    Acceptance criterion:
    - observation.any_has_conjunctive_rules is True.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=3,
        n_top_features=5,
    )

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-CONJ-RULE-001: unexpected exception {obs.exception_type}"
    assert obs.any_has_conjunctive_rules, (
        "CE-REQ-EXPL-CONJ-RULE-001: expected at least one item to have "
        "has_conjunctive_rules == True when max_rule_size=3 and n_informative=3, "
        "but no conjunctive rules were produced."
    )


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-CONJ-PARAM-001 — max_rule_size=1 suppresses multi-feature rules
# ---------------------------------------------------------------------------


def test_should_not_produce_conjunctive_rules_when_max_rule_size_one():
    """Verify CE-REQ-EXPL-CONJ-PARAM-001: max_rule_size=1 suppresses multi-feature conjunctions.

    Acceptance criteria:
    - observation.any_has_conjunctive_rules is False.
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=1,
        n_top_features=5,
    )

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-CONJ-PARAM-001: unexpected exception {obs.exception_type}"
    assert not obs.any_has_conjunctive_rules, (
        "CE-REQ-EXPL-CONJ-PARAM-001: max_rule_size=1 must not produce multi-feature "
        "conjunction rules, but has_conjunctive_rules was True for at least one item."
    )


@pytest.mark.parametrize(
    "max_rule_size,expected_conjunctions",
    [
        (1, False),
        (2, True),
        (3, True),
    ],
)
def test_should_control_conjunction_generation_via_max_rule_size(
    max_rule_size, expected_conjunctions
):
    """Verify CE-REQ-EXPL-CONJ-RULE-001 and CE-REQ-EXPL-CONJ-PARAM-001 via parametrize.

    max_rule_size=1 → no conjunctions; max_rule_size=2 or 3 → conjunctions produced.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=max_rule_size,
        n_top_features=5,
    )

    assert (
        not obs.exception_raised
    ), f"Unexpected exception {obs.exception_type} for max_rule_size={max_rule_size}"
    assert obs.any_has_conjunctive_rules == expected_conjunctions, (
        f"max_rule_size={max_rule_size}: expected any_has_conjunctive_rules="
        f"{expected_conjunctions}, got {obs.any_has_conjunctive_rules}"
    )


# ---------------------------------------------------------------------------
# Additional parameter coverage (n_top_features variant — CE-REQ-EXPL-CONJ-API-001)
# ---------------------------------------------------------------------------


def test_should_not_raise_when_individual_with_non_default_n_top_features():
    """Verify CE-REQ-EXPL-CONJ-API-001: add_conjunctions with n_top_features=2, max_rule_size=2.

    Acceptance criterion:
    - observation.exception_raised is False.
    """
    obs = run_conjunction_tif_scenario(
        explanation_mode="alternative",
        object_level="individual",
        max_rule_size=2,
        n_top_features=2,
    )

    assert not obs.exception_raised, (
        f"CE-REQ-EXPL-CONJ-API-001: add_conjunctions(n_top_features=2, max_rule_size=2) "
        f"raised {obs.exception_type}"
    )
