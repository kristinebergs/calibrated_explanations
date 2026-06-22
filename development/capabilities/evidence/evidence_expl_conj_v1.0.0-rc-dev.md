# Curated Evidence — EXPL-CONJ Chain

**Capability:** CE-CAP-EXPL-CONJ-001 — Conjunctive multi-feature explanation rules
**Milestone:** v1.0.0-rc-dev
**Date:** 2026-06-22
**Status:** PASS (4 of 5 requirements verified; 1 TIF-exempt, documented gap)

---

## Requirements covered

| Requirement ID | Obligation type | Verification strength | Result |
|---|---|---|---|
| CE-REQ-EXPL-CONJ-API-001 | api_contract | api_contract | PASS |
| CE-REQ-EXPL-CONJ-RETURN-001 | api_contract | api_contract | PASS |
| CE-REQ-EXPL-CONJ-RULE-001 | behavioral_contract | behavioral_contract | PASS |
| CE-REQ-EXPL-CONJ-PARAM-001 | behavioral_contract | behavioral_contract | PASS |
| CE-REQ-EXPL-CONJ-DOC-001 | documentation_boundary | — | not_implemented (TIF-exempt, gap: ADR-008) |

## TIF interfaces exercised

| TIF ID | File |
|---|---|
| CE-TIF-EXPL-CONJ-001 | `development/capabilities/verification/tif/tif_conjunction.py` |

## Raw evidence records

| Evidence ID | Requirement | Result | File |
|---|---|---|---|
| CE-EVID-EXPL-CONJ-API-001-20260622 | CE-REQ-EXPL-CONJ-API-001 | pass | `reports/verification/CE-EVID-EXPL-CONJ-API-001-20260622.json` |
| CE-EVID-EXPL-CONJ-RETURN-001-20260622 | CE-REQ-EXPL-CONJ-RETURN-001 | pass | `reports/verification/CE-EVID-EXPL-CONJ-RETURN-001-20260622.json` |
| CE-EVID-EXPL-CONJ-RULE-001-20260622 | CE-REQ-EXPL-CONJ-RULE-001 | pass | `reports/verification/CE-EVID-EXPL-CONJ-RULE-001-20260622.json` |
| CE-EVID-EXPL-CONJ-PARAM-001-20260622 | CE-REQ-EXPL-CONJ-PARAM-001 | pass | `reports/verification/CE-EVID-EXPL-CONJ-PARAM-001-20260622.json` |

## Test IDs that passed

### Contract tests (`tests/capabilities/test_conjunction_contracts.py`)

| Test ID | Requirement |
|---|---|
| `test_should_not_raise_when_factual_collection_add_conjunctions` | CE-REQ-EXPL-CONJ-API-001 |
| `test_should_not_raise_when_alternative_collection_add_conjunctions` | CE-REQ-EXPL-CONJ-API-001 |
| `test_should_not_raise_when_individual_factual_add_conjunctions` | CE-REQ-EXPL-CONJ-API-001 |
| `test_should_not_raise_when_individual_alternative_add_conjunctions` | CE-REQ-EXPL-CONJ-API-001 |
| `test_should_not_raise_when_individual_with_non_default_n_top_features` | CE-REQ-EXPL-CONJ-API-001 |
| `test_should_preserve_cardinality_when_factual_collection_add_conjunctions` | CE-REQ-EXPL-CONJ-RETURN-001 |
| `test_should_preserve_cardinality_when_alternative_collection_add_conjunctions` | CE-REQ-EXPL-CONJ-RETURN-001 |
| `test_should_produce_conjunctive_rules_when_max_rule_size_two` | CE-REQ-EXPL-CONJ-RULE-001 |
| `test_should_produce_conjunctive_rules_when_max_rule_size_three` | CE-REQ-EXPL-CONJ-RULE-001 |
| `test_should_not_produce_conjunctive_rules_when_max_rule_size_one` | CE-REQ-EXPL-CONJ-PARAM-001 |
| `test_should_control_conjunction_generation_via_max_rule_size[1-False]` | CE-REQ-EXPL-CONJ-PARAM-001 |
| `test_should_control_conjunction_generation_via_max_rule_size[2-True]` | CE-REQ-EXPL-CONJ-RULE-001 |
| `test_should_control_conjunction_generation_via_max_rule_size[3-True]` | CE-REQ-EXPL-CONJ-RULE-001 |

### TIF policy tests (`tests/capabilities/test_tif_policy.py`)

| Test ID | What it enforces |
|---|---|
| `test_tif_should_import_wrap_calibrated_explainer[tif_conjunction.py]` | TIF must use public CE entry point |
| `test_tif_should_not_import_calibrated_explainer_directly[tif_conjunction.py]` | No direct core import |
| `test_tif_should_not_construct_explanation_objects_directly[tif_conjunction.py]` | No internal construction |
| `test_tif_should_not_access_private_members[tif_conjunction.py]` | No `._attr` access on CE objects |
| `test_tif_directory_should_have_readme` | TIF directory has spec index |
| `test_tif_python_file_should_have_corresponding_spec[tif_conjunction.py]` | Each TIF .py has a CE-TIF-*.md spec |

## Scenario detail

### CE-REQ-EXPL-CONJ-API-001 — callability (4 scenarios, all pass)

`add_conjunctions()` was called without raising an exception across all
`(explanation_mode, object_level)` combinations:

| Scenario | Parameters | Result |
|---|---|---|
| `api_factual_collection` | factual / collection / max_rule_size=2 | pass |
| `api_alternative_collection` | alternative / collection / max_rule_size=2 | pass |
| `api_factual_individual` | factual / individual / max_rule_size=2 | pass |
| `api_alternative_individual` | alternative / individual / max_rule_size=2 | pass |

### CE-REQ-EXPL-CONJ-RETURN-001 — cardinality contract (2 scenarios, all pass)

At `object_level=collection`, `add_conjunctions()` returns a non-None result
whose length equals `n_instances` (observed: `result_len=3`, `n_instances=3`
for both factual and alternative modes).

### CE-REQ-EXPL-CONJ-RULE-001 — behavioral contract: conjunction rules produced (2 scenarios, all pass)

With a dataset where `n_informative=3`, at least one explanation in the collection
had `has_conjunctive_rules=True` after `add_conjunctions()` for both covered values:

| Scenario | max_rule_size | Observed |
|---|---|---|
| `rule_factual_collection_max_rule_size_2` | 2 | `any_has_conjunctive_rules == True` |
| `rule_factual_collection_max_rule_size_3` | 3 | `any_has_conjunctive_rules == True` |

This confirms the requirement holds for `max_rule_size >= 2`, not only the minimum value.

### CE-REQ-EXPL-CONJ-PARAM-001 — behavioral contract: max_rule_size=1 suppresses conjunctions (1 scenario, pass)

With `max_rule_size=1`, no explanation had `has_conjunctive_rules=True`.
Observed: `any_has_conjunctive_rules == False`, `exception_raised == False`.

## Assumption boundary

This evidence explicitly does NOT prove:

- **Semantic correctness**: `api_contract` evidence proves callability, not that the rules
  produced are meaningful or well-calibrated explanations.
- **Calibration validity**: The observations are structural (non-None, correct length,
  boolean flag presence). Calibration quality requires separate verification.
- **Statistical guarantees**: The `any_has_conjunctive_rules` behavioral check used a
  fixed synthetic dataset (`n_informative=3`, `random_state=42`). It does not prove
  guarantees on real datasets with different feature structures or distributions.
- **Finite-sample bounds**: No theoretical bound on conjunction quality or rule coverage
  is verified here.
- **Documentation correctness** (CE-REQ-EXPL-CONJ-DOC-001): The documentation boundary
  requirement is TIF-exempt and not_implemented. Formal documentation of conjunction
  semantics and limitations in ADR-008 is an open gap.
- **Individual-level cardinality**: RETURN-001 was verified at `object_level=collection`
  only. Individual-level return cardinality is confirmed by the API tests (no exception)
  but not by a dedicated cardinality assertion.

## Verification metadata

| Field | Value |
|---|---|
| claim_id | CE-CAP-EXPL-CONJ-001 |
| adr_ref | ADR-008 |
| tif_id | CE-TIF-EXPL-CONJ-001 |
| evidence_level | raw_evidence + curated_summary |
| commit_sha | f1af9628a4c4f5073f380592070b1773ce9e16ff |
| package_version | 0.11.3.dev0 (installed; source milestone v1.0.0-rc-dev) |
| python_version | 3.14.4 |
| platform | Windows-11-10.0.26200-SP0 |
| dataset | sklearn make_classification n_samples=120 n_features=4 n_informative=3 n_redundant=1 random_seed=42 |
| random_seed | 42 |
| generator | `python scripts/generate_capability_evidence.py` |
| tests_run | 19 (13 contract + 6 TIF policy) |
| tests_passed | 19 |
| tests_failed | 0 |

## Open gap

| Gap | Requirement | Status |
|---|---|---|
| ADR-008 conjunction semantics not documented | CE-REQ-EXPL-CONJ-DOC-001 | not_implemented |
