# CE-REQ-EXPL-CONJ-API-001 — Conjunction API Availability

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | active |
| verification_status | verified |
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) and individual (FactualExplanation, AlternativeExplanation) |
| supersedes | CE-REQ-EXPL-CONJ-001 (partial) |
| tif_refs | CE-TIF-EXPL-CONJ-001 |
| verification_strength | api_contract |
| evidence_level | raw_evidence |

## Scope

Public API:
- `CalibratedExplanations.add_conjunctions(n_top_features, max_rule_size)` — factual collection
- `AlternativeExplanations.add_conjunctions(n_top_features, max_rule_size)` — alternative collection
- `FactualExplanation.add_conjunctions(n_top_features, max_rule_size)` — individual factual
- `AlternativeExplanation.add_conjunctions(n_top_features, max_rule_size)` — individual alternative

Applicable task types: binary classification, multiclass classification, regression, probabilistic_regression.

Applicable workflow: the TIF `CE-TIF-EXPL-CONJ-001` `run_conjunction_tif_scenario()` function, which
exercises `WrapCalibratedExplainer` → `fit` → `calibrate` → `explain_factual` / `explore_alternatives`
→ `add_conjunctions`.

## Observable behavior

`add_conjunctions()` is callable on explanation objects produced through a valid
`WrapCalibratedExplainer` workflow. The call completes without raising an exception
for valid inputs (valid parameters, fitted and calibrated explainer, non-empty dataset).

## Acceptance criterion

For every object type listed in `applicable_on`, and for every task type listed in scope:

1. `observation.exception_raised` is `False`.
2. No exception is raised during the TIF scenario execution.

This criterion is verified by CE-TIF-EXPL-CONJ-001 with default parameters
(`max_rule_size=2`, `n_top_features=5`) across collection and individual object levels.

## TIF reference

**TIF ID:** CE-TIF-EXPL-CONJ-001

TIF scenario function: `run_conjunction_tif_scenario()`
Observation field verified: `exception_raised` (must be `False`)

## Verification method

Automated pytest tests calling CE-TIF-EXPL-CONJ-001, asserting `exception_raised == False`.

## Verification targets

- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_not_raise_when_factual_collection_add_conjunctions`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_not_raise_when_alternative_collection_add_conjunctions`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_not_raise_when_individual_factual_add_conjunctions`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_not_raise_when_individual_alternative_add_conjunctions`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| tif_ids | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies API availability only. It does not verify:
- That `add_conjunctions` produces semantically meaningful rules.
- That conjunction rules combine multiple features (see CE-REQ-EXPL-CONJ-RULE-001).
- That `max_rule_size=1` suppresses multi-feature rules (see CE-REQ-EXPL-CONJ-PARAM-001).
- The return type contract (see CE-REQ-EXPL-CONJ-RETURN-001).
- Any statistical or calibration properties.
