# Capability Verification Evidence - TIF Architecture Pass

**Task:** Full CE capability-verification pass - TIF architecture introduction
**Date:** 2026-06-22
**Status:** CLOSED

---

## Gate command and result

```
pytest tests/capabilities/ --no-cov
```

**Result:** 119 passed, 0 failed

---

## Verification metadata

| Field | Value |
|---|---|
| package_version | calibrated_explanations v1.0.0-rc-dev |
| commit_sha | ba0f95e1 |
| test_command | `pytest tests/capabilities/ --no-cov` |
| tests_run | 119 |
| tests_passed | 119 |
| tests_failed | 0 |
| dataset | sklearn make_classification / make_regression (seed=42, in-test, no disk reads) |
| random_seed | 42 |
| date | 2026-06-22 |
| python_version | 3.14.4 |
| platform | win32 |

---

## Scope of this evidence record

This record documents the closure of the full CE capability-verification pass that
introduced the TIF (Test Interface Framework) architecture across all CE capability
claims. It covers:

1. **New TIF interfaces** - 10 new TIF Python files and MD specs created
2. **New behavioral requirements** - 8 new requirement files decomposed from claims
3. **tif_refs additions** - 15 existing requirement files updated with tif_refs
4. **New capability tests** - 4 new test functions added via TIF layer
5. **Phase 8 validation test** - new Rule 7 in test_tif_policy.py

---

## TIF interfaces introduced

| TIF ID | Python file | Requirements served |
|---|---|---|
| CE-TIF-EXPL-001 | `tif_explanation.py` | CE-REQ-EXPL-API-001, CE-REQ-EXPL-RETURN-001, CE-REQ-EXPL-API-002, CE-REQ-EXPL-ALT-RETURN-001 |
| CE-TIF-PRED-001 | `tif_prediction.py` | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 |
| CE-TIF-PRED-CLASS-001 | `tif_classification.py` | CE-REQ-PRED-CLASS-API-001, CE-REQ-PRED-CLASS-BOUNDS-001 |
| CE-TIF-PRED-PROB-001 | `tif_prob_regression.py` | CE-REQ-PRED-PROB-API-001, CE-REQ-PRED-PROB-BOUNDS-001 |
| CE-TIF-GUARD-001 | `tif_guard.py` | CE-REQ-GUARD-API-001 |
| CE-TIF-REJECT-001 | `tif_reject.py` | CE-REQ-REJECT-API-001 |
| CE-TIF-MOND-001 | `tif_mondrian.py` | CE-REQ-MOND-API-001 |
| CE-TIF-NARR-001 | `tif_narrative.py` | CE-REQ-NARR-API-001 |
| CE-TIF-VIZ-001 | `tif_visualization.py` | CE-REQ-VIZ-SMOKE-001 |
| CE-TIF-FILTER-001 | `tif_filter.py` | CE-REQ-EXPL-FILTER-SUPER/SEMI/COUNTER/ENSURED/PARETO-001 |

Pre-existing TIF (reference pattern, unchanged):

| TIF ID | Python file | Requirements served |
|---|---|---|
| CE-TIF-EXPL-CONJ-001 | `tif_conjunction.py` | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-PARAM-001, CE-REQ-EXPL-CONJ-RULE-001 |

---

## New requirement files

| Requirement ID | Obligation type | Claim | TIF |
|---|---|---|---|
| CE-REQ-EXPL-RETURN-001 | output_contract | CE-CAP-EXPL-001 | CE-TIF-EXPL-001 |
| CE-REQ-EXPL-DOC-001 | documentation_boundary | CE-CAP-EXPL-001 | tif_exemption: documentation_boundary |
| CE-REQ-EXPL-ALT-RETURN-001 | output_contract | CE-CAP-EXPL-002 | CE-TIF-EXPL-001 |
| CE-REQ-EXPL-ALT-DOC-001 | documentation_boundary | CE-CAP-EXPL-002 | tif_exemption: documentation_boundary |
| CE-REQ-PRED-CLASS-BOUNDS-001 | output_contract | CE-CAP-PRED-CLASS-001 | CE-TIF-PRED-CLASS-001 |
| CE-REQ-PRED-PROB-BOUNDS-001 | output_contract | CE-CAP-PRED-PROB-001 | CE-TIF-PRED-PROB-001 |

---

## New capability tests (Phase 6 - TIF-layer tests)

| Test function | File | Requirements verified |
|---|---|---|
| `test_should_preserve_cardinality_when_factual_explain` | `test_explanation_contracts.py` | CE-REQ-EXPL-RETURN-001 |
| `test_should_return_accessible_feature_weights_when_factual_explain` | `test_explanation_contracts.py` | CE-REQ-EXPL-RETURN-001 |
| `test_should_preserve_cardinality_when_alternative_explain` | `test_explanation_contracts.py` | CE-REQ-EXPL-ALT-RETURN-001 |
| `test_should_return_alternative_explanations_type_when_explore_alternatives` | `test_explanation_contracts.py` | CE-REQ-EXPL-ALT-RETURN-001 |

---

## Phase 8 validation test added

**File:** `tests/capabilities/test_tif_policy.py`
**Rule:** Rule 7 - `test_capability_requirements_should_declare_tif_refs_or_exemption`

This test enforces that every requirement citing a `tests/capabilities/` target must
declare either `tif_refs` (pointing to a CE-TIF interface) or `tif_exemption` (for
requirements that cannot be exercised through WrapCalibratedExplainer).

---

## Documentation gaps recorded

| Requirement | Gap type | Intended closure |
|---|---|---|
| CE-REQ-EXPL-DOC-001 | documentation_boundary; not_implemented | Manual review v0.12.x |
| CE-REQ-EXPL-ALT-DOC-001 | documentation_boundary; not_implemented | Manual review v0.12.x |
| CE-REQ-EXPL-CONJ-DOC-001 | documentation_boundary; not_implemented | Manual review v0.12.x |

All three are registered in `development/current-work/RELEASE_PLAN_status_appendix.md`
under the TIF architecture gap inventory.

---

## Claim decomposition summary

All 30 CE capability claims were inventoried and assessed:

- **18 governance/policy claims** - each given `atomic_rationale` (one-to-one claim->requirement mapping justified as governance scope)
- **12 behavioral claims** - decomposed into requirements (most have 2+ requirements)
- **All behavioral claims** now have =>1 requirement with tif_refs pointing to a TIF interface
- **Documentation boundary requirements** carry tif_exemption instead of tif_refs

---

## Assumption boundary

This evidence record documents the structural completeness of the TIF architecture
and test pass. It does NOT verify:

- Statistical calibration validity of any CE output
- Numerical accuracy of feature weights or prediction intervals beyond the structural bounds verified
- Narrative quality, visual correctness, or domain correctness of outputs
- Finite-sample theoretical coverage guarantees from conformal prediction
- Plugin behavior beyond importability and API contract compliance

---

## Raw evidence records

Raw evidence JSON files produced by executing the TIF scenarios (via
`python scripts/generate_tif_evidence.py`) are in `reports/verification/`:

| Evidence ID | TIF | Requirements covered | Result |
|---|---|---|---|
| CE-EVID-EXPL-001-20260622 | CE-TIF-EXPL-001 | CE-REQ-EXPL-API-001, CE-REQ-EXPL-RETURN-001, CE-REQ-EXPL-API-002, CE-REQ-EXPL-ALT-RETURN-001 | pass |
| CE-EVID-PRED-001-20260622 | CE-TIF-PRED-001 | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 | pass |
| CE-EVID-PRED-CLASS-001-20260622 | CE-TIF-PRED-CLASS-001 | CE-REQ-PRED-CLASS-API-001, CE-REQ-PRED-CLASS-BOUNDS-001 | pass |
| CE-EVID-PRED-PROB-001-20260622 | CE-TIF-PRED-PROB-001 | CE-REQ-PRED-PROB-API-001, CE-REQ-PRED-PROB-BOUNDS-001 | pass |
| CE-EVID-GUARD-001-20260622 | CE-TIF-GUARD-001 | CE-REQ-GUARD-API-001 | pass |
| CE-EVID-REJECT-001-20260622 | CE-TIF-REJECT-001 | CE-REQ-REJECT-API-001 | pass |
| CE-EVID-MOND-001-20260622 | CE-TIF-MOND-001 | CE-REQ-MOND-API-001 | pass |
| CE-EVID-NARR-001-20260622 | CE-TIF-NARR-001 | CE-REQ-NARR-API-001 | pass |
| CE-EVID-VIZ-001-20260622 | CE-TIF-VIZ-001 | CE-REQ-VIZ-SMOKE-001 | pass |
| CE-EVID-FILTER-001-20260622 | CE-TIF-FILTER-001 | CE-REQ-EXPL-FILTER-{SUPER,SEMI,COUNTER,ENSURED,PARETO}-001 | pass |

Pre-existing CONJ evidence (reference pattern, produced in the prior pass):

| Evidence ID | Requirements covered | Result |
|---|---|---|

Capability-test gate run (structural / policy evidence):

```
pytest tests/capabilities/ --no-cov
119 passed in 3.41s
```

Test session date: 2026-06-22. Commit: ba0f95e1. Package: v1.0.0-rc-dev.
