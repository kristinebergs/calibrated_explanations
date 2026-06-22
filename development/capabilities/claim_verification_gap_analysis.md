# CE Capability Claim Verification Gap Analysis

**Date:** 2026-06-22
**Status:** Active - synchronized after TIF evidence hardening
**Task:** Capability-verification drift and evidence-hardening pass

## Purpose

This document tracks CE capability claims against the verification chain:

```text
ADR / Standard -> capability claim -> requirement -> TIF verification interface -> test / verification execution -> raw evidence -> summary evidence
```

The goal of this inventory is to distinguish what is closed, still open,
deferred with rationale, and not applicable. Repository state is authoritative;
this document should not retain stale `NONE` entries when TIF specs, tests, or
evidence records exist.

## Classification

| Status | Meaning |
|---|---|
| closed | TIF, tests, and raw evidence exist for the scoped behavioral requirement set. |
| still open | A behavioral requirement still lacks executable verification or evidence. |
| deferred with rationale | A known gap is intentionally outside this hardening pass and has a stated reason. |
| not applicable | The requirement is governance, documentation, schema, or policy oriented and is TIF-exempt or verified outside the behavioral TIF layer. |

## Closed Behavioral Chains

| Claim | Requirements covered | TIF | Tests | Raw evidence |
|---|---|---|---|---|
| CE-CAP-EXPL-001 | CE-REQ-EXPL-API-001, CE-REQ-EXPL-RETURN-001 | CE-TIF-EXPL-001 (`tif_explanation.py`) | `tests/capabilities/test_explanation_contracts.py` | `reports/verification/CE-EVID-EXPL-001-*.json` |
| CE-CAP-EXPL-002 | CE-REQ-EXPL-API-002, CE-REQ-EXPL-ALT-RETURN-001 | CE-TIF-EXPL-001 (`tif_explanation.py`) | `tests/capabilities/test_explanation_contracts.py` | `reports/verification/CE-EVID-EXPL-001-*.json` |
| CE-CAP-EXPL-CONJ-001 | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, CE-REQ-EXPL-CONJ-PARAM-001 | CE-TIF-EXPL-CONJ-001 (`tif_conjunction.py`) | `tests/capabilities/test_conjunction_contracts.py` | `reports/verification/CE-EVID-EXPL-CONJ-001-*.json` |
| CE-CAP-EXPL-FILTER-001 | CE-REQ-EXPL-FILTER-SUPER-001, CE-REQ-EXPL-FILTER-SEMI-001, CE-REQ-EXPL-FILTER-COUNTER-001, CE-REQ-EXPL-FILTER-ENSURED-001, CE-REQ-EXPL-FILTER-PARETO-001 | CE-TIF-FILTER-001 (`tif_filter.py`) | `tests/capabilities/test_filter_contracts.py` | `reports/verification/CE-EVID-FILTER-001-*.json` |
| CE-CAP-GUARD-001 | CE-REQ-GUARD-API-001 | CE-TIF-GUARD-001 (`tif_guard.py`) | `tests/capabilities/test_guard_contracts.py` | `reports/verification/CE-EVID-GUARD-001-*.json` |
| CE-CAP-MOND-001 | CE-REQ-MOND-API-001 | CE-TIF-MOND-001 (`tif_mondrian.py`) | `tests/capabilities/test_mondrian_contracts.py` | `reports/verification/CE-EVID-MOND-001-*.json` |
| CE-CAP-NARR-001 | CE-REQ-NARR-API-001 | CE-TIF-NARR-001 (`tif_narrative.py`) | `tests/capabilities/test_narrative_contracts.py` | `reports/verification/CE-EVID-NARR-001-*.json` |
| CE-CAP-PRED-001 | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 | CE-TIF-PRED-001 (`tif_prediction.py`) | `tests/capabilities/test_prediction_contracts.py` | `reports/verification/CE-EVID-PRED-001-*.json` |
| CE-CAP-PRED-CLASS-001 | CE-REQ-PRED-CLASS-API-001, CE-REQ-PRED-CLASS-BOUNDS-001 | CE-TIF-PRED-CLASS-001 (`tif_classification.py`) | `tests/capabilities/test_classification_contracts.py` | `reports/verification/CE-EVID-PRED-CLASS-001-*.json` |
| CE-CAP-PRED-PROB-001 | CE-REQ-PRED-PROB-API-001, CE-REQ-PRED-PROB-BOUNDS-001 | CE-TIF-PRED-PROB-001 (`tif_prob_regression.py`) | `tests/capabilities/test_probabilistic_regression_contracts.py` | `reports/verification/CE-EVID-PRED-PROB-001-*.json` |
| CE-CAP-REJECT-001 | CE-REQ-REJECT-API-001 | CE-TIF-REJECT-001 (`tif_reject.py`) | `tests/capabilities/test_reject_policy_contracts.py` | `reports/verification/CE-EVID-REJECT-001-*.json` |
| CE-CAP-VIZ-001 | CE-REQ-VIZ-SMOKE-001 | CE-TIF-VIZ-001 (`tif_visualization.py`) | `tests/capabilities/test_visualization_contracts.py` | `reports/verification/CE-EVID-VIZ-001-*.json` |

## Still Open

| Gap | Status | Rationale / next action |
|---|---|---|
| Documentation-boundary requirements for factual and alternative explanations | still open | CE-REQ-EXPL-DOC-001 and CE-REQ-EXPL-ALT-DOC-001 are TIF-exempt documentation-boundary requirements. They need documentation review evidence, not a behavioral TIF. |
| CE-REQ-EXPL-CONJ-DOC-001 | still open | TIF-exempt documentation-boundary requirement. Curated EXPL-CONJ evidence records it as not implemented. |

## Deferred With Rationale

| Gap | Status | Rationale |
|---|---|---|
| Guarded explanation semantic filtering strength beyond API callability | deferred with rationale | Current CE-TIF-GUARD-001 verifies the public guarded-options API path. Stronger semantic in-distribution guarantees are outside this pass and require separate ADR-032 scoped criteria. |
| Reject/defer tag semantics beyond API callability | deferred with rationale | Current CE-TIF-REJECT-001 verifies the public reject policy call path. Deeper rejection tag semantics remain governed by ADR-029 follow-up scope. |
| Statistical validity of intervals and probabilities | deferred with rationale | Current TIFs verify API, structural, numerical, and empirical-smoke contracts only. They do not prove finite-sample or distribution-shift guarantees. |

## Not Applicable / Governance Claims

Governance and policy claims whose requirements verify repository structure,
configuration, CI policy, schemas, documentation, or metadata linkage are not
behavioral TIF chains. They are validated through policy scripts, unit or
integration tests, schema checks, or TIF exemptions as appropriate.

| Claim family | Status | Verification route |
|---|---|---|
| Distribution, deprecation, test, config, schema, core-boundary, docs, validity, CI, modality governance | not applicable | Repository policy, schema, metadata-linkage, or quality-gate checks. |
| Plugin documentation boundary | not applicable | Documentation/static importability checks with TIF exemption. |
| Preprocessing, observability, parallel, cache, PlotSpec, serialization governance | deferred with rationale | Existing non-capability tests cover parts of these areas; dedicated behavioral TIF layering is outside this hardening pass unless a future task scopes it explicitly. |

## Maintenance Rules

- Update `development/capabilities/verification/tif/README.md` whenever active TIF specs change.
- Regenerate raw evidence with `python scripts/generate_tif_evidence.py` after changing TIF behavior.
- Use `python scripts/generate_tif_evidence.py --check-current` at release closure when raw evidence must match the release commit.
- Do not leave stale `NONE` entries for TIF, tests, or evidence when files exist.
