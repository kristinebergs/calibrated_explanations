# CE Capability Claim Verification Gap Analysis

**Date:** 2026-06-22
**Status:** Active — produced by capability-verification pass
**Task:** Full CE capability-verification pass (post v1.0.0-rc, EXPL-CONJ as reference)

---

## Purpose

This document captures the complete inventory of all CE capability claims, their
current verification status, and concrete gaps. It drives Phases 2–8 of the
capability-verification pass.

The EXPL-CONJ chain (`CE-CAP-EXPL-CONJ-001`) is the reference model. Every other
behavioral claim is measured against it.

---

## Reference model: EXPL-CONJ

The completed EXPL-CONJ chain demonstrates the intended architecture:

```
CE-CAP-EXPL-CONJ-001 (claim)
  → CE-REQ-EXPL-CONJ-API-001    (api_contract)       → CE-TIF-EXPL-CONJ-001 ✓
  → CE-REQ-EXPL-CONJ-RETURN-001 (output_contract)    → CE-TIF-EXPL-CONJ-001 ✓
  → CE-REQ-EXPL-CONJ-RULE-001   (semantic_behavior)  → CE-TIF-EXPL-CONJ-001 ✓
  → CE-REQ-EXPL-CONJ-PARAM-001  (parameter_behavior) → CE-TIF-EXPL-CONJ-001 ✓
  → CE-REQ-EXPL-CONJ-DOC-001    (documentation_boundary) → tif_exemption ✓
TIF: development/capabilities/verification/tif/tif_conjunction.py
Tests: tests/capabilities/test_conjunction_contracts.py
Raw evidence: reports/verification/CE-EVID-EXPL-CONJ-*-20260622.json
```

---

## Classification key

| Category | Meaning |
|---|---|
| A | Already matches the new model |
| B | Claim text too detailed — needs generalization |
| C | Claim too broad — needs splitting |
| D | Valid claim — needs multiple requirements |
| E | Requirements too weak or too test-like |
| F | Requirements exist but no TIF |
| G | Tests exist but no raw/curated evidence |
| H | No executable verification |
| I | Unsupported, roadmap, stale |

---

## Complete claim inventory

### CE-CAP-EXPL-CONJ-001 — Conjunctive multi-feature explanation rules

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-008 |
| Requirements | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, CE-REQ-EXPL-CONJ-PARAM-001, CE-REQ-EXPL-CONJ-DOC-001 |
| TIF | CE-TIF-EXPL-CONJ-001 (tif_conjunction.py) |
| Tests | test_conjunction_contracts.py |
| Raw evidence | CE-EVID-EXPL-CONJ-*-20260622.json ✓ |
| **Category** | **A — Already matches new model** |

No action required.

---

### CE-CAP-EXPL-001 — Factual explanations

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-008, ADR-015, ADR-026 |
| Requirements | CE-REQ-EXPL-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_explanation_contracts.py (direct WrapCalibratedExplainer, no TIF) |
| Raw evidence | NONE |
| **Category** | **D+F — valid claim needing multiple requirements; requirement has no TIF** |

**Issues:**
1. One-to-one claim-to-requirement with no `atomic_rationale` field.
2. CE-REQ-EXPL-API-001 has no `tif_refs`.
3. Claim text mentions specific return details ("calibrated probability uncertainty") — borderline too detailed.
4. Missing requirement dimensions: output_contract, documentation_boundary.

**Required actions:**
- Add `atomic_rationale` to CE-CAP-EXPL-001 OR decompose into multiple requirements.
- Decision: Decompose. Add CE-REQ-EXPL-RETURN-001 (output_contract) and CE-REQ-EXPL-DOC-001 (documentation_boundary).
- Create CE-TIF-EXPL-001 and tif_explanation.py.
- Update CE-REQ-EXPL-API-001 to add `tif_refs: CE-TIF-EXPL-001`.
- Update CE-CAP-EXPL-001 to list new requirements.
- Refactor test_explanation_contracts.py to call TIF.
- Document raw evidence gap.

---

### CE-CAP-EXPL-002 — Alternative (counterfactual-style) explanations

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-008, ADR-015, ADR-026 |
| Requirements | CE-REQ-EXPL-API-002 (1 requirement) |
| TIF | NONE |
| Tests | test_explanation_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **D+F — same as CE-CAP-EXPL-001** |

**Required actions:**
- Decompose. Add CE-REQ-EXPL-ALT-RETURN-001 (output_contract) and CE-REQ-EXPL-ALT-DOC-001 (documentation_boundary).
- CE-TIF-EXPL-001 covers both factual and alternative (explanation_mode parameter).
- Update CE-REQ-EXPL-API-002 to add `tif_refs: CE-TIF-EXPL-001`.
- Update CE-CAP-EXPL-002 to list new requirements.
- Document raw evidence gap.

---

### CE-CAP-EXPL-FILTER-001 — Alternative explanation filter operations

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-027 |
| Requirements | CE-REQ-EXPL-FILTER-SUPER-001, CE-REQ-EXPL-FILTER-SEMI-001, CE-REQ-EXPL-FILTER-COUNTER-001, CE-REQ-EXPL-FILTER-ENSURED-001, CE-REQ-EXPL-FILTER-PARETO-001 |
| TIF | NONE |
| Tests | test_filter_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+F — claim text too detailed; 5 requirements exist but no TIF** |

**Issues:**
1. Claim text is too detailed — it names and describes all 5 filter operations with their semantics. These details belong in requirements, not the claim.
2. All 5 filter requirements have no `tif_refs`.
3. No raw evidence.

**Required actions:**
- Simplify claim text to capability level.
- Create CE-TIF-FILTER-001 and tif_filter.py covering all 5 operations.
- Update all 5 filter requirements to add `tif_refs: CE-TIF-FILTER-001`.
- Refactor test_filter_contracts.py to use TIF.
- Document raw evidence gap.

---

### CE-CAP-GUARD-001 — Guarded explanations

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-032, ADR-038 |
| Requirements | CE-REQ-GUARD-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_guard_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **D+F — needs behavioral requirement + TIF** |

**Issues:**
1. One-to-one with no `atomic_rationale`.
2. CE-REQ-GUARD-API-001 has no `tif_refs`.
3. Claim text mentions specific internal mechanism ("nearest-neighbour strategy") — borderline.
4. Missing: behavioral requirement for filter semantics.

**Required actions:**
- Add `atomic_rationale` OR decompose (decision: add atomic_rationale — the api_contract covers the observable behavior adequately for initial verification; behavioral filtering semantics are a gap deferred with rationale).
- Create CE-TIF-GUARD-001 and tif_guard.py.
- Update CE-REQ-GUARD-API-001 to add `tif_refs`.
- Refactor/create TIF-based test.
- Document raw evidence gap.

---

### CE-CAP-MOND-001 — Mondrian conditional calibration

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-013 |
| Requirements | CE-REQ-MOND-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_mondrian_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **D+F** |

**Required actions:**
- Add `atomic_rationale` (the only observable output of Mondrian calibration through the public API is successful completion and `calibrated == True`; conditional validity per category cannot be verified through the public API without statistical tests beyond the scope of this pass).
- Create CE-TIF-MOND-001 and tif_mondrian.py.
- Update CE-REQ-MOND-API-001 to add `tif_refs`.
- Refactor test to use TIF.

---

### CE-CAP-NARR-001 — Natural language narratives

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-008 |
| Requirements | CE-REQ-NARR-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_narrative_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+D+F — claim text too detailed; needs decomposition; no TIF** |

**Issues:**
1. Claim text lists specific return types: "string, DataFrame, HTML, or dict" — these are return type details, not a capability description.
2. One-to-one with no `atomic_rationale`.
3. CE-REQ-NARR-API-001 has no `tif_refs`.

**Required actions:**
- Simplify claim text.
- Add atomic_rationale (the claim is behaviorally narrow: `to_narrative()` returns non-None; format-specific contracts are deferred as gaps).
- Create CE-TIF-NARR-001 and tif_narrative.py.
- Update CE-REQ-NARR-API-001 to add `tif_refs`.
- Refactor test.

---

### CE-CAP-PRED-001 — Uncertainty prediction intervals

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-013, ADR-021 |
| Requirements | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 (2 requirements) |
| TIF | NONE |
| Tests | test_prediction_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+F — claim text too detailed; 2 requirements but no TIF** |

**Issues:**
1. Claim text mentions specific parameter (`uq_interval=True`) and specific semantics ("low and high bounds") — these details belong in requirements.
2. Both requirements have no `tif_refs`.

**Required actions:**
- Simplify claim text to capability level.
- Create CE-TIF-PRED-001 and tif_prediction.py covering both requirements.
- Update both requirements to add `tif_refs: CE-TIF-PRED-001`.
- Refactor test_prediction_contracts.py to use TIF.

---

### CE-CAP-PRED-CLASS-001 — Classification calibrated probabilities

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-021 |
| Requirements | CE-REQ-PRED-CLASS-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_classification_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+D+F — claim text too detailed; needs decomposition; no TIF** |

**Issues:**
1. Claim text specifies shapes ("array bounded in [0, 1] with shape matching the input") and method-level details.
2. One-to-one with no `atomic_rationale`.
3. CE-REQ-PRED-CLASS-API-001 has no `tif_refs`.
4. Missing probability bounds requirement as a separate requirement.

**Required actions:**
- Simplify claim text.
- Add CE-REQ-PRED-CLASS-BOUNDS-001 (output_contract: probability values in [0,1]).
- Create CE-TIF-PRED-CLASS-001 and tif_classification.py.
- Update CE-REQ-PRED-CLASS-API-001 to add `tif_refs`.
- Refactor test.

---

### CE-CAP-PRED-PROB-001 — Probabilistic regression threshold queries

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-021 |
| Requirements | CE-REQ-PRED-PROB-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_probabilistic_regression_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+D+F — claim text too detailed; needs decomposition; no TIF** |

**Issues:**
1. Claim text specifies the threshold parameter signature, array bounds, and shape — these are requirement-level details.
2. One-to-one with no `atomic_rationale`.
3. CE-REQ-PRED-PROB-API-001 has no `tif_refs`.

**Required actions:**
- Simplify claim text.
- Add CE-REQ-PRED-PROB-BOUNDS-001 (output_contract: P in [0,1]).
- Create CE-TIF-PRED-PROB-001 and tif_prob_regression.py.
- Update CE-REQ-PRED-PROB-API-001 to add `tif_refs`.
- Refactor test.

---

### CE-CAP-REJECT-001 — Reject/defer policies

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-029, ADR-038 |
| Requirements | CE-REQ-REJECT-API-001 (1 requirement) |
| TIF | NONE |
| Tests | test_reject_policy_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **B+D+F** |

**Issues:**
1. Claim text names specific enum values (FLAG, ONLY_REJECTED, ONLY_ACCEPTED) — too detailed.
2. One-to-one with no `atomic_rationale`.
3. CE-REQ-REJECT-API-001 has no `tif_refs`.

**Required actions:**
- Simplify claim text.
- Add `atomic_rationale` (tagging semantic behavior requires introspecting the rejection tags on individual explanations, which is a behavioral gap deferred per ADR-029 scope).
- Create CE-TIF-REJECT-001 and tif_reject.py.
- Update CE-REQ-REJECT-API-001 to add `tif_refs`.
- Refactor test.

---

### CE-CAP-VIZ-001 — Visualization output

| Field | Value |
|---|---|
| Status | current |
| ADR links | ADR-023, ADR-036, ADR-037 |
| Requirements | CE-REQ-VIZ-SMOKE-001 (1 requirement, empirical_smoke) |
| TIF | NONE |
| Tests | test_visualization_contracts.py (direct, no TIF) |
| Raw evidence | NONE |
| **Category** | **F — one-requirement claim justified as atomic; but no TIF** |

**Issues:**
1. The empirical_smoke obligation makes one-to-one justifiable (visual correctness is not verifiable through TIF).
2. CE-REQ-VIZ-SMOKE-001 has no `tif_refs`.

**Required actions:**
- Add `atomic_rationale` to CE-CAP-VIZ-001.
- Create CE-TIF-VIZ-001 and tif_visualization.py.
- Update CE-REQ-VIZ-SMOKE-001 to add `tif_refs`.
- Refactor test.

---

## Governance claims

The following claims are governance/policy claims. Their one-to-one mapping is justified
as atomic for governance obligations where decomposition would not add testable value.
Each needs `atomic_rationale` added and governance requirements need `tif_exemption` where
the obligation is not behavioral through `WrapCalibratedExplainer`.

| Claim | Requirement | Obligation type | TIF status |
|---|---|---|---|
| CE-CAP-DIST-001 | CE-REQ-DIST-GOV-001 | quality_gate | needs tif_exemption (repository_policy) |
| CE-CAP-DEPREC-001 | CE-REQ-DEPREC-GOV-001 | quality_gate | needs tif_exemption (repository_policy) |
| CE-CAP-PREPROC-001 | CE-REQ-PREPROC-GOV-001 | runtime_behavior | verified via integration tests — gap: no TIF |
| CE-CAP-TEST-001 | CE-REQ-TEST-GOV-001 | quality_gate | needs tif_exemption (repository_policy) |
| CE-CAP-OBS-001 | CE-REQ-OBS-GOV-001 | runtime_behavior | gap: needs TIF or tif_exemption |
| CE-CAP-CONFIG-001 | CE-REQ-CONFIG-GOV-001 | static_policy | needs tif_exemption (repository_policy) |
| CE-CAP-SCHEMA-001 | CE-REQ-SCHEMA-GOV-001 | payload_schema | needs tif_exemption (schema_validation) |
| CE-CAP-CORE-001 | CE-REQ-CORE-BOUNDARY-001 | static_policy | needs tif_exemption (repository_policy) |
| CE-CAP-LEGACY-001 | CE-REQ-LEGACY-GOV-001 | api_contract | gap: behavioral but tested through unit tests not TIF |
| CE-CAP-DOCS-001 | CE-REQ-DOCS-GOV-001 | quality_gate | needs tif_exemption (repository_policy) |
| CE-CAP-PLUGIN-001 | CE-REQ-PLUGIN-DOC-001 | documentation_boundary | needs tif_exemption (documentation_boundary) |
| CE-CAP-PARALLEL-001 | CE-REQ-PARALLEL-GOV-001 | runtime_behavior | gap: needs TIF or tif_exemption |
| CE-CAP-CACHE-001 | CE-REQ-CACHE-GOV-001 | runtime_behavior | gap: needs TIF or tif_exemption |
| CE-CAP-VALID-001 | CE-REQ-VALID-EXCEPTION-001 | static_policy | needs tif_exemption (repository_policy) |
| CE-CAP-CI-001 | CE-REQ-CI-GOV-001 | quality_gate | needs tif_exemption (repository_policy) |
| CE-CAP-PLOTSPEC-001 | CE-REQ-PLOTSPEC-GOV-001 | visualization_behavior | gap: needs TIF or tif_exemption |
| CE-CAP-MODALITY-001 | CE-REQ-MODALITY-GOV-001 | plugin_behavior | needs tif_exemption (repository_policy) |
| CE-CAP-SERIAL-001 | CE-REQ-SERIAL-GOV-001 | serialization_contract | gap: needs TIF or tif_exemption |

**Governance TIF exemption rationale:** Governance claims whose requirements verify
static import graphs, CI policy files, YAML/JSON schemas, documentation content,
or repository configuration are exempt from TIF because their behavior is not
exercisable through `WrapCalibratedExplainer`. Runtime-behavioral governance claims
(PREPROC, OBS, PARALLEL, CACHE, PLOTSPEC, SERIAL) have behavioral tests in
`tests/unit/` or `tests/integration/` that verify through or around the public API;
creating a TIF layer for them in the capabilities path is deferred as a follow-up gap.

---

## Summary of gaps

### Critical gaps (behavioral requirements without TIF)

| Requirement | Claim | Gap type |
|---|---|---|
| CE-REQ-EXPL-API-001 | CE-CAP-EXPL-001 | No tif_refs; tests exist but bypass TIF layer |
| CE-REQ-EXPL-API-002 | CE-CAP-EXPL-002 | No tif_refs; tests exist but bypass TIF layer |
| CE-REQ-EXPL-FILTER-SUPER-001 | CE-CAP-EXPL-FILTER-001 | No tif_refs |
| CE-REQ-EXPL-FILTER-SEMI-001 | CE-CAP-EXPL-FILTER-001 | No tif_refs |
| CE-REQ-EXPL-FILTER-COUNTER-001 | CE-CAP-EXPL-FILTER-001 | No tif_refs |
| CE-REQ-EXPL-FILTER-ENSURED-001 | CE-CAP-EXPL-FILTER-001 | No tif_refs |
| CE-REQ-EXPL-FILTER-PARETO-001 | CE-CAP-EXPL-FILTER-001 | No tif_refs |
| CE-REQ-GUARD-API-001 | CE-CAP-GUARD-001 | No tif_refs |
| CE-REQ-PRED-API-001 | CE-CAP-PRED-001 | No tif_refs |
| CE-REQ-PRED-INTERVAL-BOUNDS-001 | CE-CAP-PRED-001 | No tif_refs |
| CE-REQ-PRED-CLASS-API-001 | CE-CAP-PRED-CLASS-001 | No tif_refs |
| CE-REQ-PRED-PROB-API-001 | CE-CAP-PRED-PROB-001 | No tif_refs |
| CE-REQ-MOND-API-001 | CE-CAP-MOND-001 | No tif_refs |
| CE-REQ-NARR-API-001 | CE-CAP-NARR-001 | No tif_refs |
| CE-REQ-REJECT-API-001 | CE-CAP-REJECT-001 | No tif_refs |
| CE-REQ-VIZ-SMOKE-001 | CE-CAP-VIZ-001 | No tif_refs |

### One-to-one mappings without atomic_rationale

| Claim | Requirement | Action |
|---|---|---|
| CE-CAP-EXPL-001 | CE-REQ-EXPL-API-001 | Decompose: add RETURN-001, DOC-001 |
| CE-CAP-EXPL-002 | CE-REQ-EXPL-API-002 | Decompose: add RETURN-001, DOC-001 |
| CE-CAP-GUARD-001 | CE-REQ-GUARD-API-001 | Add atomic_rationale |
| CE-CAP-MOND-001 | CE-REQ-MOND-API-001 | Add atomic_rationale |
| CE-CAP-NARR-001 | CE-REQ-NARR-API-001 | Add atomic_rationale |
| CE-CAP-PRED-CLASS-001 | CE-REQ-PRED-CLASS-API-001 | Decompose: add BOUNDS-001 |
| CE-CAP-PRED-PROB-001 | CE-REQ-PRED-PROB-API-001 | Decompose: add BOUNDS-001 |
| CE-CAP-REJECT-001 | CE-REQ-REJECT-API-001 | Add atomic_rationale |
| CE-CAP-VIZ-001 | CE-REQ-VIZ-SMOKE-001 | Add atomic_rationale |
| CE-CAP-DIST-001 | CE-REQ-DIST-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-DEPREC-001 | CE-REQ-DEPREC-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-PREPROC-001 | CE-REQ-PREPROC-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-TEST-001 | CE-REQ-TEST-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-OBS-001 | CE-REQ-OBS-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-CONFIG-001 | CE-REQ-CONFIG-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-SCHEMA-001 | CE-REQ-SCHEMA-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-CORE-001 | CE-REQ-CORE-BOUNDARY-001 | Add atomic_rationale (governance) |
| CE-CAP-LEGACY-001 | CE-REQ-LEGACY-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-DOCS-001 | CE-REQ-DOCS-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-PLUGIN-001 | CE-REQ-PLUGIN-DOC-001 | Add atomic_rationale (governance) |
| CE-CAP-PARALLEL-001 | CE-REQ-PARALLEL-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-CACHE-001 | CE-REQ-CACHE-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-VALID-001 | CE-REQ-VALID-EXCEPTION-001 | Add atomic_rationale (governance) |
| CE-CAP-CI-001 | CE-REQ-CI-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-PLOTSPEC-001 | CE-REQ-PLOTSPEC-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-MODALITY-001 | CE-REQ-MODALITY-GOV-001 | Add atomic_rationale (governance) |
| CE-CAP-SERIAL-001 | CE-REQ-SERIAL-GOV-001 | Add atomic_rationale (governance) |

### Claim texts too detailed

| Claim | Issue |
|---|---|
| CE-CAP-EXPL-FILTER-001 | Names/describes all 5 filter operations with semantic details |
| CE-CAP-PRED-001 | Mentions `uq_interval=True` parameter and bound semantics |
| CE-CAP-PRED-CLASS-001 | Specifies array shapes and [0,1] bounds in claim text |
| CE-CAP-PRED-PROB-001 | Specifies threshold parameter signature and return shape |
| CE-CAP-NARR-001 | Lists specific return type variants (string, DataFrame, HTML, dict) |
| CE-CAP-REJECT-001 | Names specific enum values (FLAG, ONLY_REJECTED, ONLY_ACCEPTED) |
| CE-CAP-GUARD-001 | Mentions internal mechanism (nearest-neighbour strategy) |

### Evidence gaps

Raw evidence exists only for CE-CAP-EXPL-CONJ-001. All other behavioral claims
lack raw evidence records under `reports/verification/`.

**Evidence gap status for behavioral claims:**

| Claim | Tests exist | Raw evidence | Gap |
|---|---|---|---|
| CE-CAP-EXPL-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-EXPL-002 | ✓ | ✗ | Evidence gap |
| CE-CAP-EXPL-FILTER-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-GUARD-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-PRED-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-PRED-CLASS-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-PRED-PROB-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-MOND-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-NARR-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-REJECT-001 | ✓ | ✗ | Evidence gap |
| CE-CAP-VIZ-001 | ✓ | ✗ | Evidence gap |

Closure path: evidence emission will be added as part of TIF creation (each TIF
returns observation fields that can be serialized as raw evidence).

---

## TIF creation plan

| TIF ID | Executable | Covers requirements |
|---|---|---|
| CE-TIF-EXPL-001 | tif_explanation.py | CE-REQ-EXPL-API-001, CE-REQ-EXPL-API-002, CE-REQ-EXPL-RETURN-001, CE-REQ-EXPL-ALT-RETURN-001 |
| CE-TIF-PRED-001 | tif_prediction.py | CE-REQ-PRED-API-001, CE-REQ-PRED-INTERVAL-BOUNDS-001 |
| CE-TIF-PRED-CLASS-001 | tif_classification.py | CE-REQ-PRED-CLASS-API-001, CE-REQ-PRED-CLASS-BOUNDS-001 |
| CE-TIF-PRED-PROB-001 | tif_prob_regression.py | CE-REQ-PRED-PROB-API-001, CE-REQ-PRED-PROB-BOUNDS-001 |
| CE-TIF-GUARD-001 | tif_guard.py | CE-REQ-GUARD-API-001 |
| CE-TIF-REJECT-001 | tif_reject.py | CE-REQ-REJECT-API-001 |
| CE-TIF-MOND-001 | tif_mondrian.py | CE-REQ-MOND-API-001 |
| CE-TIF-NARR-001 | tif_narrative.py | CE-REQ-NARR-API-001 |
| CE-TIF-VIZ-001 | tif_visualization.py | CE-REQ-VIZ-SMOKE-001 |
| CE-TIF-FILTER-001 | tif_filter.py | CE-REQ-EXPL-FILTER-SUPER-001, CE-REQ-EXPL-FILTER-SEMI-001, CE-REQ-EXPL-FILTER-COUNTER-001, CE-REQ-EXPL-FILTER-ENSURED-001, CE-REQ-EXPL-FILTER-PARETO-001 |
