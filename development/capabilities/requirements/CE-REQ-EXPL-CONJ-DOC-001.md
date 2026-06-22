# CE-REQ-EXPL-CONJ-DOC-001 — Conjunction Documentation Boundary

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-DOC-001 |
| obligation_type | documentation_boundary |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | active |
| verification_status | not_implemented |
| gap_ref | ADR-008 |
| applicable_on | CE documentation (RTD, docstrings) |
| supersedes | — |
| tif_exemption | documentation_boundary |
| verification_strength | documentation_boundary |
| evidence_level | metadata_only |

## Scope

CE documentation covering the conjunction capability:
- RTD pages referencing `add_conjunctions`
- Docstrings on `CalibratedExplanations.add_conjunctions`,
  `AlternativeExplanations.add_conjunctions`, `FactualExplanation.add_conjunctions`,
  `AlternativeExplanation.add_conjunctions`

## TIF exemption

```
tif_exemption: documentation_boundary
tif_exemption_rationale: >
  This requirement governs what CE documentation states about the conjunction
  capability, not observable API or runtime behavior. Documentation review cannot
  be exercised through WrapCalibratedExplainer. Behavioral requirements for
  conjunction API behavior are covered by CE-REQ-EXPL-CONJ-API-001,
  CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, and
  CE-REQ-EXPL-CONJ-PARAM-001.
```

## Observable behavior

CE documentation must state:

1. What `add_conjunctions` does (extends explanation objects with multi-feature rules).
2. The scope of verification: which requirement IDs provide executable evidence.
3. What the documentation does NOT claim to prove, including:
   - That conjunction rules are semantically superior to single-feature rules.
   - That the calibration validity of conjunction rules has been tested beyond API smoke.
   - Runtime performance guarantees.

## Acceptance criterion

A human reviewer can confirm that:

1. The conjunction documentation does not assert scientific superiority of conjunction rules
   without qualification.
2. The docstrings reference what the API guarantees and what is out of scope.

## Verification method

Human review of RTD pages and public docstrings.

## Verification targets

Gap: `adr_gap_open` — human review of conjunction documentation has not been
executed as a formal verification step. Documentation content may be accurate but
has not been reviewed against this requirement.

```
gap_ref: CE-REQ-EXPL-CONJ-DOC-001
adr_gap_ref: ADR-008
missing_behavior: formal documentation review confirming boundary claims
why_not_verified: documentation review is not currently an automated CI gate
intended_closure: manual review during v0.12.x documentation audit
```

## Evidence required

| Field | Required |
|---|---|
| reviewer | yes |
| package_version | yes |
| reviewed_doc_paths | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement documents the obligation to state documentation boundaries.
It does not prove that the documentation is correct, complete, or that users
read it. It does not replace behavioral verification provided by
CE-REQ-EXPL-CONJ-API-001 through CE-REQ-EXPL-CONJ-PARAM-001.
