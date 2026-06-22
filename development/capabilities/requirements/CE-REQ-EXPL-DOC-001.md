# CE-REQ-EXPL-DOC-001 — Factual Explanation Documentation Boundary

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-DOC-001 |
| obligation_type | documentation_boundary |
| claim_refs | CE-CAP-EXPL-001 |
| adr_refs | ADR-008, ADR-026 |
| status | active |
| verification_status | not_implemented |
| gap_ref | CE-REQ-EXPL-DOC-001 |
| applicable_on | CE documentation (RTD, docstrings) |
| tif_exemption | documentation_boundary |
| verification_strength | documentation_boundary |
| evidence_level | metadata_only |

## Scope

CE documentation covering the factual explanation capability:
- RTD pages referencing `explain_factual`
- Docstrings on `WrapCalibratedExplainer.explain_factual` and `CalibratedExplanations`

## TIF exemption

```
tif_exemption: documentation_boundary
tif_exemption_rationale: >
  This requirement governs what CE documentation states about the factual explanation
  capability, not observable API or runtime behavior. Documentation review cannot be
  exercised through WrapCalibratedExplainer. Behavioral requirements for factual
  explanation API behavior are covered by CE-REQ-EXPL-API-001 and CE-REQ-EXPL-RETURN-001.
```

## Observable behavior

CE documentation must state:

1. What `explain_factual` does at the capability level.
2. The assumption boundary: what the API contract verifies and what it does not prove
   (calibration validity, finite-sample coverage, feature attribution accuracy).
3. That the explainer must be fitted and calibrated before calling `explain_factual`.

## Acceptance criterion

A human reviewer can confirm that:

1. The factual explanation documentation does not assert scientific validity of
   feature attributions without qualification.
2. The calibration assumption (exchangeability) is stated in docstrings or RTD.

## Verification method

Human review of RTD pages and public docstrings.

## Verification targets

Gap: `not_implemented` — documentation review has not been executed as a formal
verification step for this requirement.

```
gap_ref: CE-REQ-EXPL-DOC-001
missing_behavior: formal documentation review confirming assumption boundary claims
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

This requirement documents the obligation to state documentation boundaries. It does
not prove that the documentation is correct, complete, or that users read it. It does
not replace behavioral verification provided by CE-REQ-EXPL-API-001 and CE-REQ-EXPL-RETURN-001.
