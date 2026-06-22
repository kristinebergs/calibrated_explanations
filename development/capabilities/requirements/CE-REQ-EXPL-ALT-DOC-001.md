# CE-REQ-EXPL-ALT-DOC-001 — Alternative Explanation Documentation Boundary

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-ALT-DOC-001 |
| obligation_type | documentation_boundary |
| claim_refs | CE-CAP-EXPL-002 |
| adr_refs | ADR-008, ADR-026 |
| status | active |
| verification_status | not_implemented |
| gap_ref | CE-REQ-EXPL-ALT-DOC-001 |
| applicable_on | CE documentation (RTD, docstrings) |
| tif_exemption | documentation_boundary |
| verification_strength | documentation_boundary |
| evidence_level | metadata_only |

## Scope

CE documentation covering the alternative explanation capability:
- RTD pages referencing `explore_alternatives`
- Docstrings on `WrapCalibratedExplainer.explore_alternatives` and `AlternativeExplanations`

## TIF exemption

```
tif_exemption: documentation_boundary
tif_exemption_rationale: >
  This requirement governs what CE documentation states about the alternative explanation
  capability, not observable API or runtime behavior. Documentation review cannot be
  exercised through WrapCalibratedExplainer. Behavioral requirements for alternative
  explanation API behavior are covered by CE-REQ-EXPL-API-002 and CE-REQ-EXPL-ALT-RETURN-001.
```

## Observable behavior

CE documentation must state:

1. What `explore_alternatives` provides at the capability level.
2. The assumption boundary: that alternative explanations describe feature changes relative
   to the current instance and do not guarantee achievability in a new model deployment.
3. That the explainer must be fitted and calibrated before calling `explore_alternatives`.

## Acceptance criterion

A human reviewer can confirm that:

1. The alternative explanation documentation does not assert that counterfactual scenarios
   are physically or distributionally achievable without qualification.
2. The exchangeability assumption is stated in docstrings or RTD.

## Verification method

Human review of RTD pages and public docstrings.

## Verification targets

Gap: `not_implemented` — documentation review has not been executed as a formal
verification step for this requirement.

```
gap_ref: CE-REQ-EXPL-ALT-DOC-001
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
not replace behavioral verification provided by CE-REQ-EXPL-API-002 and CE-REQ-EXPL-ALT-RETURN-001.
