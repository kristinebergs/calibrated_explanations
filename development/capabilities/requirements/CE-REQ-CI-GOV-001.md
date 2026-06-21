# CE-REQ-CI-GOV-001 — ADR Governance Linkage Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CI-GOV-001 |
| obligation_type | documentation_boundary |
| claim_refs | CE-CAP-CI-001 |
| adr_refs | ADR-035 |
| status | active |

## Scope

Development governance artifacts under `development/adrs/` and
`development/capabilities/` for ADR-035.

## Observable behavior

The governed ADR claim chain MUST remain navigable in-place:

1. Each owning ADR MUST list `CE-CAP-CI-001` in its `## Governed claims` section.
2. `CE-CAP-CI-001` MUST list its owning ADRs in `adr_links`.
3. `CE-CAP-CI-001` MUST list `CE-REQ-CI-GOV-001` in `requirements`.
4. This requirement MUST list `CE-CAP-CI-001` in `claim_refs`.
5. Linked test references MUST point to existing tests or explicit metadata checks.

## Acceptance criterion

The capability traceability validation test passes for this ADR/claim/requirement
chain without relying on a standalone traceability table or generated matrix.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_validate_adr_capability_links_when_metadata_changes`

(in `tests/capabilities/test_adr_capability_links.py`)

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies governance linkage and metadata consistency. It does not
prove every runtime behavior implied by the owning ADR; runtime behavior remains
covered by the specific implementation tests referenced by feature requirements.
