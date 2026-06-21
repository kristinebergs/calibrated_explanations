# CE-REQ-EXPL-FAST-FILTER-001 — Internal FAST-based feature filtering is opt-in, fail-open, telemetry-visible, and preserves factual and alternative explanation semantics.

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FAST-FILTER-001 |
| obligation_type | static_policy |
| claim_refs | CE-CAP-EXPL-FAST-FILTER-001 |
| adr_refs | ADR-027 |
| status | active |
| verification_status | manual_review_required |

## Scope

ADR-governed implementation, documentation, and verification artifacts for `ADR-027`.

## Observable behavior

FAST filtering MUST remain internal, MUST NOT alter the public CalibratedExplainer API, MUST apply only to factual and alternative modes, MUST proceed without filtering on failure, MUST preserve baseline ignore sets, MUST treat per-instance masks as best-effort metadata, and MUST expose effective config and filter events where telemetry is configured.

## Acceptance criterion

Behavioral acceptance for this requirement is the ADR-specific obligation itself:

- FAST filtering MUST remain internal, MUST NOT alter the public CalibratedExplainer API, MUST apply only to factual and alternative modes, MUST proceed without filtering on failure, MUST preserve baseline ignore sets, MUST treat per-instance masks as best-effort metadata, and MUST expose effective config and filter events where telemetry is configured.
- Automated metadata tests MAY prove that the requirement is linked, but MUST NOT be
  treated as proof that the behavioral obligation is satisfied.
- Until a named runtime, static-policy, serialization, schema, or quality-gate check below
  directly verifies the obligation, the verification status remains `manual_review_required`.

## Verification method

Verification method: `manual_review` plus the concrete source, documentation, schema,
quality-script, or pytest targets listed below. Metadata tests are linkage guards only.

## Verification targets

- source: `development/adrs/ADR-027*.md`
- docs: `docs/practitioner/performance-tuning.md`
- docs: `notebooks/advanced/fast_feature_filtering_demo.ipynb`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`
- manual_review: `development/current-work/RELEASE_PLAN_status_appendix.md` (ADR-027 section)

Test IDs:
- `test_should_validate_adr_claim_requirement_link_metadata`
- `test_should_validate_curated_semantic_claim_presence`
- `test_should_require_behavioral_requirements_to_name_concrete_evidence`
- `test_should_prevent_runtime_obligations_from_using_documentation_boundary_type`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail/partial) |

## Assumption boundary

This requirement captures the normative ADR obligation and its evidence chain. It does not
claim that metadata checks alone prove runtime correctness; behavior remains verified by the
specific source, test, documentation, or quality-script evidence maintained for `ADR-027`.
