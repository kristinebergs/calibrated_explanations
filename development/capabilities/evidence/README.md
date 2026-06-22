# Capability Evidence — Curated Summaries

This directory contains human-curated evidence summaries for CE capability
verification runs. Curated evidence records here are produced after a
verification run in `reports/verification/` and capture a human-reviewed
snapshot of what was verified, which test IDs passed, and against which
package version.

## Two evidence layers

Evidence is split into two complementary layers:

### Raw evidence

Machine-readable output from verification execution. Raw evidence belongs under:

```
reports/verification/
```

Raw evidence should be JSON or JSONL where feasible. It is produced automatically
by verification scripts or CI pipelines. Do **not** place raw evidence in this
directory — use `reports/verification/`.

#### Minimum raw evidence fields

```yaml
evidence_id: CE-EVID-<AREA>-<NNN>-<YYYYMMDD>
claim_ids:
  - CE-CAP-...
requirement_ids:
  - CE-REQ-...
adr_refs:
  - ADR-...        # if applicable
standard_refs:
  - STD-...        # if applicable
tif_ids:
  - CE-TIF-...     # if applicable; empty for TIF-exempt requirements
verification_type: api_contract  # see verification_strength values
test_id: tests/capabilities/test_<area>_contracts.py::test_should_...
result: pass
timestamp: <ISO 8601>
commit_sha: <git sha>
package_version: <calibrated_explanations version>
python_version: <python version>
platform: <os/arch>
dataset_id: <e.g., sklearn make_classification, random_seed=42>
random_seed: 42
configuration: {}  # any non-default parameters used
acceptance:
  criterion_ref: CE-REQ-...  # which acceptance criterion this satisfies
  expected: <what was expected>
  observed: <what was observed>
artifacts:
  logs: null
  raw_output: null
```

### Curated evidence

Human-readable release or closure summaries derived from raw evidence. Curated
evidence belongs in this directory (`development/capabilities/evidence/`).

Curated evidence must:
- Reference raw evidence files or verification runs (by `evidence_id`, test ID,
  or run timestamp)
- State which requirement IDs are covered
- Note what was verified and what was explicitly NOT verified

Curated evidence must not:
- Replace raw evidence for behavioral requirements
- Assert that a requirement is verified without citing a raw evidence record
  or an executable test

## Role in the verification chain

```text
ADR / Standard
    -> constrains
Capability claim         -> development/capabilities/claims/
    -> decomposes into
Requirement              -> development/capabilities/requirements/
    -> is exercised through
TIF verification interface -> development/capabilities/verification/tif/
    -> is executed by
Test / verification gate -> tests/capabilities/
    -> produces
Evidence record          -> reports/verification/      (raw — machine-readable)
                         -> development/capabilities/evidence/ (this directory — curated)
```

## Verification strength model

Avoid generic `verified: true` without specifying what dimension was tested.
The `verification_type` field in raw evidence and `verification_strength` in
curated evidence must use one of these values:

| Strength | What it proves | What it does NOT prove |
|---|---|---|
| `api_contract` | Method exists and is callable from the public workflow | Semantic correctness, numerical validity |
| `behavioral_contract` | Observable behavior matches the requirement's acceptance criterion | Statistical guarantees, scientific meaning |
| `numerical_behavior` | Numeric output values satisfy documented invariants | Calibration validity, theoretical bounds |
| `statistical_method_alignment` | Statistical properties satisfy documented assumptions | Finite-sample guarantees beyond stated assumptions |
| `empirical_smoke` | Basic end-to-end execution completed | Any of the above |
| `documentation_boundary` | Documentation states what is and is not proven | Behavioral correctness |
| `visualization_structure` | Visualization output matches structural contract | Scientific meaning of the visualization |
| `policy_check` | Repository or CI policy compliance | Behavioral requirements |

A requirement can be verified as `api_contract` while not verified as
`statistical_method_alignment` or semantic explanation quality.

Use `evidence_level` to record what form of evidence exists:

| Level | Meaning |
|---|---|
| `raw_evidence` | Machine-readable output in `reports/verification/` |
| `curated_summary` | Human-reviewed summary in this directory |
| `ci_gate` | CI pass/fail recorded in pipeline artifacts |
| `metadata_only` | Traceability links only — cannot prove behavioral requirements |

## Do not overclaim

Evidence records must not assert more than what was tested:

- API liveness does not prove semantic correctness.
- Output shape does not prove calibration validity.
- Empirical smoke tests do not prove finite-sample theoretical guarantees.
- Visualization structure does not prove scientific meaning.
- Metadata/linkage tests do not prove behavioral requirements.
- Curated evidence summaries do not replace raw verification evidence.

## When to add a curated evidence record

Add a curated summary here when:
- A capability release milestone is reached.
- A failing requirement was fixed and a re-run confirms it now passes.
- A regression was investigated and confirmed to be absent.

Do NOT add raw pytest output here. Raw output belongs in `reports/verification/`.

## File naming

```
evidence_<area>_<version>.md
```

Examples: `evidence_expl_conj_v0.11.4.md`, `evidence_filter_ops_v0.11.4.md`

## Required content per curated evidence record

Each file must state:

- **requirement_ids**: which CE-REQ-... IDs are covered
- **tif_ids**: which CE-TIF-... interfaces were exercised (if applicable)
- **verification_strength**: one of the strength values above
- **evidence_level**: one of the level values above
- **package_version**: the calibrated_explanations version under test
- **commit_sha**: the commit at which the run was executed
- **test_ids**: named test functions that passed
- **raw_evidence_ref**: reference to the raw evidence file(s) in `reports/verification/`
  (required for behavioral requirements; may be `none` for policy/documentation checks)
- **dataset_id**: dataset used (e.g., sklearn make_classification, random_seed=42)
- **result**: PASS / FAIL / PARTIAL
- **assumption_boundary**: what this evidence explicitly does NOT prove
- **notes**: any reviewer observations (optional)

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| TIF verification interfaces | `development/capabilities/verification/tif/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) run outputs | `reports/verification/` |
