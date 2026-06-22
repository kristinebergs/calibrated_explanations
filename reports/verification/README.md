# Generated Verification Run Outputs

This directory contains machine-generated raw evidence from capability verification
runs: structured JSON/JSONL evidence records, raw pytest output, coverage reports,
and result files produced by automated verification scripts or CI pipelines.

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
Evidence record          -> reports/verification/      (this directory - raw, generated)
                         -> development/capabilities/evidence/ (curated summaries)
```

## Raw evidence vs curated evidence

**Raw evidence (this directory)** is machine-generated output from a verification run.
It is the primary evidence record for behavioral requirements. Raw evidence must be
present before a curated summary can be written.

**Curated evidence** (`development/capabilities/evidence/`) is a human-reviewed summary
derived from raw evidence. It may not replace raw evidence for behavioral requirements.

## Minimum raw evidence fields (JSON)

```yaml
evidence_id: CE-EVID-<AREA>-<NNN>-<YYYYMMDD>
claim_ids:
  - CE-CAP-...
requirement_ids:
  - CE-REQ-...
adr_refs:
  - ADR-...        # if applicable
tif_ids:
  - CE-TIF-...     # if applicable; empty for TIF-exempt requirements
verification_type: api_contract  # see verification_strength values in evidence/README.md
test_id: tests/capabilities/test_<area>_contracts.py::test_should_...
result: pass
timestamp: <ISO 8601>
commit_sha: <git sha>
package_version: <calibrated_explanations version>
python_version: <python version>
platform: <os/arch>
dataset_id: <e.g., sklearn make_classification, random_seed=42>
random_seed: 42
configuration: {}
acceptance:
  criterion_ref: CE-REQ-...
  expected: <what was expected>
  observed: <what was observed>
```

See `development/capabilities/evidence/README.md` for the full schema, the
`verification_strength` and `evidence_level` value sets, and the do-not-overclaim rules.

## What belongs here

- Structured JSON/JSONL evidence records produced by verification scripts or CI
- Raw `pytest --tb=short -v` output captured to file
- `coverage.xml` or `coverage.json` from capability test runs
- CI artifact outputs from capability gates

## What does NOT belong here

- Human-written evidence summaries â†’ `development/capabilities/evidence/`
- Claim or requirement files â†’ `development/capabilities/claims/` or `requirements/`
- TIF interface specifications â†’ `development/capabilities/verification/tif/`
- Verification scenarios and helpers â†’ `development/capabilities/verification/`

## File naming

```
<evidence_id>.<ext>           # for structured evidence records
run_<area>_<date>.<ext>       # for raw pytest/script output
```

Examples:
- `CE-EVID-EXPL-CONJ-001-20260622.json`
- `run_expl_conj_2026-06-22.txt`

## Evidence freshness policy

Committed raw evidence is allowed to be historical. Historical evidence remains valid when it records a full git SHA, has a parseable ISO 8601 timestamp, uses valid references, and has internally consistent pass/fail results.

Release closure should regenerate raw evidence at the release commit. Current-commit enforcement is explicit and opt-in:

```bash
python scripts/generate_tif_evidence.py --check-current
```

For pytest-only validation, maintainers may also opt in with:

```bash
CE_REQUIRE_CURRENT_EVIDENCE=1 pytest tests/capabilities/test_evidence_refs.py -q
```
