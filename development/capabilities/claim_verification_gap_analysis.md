# Claim Verification Gap Analysis

This file documents the structure of the CE capability-verification chain and
how to inspect its current state. It is explanatory documentation, not an
authoritative inventory. Active chain nodes are discovered at runtime from the
repository files.

---

## Chain model

Each active capability claim is connected to the evidence that verifies it
through the following chain:

```text
ADR / Standard refs   (optional governing constraints)
  -> Capability claim  (development/capabilities/claims/CE-CAP-*.yaml)
  -> Requirement       (development/capabilities/requirements/CE-REQ-*.md)
  -> TIF spec          (development/capabilities/verification/tif/CE-TIF-*.md)
  -> TIF executable    (development/capabilities/verification/tif/tif_*.py)
  -> Raw evidence      (reports/verification/CE-EVID-*.json)
  -> Curated summary   (development/capabilities/evidence/evidence_*.md)
```

All chain nodes are discovered dynamically from the files above. No manifest,
catalog, or sidecar mapping file is required.

---

## Discovering the chain

To inspect the current chain and validate all links:

```bash
python scripts/quality/validate_capability_chain.py --check
```

To regenerate raw evidence from all active TIF specs:

```bash
python scripts/generate_tif_evidence.py --check-current
```

To validate all committed evidence files without re-running TIF scenarios:

```bash
python scripts/generate_tif_evidence.py --validate-existing
```

To run all capability contract tests:

```bash
pytest tests/capabilities -q
```

---

## What the validator checks

The chain validator (`scripts/quality/validate_capability_chain.py`) verifies:

- Claim filenames match `claim_id`; each active claim has at least one requirement.
- Requirement filenames match `requirement_id`; claim↔requirement links are reciprocal.
- Behavioral requirements have valid `tif_refs` or a documented `tif_exemption`.
- TIF filenames match `tif_id`; each active TIF has an executable with `build_evidence_payload()`.
- TIF↔requirement links are reciprocal; TIF served claims are reachable through served requirements.
- Every active TIF has at least one committed raw evidence record.
- Raw evidence references valid claims, requirements, and TIFs.
- Curated summaries do not overclaim beyond raw evidence.

---

## Open gaps

No known open gaps. Run `python scripts/quality/validate_capability_chain.py --check` to
see the current state.
