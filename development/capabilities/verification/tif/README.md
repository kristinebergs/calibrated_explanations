# TIF — Test Interface Framework

This directory contains TIF (Test Interface Framework) verification interface
specifications and executable TIF scenarios for CE capability verification.

TIF is the layer between requirements and tests. It is not pytest. TIF defines
the interface between requirements and executable verification.

See `development/capabilities/verification/README.md` for the full TIF definition.

## What TIF is

A TIF interface:

1. Stimulates CE exclusively through `WrapCalibratedExplainer`.
2. Captures structured observations from the stimulus.
3. Returns those observations as a typed dataclass or dictionary.
4. Does **not** perform final pytest assertions (except local sanity checks).
5. Enables multiple tests and requirements to share one verified stimulus.

## What TIF is not

- TIF is not a pytest test. Tests call TIF; TIF observes.
- TIF is not a fixture. TIF is parameterized by test scenarios, not by
  pytest fixture injection (though it may use seeded data internally).
- TIF is not a mock. TIF runs the actual CE public workflow.

## Non-negotiable TIF rule

Every behavioral CE TIF scenario must stimulate CE through `WrapCalibratedExplainer`.

TIF scenarios must not:
- Construct explanation objects directly
- Use private/internal CE APIs (`._private`, `CalibratedExplainer` imports)
- Bypass the `fit -> calibrate -> explain` lifecycle

## File naming

```
CE-TIF-<AREA>-<NNN>.md    # interface specification
tif_<area>.py              # executable scenario implementing the spec
```

## Current TIF interfaces

| TIF ID | Specification | Executable | Requirements served |
|---|---|---|---|
| CE-TIF-EXPL-CONJ-001 | [CE-TIF-EXPL-CONJ-001.md](CE-TIF-EXPL-CONJ-001.md) | [tif_conjunction.py](tif_conjunction.py) | CE-REQ-EXPL-CONJ-API-001, CE-REQ-EXPL-CONJ-RETURN-001, CE-REQ-EXPL-CONJ-RULE-001, CE-REQ-EXPL-CONJ-PARAM-001 |

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
