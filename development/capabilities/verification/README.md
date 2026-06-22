# Verification Scenarios and Helpers — Capabilities

This directory contains executable verification scenarios, helpers, and TIF
(Test Interface Framework) definitions for CE capability verification.

Verification scenarios are distinct from pytest tests. They may be runnable
standalone scripts, parameterized helpers, or scenario libraries that back
multiple tests. They belong here because they are part of the verification
infrastructure, not test assertions.

TIF verification interface definitions and executables belong in the `tif/`
subdirectory. See `tif/README.md` for the TIF definition.

Pytest test files belong in `tests/capabilities/`.

## Role in the verification chain

```text
ADR / Standard
    -> constrains
Capability claim         -> development/capabilities/claims/
    -> decomposes into
Requirement              -> development/capabilities/requirements/
    -> is exercised through
TIF verification interface -> development/capabilities/verification/tif/ (this directory)
    -> is executed by
Test / verification gate -> tests/capabilities/
    -> produces
Evidence record          -> reports/verification/ (raw)
                         -> development/capabilities/evidence/ (curated)
```

## The TIF layer

**TIF (Test Interface Framework)** is the layer between requirements and tests.
A TIF interface:

1. Stimulates CE exclusively through `WrapCalibratedExplainer` (and the public
   CE API methods it exposes).
2. Captures structured observations from that stimulus.
3. Returns those observations as a structured object or dictionary to the calling test.
4. Does **not** perform final pytest assertions — it observes, not judges.
5. Enables multiple tests to cover multiple requirements by calling the same
   TIF interface with different parameter configurations.

TIF is not pytest. Tests call TIF; TIF observes and returns; tests assert on
the observations against acceptance criteria from the requirement file.

### Non-negotiable TIF rule

Every behavioral CE TIF scenario must stimulate CE through `WrapCalibratedExplainer`,
either directly or through a small public-API helper that itself uses
`WrapCalibratedExplainer`.

TIF scenarios must not:
- Construct explanation objects directly (e.g., `FactualExplanation(...)`)
- Call internal orchestrators
- Use private members (`._something`)
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Bypass the `fit -> calibrate -> predict/explain` workflow unless the
  requirement is specifically about documented error behavior before fit/calibrate

The valid TIF workflow entry is:

```python
explainer = WrapCalibratedExplainer(model)
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)
prediction = explainer.predict(...)
probability = explainer.predict_proba(...)
factual = explainer.explain_factual(...)
alternatives = explainer.explore_alternatives(...)
```

TIF may call public methods on the returned objects (explanation collection
methods, individual explanation methods, plotting methods, narrative methods,
filtering methods, etc.), but the scenario must originate from a valid
`WrapCalibratedExplainer` workflow.

### TIF interface definition

TIF interfaces are defined in `tif/CE-TIF-<AREA>-<NNN>.md` and implemented in
`tif/tif_<area>.py`. Each TIF interface specification must define:

- **tif_id**: unique `CE-TIF-...` identifier
- **requirement_ids served**: list of CE-REQ-... IDs exercised by this TIF
- **claim_ids served** (optional): CE-CAP-... IDs for navigation
- **adr_refs / standard_refs** (if applicable): ADRs / Standards governing this TIF
- **public API surface under test**: which public methods are stimulated
- **fixture/data contract**: what input data is used (deterministic, seeded)
- **WrapCalibratedExplainer workflow used**: the exact sequence of calls
- **stimulus**: what is varied across TIF invocations (parameters, configurations)
- **observation fields**: what structured data is captured and returned
- **acceptance fields**: what fields tests should assert against (the mapping from
  requirement acceptance criteria to observation field names)
- **evidence fields**: what fields should appear in raw evidence records

TIF interfaces must return a structured observation object or dictionary.
They must not perform final pytest assertions. Local sanity checks (e.g., assert
the explainer was successfully calibrated before proceeding) are permitted to
prevent invalid observations from being returned.

## Naming

TIF files:
```
tif/CE-TIF-<AREA>-<NNN>.md    # specification
tif/tif_<area>.py              # executable
```

General scenario files:
```
scenario_<area>.py
helpers_<area>.py
```

## Rules

1. Scenarios implement requirements, not claims directly.
2. Each scenario or helper should reference the requirement ID(s) it serves
   in its module or function docstring.
3. Scenarios must be runnable in isolation and must not depend on test fixtures
   from `tests/conftest.py`.
4. Do not encode acceptance criteria only in scenario code — the criteria must
   be visible in the requirements files in `development/capabilities/requirements/`.
5. TIF scenarios must use `WrapCalibratedExplainer` as the CE entry point.
   See the Non-negotiable TIF rule above.

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| TIF specifications | `development/capabilities/verification/tif/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
