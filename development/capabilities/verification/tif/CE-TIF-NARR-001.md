# CE-TIF-NARR-001 — Narrative Explanation Output TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-NARR-001 |
| executable | `development/capabilities/verification/tif/tif_narrative.py` |
| entry_functions | `run_narrative_tif_scenario()` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | NARR-001 |
| verification_type | api_contract |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-NARR-API-001 | `exception_raised`, `result_is_none`, `result_is_str`, `result_len` |

## Claims served

- CE-CAP-NARR-001

## ADR refs

- ADR-008

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explain_factual(X_test)` → `CalibratedExplanations`
- `explanations.to_narrative(output_format='text')` → `str`

## Fixture / data contract

The TIF scenario creates a deterministic binary classification dataset using `sklearn.datasets.make_classification`:

```python
make_classification(
    n_samples=120,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    random_state=42,
)
```

The dataset is split deterministically:
- 3 instances for `X_test`
- Remaining 117 split 65%/35% into proper-train and calibration

The model used is `RandomForestClassifier(n_estimators=10, random_state=42)`.

No external data, network, or clock access. All randomness is seeded.

**Dependency note:** `to_narrative()` requires `pyyaml`. If not installed, the TIF will
set `exception_raised=True` with `exception_type='ModuleNotFoundError'`. Tests should
use `pytest.importorskip('yaml')` before calling this TIF.

## WrapCalibratedExplainer workflow

```python
explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

explanations = explainer.explain_factual(X_test)
result = explanations.to_narrative(output_format='text')
```

## Observation fields

The TIF returns a `NarrativeObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the result is `None` |
| `result_is_str` | `bool` | Whether `isinstance(result, str)` |
| `result_len` | `int \| None` | `len(result)` if result is a str, else `None` |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-NARR-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-NARR-API-001 | Result is not None | `result_is_none` | `False` |
| CE-REQ-NARR-API-001 | Result is a string | `result_is_str` | `True` |
| CE-REQ-NARR-API-001 | Result is non-empty | `result_len` | `> 0` |

## TIF constraints

This TIF does NOT:
- Construct explanation objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
