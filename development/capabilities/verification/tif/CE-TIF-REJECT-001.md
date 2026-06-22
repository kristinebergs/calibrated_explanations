# CE-TIF-REJECT-001 — Reject Policy Explanation TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-REJECT-001 |
| executable | `development/capabilities/verification/tif/tif_reject.py` |
| entry_functions | `run_reject_tif_scenario()` |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-REJECT-API-001 | `exception_raised`, `result_is_none`, `result_len` |

## Claims served

- CE-CAP-REJECT-001

## ADR refs

- ADR-029

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explain_factual(X_test, reject_policy=RejectPolicySpec.flag())`
- `calibrated_explanations.RejectPolicySpec`

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

## WrapCalibratedExplainer workflow

```python
from calibrated_explanations import RejectPolicySpec

explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

result = explainer.explain_factual(X_test, reject_policy=RejectPolicySpec.flag())
```

## Observation fields

The TIF returns a `RejectObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the result is `None` |
| `result_len` | `int \| None` | `len(result)` if result supports it, else `None` |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-REJECT-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-REJECT-API-001 | Returns non-None | `result_is_none` | `False` |
| CE-REQ-REJECT-API-001 | Cardinality preserved | `result_len` | `== n_instances` |

## TIF constraints

This TIF does NOT:
- Construct explanation objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
