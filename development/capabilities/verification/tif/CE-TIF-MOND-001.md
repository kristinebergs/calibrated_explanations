# CE-TIF-MOND-001 — Mondrian Conditional Calibration TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-MOND-001 |
| executable | `development/capabilities/verification/tif/tif_mondrian.py` |
| entry_functions | `run_mondrian_tif_scenario()` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | MOND-001 |
| verification_type | api_contract |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-MOND-API-001 | `exception_raised`, `calibrated` |

## Claims served

- CE-CAP-MOND-001

## ADR refs

- ADR-013

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y)`
- `explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)` where `mondrian_fn` is a callable
- `explainer.calibrated` — boolean property

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

The Mondrian categorizer partitions by sign of the first feature (2 categories):

```python
def mondrian_fn(x):
    return (np.asarray(x)[:, 0] >= 0).astype(int)
```

The model used is `RandomForestClassifier(n_estimators=10, random_state=42)`.

No external data, network, or clock access. All randomness is seeded.

## WrapCalibratedExplainer workflow

```python
explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

assert explainer.calibrated is True
```

## Observation fields

The TIF returns a `MondrianObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during `calibrate` |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `calibrated` | `bool` | Whether `wrapper.calibrated` is `True` after calibration |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-MOND-API-001 | calibrate completes without exception | `exception_raised` | `False` |
| CE-REQ-MOND-API-001 | wrapper reports calibrated | `calibrated` | `True` |

## TIF constraints

This TIF does NOT:
- Construct calibration objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
