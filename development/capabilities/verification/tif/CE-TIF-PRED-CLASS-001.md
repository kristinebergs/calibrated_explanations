# CE-TIF-PRED-CLASS-001 — Classification Prediction TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-PRED-CLASS-001 |
| executable | `development/capabilities/verification/tif/tif_classification.py` |
| entry_functions | `run_classification_tif_scenario()` |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-PRED-CLASS-API-001 | `exception_raised`, `proba_len`, `labels_len` |
| CE-REQ-PRED-CLASS-BOUNDS-001 | `proba_min`, `proba_max` |

## Claims served

- CE-CAP-PRED-CLASS-001

## ADR refs

- ADR-021

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.predict_proba(X_test)` → probability array
- `explainer.predict(X_test)` → label array

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
- 5 instances for `X_test`
- Remaining 115 split 65%/35% into proper-train and calibration

The model used is `RandomForestClassifier(n_estimators=10, random_state=42)`.

No external data, network, or clock access. All randomness is seeded.

## WrapCalibratedExplainer workflow

```python
explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

probas = explainer.predict_proba(X_test)
labels = explainer.predict(X_test)
```

## Observation fields

The TIF returns a `ClassificationObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the calls |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `proba_is_none` | `bool` | Whether `predict_proba` returned `None` |
| `proba_len` | `int \| None` | Length of the probability array |
| `proba_min` | `float \| None` | Minimum value in the probability array |
| `proba_max` | `float \| None` | Maximum value in the probability array |
| `labels_is_none` | `bool` | Whether `predict` returned `None` |
| `labels_len` | `int \| None` | Length of the label array |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-PRED-CLASS-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-PRED-CLASS-API-001 | Returns probability array | `proba_len` | `== n_instances` |
| CE-REQ-PRED-CLASS-API-001 | Returns label array | `labels_len` | `== n_instances` |
| CE-REQ-PRED-CLASS-BOUNDS-001 | Probabilities ≥ 0 | `proba_min` | `>= 0.0` |
| CE-REQ-PRED-CLASS-BOUNDS-001 | Probabilities ≤ 1 | `proba_max` | `<= 1.0` |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-PRED-CLASS-<YYYYMMDD>
claim_ids:
  - CE-CAP-PRED-CLASS-001
requirement_ids:
  - CE-REQ-PRED-CLASS-API-001
  - CE-REQ-PRED-CLASS-BOUNDS-001
adr_refs:
  - ADR-021
tif_ids:
  - CE-TIF-PRED-CLASS-001
verification_type: behavioral_contract
dataset_id: sklearn make_classification n_samples=120 n_features=4 n_informative=3 n_redundant=1 random_seed=42
random_seed: 42
```

## TIF constraints

This TIF does NOT:
- Construct prediction objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
