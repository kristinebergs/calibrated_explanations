# CE-TIF-PRED-PROB-001 — Probabilistic Regression Threshold TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-PRED-PROB-001 |
| executable | `development/capabilities/verification/tif/tif_prob_regression.py` |
| entry_functions | `run_prob_regression_tif_scenario(threshold=0.0)` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | PRED-PROB-001 |
| verification_type | numerical_behavior |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-PRED-PROB-API-001 | `exception_raised`, `proba_len` |
| CE-REQ-PRED-PROB-BOUNDS-001 | `proba_min`, `proba_max` |

## Claims served

- CE-CAP-PRED-PROB-001

## ADR refs

- ADR-021

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.predict_proba(X_test, threshold=t)` → probability array `P(Y > t | X)`

## Fixture / data contract

The TIF scenario creates a deterministic regression dataset using `sklearn.datasets.make_regression`:

```python
make_regression(
    n_samples=150,
    n_features=4,
    n_informative=3,
    random_state=42,
    noise=10.0,
)
```

The dataset is split deterministically:
- 5 instances for `X_test`
- Remaining 145 split 65%/35% into proper-train and calibration

The model used is `RandomForestRegressor(n_estimators=10, random_state=42)`.

No external data, network, or clock access. All randomness is seeded.

## WrapCalibratedExplainer workflow

```python
explainer = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

# predict P(Y > threshold | X)
result = explainer.predict_proba(X_test, threshold=0.0)
```

## Observation fields

The TIF returns a `ProbRegressionObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the result is None |
| `proba_len` | `int \| None` | Length of the returned probability array |
| `proba_min` | `float \| None` | Minimum value in the probability array |
| `proba_max` | `float \| None` | Maximum value in the probability array |
| `threshold` | `float` | The threshold used |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-PRED-PROB-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-PRED-PROB-API-001 | Returns probability array | `proba_len` | `== n_instances` |
| CE-REQ-PRED-PROB-BOUNDS-001 | Probabilities ≥ 0 | `proba_min` | `>= 0.0` |
| CE-REQ-PRED-PROB-BOUNDS-001 | Probabilities ≤ 1 | `proba_max` | `<= 1.0` |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-PRED-PROB-<YYYYMMDD>
claim_ids:
  - CE-CAP-PRED-PROB-001
requirement_ids:
  - CE-REQ-PRED-PROB-API-001
  - CE-REQ-PRED-PROB-BOUNDS-001
adr_refs:
  - ADR-021
tif_ids:
  - CE-TIF-PRED-PROB-001
verification_type: behavioral_contract
dataset_id: sklearn make_regression n_samples=150 n_features=4 n_informative=3 random_seed=42 noise=10
random_seed: 42
```

## TIF constraints

This TIF does NOT:
- Construct prediction objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
