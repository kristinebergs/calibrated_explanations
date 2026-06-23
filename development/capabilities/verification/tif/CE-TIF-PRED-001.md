# CE-TIF-PRED-001 — Uncertainty Prediction Interval TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-PRED-001 |
| executable | `development/capabilities/verification/tif/tif_prediction.py` |
| entry_functions | `run_prediction_tif_scenario(low_high_percentiles=None)` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | PRED-001 |
| verification_type | behavioral_contract |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-PRED-API-001 | `exception_raised` (must be `False`), `y_hat_len`, `low_is_none`, `high_is_none` |
| CE-REQ-PRED-INTERVAL-BOUNDS-001 | `bounds_ordered`, `low_values`, `high_values`, `y_hat_len` |

## Claims served

- CE-CAP-PRED-001

## ADR refs

- ADR-021

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.predict(X_test, uq_interval=True)` → `(y_hat, (low, high))`
- `explainer.predict(X_test, uq_interval=True, low_high_percentiles=(p1, p2))`

## Fixture / data contract

The TIF scenario creates a deterministic dataset using `sklearn.datasets.make_regression`:

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

# default percentiles
result = explainer.predict(X_test, uq_interval=True)
y_hat, (low, high) = result

# custom percentiles
result = explainer.predict(X_test, uq_interval=True, low_high_percentiles=(10, 90))
```

## Observation fields

The TIF returns a `PredictionObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the predict call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the result is None |
| `y_hat_len` | `int \| None` | Length of the point prediction array |
| `low_is_none` | `bool` | Whether the lower bound array is None |
| `high_is_none` | `bool` | Whether the upper bound array is None |
| `bounds_ordered` | `bool` | Whether `low[i] <= high[i]` for all i |
| `low_lte_yhat` | `bool \| None` | Whether `low[i] <= y_hat[i]` for all i |
| `low_high_percentiles` | `tuple \| None` | The percentile tuple used, or None for default |
| `n_instances` | `int` | Number of test instances |
| `low_values` | `list \| None` | Actual lower bound values |
| `high_values` | `list \| None` | Actual upper bound values |
| `y_hat_values` | `list \| None` | Point prediction values |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-PRED-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-PRED-API-001 | Returns point predictions | `y_hat_len` | `== n_instances` |
| CE-REQ-PRED-API-001 | Returns lower bounds | `low_is_none` | `False` |
| CE-REQ-PRED-API-001 | Returns upper bounds | `high_is_none` | `False` |
| CE-REQ-PRED-INTERVAL-BOUNDS-001 | Bounds ordered low ≤ high | `bounds_ordered` | `True` |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-PRED-<YYYYMMDD>
claim_ids:
  - CE-CAP-PRED-001
requirement_ids:
  - CE-REQ-PRED-API-001
  - CE-REQ-PRED-INTERVAL-BOUNDS-001
adr_refs:
  - ADR-021
tif_ids:
  - CE-TIF-PRED-001
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
