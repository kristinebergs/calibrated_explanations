# CE-TIF-FILTER-001 — Alternative Explanation Filter TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-FILTER-001 |
| executable | `development/capabilities/verification/tif/tif_filter.py` |
| entry_functions | `run_filter_tif_scenario(filter_type)` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | FILTER-001 |
| verification_type | api_contract |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-EXPL-FILTER-SUPER-001 | `exception_raised`, `collection_result_is_none`, `collection_result_len`, `individual_result_is_none`, `alias_result_is_none` |
| CE-REQ-EXPL-FILTER-SEMI-001 | same |
| CE-REQ-EXPL-FILTER-COUNTER-001 | same |
| CE-REQ-EXPL-FILTER-ENSURED-001 | same |
| CE-REQ-EXPL-FILTER-PARETO-001 | same |

## Claims served

- CE-CAP-EXPL-FILTER-001

## ADR refs

- ADR-027

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explore_alternatives(X_test)` → `AlternativeExplanations`
- `alternatives.super_explanations()` / `.super()` (collection + alias)
- `alternatives.semi_explanations()` / `.semi()` (collection + alias)
- `alternatives.counter_explanations()` / `.counter()` (collection + alias)
- `alternatives.ensured_explanations()` / `.ensured()` (collection + alias)
- `alternatives.pareto_explanations()` / `.pareto()` (collection + alias)
- `alternatives[0].<method>()` (individual level for each filter)

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
explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

alternatives = explainer.explore_alternatives(X_test)

# e.g. for filter_type="super":
col_result  = alternatives.super_explanations()
ind_result  = alternatives[0].super_explanations()
alias_result = alternatives.super()
```

## Observation fields

The TIF returns a `FilterObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `filter_type` | `str` | Which filter operation was exercised |
| `exception_raised` | `bool` | Whether an exception was raised |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `collection_result_is_none` | `bool` | Whether the collection-level result is `None` |
| `collection_result_len` | `int \| None` | `len()` of collection result |
| `individual_result_is_none` | `bool` | Whether the individual-level result is `None` |
| `alias_result_is_none` | `bool` | Whether the alias method result is `None` |
| `alias_result_len` | `int \| None` | `len()` of alias result |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-EXPL-FILTER-* | API callable without exception | `exception_raised` | `False` |
| CE-REQ-EXPL-FILTER-* | Collection result not None | `collection_result_is_none` | `False` |
| CE-REQ-EXPL-FILTER-* | Collection cardinality preserved | `collection_result_len` | `== n_instances` |
| CE-REQ-EXPL-FILTER-* | Individual result not None | `individual_result_is_none` | `False` |
| CE-REQ-EXPL-FILTER-* | Alias result not None | `alias_result_is_none` | `False` |
| CE-REQ-EXPL-FILTER-* | Alias cardinality preserved | `alias_result_len` | `== n_instances` |

## TIF constraints

This TIF does NOT:
- Verify that filtered sets are non-empty (this is data-dependent, not contractual)
- Construct explanation objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
