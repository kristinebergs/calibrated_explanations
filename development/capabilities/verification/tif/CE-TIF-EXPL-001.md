# CE-TIF-EXPL-001 — Factual and Alternative Explanation TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-EXPL-001 |
| executable | `development/capabilities/verification/tif/tif_explanation.py` |
| entry_functions | `run_factual_tif_scenario()`, `run_alternative_tif_scenario()` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | EXPL-001 |
| verification_type | behavioral_contract |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-EXPL-API-001 | `exception_raised` (must be `False`) |
| CE-REQ-EXPL-RETURN-001 | `result_is_none`, `result_len`, `first_item_is_none`, `feature_weights_accessible`, `result_type_name` |
| CE-REQ-EXPL-API-002 | `exception_raised` (must be `False`) |
| CE-REQ-EXPL-ALT-RETURN-001 | `result_is_none`, `result_len`, `first_item_is_none`, `result_type_name` |

## Claims served

- CE-CAP-EXPL-001
- CE-CAP-EXPL-002

## ADR refs

- ADR-008
- ADR-015
- ADR-026

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explain_factual(X_test)` → `CalibratedExplanations`
- `explainer.explore_alternatives(X_test)` → `AlternativeExplanations`
- `result[0].feature_weights` → public attribute on `FactualExplanation`

## Fixture / data contract

The TIF scenario creates a deterministic dataset using `sklearn.datasets.make_classification`:

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

# factual
factual_result = explainer.explain_factual(X_test)

# alternative
alt_result = explainer.explore_alternatives(X_test)
```

## Observation fields

The TIF returns an `ExplanationObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the returned object is `None` |
| `result_len` | `int \| None` | `len(result)` if result supports it, else `None` |
| `result_type_name` | `str \| None` | `type(result).__name__` if result is not None |
| `first_item_is_none` | `bool` | Whether `result[0]` is `None` |
| `feature_weights_accessible` | `bool` | Whether `result[0].feature_weights` is accessible (factual only) |
| `explanation_mode` | `str` | `"factual"` or `"alternative"` |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-EXPL-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-EXPL-RETURN-001 | Return value not None | `result_is_none` | `False` |
| CE-REQ-EXPL-RETURN-001 | Cardinality preserved | `result_len` | `== n_instances` |
| CE-REQ-EXPL-RETURN-001 | First item accessible | `first_item_is_none` | `False` |
| CE-REQ-EXPL-RETURN-001 | Feature weights accessible | `feature_weights_accessible` | `True` |
| CE-REQ-EXPL-API-002 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-EXPL-ALT-RETURN-001 | Return value not None | `result_is_none` | `False` |
| CE-REQ-EXPL-ALT-RETURN-001 | Cardinality preserved | `result_len` | `== n_instances` |
| CE-REQ-EXPL-ALT-RETURN-001 | Return type correct | `result_type_name` | `"AlternativeExplanations"` |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-EXPL-<YYYYMMDD>
claim_ids:
  - CE-CAP-EXPL-001
  - CE-CAP-EXPL-002
requirement_ids:
  - CE-REQ-EXPL-API-001
  - CE-REQ-EXPL-RETURN-001
  - CE-REQ-EXPL-API-002
  - CE-REQ-EXPL-ALT-RETURN-001
adr_refs:
  - ADR-008
  - ADR-015
  - ADR-026
tif_ids:
  - CE-TIF-EXPL-001
verification_type: behavioral_contract
dataset_id: sklearn make_classification n_samples=120 n_features=4 n_informative=3 random_seed=42
random_seed: 42
```

## TIF constraints

This TIF does NOT:
- Construct explanation objects directly (no `FactualExplanation(...)`)
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
