# CE-TIF-EXPL-CONJ-001 — Conjunction TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-EXPL-CONJ-001 |
| executable | `development/capabilities/verification/tif/tif_conjunction.py` |
| entry_function | `run_conjunction_tif_scenario()` |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-EXPL-CONJ-API-001 | `exception_raised` (must be `False`) |
| CE-REQ-EXPL-CONJ-RETURN-001 | `result_is_none`, `result_len` |
| CE-REQ-EXPL-CONJ-RULE-001 | `any_has_conjunctive_rules` (must be `True` when `max_rule_size >= 2`) |
| CE-REQ-EXPL-CONJ-PARAM-001 | `any_has_conjunctive_rules` (must be `False` when `max_rule_size == 1`) |

## Claims served

- CE-CAP-EXPL-CONJ-001

## ADR refs

- ADR-008

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explain_factual(X_test)` → `CalibratedExplanations`
- `explainer.explore_alternatives(X_test)` → `AlternativeExplanations`
- `collection.add_conjunctions(n_top_features, max_rule_size)` → same collection type
- `collection[i].add_conjunctions(n_top_features, max_rule_size)` → individual explanation
- `explanation.has_conjunctive_rules` → `bool` (public attribute)

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

This provides 4 features with 3 informative, which is sufficient for conjunction
generation when `max_rule_size >= 2` and `n_top_features >= 2`.

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

# collection-level
collection = explainer.explain_factual(X_test)       # or explore_alternatives
result = collection.add_conjunctions(n_top_features, max_rule_size)

# individual-level (only when object_level == "individual")
individual = collection[0]
result_ind = individual.add_conjunctions(n_top_features, max_rule_size)
```

## Parameters (stimulus)

| Parameter | Type | Values covered by tests | Meaning |
|---|---|---|---|
| `task_type` | str | `"binary_classification"` | classifier task type |
| `explanation_mode` | str | `"factual"`, `"alternative"` | which explain method |
| `object_level` | str | `"collection"`, `"individual"` | collection or individual |
| `n_top_features` | int | 5 (default), 2 | top feature limit |
| `max_rule_size` | int | 1, 2, 3 | max conjunction size |

## Observation fields

The TIF returns a `ConjunctionObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during the call |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `result_is_none` | `bool` | Whether the returned object is `None` |
| `result_len` | `int \| None` | `len(result)` if the result supports it, else `None` |
| `result_type_name` | `str \| None` | `type(result).__name__` if result is not None |
| `any_has_conjunctive_rules` | `bool` | Whether any item in the collection has `has_conjunctive_rules == True` |
| `object_level` | `str` | The `object_level` parameter passed to the scenario |
| `max_rule_size` | `int` | The `max_rule_size` parameter used |
| `n_top_features` | `int` | The `n_top_features` parameter used |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-EXPL-CONJ-API-001 | API callable without exception | `exception_raised` | `False` |
| CE-REQ-EXPL-CONJ-RETURN-001 | Return value is not None | `result_is_none` | `False` |
| CE-REQ-EXPL-CONJ-RETURN-001 | Collection cardinality preserved | `result_len` | `== n_instances` |
| CE-REQ-EXPL-CONJ-RULE-001 | At least one conjunction rule produced | `any_has_conjunctive_rules` | `True` (when `max_rule_size >= 2`) |
| CE-REQ-EXPL-CONJ-PARAM-001 | No conjunction rules when `max_rule_size=1` | `any_has_conjunctive_rules` | `False` (when `max_rule_size == 1`) |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-EXPL-CONJ-<YYYYMMDD>
claim_ids:
  - CE-CAP-EXPL-CONJ-001
requirement_ids:
  - CE-REQ-EXPL-CONJ-API-001
  - CE-REQ-EXPL-CONJ-RETURN-001
  - CE-REQ-EXPL-CONJ-RULE-001
  - CE-REQ-EXPL-CONJ-PARAM-001
adr_refs:
  - ADR-008
tif_ids:
  - CE-TIF-EXPL-CONJ-001
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
