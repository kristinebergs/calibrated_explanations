# CE-TIF-VIZ-001 — Visualization Smoke Test TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-VIZ-001 |
| executable | `development/capabilities/verification/tif/tif_visualization.py` |
| entry_functions | `run_visualization_tif_scenario()` |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-VIZ-SMOKE-001 | `exception_raised` (must be `False`) |

## Claims served

- CE-CAP-VIZ-001

## ADR refs

- ADR-023

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.explain_factual(X_test)` → `CalibratedExplanations`
- `explanations.plot(show=False)` — no-raise smoke test with Agg backend

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
- 2 instances for `X_test`
- Remaining 118 split 65%/35% into proper-train and calibration

The model used is `RandomForestClassifier(n_estimators=10, random_state=42)`.

The TIF sets `matplotlib.use('Agg')` before any plot calls and calls `plt.close('all')`
after the test to clean up figure state.

**Dependency note:** requires `matplotlib`. If not installed, the TIF will set
`exception_raised=True` with `exception_type='ImportError'`. Tests should use
`pytest.importorskip('matplotlib')` before calling this TIF.

## WrapCalibratedExplainer workflow

```python
import matplotlib
matplotlib.use('Agg')

explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

explanations = explainer.explain_factual(X_test)
explanations.plot(show=False)
plt.close('all')
```

## Observation fields

The TIF returns a `VizObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `exception_raised` | `bool` | Whether an exception was raised during `plot()` |
| `exception_type` | `str \| None` | Exception class name if raised, else `None` |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-VIZ-SMOKE-001 | plot() completes without exception | `exception_raised` | `False` |

## TIF constraints

This TIF does NOT:
- Assert visual correctness (colors, layout, axes, labels)
- Construct explanation objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
