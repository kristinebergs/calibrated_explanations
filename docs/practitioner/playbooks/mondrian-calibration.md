# Mondrian (Conditional) Calibration Playbook

Mondrian calibration fits separate calibrators for user-defined category labels.
Use it when uncertainty must be inspected by subgroup, segment, region, or another
auditable partition.

## Inline Category Labels

Use inline `bins` when the category labels are already known for both calibration
and inference instances.

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)

group_cal = x_cal[:, protected_feature_idx].astype(int)
group_test = x_test[:, protected_feature_idx].astype(int)

explainer.calibrate(x_cal, y_cal, bins=group_cal)
factual = explainer.explain_factual(x_test, bins=group_test)
probabilities, interval = explainer.predict_proba(
    x_test,
    uq_interval=True,
    bins=group_test,
)
```

## MondrianCategorizer

Use `mc=` when category labels should be derived from the input matrix. The
categorizer is stored on the wrapper and applied automatically at inference time.

```python
from crepes.extras import MondrianCategorizer

from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)

mc = MondrianCategorizer()
mc.fit(x_cal, f=lambda X: model.predict_proba(X)[:, 1], no_bins=5)

explainer.calibrate(x_cal, y_cal, mc=mc)
factual = explainer.explain_factual(x_test)
probabilities, interval = explainer.predict_proba(x_test, uq_interval=True)
```

## Consistency Rules

- Calibrate with exactly one conditional channel: `bins=`, `mc=`, or
  `reuse_conditional=True`.
- If calibration used inline `bins=`, every inference call must pass matching
  per-instance `bins=`.
- If calibration used `mc=`, inference calls omit `bins=` because labels are
  derived automatically.
- If calibration was global, inference calls must not pass `bins=`.
- Test-time labels must belong to the label vocabulary seen during calibration.

Recalibrating without `bins=` or `mc=` resets the wrapper to global calibration.
Use `calibrate(..., reuse_conditional=True)` only when you intentionally want to
reuse a stored categorizer on a new calibration set.

## Persistence Caveat

Calibrator-level Mondrian bins round-trip through persistence, but arbitrary
`mc` categorizer objects are not portable. Pickle and `save_state()` warn when a
configured `mc` is dropped. After loading, pass explicit `bins=` at inference time
for wrappers that were saved from an `mc`-calibrated state.

`save_state()`/`load_state()` persist only JSON-safe declarative data (ADR-031);
no wrapper or calibrator pickle bytes are written. `load_state()` requires the
original (or an equivalent, already fitted) learner via `load_state(path,
learner=...)`, since learner bytes are never persisted. Artifacts saved by a
pre-hardening release (schema v1/v2) are rejected by `load_state()`; migrate
them with a trusted older calibrated-explanations environment and re-save.

## Minimum Category Size

Each category needs enough calibration samples for useful intervals. As a rule of
thumb, aim for at least 30-50 calibration samples per category and document any
smaller category as an assumption boundary.

See also: [](../../foundations/how-to/conditional_calibration.md).
