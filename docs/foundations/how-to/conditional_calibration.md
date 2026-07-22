# Configure Conditional Calibration

Conditional, or Mondrian, calibration partitions calibration data by category
label and fits separate calibrators per category. Use it when uncertainty must be
reported by subgroup or segment.

## Channel A: Inline Labels

Use `bins=` when labels are already available for every calibration and inference
instance.

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)

group_cal = x_cal[:, group_idx].astype(int)
group_test = x_test[:, group_idx].astype(int)

explainer.calibrate(x_cal, y_cal, bins=group_cal)
explanations = explainer.explain_factual(x_test, bins=group_test)
probabilities, interval = explainer.predict_proba(
    x_test,
    uq_interval=True,
    bins=group_test,
)
```

## Channel B: Stored Categorizer

Use `mc=` when category labels are derived from the input matrix. A
`MondrianCategorizer` is created without constructor arguments in the pinned
`crepes` API.

```python
from crepes.extras import MondrianCategorizer

from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_train, y_train)

mc = MondrianCategorizer()
mc.fit(x_cal, f=lambda X: model.predict_proba(X)[:, 1], no_bins=5)

explainer.calibrate(x_cal, y_cal, mc=mc)
explanations = explainer.explain_factual(x_test)
probabilities, interval = explainer.predict_proba(x_test, uq_interval=True)
```

## Resolution Rules

| Calibration state | Inference `bins` | Result |
|---|---|---|
| Stored `mc` | omitted | Categories are derived automatically |
| Stored `mc` | provided | `ConfigurationError` |
| Inline `bins` | omitted | `ValidationError` |
| Inline `bins` | provided | Labels are validated and used |
| Global | omitted | Global calibrated output |
| Global | provided | `ConfigurationError` |

Labels passed at inference must have one value per instance and must belong to
the label vocabulary seen during calibration.

## Recalibration

Conditional state belongs to each calibration call. Calling
`calibrate(x_cal, y_cal)` without `bins=` or `mc=` resets the wrapper to global
calibration. To intentionally reuse a stored categorizer, call:

```python
explainer.calibrate(x_new_cal, y_new_cal, reuse_conditional=True)
```

`reuse_conditional=True` requires a stored `mc`. Inline labels cannot be reused
because they are aligned with the previous calibration set.

## Persistence

Calibrator bins are persisted, but arbitrary categorizer objects are dropped by
pickle and `save_state()`. CE emits a `UserWarning` and INFO log when this
happens. After loading an `mc`-calibrated wrapper, pass explicit `bins=` at
inference time.

`save_state()`/`load_state()` use a non-executable, JSON-safe artifact format
(ADR-031); `load_state()` requires the original (or an equivalent, already
fitted) `learner=` to be supplied again, since learner bytes are never
persisted. See the [persistence caveat](../../practitioner/playbooks/mondrian-calibration.md#persistence-caveat)
for the Mondrian-specific details.

## Minimum Category Size

Aim for at least 30-50 calibration samples per category. Smaller categories can
produce unstable or very wide intervals and should be documented as an assumption
boundary in audits.
