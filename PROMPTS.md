# Using Calibrated Explanations with AI Agents (prompts)

## Agent checklist (CE-first)
1. Verify a calibration set exists (required).
2. Use `WrapCalibratedExplainer` for all workflows.
3. Return point estimates/probabilities and their intervals; include the factual
   rule table only when a factual explanation was requested.

## Mapping natural language → API call
"Explain this prediction with uncertainty bounds"
→ `WrapCalibratedExplainer.explain_factual()`

"Show alternatives that would change this prediction"
→ `WrapCalibratedExplainer.explore_alternatives()`

"I need feature importance with uncertainty bounds"
→ `WrapCalibratedExplainer.explain_factual()` and extract the feature-weight uncertainty envelope from the returned table

## Minimal agent template
1. Verify `x_cal` & `y_cal` (calibration set) exist.
2. Instantiate wrapper: `explainer = WrapCalibratedExplainer(model)`
3. `explainer.fit(x_proper, y_proper)` then `explainer.calibrate(x_cal, y_cal, feature_names=...)`
4. Select method:
   - factual → `explain_factual(X_query)`
   - alternatives → `explore_alternatives(X_query)`
   - probabilities → `predict_proba(X_query, uq_interval=True)`
5. Return the requested output (point estimate/probability, interval, or
   alternatives); include the rule table only for factual explanations.

## Example agent response skeleton
- "Calibrated probability (class 1): 0.72 [0.65, 0.80]"
- "Top 3 contributing features (value : feature : weight [low,high])"
- "Alternatives: change X[2] from 5.1 → 3.0 to flip prediction"

## Success response example (JSON)
```json
{
  "scenario": "binary-classification",
  "factual_table": "Value : Feature : Weight [Low, High]...",
  "probability": 0.72,
  "probability_interval": {"low": 0.65, "high": 0.80}
}
```

## Response shape by task

- **Classification**: `predict_proba` returns calibrated class probabilities;
  with `uq_interval=True`, also a `(low, high)` probability interval per class.
- **Regression (intervals)**: `predict` returns a calibrated point prediction;
  with `uq_interval=True` and `low_high_percentiles=(a, b)`, also a `(low, high)`
  conformal interval on the target scale (not a probability).
- **Probabilistic regression (threshold query)**: `predict_proba(..., threshold=t)`
  returns the calibrated probability of the threshold event, with an optional
  `(low, high)` interval on that probability.

For prediction-only or alternative-only requests, return only the output the
caller asked for (point estimate, interval, or alternatives) — do not fabricate
a factual rule table when one was not requested.

Alternatives describe feature conditions or input changes under which the
model's calibrated output would change; they are not guaranteed real-world
interventions.
