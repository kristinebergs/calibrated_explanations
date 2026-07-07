# Documentation Boundary Evidence — v0.11.4 Hardening Pass

**Date:** 2026-06-22
**Status:** CLOSED — documentation review completed and boundary statements added

---

## Summary

This record closes the three outstanding documentation-boundary requirements:

- `CE-REQ-EXPL-DOC-001` — Factual explanation documentation boundary
- `CE-REQ-EXPL-ALT-DOC-001` — Alternative explanation documentation boundary
- `CE-REQ-EXPL-CONJ-DOC-001` — Conjunction documentation boundary

All three are TIF-exempt (`tif_exemption: documentation_boundary`). They are verified
through documentation review, not through a behavioral TIF scenario.

---

## Curated Evidence Fields

| Field | Value |
|---|---|
| requirement_ids | CE-REQ-EXPL-DOC-001, CE-REQ-EXPL-ALT-DOC-001, CE-REQ-EXPL-CONJ-DOC-001 |
| tif_ids | none — TIF-exempt documentation-boundary requirements |
| verification_strength | documentation_boundary |
| evidence_level | curated_summary |
| package_version | v0.11.5-dev |
| commit_sha | 5d02a2c282256ab445a11c864a1bbbc59c5082c6 |
| reviewer | Tuwe Löfström-Cavallin |
| result | PASS |
| raw_evidence_ref | none — TIF-exempt documentation-boundary review |

---

## Reviewed doc paths

| Path | Type |
|---|---|
| `src/calibrated_explanations/core/wrap_explainer.py` → `WrapCalibratedExplainer.explain_factual` | Docstring |
| `src/calibrated_explanations/core/wrap_explainer.py` → `WrapCalibratedExplainer.explore_alternatives` | Docstring |
| `src/calibrated_explanations/explanations/explanations.py` → `CalibratedExplanations.add_conjunctions` | Docstring |
| `src/calibrated_explanations/core/calibrated_explainer.py` → `CalibratedExplainer.explain_factual` | Docstring |
| `src/calibrated_explanations/core/calibrated_explainer.py` → `CalibratedExplainer.explore_alternatives` | Docstring |

---

## Review findings

### CE-REQ-EXPL-DOC-001 — Factual explanation documentation boundary

**Acceptance criteria checked:**
1. Documentation does not assert scientific validity of feature attributions without qualification. ✓
2. Calibration assumption (exchangeability) is stated in docstrings or RTD. ✓

**Findings:**
- `CalibratedExplainer.explain_factual` has a detailed parameter docstring that covers
  threshold semantics and fit/calibrate preconditions.
- `WrapCalibratedExplainer.explain_factual` had no explicit assumption boundary statement.
  **Action taken:** Added a `Notes` section stating the API-contract scope, the
  exchangeability assumption dependency, and that feature attribution magnitudes are not
  causal importances.
- Neither docstring asserted scientific validity without qualification. ✓

**Result:** PASS

---

### CE-REQ-EXPL-ALT-DOC-001 — Alternative explanation documentation boundary

**Acceptance criteria checked:**
1. Documentation does not assert that counterfactual scenarios are physically or
   distributionally achievable without qualification. ✓
2. The exchangeability assumption is stated in docstrings or RTD. ✓

**Findings:**
- `CalibratedExplainer.explore_alternatives` delegates to the underlying explainer and
  carries a "See Also" reference.
- `WrapCalibratedExplainer.explore_alternatives` had no explicit assumption boundary
  statement. **Action taken:** Added a `Notes` section stating that alternative
  explanations describe feature changes relative to the current instance, that
  achievability is not guaranteed, and that the exchangeability assumption applies.
- Neither docstring asserted physical or distributional achievability without
  qualification. ✓

**Result:** PASS

---

### CE-REQ-EXPL-CONJ-DOC-001 — Conjunction documentation boundary

**Acceptance criteria checked:**
1. Documentation does not assert scientific superiority of conjunction rules without
   qualification. ✓
2. Docstrings reference what the API guarantees and what is out of scope. ✓

**Findings:**
- `CalibratedExplanations.add_conjunctions` documented the operation and parameters but
  did not state that conjunction rule superiority is not asserted, nor that performance
  guarantees are absent. **Action taken:** Added a `Notes` section stating the API
  contract scope, that conjunction rules are not asserted to be superior to single-feature
  rules, and that calibration validity depends on the same exchangeability assumption as
  the underlying factual explanations.
- `FactualExplanation.add_conjunctions` and `AlternativeExplanation.add_conjunctions` are
  the per-instance implementations; their docstrings are adequate for the behavioral
  contract; the collection-level docstring is the primary documentation surface.
- No docstring asserted scientific superiority of conjunction rules. ✓

**Result:** PASS

---

## Assumption boundary

This curated evidence records a documentation-boundary review, not a behavioral
verification. It confirms that the reviewed docstrings:

- Do not overclaim (no assertions of scientific validity, achievability, or superiority
  without explicit qualification).
- State the calibration assumption (exchangeability) for factual and alternative
  explanations.
- State the API-contract scope and what the implementation does not guarantee.

This evidence does NOT prove:

- Statistical validity of calibrated outputs for any specific instance or dataset.
- That feature attributions are causally correct.
- That alternative explanations describe achievable real-world changes.
- That conjunction rules outperform single-feature rules on any task.
- That documentation is complete or that users have read it.

Behavioral verification for these capabilities is provided by the separate behavioral TIF
chains: CE-TIF-EXPL-001 (factual/alternative), CE-TIF-EXPL-CONJ-001 (conjunctions).
