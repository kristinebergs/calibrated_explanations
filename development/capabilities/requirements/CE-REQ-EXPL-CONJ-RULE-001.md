# CE-REQ-EXPL-CONJ-RULE-001 — Multi-Feature Conjunction Rule Semantics

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-RULE-001 |
| obligation_type | behavioral_contract |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | active |
| verification_status | verified |
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) |
| supersedes | CE-REQ-EXPL-CONJ-001 (partial) |
| tif_refs | CE-TIF-EXPL-CONJ-001 |
| verification_strength | behavioral_contract |
| evidence_level | raw_evidence |

## Scope

Public API:
- `CalibratedExplanations.add_conjunctions(n_top_features, max_rule_size)` — factual collection
- `AlternativeExplanations.add_conjunctions(n_top_features, max_rule_size)` — alternative collection

Applicable condition: `max_rule_size >= 2` and `n_top_features >= 2` (sufficient features for
conjunction generation to be possible).

Applicable workflow: `WrapCalibratedExplainer` → `fit` → `calibrate` → `explain_factual` /
`explore_alternatives` → `add_conjunctions(max_rule_size=2, n_top_features=5)`.

## Observable behavior

When `add_conjunctions(max_rule_size >= 2)` is called on a collection with sufficient features
(`n_top_features >= 2`), and the dataset has at least 2 informative features, at least one
explanation item in the collection must have `has_conjunctive_rules == True`.

This must hold for any `max_rule_size >= 2`, not only the minimum value. A caller passing
`max_rule_size=3` is requesting rules that may combine up to 3 features; the API must
accept that value and produce at least one conjunction rule given a suitably informative
dataset.

`has_conjunctive_rules` is a public boolean attribute on individual explanation items
(FactualExplanation, AlternativeExplanation) that is set to `True` by `add_conjunctions`
when at least one multi-feature conjunction rule was successfully created for that item.

## Acceptance criterion

For `factual = explainer.explain_factual(X_test)` after
`factual.add_conjunctions(max_rule_size=N)` where `N >= 2`:

1. At least one item in `[factual[i] for i in range(len(factual))]` has
   `item.has_conjunctive_rules == True`.

This is verified by CE-TIF-EXPL-CONJ-001 with both `max_rule_size=2` and `max_rule_size=3`,
`n_top_features=5`, using a dataset with `n_informative=3`.

The fixture is designed so that conjunction generation is expected to succeed for at
least one instance. If the fixture produces no conjunctions (all instances have
`has_conjunctive_rules == False`), the TIF scenario should surface that as a gap rather
than a pass.

## TIF reference

**TIF ID:** CE-TIF-EXPL-CONJ-001

TIF scenario function: `run_conjunction_tif_scenario()`
Observation field verified: `any_has_conjunctive_rules` (must be `True` when `max_rule_size >= 2`)

## Verification method

Automated pytest tests calling CE-TIF-EXPL-CONJ-001 with both `max_rule_size=2` and
`max_rule_size=3`, each asserting `any_has_conjunctive_rules == True`.

## Verification targets

- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_produce_conjunctive_rules_when_max_rule_size_two`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_produce_conjunctive_rules_when_max_rule_size_three`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_control_conjunction_generation_via_max_rule_size[1-False]`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_control_conjunction_generation_via_max_rule_size[2-True]`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_control_conjunction_generation_via_max_rule_size[3-True]`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| tif_ids | yes |
| dataset_id | yes (sklearn make_classification, n_informative=3, n_features=4, random_seed=42) |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies that conjunctive rules are produced for at least one instance
when `max_rule_size >= 2` and the fixture has sufficient features.

It does not verify:
- That conjunction rules are semantically superior to single-feature rules.
- The exact number of conjunction rules per instance.
- Behavior on datasets with fewer features than `n_top_features`.
- Runtime performance.
- Calibration or statistical validity of conjunctive rules.
