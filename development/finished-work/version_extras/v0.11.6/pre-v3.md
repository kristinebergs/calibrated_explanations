# Pre-v1 Unknown-Unknowns Audit — calibrated_explanations

> **Status: COMPLETE (2026-07-10) — dispositioned into `v0.11.6_plan.md` Tasks 39–44
> the same day** (release preparation renumbered 39 → 45). Mapping: B1/M7/M8 →
> Task 39 (blocker, non-deferrable); H1/M2·S3 → Task 40; H2/H3 + QUICK_API reject
> claim → Task 41; M1/M3/L1 → Task 42; H4/L6 → Task 43; M4/M5 → Task 44.
> M6 stays ledgered under Task 38 item 8; L2–L5 and S2 are recorded as non-gating
> post-v1 leftovers at the end of Task 44.
> All findings are based on current repo evidence and runnable repros against
> commit `fe9b9478` (working tree).

## Executive summary

- **Current commit audited:** `fe9b9478` on `main` (2026-07-10), with uncommitted
  working-tree changes present (docs, `scripts/local_checks.py`, plan file).
- **Version state:** dev window (`0.11.6-dev` / `0.11.6.dev0`); version alignment
  check passes per Task 36 policy. Not treated as a defect.
- **Commands run:** full `pytest -q` (exit 0, ~3,070 tests), `make deprecation-closure`
  (exit 0), `make warning-policy` (exit 0), `make capability-chain-check` (pass),
  `python scripts/quality/check_version_alignment.py --check` (pass),
  `python scripts/quality/check_packaging_artifacts.py --clean-dist` (pass),
  `python scripts/quality/snapshot_public_api.py` (wrote snapshot),
  `python -m sphinx -W --keep-going docs docs/_build/html` (**exit 1**),
  `make local-checks-release` (**exit 2**), anti-pattern/test-helper/meta-test/
  import-graph quality scripts (all pass), plus ~10 targeted behavioral repro scripts.
- **Commands not run:** `make local-checks-full` (its Ruff step shares the same
  `scripts/` failure that already reddens `local-checks-release`; rerunning adds no
  information), `make local-checks-task TASK=1..38` full sweep (plan's own status
  note already records TASK=38 red on the same Ruff findings).
- **Overall release risk: YELLOW** — core prediction/threshold/label behavior is
  demonstrably solid (the v0.11.6 hardening landed), but the *presentation* surfaces
  are not v1-ready: one silent-wrong-output defect on the README front-page path
  (B1), a broken multiclass narrative (M7), a Windows-fatal plugin CLI (H4), one
  public reject-contract mismatch (H1), and two currently red pre-tag gates (H2, H3)
  must close before tag.
- **Highest-risk finding:** B1 — `to_narrative()` reports the *explained class* as
  "Prediction" for binary classification, printing `Prediction: 1` for instances the
  model actually predicts as class 0.

---

## Blockers

### B1 — `to_narrative()` states the wrong predicted label for binary classification

- **ID:** B1
- **Title:** Narrative "Prediction:" line renders the explained class, not the prediction
- **Severity:** blocker (silent wrong output on the README quickstart path)
- **Current evidence:** With the README Quick Start (breast_cancer, RF, seed 42),
  instance `X_te[:1]`: `wrapper.predict` → `0` (correct; true label 0;
  `predict_proba` → `[0.971, 0.029]`), but
  `exp[0].to_narrative(output_format="text", expertise_level="advanced")` prints:

  ```
  Prediction: 1
  Calibrated Probability: 0.029
  ```

  The narrative asserts the opposite class of the actual calibrated prediction.
- **Minimal repro:**

  ```python
  from sklearn.datasets import load_breast_cancer
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from calibrated_explanations import WrapCalibratedExplainer
  d = load_breast_cancer()
  X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=42)
  X_pr, X_cal, y_pr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
  w = WrapCalibratedExplainer(RandomForestClassifier(random_state=42))
  w.fit(X_pr, y_pr); w.calibrate(X_cal, y_cal, feature_names=d.feature_names)
  print(w.predict(X_te[:1]))                       # [0]
  print(w.explain_factual(X_te[:1])[0].to_narrative(
      output_format="text", expertise_level="advanced").splitlines()[0])
  # -> "Prediction: 1"   (wrong)
  ```
- **Affected public surface:** `FactualExplanation.to_narrative` /
  `AlternativeExplanation.to_narrative` (binary classification, all expertise levels;
  both wrapper and direct core, since it lives on the explanation object). README's
  front-page example prints this output.
- **Why this matters for v1:** This is exactly the class of "silent wrong output"
  that Tasks 30–32 declared non-deferrable — but those tasks fixed `predict`, not the
  narrative. A regulator-facing narrative that states the wrong predicted label is
  worse than a wrong probability.
- **Likely root cause:** `src/calibrated_explanations/core/narrative_generator.py:227`
  uses `rules_dict.get("classes")` as "the predicted class". For binary
  classification CE explanations, `classes` is the **explained class** (always 1),
  not the prediction. The template `Prediction: {label}` then renders it. The
  neighbouring fallback branch (`bp > 0.5 → "1"`) shows the intended semantics.
- **Test gap:** `tests/unit/core/test_narrative_generator.py` tests `{label}` only via
  mocks/fallbacks; `tests/capabilities/test_narrative_contracts.py` asserts the
  narrative is a non-empty string. No test asserts the narrative's stated prediction
  equals `predict(x)`.
- **Minimal remediation:** derive the label from the calibrated prediction (e.g.
  `predict >= 0.5` → positive label else negative, or thread the actual predicted
  class into the narrative context); add a behavior test asserting narrative label ==
  `wrapper.predict(x)[0]` for a class-0 and a class-1 instance.
- **Verification command:** the repro above; expected first line `Prediction: 0`.

---

## High-risk findings

### H1 — RejectResult metadata contract wrong for `predict`/`predict_proba` with subset policies

- **ID:** H1
- **Title:** `matched_count`/`source_indices` documented contract does not match prediction-path behavior
- **Severity:** high (documented public ABI contract vs. actual payload)
- **Current evidence:** `docs/practitioner/advanced/reject-policy.md` ("ABI/API
  Guarantees for RejectResult") documents `matched_count` as "Number of payload rows
  matched by `ONLY_REJECTED`/`ONLY_ACCEPTED` (`None` for `FLAG`)" and
  `source_indices` as "Source-row mapping from returned payload rows to original
  input rows". Actual behavior (binary classification, seed 42, 75-row batch):

  | policy | payload rows | `source_indices` len | `matched_count` |
  |---|---|---|---|
  | ONLY_ACCEPTED | 75 (full batch) | 43 | `None` |
  | ONLY_REJECTED | 75 (full batch) | 32 | `None` |

  The payload is the full batch, so the documented "payload rows → original rows"
  mapping is unusable (75 ≠ 43/32), and `matched_count` is `None` where docs say it is
  set. Same for `predict_proba`. Explanation entrypoints are coherent
  (13 explanations / 13 source_indices / original_count 20).
- **Minimal repro:** `wrapper.predict(X, reject_policy=RejectPolicy.ONLY_ACCEPTED)` →
  compare `len(result.prediction)`, `len(result.metadata["source_indices"])`,
  `result.metadata["matched_count"]`.
- **Affected public surface:** `WrapCalibratedExplainer.predict` / `predict_proba`
  with `reject_policy=ONLY_ACCEPTED|ONLY_REJECTED`; `RejectResult.metadata`.
- **Why this matters for v1:** the reject metadata table is written as an explicit
  stability guarantee ("These guarantees help you write robust production code").
  Consumers following it will mis-index predictions.
- **Likely root cause:** `core/reject/orchestrator.py:1231` computes the prediction on
  the full batch before subsetting; `matched_count` is only assigned inside the
  `explain_fn` branch (line 1262); `source_indices` is filtered regardless (1249–1252).
- **Test gap:** `tests/unit/core/test_reject.py` asserts `matched_count` semantics on
  the explanation path only; no test asserts prediction-path payload/subset agreement.
- **Minimal remediation (choose one, then test it):** either (a) subset the
  prediction payload for ONLY_* policies and set `matched_count`, or (b) document that
  prediction payloads are always full-batch and `source_indices`/`matched_count`
  describe the *matched subset*, and fix the docs table. (b) is the non-breaking option.
- **Verification command:** probe above; payload/`source_indices`/`matched_count`
  must satisfy whichever contract is chosen.

### H2 — Strict Sphinx docs build (Task 39 gate 7) currently fails

- **ID:** H2
- **Severity:** high (red pre-tag gate)
- **Current evidence:** `python -m sphinx -W --keep-going docs docs/_build/html` →
  exit 1, "build finished with problems, 7 warnings (with warnings treated as
  errors)". All 7 are in `docs/upgrade/v0.11.6-upgrade-checklist.md`: six
  `myst.header` "Document headings start at H2, not H1" (file begins with
  `## v0.11.6 Upgrade Checklist`) and one `myst.xref_missing` (line 108 links to
  repo-root `CHANGELOG`, outside the docs tree).
- **Why this matters for v1:** Task 39 step 7 requires this gate to exit 0; the file
  causing it is itself the v0.11.6 upgrade checklist.
- **Minimal remediation:** make the first heading H1 and demote/retitle the rest;
  point the CHANGELOG link at a resolvable target (GitHub URL is already allowlisted
  in `docs/conf.py` linkcheck).
- **Verification command:** rerun the sphinx command; expect exit 0.

### H3 — `make local-checks-release` (Task 39 gate 4) currently fails on Ruff

- **ID:** H3
- **Severity:** high (red pre-tag gate)
- **Current evidence:** `make local-checks-release` → exit 2. The Ruff step
  (`ruff check src tests scripts`) reports **105 errors**, all under `scripts/`
  (`scripts/anti-pattern-analysis/find_shared_helpers.py`,
  `scripts/over_testing/detect_redundant_tests.py`, `scripts/local_checks.py`, etc.).
  The v0.11.6 plan's Task 38 status note acknowledges "pre-existing Ruff findings
  under `scripts/`" and calls TASK=38 "not fully gate-closed" — but the same findings
  also redden `local-checks-release` itself, which is a non-negotiable Task 39 gate.
- **Minimal remediation:** fix or explicitly per-file-ignore the `scripts/` findings
  (16 are `--fix`able); do not narrow the Ruff target set to make the gate pass.
- **Verification command:** `make local-checks-release` → exit 0.

### H4 — Plugin CLI crashes with `UnicodeEncodeError` on Windows (cp1252 stdout)

- **ID:** H4
- **Title:** `ce plugins ...` and `ce.plugins ...` are unusable when stdout is not UTF-8
- **Severity:** high (shipped CLI fatally broken on the default Windows encoding)
- **Current evidence:** On Windows 11 / Python 3.14 with stdout redirected or piped
  (cp1252): both `ce plugins list` and `ce.plugins list` exit 1 with
  `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f512' in
  position 0`. Root cause: `src/calibrated_explanations/plugins/cli.py:638` prints an
  unconditional 🔒-prefixed banner in `main()`, so **every** subcommand crashes
  before doing anything.
- **Minimal repro (Windows, non-UTF-8 stdout):** `ce plugins list > out.txt` → exit 1.
- **Affected public surface:** `ce plugins` subcommand tree and the `ce.plugins`
  console script (documented in `docs/api/cli.md`, updated this cycle).
- **Why this matters for v1:** the CLI docs were just polished for v0.11.6; the
  documented commands crash on the project's own primary development platform
  whenever output is redirected (scripts, CI logs, `| findstr`).
- **Likely root cause:** emoji in CLI output with no encoding guard; Python only
  defaults to UTF-8 for interactive consoles on Windows, not pipes/files.
- **Test gap:** CLI tests capture output via pytest (UTF-8-safe internals) and never
  exercise a cp1252 stream.
- **Minimal remediation:** drop the emoji (or wrap the banner print in a
  `try/except UnicodeEncodeError` with an ASCII fallback); add a test that encodes
  CLI output to cp1252 with `errors="strict"`.
- **Verification command:** `ce plugins list > NUL` on a cp1252 console → exit 0.

---

## Medium-risk findings

### M1 — QUICK_API.md classification interval example crashes (`IndexError`)

- **Current evidence:** QUICK_API.md line 40:
  `print("P(class=1) =", probs[0, 1], "interval =", low[0, 1], high[0, 1])`.
  Runtime: `predict_proba(X, uq_interval=True)` on binary classification returns
  `low`/`high` as **1-D** arrays (shape `(n,)`), so `low[0, 1]` raises
  `IndexError: too many indices`. QUICK_API corrections were a claimed v0.11.6 item;
  this one is still wrong.
- **Remediation:** `low[0]`, `high[0]` (matching the probabilistic-regression example
  which already indexes 1-D).
- **Verification:** run the four QUICK_API snippets end-to-end.

### M2 — QUICK_API.md documents the wrong return type for reject-aware explanations

- **Current evidence:** QUICK_API.md line 13 claims any prediction/explanation API
  with a non-NONE `RejectPolicy` "returns a `RejectResult` envelope" whose
  `explanation` field carries the payload. Runtime:
  `explain_factual(..., reject_policy=FLAG)` returns `RejectCalibratedExplanations`
  (no `.prediction`/`.explanation` attributes; has `.explanations`). The
  reject-policy guide documents the collection behavior correctly; QUICK_API
  contradicts both it and the runtime.
- **Remediation:** correct QUICK_API to distinguish prediction entrypoints
  (`RejectResult`) from explanation entrypoints (reject-aware collections).

### M3 — README fairness example fails as written

- **Current evidence:** README "Fairness-Aware Explanations" shows
  `explainer.explain_factual(X_query, bins=X_query[:, gender_col_index])` with no
  surrounding context. Against an explainer calibrated without bins this raises
  `ConfigurationError: This explainer was not calibrated with Mondrian bins...`.
  The example omits the required `calibrate(..., bins=...)` step.
- **Remediation:** add the calibrate-with-bins line to the README snippet.

### M4 — `threshold=` silently ignored on classification `predict`/`predict_proba`

- **Current evidence:** On a calibrated binary classifier,
  `predict(X, threshold=0.5)` and even `predict(X, threshold=(3, 1))` (malformed,
  reversed tuple) return normal predictions identical to `predict(X)` — no error, no
  warning. The same explainer's `explain_factual(X, threshold=0.5)` correctly raises
  ("The threshold parameter is only supported for mode='regression'").
  Root cause: `core/prediction/orchestrator.py` only consults `threshold` inside the
  `"regression"` mode branch; the classification branch never validates it, and
  `threshold` is in the predict/predict_proba allow-list for both modes.
- **Why this matters:** ADR-038's stated goal is that invalid call-time configuration
  fails fast; a user who believes they set a decision threshold gets silently
  different semantics than they asked for.
- **Remediation:** raise `ValidationError` when `threshold` is supplied in
  classification mode on predict/predict_proba (mirroring the explain path).
- **Verification:** repro script in scratchpad (`probe_threshold_cls.py`).

### M5 — Calibration labels disjoint from training labels silently accepted

- **Current evidence:** model fitted on labels `{0, 1}`, then
  `calibrate(X_cal, y_cal)` with `y_cal` labels `{0, 2}` is accepted without warning;
  subsequent `predict` returns labels from `{0, 2}` — the class universe silently
  becomes the calibration set's, while the learner's `classes_` remains `{0, 1}`.
  (Single-class calibration is correctly rejected with `ValidationError`; this
  neighbouring path is not validated.)
- **Remediation:** validate `set(y_cal) ⊆ set(learner.classes_)` (or document the
  relabeling contract explicitly) at `calibrate` time.

### M6 — `serialization.to_json` still fails with bare `AttributeError` (Task 38 item 8 closed docs-only)

- **Current evidence:** `from calibrated_explanations import serialization;
  serialization.to_json(explanation)` on a runtime `FactualExplanation` →
  `AttributeError: 'FactualExplanation' object has no attribute 'metadata'`.
  Task 38 item 8 is marked "implemented", but the implementation was documentation
  routing only; the exported top-level function still raises an untyped error for the
  objects users actually hold.
- **Remediation (post-v1 acceptable if ledgered):** raise `SerializationError` (or
  `ValidationError`) with a pointer to `collection.to_json()`.

### M7 — Multiclass narratives render empty `Runner-up Class:  (margin )` placeholders

- **Current evidence:** All six multiclass templates in
  `src/calibrated_explanations/templates/explain_template.yaml` (lines 88–141) contain
  `Runner-up Class: {runner_up_class} (margin {margin_value})`, but
  `core/narrative_generator.py:285-286` hardcodes `"runner_up_class": ""` and
  `"margin_value": ""` — they are never populated. Every multiclass narrative at
  every expertise level therefore contains the literal line
  `Runner-up Class:  (margin )`. Reproduced with a 3-class RF (seed 42).
  Note the runner-up information is trivially available in the explanation's
  `__full_probabilities__` payload.
- **Why this matters:** `to_narrative` is a headline v1 feature; visibly broken
  output in the default templates undermines it, and no test catches it because
  narrative tests only assert non-empty strings.
- **Remediation:** populate both variables from the full probability vector, or
  remove the line from the templates.

### M8 — Calibrated probability intervals can exceed 1.0 and are displayed unclipped

- **Current evidence:** Multiclass factual explanation (3-class RF, seed 42):
  `explanation.prediction == {'predict': 0.944, 'low': 0.933, 'high': 1.049, ...}`,
  and the narrative prints `Prediction Interval: [0.933, 1.049]` — a probability
  interval with an upper bound above 1 presented to end users of a *calibration*
  library. The out-of-range value is in the stored explanation payload itself, so
  JSON exports carry it too.
- **Remediation:** decide the contract: clip interval bounds to [0, 1] at the
  interval-learner or presentation layer (and test it), or document why bounds may
  exceed [0, 1]. Either way the narrative should not show P > 1 without explanation.

---

## Low-risk irregularities

- **L1 — `audio.py`/`vision.py` point at non-existent extras.** Both shims raise
  `MissingExtensionError("... Install with: pip install calibrated_explanations[audio]")`
  but `pyproject.toml` defines no `audio` or `vision` extras — the instructed command
  installs nothing. Fix the message or add the extras.
- **L2 — timestamped public-API snapshots accumulate.**
  `scripts/quality/snapshot_public_api.py` writes
  `scripts/tests/benchmarks/api_public_<timestamp>.txt` (several committed, one new
  untracked from this audit; also uses deprecated `datetime.utcnow()`). There is no
  stable "current" baseline for Task 39 step 8's "diff reviewed" to diff against.
- **L3 — committed generated docs with BOMs.** 13 generated files under
  `docs/_autosummary/*.rst` are tracked and BOM-prefixed (Windows-generated).
  Harmless but regenerable cruft in the repo.
- **L4 — test-only hooks ship in the wheel.** `plugins/_testing.py`,
  `testing/parity_compare.py`, and `*_for_testing()` functions in
  `core/config_helpers.py`, `core/config_manager.py`, `plotting.py`
  (`reset_plotting_config_manager`), `core/explain/_guarded_explain.py` are packaged.
  Not called at import time; `check_no_test_helper_exports` sanctions them — but they
  are mutation hooks on process-global state exposed to production users.
- **L5 — empty `perf/` package still ships.** `perf/__init__.py` is an intentional
  empty tombstone ("v0.11.0 removed the ... facade"); consider whether an
  ImportError-with-guidance would serve users better than an empty module at v1.
- **L6 — narrative text is unprintable on cp1252 consoles.** The advanced templates
  embed `≈` (U+2248) and `⚠️`, so the README's own
  `print(exp[0].to_narrative(...))` line raises `UnicodeEncodeError` on Windows when
  stdout is redirected/piped. Same hazard class as H4 but in user code rather than
  CE's; consider ASCII-safe default templates or a docs note.

---

## Suspicious but unconfirmed

- **S2 — full pytest run prints no summary line.** `pytest -q` exits 0 but the
  captured output ends at the `[100%]` progress line with no `N passed` summary
  (the focused deprecation-closure run does print one). Possibly a reporting plugin
  or output-capture quirk; worth confirming CI sees real summaries. No evidence of
  hidden failures (exit code 0 verified directly).
- **S3 — `matched_count`-adjacent undocumented metadata keys.** `prediction_set`,
  `raw_total_examples`, `raw_reject_counts`, `threshold_source`, `schema_version`,
  `effective_w` appear in `RejectResult.metadata` but not in the documented contract
  table ("contains at least" wording makes this legal; flagging for deliberate
  inclusion/exclusion at v1).

---

## False positives / already fixed (verified against current source)

- Classification label semantics (Tasks 30/31): probed `{0,1}`, `{1,2}`, `{0,2}`,
  `{5,9}`, `{1,2,3}`, strings, booleans — predictions stay in the original label set,
  dtype preserved (int64/`<U4`/bool), proba columns align, calibrated vs uncalibrated
  and reject-envelope vs plain predictions agree. Clean.
- Threshold tuple validation (Task 32): reversed/equal/non-numeric/wrong-length
  tuples all raise `ValidationError`; reversed percentiles raise; `predict_proba`
  without threshold on regression raises. Clean.
- ADR-038 fail-fast: unknown kwargs raise `ConfigurationError` on all six wrapper
  methods and on core `predict`/`predict_proba`; removed aliases (`guarded=`,
  `normalize=`, `format=`, `narrative_format=`) raise with migration text. Core
  `explain_factual`/`explore_alternatives` pass unknown kwargs to plugins **by
  documented ADR-038 decision** (experimental-surface exception); `explain_fast` is
  fully typed (bare `TypeError`) per the same addendum. Not defects.
- Warning-policy fallback enforcement: suspected a case-sensitivity hole in
  `error:.*fall.*back.*:UserWarning`; disproven empirically — PlotSpec/legacy and
  cache fallback messages are promoted to error under repo pytest config, generic
  `UserWarning`s ignored.
- Single-class calibration: rejected with `ValidationError` on both wrapper and core.
- `make deprecation-closure`, `make warning-policy`, `make capability-chain-check`,
  packaging artifact check (LICENSE, py.typed, schemas, templates present;
  single top-level package): all green.
- Mondrian bins pickle: explicit-`bins` calibration round-trips predictions
  exactly; post-load `predict` without `bins=` fail-fasts with actionable
  `ValidationError` (not silent-global). The `mc=` callable warn-and-drop contract
  is a recorded Task 38 decision (item 9).
- `pip check` clean; version sources aligned per Task 36 dev-window policy.
- Multiclass narrative class labeling: unlike binary (B1), multiclass narratives
  correctly describe the *predicted* class ("Calibrated Probability for class 3"
  when `predict` → 3). The wrong-label defect is binary-specific.
- CWD-dependent plugin trust behaves exactly as the Task 38 item 3/4 decisions
  record: from repo root `{demo.renderer, plugin.trusted, renderer, target}` are
  trusted (test fixtures in repo `pyproject.toml`); from any other CWD the set is
  empty. Deferred-with-owner; no new evidence of production leakage beyond the
  recorded repo-root caveat.
- `Moffran/...` GitHub URLs in `pyproject.toml`, README badges, `CITATION.cff`, and
  `docs/conf.py` linkcheck allowlists are consistent with Moffran being the canonical
  public repo and `kristinebergs` a development mirror; not flagged as defects.

---

## Release-gate recommendations

1. **Fix and rerun the two red gates before tag:** strict Sphinx (H2) and
   `local-checks-release` (H3). Neither is a product bug, but both are Task 39
   hard gates and both are red today.
2. **Add a narrative-correctness behavior gate** (for B1): a test asserting
   `to_narrative` states the same label as `predict` for at least one class-0 and
   one class-1 binary instance, plus a multiclass case. This is a durable product
   gate, not a release-plan meta-test.
3. **Add a reject prediction-path contract test** (for H1) asserting whichever
   payload/subset contract is chosen.
4. **Add a docs-example smoke** that executes QUICK_API.md and README code blocks
   verbatim (M1–M3 would all have been caught); nightly is fine, but it must exist.
5. **Add a Windows-encoding gate for user-facing text** (for H4/L6): encode CLI
   output and default narrative output to cp1252 with `errors="strict"` in a test.
6. Keep the existing green gates as-is: full pytest, deprecation-closure,
   warning-policy, capability-chain, packaging artifact check, version alignment.
