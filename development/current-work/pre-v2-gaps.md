# Pre-v1 Second-Pass Gap Audit (pre-v2-gaps)

Status: complete — dispositioned into `v0.11.6_plan.md` Tasks 23–28 (2026-07-09;
originally numbered 24–29, renumbered when the plan moved release preparation to
Task 30 and added the Task 29 test-quality correction; release preparation has since
moved again to Task 39 when the 2026-07-09 third-pass audit inserted Tasks 30–38,
and to Task 45 when the 2026-07-10 fourth-pass audit inserted Tasks 39–44)
Date: 2026-07-09

> **Disposition (2026-07-09; task ids updated to the renumbered plan):** the findings
> below were converted into release-plan
> work the same day. P3 → Task 23; P2 → Task 24; P6 → Task 25; P4/P5/P7 → Task 26;
> P8 + the Task 5 plugin-seam follow-up → Task 27; P9/P10/P12 → Task 28. P1 was
> closed directly: `scripts/local_checks.py` now maps every non-optional task
> (1–21, 23–29) from the plan's `toml ce-task-verification` block. (The original
> mapping-completeness test in `tests/scripts/test_local_checks_profiles.py` was
> later removed by plan Task 29 as prohibited release-plan meta-testing; the
> engine's fail-fast unmapped-id error is the durable enforcement.) P11 was folded
> into the pending Tasks 14 and 21 as scope expansions. Five new gate scripts under
> `scripts/quality/` back the pending-task mappings and were validated to fail on the
> exact confirmed defects; two of them widened the evidence: the local editable
> install's package metadata reports `0.11.3.dev0` (Task 13 will catch this via
> `check_version_alignment.py`), and seven `.github/skills` SKILL.md files reference
> non-existent `references/` materials (Task 21 scope). Task 12 closed independently
> while this audit ran; section 4/P9's "Task 12 marked Not started" observation is
> therefore resolved and retained only as history.
Scope: red-team of the coverage of `pre-v1.md` and the updated
`development/current-work/v0.11.6_plan.md`. This is not a restatement of the
first audit; only new gaps, second-order regressions, and false-confidence
risks are recorded. Every serious finding below was verified against the
repository checkout with a command or a file/line citation.

---

## 1. Verdict

**The plan is directionally good but has important audit gaps.**

The completed Tasks 1–11 are substantively real: the fail-fast kwarg gates are
wired at both wrapper and direct-core boundaries and behave as documented on
predict/explain surfaces (verified by live repro), the license fix is in the
fresh artifacts, and the coercer/warnings work matches its claims. But the
verification fabric around the plan has three structural weaknesses: (a) the
per-task closure instrument (`make local-checks-task TASK=<n>`) does not exist
for any pending task (mappings 12–21 are unimplemented, contradicting Task 9's
checked verification box), so completion evidence will be authored ad hoc by
the same person closing each task; (b) two of the plan's central gates prove
less than they appear to — the deprecation-closure gate audits only a markdown
ledger while a live runtime `DeprecationWarning` alias exists in `src/`, and
the suite-wide fallback-visibility warning filter is dead; and (c) a genuine
public-API correctness defect (numeric class labels not equal to `0..K-1`
break calibration with misleading errors) is outside the scope of every
pending task. Completing Tasks 11–21 exactly as written would still tag
v0.11.6 with all three problems intact.

## 2. Strongest remaining risk

**Numeric class labels that are not `0..K-1` break `calibrate()` with
misleading internal errors, and no task covers it.**

Verified repro (RandomForestClassifier, seed 42):

- binary labels `{1, 2}` → `calibrate()` raises
  `ValidationError: Interval calibrator 'core.interval.legacy' probability
  output is outside [0, 1]` — blaming the model's probabilities, which are
  fine;
- labels `{2, 5, 9}` → `ConfigurationError: Interval plugin execution failed
  for default mode: index 5 is out of bounds for axis 1 with size 3`;
- string labels `{"no", "yes"}` work correctly.

Root cause: `src/calibrated_explanations/core/calibrated_explainer.py:401-411`
routes targets through `convert_targets_to_numeric` only when labels are
str/object; numeric labels flow through raw and are later used as probability
column indices. sklearn accepts these labels, so users with 1/2-coded or
sparse-coded datasets hit this immediately, and the error text sends them
debugging the wrong thing. `pre-v1.md` probed single-class calibration
(Task 21) but never non-contiguous labels; Tasks 11–21 do not touch it. This
is a v1 API-freeze-relevant correctness defect: either fix the encoding (route
all classification targets through an encoder) or fail fast at `calibrate()`
with a clear `ValidationError` naming the label requirement.

## 3. Gaps in the previous audit prompt

Dimensions the first broad audit did not force strongly enough, each verified
to have hidden at least one real finding:

1. **Class-label edge-space beyond single-class.** The first audit tested
   single-class calibration but not non-contiguous / non-zero-based numeric
   labels. That is where the strongest remaining defect sat (Section 2). A new
   prompt should enumerate label spaces explicitly: `{0,1}`, `{1,2}`,
   `{2,5,9}`, strings, single-class, and pandas categoricals.
2. **Test-harness configuration as an audit object.** Nobody audited
   `pyproject.toml` `filterwarnings` ordering. The
   `error:.*fall.*back.*:UserWarning` entry is shadowed by the later
   `ignore::UserWarning` (pytest gives later entries precedence). Verified: a
   synthetic test emitting `UserWarning("falling back to default template")`
   **passes** the suite. The fallback-visibility policy
   (CONTRIBUTOR_INSTRUCTIONS §5) has no suite-wide enforcement.
3. **Runtime deprecation emissions vs. the ledger.** The first audit ran
   `--deprecation-closure` but never grepped `src/` for live `deprecate(`
   call sites. `src/calibrated_explanations/ce_agent_utils.py:580-586` emits a
   `DeprecationWarning` for the live `narrative_format=` alias; the Active
   ledger is empty. "Zero active deprecations" is currently true of the ledger
   and false of the runtime.
4. **Verification-harness routing.** The first audit did not (and could not,
   pre-Task-9) audit `_task_specific_steps`; the new prompt correctly forces
   this, and it found that mappings 12–21 do not exist (see Finding P1).
5. **Package-metadata sanity beyond the license.** `requires-python = ">=3.8"`
   (pyproject.toml:13) is unsatisfiable with `pandas>=2.0` (needs ≥3.9) and
   contradicts ruff `target-version = "py310"` / mypy 3.11;
   `[tool.setuptools.packages.find]` lists a non-existent `external_plugins`
   directory; stale `0.11.2.dev0` artifacts sit in `dist/` beside the fresh
   ones. Task 11 checked only the license.
6. **Wrapper vs. direct-core parity on the plugin seam.** Verified divergence:
   `WrapCalibratedExplainer.explain_factual(features_to_ignor=[0])` raises
   `ConfigurationError`, while `CalibratedExplainer.explain_factual` with the
   same typo **silently succeeds with zero warnings** (ADR-038 §3 forwarding).
   The first audit checked each surface but not the asymmetry or its
   visibility.
7. **Non-explain public surfaces as typo sinks.** `to_narrative(
   expertise_lvl=...)` and `plot(x, filter_topp=3)` are silently ignored (0
   warnings, verified). The first audit only found the specific `format=` case.
8. **Skill mirrors as structures, not files.** Registry entry counts diverge:
   `.github/skills` 72, `.agents/skills` 54, `.codex/skills` 74,
   `.claude/skills` 54. The first audit spotted one missing references file.
9. **Evidence retention.** `reports/local_checks/task_profile_report.json` is
   one mutable file holding only the last run (currently task 11); evidence
   for Tasks 1–10 closures no longer exists. Stale mis-named logs
   (`reports/local_checks_task11.log`, `..._task12.log`, dated 2026-06-17,
   v0.11.4-era numbering) sit beside it and invite misreading.
10. **Promised-vs-implemented profile contents.** The skip-reason ledger in
    `local_checks.py` names a `release_bookkeeping` gate "Reserved for release
    validation" that no profile implements, and `make local-checks-release`
    contains no artifact build/inspection step at all (see P4/P5).

## 4. Gaps in `v0.11.6_plan.md`

### P1. Task-verification instrument missing for all pending tasks; Task 9 checkbox falsely green

- Severity: **high**.
- Evidence: `scripts/local_checks.py` `_task_specific_steps` (918–1083) and
  `_task_specific_lint_targets` (1086–1153) map tasks 1–11 only.
  `python scripts/local_checks.py --profile task --task 12` exits 1 with
  `ValueError: Unsupported task profile mapping: 12` (verified; it fails
  loudly, via unhandled traceback rather than the argparse-clean exit 2).
  Plan §9.2 promises mappings for Tasks 1–21; Task 9's checklist is `[x]` for
  that behavior; Task 10's own status note admits its mapping "did not exist
  yet" when Task 10 began.
- Why the plan misses it: each pending task's last checklist line ("run
  `make local-checks-task TASK=<n>`") *assumes* the mapping exists and says
  nothing about who writes it or what it must contain. The person closing a
  task will author their own acceptance instrument with no independent check
  that it includes the plan-promised steps (e.g. Task 15's "wheel-only docs
  snippet smoke", Task 14's grep gate).
- Recommended amendment: add to Task 9 (reopened as a subcheck) or Task 23 a
  parametrized test asserting `_task_specific_steps(n)` exists for every
  n in 1..21 **and** a review rule that each mapping's steps are named in the
  plan task before the checkbox is ticked.
- Disposition: Task 9 subcheck + Task 23 release-gate item.

### P2. Deprecation closure is a ledger property, not a runtime property

- Severity: **high**.
- Evidence: `_active_deprecation_rows` (local_checks.py:322-352) parses only
  `docs/migration/deprecations.md`.
  `src/calibrated_explanations/ce_agent_utils.py:580-586` emits
  `DeprecationWarning` for the live `narrative_format=` alias
  (`explain_and_narrate`), absent from the ledger. The pytest config also
  pre-ignores several package deprecation messages
  (pyproject.toml:129-137), so the suite cannot surface them either.
- Why the plan misses it: Task 1 fixed the parser's permissiveness, not its
  scope. D5's "zero active deprecations" gate can be green forever while
  `src/` keeps emitting.
- Recommended amendment: new subcheck (Task 1 follow-up or Task 23 gate): scan
  `src/` for `deprecate(` / `warnings.warn(..., DeprecationWarning)` call
  sites outside `utils/deprecation*.py` and fail on any hit not matched to a
  ledger row. Decide `narrative_format=` now: remove it fail-fast (consistent
  with the milestone posture — no new deprecation cycles also means no
  *residual* ones) or add a ledger row and accept the gate failing.
- Disposition: new task (small) + Task 23 gate item.

### P3. Numeric non-`0..K-1` class labels break calibration (Section 2)

- Severity: **blocker for v1 freeze; high for v0.11.6** (fix or explicit
  fail-fast + deferral with target).
- Evidence: Section 2; root cause `core/calibrated_explainer.py:401-411`.
- Disposition: new task; at minimum a `ValidationError` at the `calibrate()` /
  `__init__` boundary in v0.11.6, full encoding fix acceptable in RC only if
  classified as emergency-patch-eligible.

### P4. `make local-checks-release` never builds or inspects artifacts

- Severity: **medium**.
- Evidence: `_release_steps` (local_checks.py:824-900) contains docs,
  notebooks, linkcheck, warning policy, deprecation closure, evidence refresh,
  API snapshot — but no `python -m build` and no
  `check_packaging_artifacts.py`. Artifact smoke exists only in the TASK=11
  mapping and as a manual Task 23 item.
- Why the plan misses it: Task 23's pre-tag list includes "Artifact smoke" as
  a prose item; nothing wires it into the release profile the plan calls the
  release-verification instrument.
- Recommended amendment: append the Task 11 build+inspect steps to
  `_release_steps`, preceded by a `dist/` clean (see P7).
- Disposition: Task 23 release-gate item (tooling change now).

### P5. Phantom `release_bookkeeping` gate

- Severity: **low**.
- Evidence: `_common_skipped_heavy_gates` (local_checks.py:903-915) reports
  `release_bookkeeping` as "Reserved for release validation"; no profile
  implements any step by that name.
- Recommended amendment: implement it (CHANGELOG-entry presence, version-string
  alignment check) or delete the skip row so reports stop over-promising.
- Disposition: Task 9 subcheck.

### P6. Fallback-visibility pytest filter is dead

- Severity: **medium** (it silently weakens Task 19 and all fallback tests).
- Evidence: verified repro — a test emitting
  `UserWarning("falling back to default template")` passes the suite;
  `ignore::UserWarning` (pyproject.toml:126) shadows
  `error:.*fall.*back.*:UserWarning` (line 122) because pytest applies later
  entries with precedence.
- Recommended amendment: reorder so the fallback-error filter comes after the
  blanket ignore, add a meta-test that a synthetic fallback warning fails, and
  note in Task 19 that its fallback-warning assertions must use
  `pytest.warns` explicitly until then.
- Disposition: Task 19 subcheck + tiny standalone fix.

### P7. Stale artifacts and metadata sanity unowned

- Severity: **medium**.
- Evidence: `dist/` contains `calibrated_explanations-0.11.2.dev0*`
  (2026-05-11) beside the fresh 0.11.6.dev0 files;
  `check_packaging_artifacts.py` picks newest-by-mtime (safe for the check,
  not for `twine upload dist/*`). `requires-python = ">=3.8"` vs
  `pandas>=2.0`; no Python classifiers; `packages.find` lists non-existent
  `external_plugins`; the check validates license only — not `py.typed`,
  templates, schemas, RECORD, or unexpected top-level packages.
- Recommended amendment: extend Task 11's checker to (a) require a clean
  `dist/`, (b) assert expected package-data manifest (py.typed, templates,
  schemas), (c) assert no unexpected top-level packages, (d) validate
  `Requires-Python` against the actual support floor. Fix `requires-python`
  and drop `external_plugins` from `packages.find` now.
- Disposition: Task 11 reopened subchecks + Task 23 gate item.

### P8. Narrative/plot surfaces remain silent typo sinks; docstring contract wrong

- Severity: **medium**.
- Evidence: verified — `to_narrative(output_format="text",
  expertise_lvl="advanced")` returns silently (single and collection);
  `plot(x, filter_topp=3)` silently ignored. The Task 12 guard
  (`api/params.py:184-209`) is a denylist of exactly one name (`format`).
  Invalid values do raise, but as `ValidationError`
  (narrative_plugin.py:417) while the `to_narrative` docstring promises
  `ValueError` and still lists `FileNotFoundError`
  (explanation.py:616-623).
- Why the plan misses it: Task 12's checklist asks only about `format=`;
  Task 19 covers only the template-file docs.
- Recommended amendment: extend Task 12 to close or explicitly document the
  narrative kwarg seam (allowlist like the explain surfaces, or a logged
  forward), and fix the `Raises` docstrings; add `plot()` kwargs to the
  Task 5 documented-seam inventory.
- Disposition: Task 12 expansion + Task 19 subcheck.

### P9. Plan status ledger drifts in both directions

- Severity: **medium** (evidence integrity).
- Evidence: (a) Task 11 is marked completed in the gate summary
  (v0.11.6_plan.md:1271,1288) though outside prompts still describe 11 as
  pending; (b) Task 12 is marked "Not started" while its runtime rejection,
  tests (`tests/unit/explanations/test_explanation_unit.py:215-219` etc.),
  docs migration (`AGENTS.md:129`, `ce_first_agent_guide.md`), and CHANGELOG
  entry (line 14) are all merged — `api/params.py`'s docstring even cites
  "v0.11.6 Task 12".
- Recommended amendment: reconcile the gate summary now; adopt the rule that
  merging task work and ticking the plan happen in the same PR.
- Disposition: bookkeeping fix + Task 23 gate item ("no task marked pending
  whose implementation is already merged, and vice versa").

### P10. Ledger and CHANGELOG drift on reject-NCF removals

- Severity: **low/medium**.
- Evidence: `docs/migration/deprecations.md:292` records that explicit
  `ncf="hinge"` / `ncf="margin"` now raise `ValidationError`; the CHANGELOG
  Unreleased bullet (line 15) mentions only `ncf="entropy"`. The internal
  legacy scorer docstring still advertises
  `ncf : {'ensured', 'entropy', 'margin'}`
  (core/reject/orchestrator.py:366) though the public boundary rejects them.
- Recommended amendment: add the hinge/margin migration note to the CHANGELOG
  (plan rule: every behavior change gets one) and align the internal
  docstring.
- Disposition: Task 8 reopened subcheck.

### P11. Mirror/instruction gates are narrower than the tasks assume

- Severity: **medium**.
- Evidence: `scripts/quality/check_agent_instruction_consistency.py` checks a
  fixed set of nine docs plus `.github/prompts` and `.github/instructions` —
  no skills trees, no ADRs, no `.github/CONTRIBUTING.md` (so Task 18's dead
  link is invisible to it). Skill mirrors diverge structurally (72/54/74/54
  entries across `.github`/`.agents`/`.codex`/`.claude`); `.agents/` and
  `.claude/` skill files carry the same stale "guarded=True … removed v1.0.0"
  wording as `.github/skills`, but Task 14 step 2 names only
  `.github/skills/*`.
- Recommended amendment: Task 14's grep gate must cover all four mirror trees
  and ADRs; Task 21 must compare mirror trees structurally (file lists), not
  just one references file.
- Disposition: Task 14 + Task 21 expansions.

### P12. Plan-of-record documents disagree

- Severity: **low**.
- Evidence: `RELEASE_PLAN_v1.md:682-745` — the section the v0.11.6 plan calls
  its "authoritative task source" — lists only Tasks 1–8 (no 9–23) and its
  gate line still says "`make local-checks` passes" (pre-Task-9 semantics).
  `docs/migration/deprecations.md`'s Active section says "All active
  deprecations were removed in 1.0.0" — v1.0.0 has not shipped; the removals
  were v0.11.5.
- Disposition: bookkeeping sweep in Task 23.

## 5. Risks introduced by completed Tasks 1–10

1. **Task 5 — wrapper/core plugin-seam divergence with zero visibility.**
   Verified: the wrapper rejects genuinely-unknown explain kwargs; direct core
   forwards them to plugins silently (no warning, no INFO log). Two
   consequences: plugin-defined call-time kwargs are unusable through the
   recommended wrapper entry point, and the original bug class (typo silently
   vanishes) survives on the direct-core explain surface. Recommended gate: an
   INFO log (per §5 visibility policy) listing forwarded unknown keys, plus a
   documented statement of the wrapper limitation; a test asserting the log.
2. **Task 5/D3 — third-party breakage surface is broad and only
   CHANGELOG-mitigated.** Six previously-accepted-but-inert names now raise
   (`condition`, `condition_label`, `condition_labels`,
   `include_reject_details`, `output_interval`, `y_threshold`) and unknown
   kwargs went from warn to raise. Any downstream wrapper passing a kwargs
   dict through will hard-fail on upgrade. This is the accepted policy, but
   there is no upgrade-checklist entry equivalent to
   `docs/upgrade/v0.11.4-upgrade-checklist.md`; recommend one for v0.11.6.
3. **Task 9 — lint scope narrows silently outside task profiles.**
   `_changed_python_targets` (local_checks.py:524-548) lints only changed
   files for quick/pr/full/**release**; on a dirty tree,
   `make local-checks-release` ruff-checks only the diff. Release lint should
   pin to full `src`/`tests`/`scripts`.
4. **Task 9 — offline pre-commit downgraded to pass.**
   `_run_step` (local_checks.py:102-108) converts pre-commit network-fetch
   failures into rc 0 in pr/full/release profiles. Acceptable for dev,
   but release evidence should not include a skipped-hook pass; gate on
   network availability in the release profile.
5. **Task 10 — detector-driven test editing.** The
   `test_local_checks_profiles.py:122` order-insensitive rewrite was done to
   stop tripping the detector, not to strengthen the assertion. Harmless here
   (lint-target order is not semantic), but it demonstrates the incentive; a
   reviewer rule ("assertion edits prompted by the detector must strengthen,
   not restructure") is worth writing into the plan.
6. **Task 4 — mock-fixture class of failure.** The two MagicMock fixtures the
   task fixed were found by breakage, not search. A one-off grep for
   `MagicMock()` explainers whose attributes feed coercers/validators would
   close the class; not urgent.

## 6. Pending task hardening recommendations

| Task | Assessment |
|---|---|
| 12 | **Too narrow — expand.** Runtime `format=` rejection is already merged; the remaining risk is the open narrative/plot kwarg seam and the wrong `Raises` docstrings (P8). Also reconcile status (P9). |
| 13 | **Strong enough as written**, with one addition: assert plugin descriptors and provenance payloads under an *installed wheel* (editable installs mask metadata drift). Current drift confirmed: `__version__ = "v0.11.6-dev"` vs metadata `0.11.6.dev0`. |
| 14 | **Needs a stronger verification checklist.** The grep gate must cover `.agents/`, `.codex/`, `.claude/` mirrors and ADR-026/-032/-038 text (all still say deprecated/current, verified), all five kwargs, and must whitelist historical sections structurally (path- or heading-based), not by eyeball. No existing checker covers any of this (P11). |
| 15 | **Strong enough**, but the "wheel-only docs snippet smoke" must be an actual executed check (fresh wheel, temp cwd, repo not on `sys.path`) — the promised TASK=15 mapping does not exist yet (P1). |
| 16 | **Strong enough as written.** Repro re-confirmed today (`IndexError` on empty calibration; mismatched empty/non-empty already raises `DataShapeError` correctly). Add the `(0,)`-shaped-X and pandas-empty variants from the prompt. |
| 17 | **Strong enough**; prefer re-execution over clearing for the quickstart notebooks since outputs are user-facing docs; the hygiene grep on stale warning text is cheap either way. |
| 18 | **Strong enough** (dead link re-confirmed at `.github/CONTRIBUTING.md:67`); add the file to `check_agent_instruction_consistency.py`'s REQUIRED_DOCS so the class of defect is gated, not just this instance. |
| 19 | **Needs strengthening**: the fallback contract cannot be trusted while the fallback warning filter is dead (P6); also cover the `ValueError`→`ValidationError` docstring drift (P8) and verify the packaged default template loads via `importlib.resources` (wheel-only load verified OK today). |
| 20 | **Strong enough as written**; confirmed still open (wheel ships only the three top-level schemas; `schemas/v1/plotspec_schema.json` absent). |
| 21 | **Too narrow — expand.** Mirror drift is structural (72/54/74/54 entries), not one file; single-class needs a *policy decision*, and the non-contiguous-label defect (P3) belongs in the same class-label policy discussion. Deferrals must carry owner/date/milestone, and any public-contract ambiguity deferral should block `1.0.0rc1` even if it doesn't block v0.11.6. |

## 7. Suggested new audit/gate checks

Only checks corresponding to verified risks:

```bash
# P2: runtime deprecation scan (fail on un-ledgered deprecation emitters)
python scripts/quality/check_runtime_deprecations.py --src src/calibrated_explanations \
    --ledger docs/migration/deprecations.md --check

# P1: mapping completeness (pytest, parametrized 1..21)
pytest tests/scripts/test_local_checks_profiles.py -k task_mapping_completeness

# P6: meta-test that the fallback filter actually errors
pytest tests/meta/test_fallback_warning_filter_is_live.py

# P7: extended artifact check (clean dist, package-data manifest, metadata)
python scripts/quality/check_packaging_artifacts.py --clean-dist \
    --expect py.typed --expect templates/explain_template.yaml \
    --expect schemas/explanation_schema_v1.json --requires-python ">=3.9"

# P3: class-label boundary tests
pytest tests/unit/core/test_class_label_encoding.py  # {0,1},{1,2},{2,5,9},str,single

# P11: skill-mirror structural sync
python scripts/quality/check_skill_mirror_sync.py --trees .github/skills .agents/skills .codex/skills .claude/skills

# P8: narrative surface closure regression
pytest tests/unit/explanations -k "narrative_kwarg"  # typo kwargs raise or are logged
```

## 8. Final action list

**1. Must add before continuing v0.11.6**

- P1: implement TASK=12..21 mappings (or a mapping-completeness test) before
  any further task is closed with a `local-checks-task` checkbox.
- P9: reconcile the gate summary (Task 12 state) so the plan is a truthful
  control surface again.
- P3: decide the numeric-label policy (fail fast now vs. full fix) and open a
  task for it.

**2. Should add before the v0.11.6 tag**

- P2: runtime deprecation scan + resolve `narrative_format=`.
- P6: fix the dead fallback filter and add the meta-test.
- P7: dist-clean + extended artifact manifest check; fix `requires-python`
  and `packages.find`.
- P4/P5: add build+artifact smoke to `_release_steps`; implement or delete
  `release_bookkeeping`.
- P8: close or log the narrative/plot typo seams; fix `Raises` docstrings.
- P10: CHANGELOG note for `ncf="hinge"/"margin"`.
- Task 5 follow-up: INFO log for plugin-forwarded unknown kwargs + documented
  wrapper limitation.
- v0.11.6 upgrade checklist page (mirroring the v0.11.4 one) for the
  warn→raise reversal.

**3. Can defer to v1.0.0-rc**

- P11 structural mirror sync (if v0.11.6 ships the Task 14 grep for the five
  kwargs first).
- P12 plan-of-record reconciliation (RELEASE_PLAN task list, ledger framing
  text).
- Release-profile lint pinning and offline pre-commit gating (items 5.3/5.4).

**4. Can defer post-v1**

- Single-class calibration full policy implementation, provided v1 documents
  the current behavior and Task 21 records the deferral with owner/date.
- MagicMock-fixture sweep (item 5.6).
- Per-task evidence retention (C6-style per-task report files) — process
  quality, not release correctness.

---

## Appendix: verification commands run (selection)

- `python scripts/local_checks.py --profile task --task 12` → exit 1,
  `ValueError: Unsupported task profile mapping: 12`.
- Synthetic pytest file with
  `warnings.warn("falling back to default template", UserWarning)` → 1 passed.
- Live repros with RandomForestClassifier/seed 42: empty calibration
  (`IndexError`), labels `{1,2}` (`ValidationError` — misleading), labels
  `{2,5,9}` (`ConfigurationError` — index out of bounds), string labels (OK),
  single-class (silently incoherent `class_labels`), wrapper vs core
  `features_to_ignor=` typo (raise vs silent), `to_narrative(expertise_lvl=)`
  and `plot(filter_topp=)` (silent), `to_narrative(format=)` (raises —
  Task 12 runtime already merged).
- Wheel-only install smoke of `dist/calibrated_explanations-0.11.6.dev0`:
  imports, both CLI modules, templates via `importlib.resources`, `py.typed`
  present; `__version__` `v0.11.6-dev` vs metadata `0.11.6.dev0` (H2 still
  open); `schemas/v1/plotspec_schema.json` absent (M4 still open).
- `reports/local_checks/task_profile_report.json` holds only the last run
  (task 11); `dist/` contains stale 0.11.2.dev0 artifacts.
