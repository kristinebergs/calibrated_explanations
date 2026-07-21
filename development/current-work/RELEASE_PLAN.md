> **Active scope:** Canonical, evergreen master release-planning document for
> Calibrated Explanations after v1.0.0 GA. Governs current and prospective OSS
> CE milestones. Superseded the archived `development/finished-work/RELEASE_PLAN_v1.md`
> (the v0.6.0 → v1.0.0 master control surface) when that series closed. Kept
> lightweight by design; version-specific execution detail lives in
> `development/current-work/vX.Y.Z_plan.md` files, not here.

# Release Plan (Master)

## A. Authority and purpose

This file governs current and prospective **OSS `calibrated_explanations`**
milestones. It is the single canonical entry point for "what is committed,
what is candidate, and what is deliberately deferred or out of scope."

- Version-specific execution detail (task breakdowns, verification
  checklists, exact release/development version declarations) belongs in
  `development/current-work/vX.Y.Z_plan.md`, not here.
- Completed version plans move to `development/finished-work/` when their
  release ships (see `release.md` step 16).
- ADRs (`development/adrs/`) and Standards (`development/standards/`) govern
  architecture and engineering rules. They are authoritative over this plan;
  where this plan and an ADR/Standard conflict, the ADR/Standard wins. Being
  listed as a candidate here does not itself authorize an architectural
  change — a candidate that changes architecture still needs a governing ADR
  before or during implementation.
- Research questions (see §D.3) are not release commitments. Listing a
  research direction here does not promise a version, a timeline, or an
  implementation.
- Git history — not this directory — is the archive for superseded planning
  drafts. Do not keep obsolete plan documents around "for historical
  comfort"; `development/finished-work/` is reserved for plans tied to a
  shipped release or for material still load-bearing for current tooling.

## B. Current release state

- **Latest released version:** `1.0.0` (GA, 2026-07-21; tag `v1.0.0`).
- **Active development version:** `1.0.1-dev` (`pyproject.toml`).
- **Active version plan:** `development/current-work/v1.0.1_plan.md` —
  first-patch-release maintenance checkpoint (docstring/nomenclature
  re-audit plus a caching/parallel telemetry regression sweep against the
  RC/GA baseline). No feature work is in scope for `1.0.1`.
- **Next milestone:** v1.0.1 (`development/current-work/v1.0.1_plan.md`).
- No successor milestone beyond `1.0.1` is decided. Do not infer a `1.1.0` or
  later scope from this document; a future minor/major cycle is a separate
  maintainer decision that will update this section when made.

## C. OSS CE scope

**In scope** — the reusable, installable library:

- Calibrated prediction and uncertainty interfaces (classification and
  regression, probabilistic and interval regression semantics).
- Factual and alternative calibrated explanations, including conjunctions.
- Reject/defer behavior implemented as a library capability (`RejectPolicy`,
  `RejectOrchestrator`, `RejectResult`) — a caller-invoked decision aid, not an
  operational deployment gate.
- Preprocessing required by the public CE API (wrapper-level encoding,
  mapping persistence).
- Explanation, interval, plot, and modality extension contracts (plugin
  registry, trust model, PlotSpec).
- Serialization and schema contracts, configuration (`ConfigManager`), plugin
  trust, documentation, and library-level observability (logging, optional
  telemetry a caller opts into and reads locally).
- Performance work that affects the local library's own execution (caching,
  parallel execution, feature filtering) — not a hosted service's throughput.

**Out of scope** — operational/deployment/enterprise concerns. These do not
belong in this repository's roadmap regardless of how they are phrased
(telemetry, governance, monitoring, etc. are legitimate CE library concepts;
the boundary is library vs. operational layer):

- Online/semi-online/continuously adaptive calibration; drift detection and
  drift-triggered recalibration.
- Checkpointing, rollback, state recovery, or event replay.
- MLflow-backed lifecycle management, KServe or other production-serving
  infrastructure integration.
- Enterprise governance telemetry, evidence packs, manual-review queues,
  capacity-aware deferral, escalation operations, or decision ledgers.
- Deployment monitoring, production incident reconstruction, commercial
  package composition or licensing.

## D. Milestone categories

### D.1 Committed milestone

Has an active version plan, approved scope, and verification gates.

- **v1.0.1** — see `development/current-work/v1.0.1_plan.md`. Maintenance
  checkpoint only (Standard-001/Standard-002 re-audit, caching/parallel
  telemetry regression sweep, release-tool execution). No new tooling, no
  runtime behavior changes planned.

### D.2 Candidate milestones

Not yet a commitment. Each entry states the problem, the source evidence,
why it belongs in OSS CE, the governing ADR, entry criteria, the intended
observable result, and why it isn't committed yet.

1. **Complete ADR-009 wrapper preprocessing: `auto_encode='auto'` and
   unseen-category policy.**
   - Problem: users who need automatic categorical encoding with
     deterministic, persisted mappings still fall back to the standalone
     `transform_to_numeric` helper; the wrapper's automatic mode is
     unfinished.
   - Evidence: ADR-009 §"Adoption Progress"/§"Implementation status" lists
     `auto_encode='auto'` and the unseen-category (`'error'`/`'ignore'`)
     policy as pending; `WrapCalibratedExplainer` currently implements
     user-supplied-preprocessor wiring and JSON-safe mapping export/import
     but not the automatic-encoding path.
   - Governing ADR: ADR-009 (amendment, not a new ADR).
   - Entry criteria: a concrete design for deterministic automatic mapping
     storage and the default `'error'` unseen-category behavior, reviewed
     against existing `export_preprocessor_mapping()`/
     `import_preprocessor_mapping()` contracts.
   - Intended observable result: `WrapCalibratedExplainer(auto_encode='auto')`
     learns and persists mappings without a user-supplied transformer.
   - Not yet committed: design not started; no version plan exists.

2. **Deprecate `transform_to_numeric` from the root namespace.**
   - Problem: `transform_to_numeric` remains in the root `__all__` even though
     its purpose is now mostly covered by wrapper preprocessing.
   - Evidence: ADR-009 §"Post-v1.0 open item" states this is blocked on
     candidate item D.2.1 above (the wrapper must be a complete substitute
     first).
   - Governing ADR: ADR-009 + ADR-011 (deprecation/migration policy) for the
     removal cycle itself.
   - Entry criteria: candidate D.2.1 ships and is documented as the
     recommended path.
   - Intended observable result: `transform_to_numeric` deprecated out of the
     root namespace (moved under `calibrated_explanations.utils`) with a
     standard ADR-011 migration window.
   - Not yet committed: blocked on D.2.1.

3. **Consolidate core ranking semantics and provenance.**
   - Problem: feature-ranking logic (`Explanation.rank_features`, plot-layer
     `rnk_metric`/`rnk_weight` options, narrative ordering) is implemented
     across several call sites (`explanations/explanation.py`,
     `explanations/explanations.py`, plot builders) without one place that
     states the deterministic tie-breaking rule and provenance guarantee.
   - Evidence: `rank_features` is called independently from at least 10 sites
     across factual/alternative/plot code paths; no single document states
     the canonical ordering contract.
   - Governing ADR: none yet; likely an addendum to ADR-008 (explanation
     domain model) rather than a new plugin contract.
   - Entry criteria: an audit confirming whether the multiple call sites are
     genuinely consistent (same tie-break rule) or have quietly diverged.
   - Intended observable result: one documented ranking contract that
     `rank_features` and plot-layer sorting both cite.
   - Not yet committed: audit not started. This is presentation/consolidation
     work, not a new extensibility point — do not read it as a precursor to
     a public ranking-plugin category (see §D.4).

4. **Plugin inspection and diagnostics for plugin authors.**
   - Problem: third-party plugin authors have the trust/registration contract
     (ADR-006) but no first-party CLI/test template for verifying a plugin's
     registration and trust status before publishing.
   - Evidence: `ce.plugins` CLI exists (`plugins/cli.py`) but plugin-author
     templates/examples are not part of the public docs surface.
   - Governing ADR: ADR-006 (no amendment expected; documentation/tooling
     only).
   - Entry criteria: a concrete author pain point demonstrated (e.g. a
     support issue or external plugin repo needing this).
   - Intended observable result: a documented `ce.plugins` diagnostic
     workflow and/or a plugin-author test template.
   - Not yet committed: no demonstrated user request yet; tracked here so it
     isn't lost, not because it is scheduled.

### D.3 Research directions

Require scientific validation; carry no release version; must not imply a
guarantee beyond what is currently established.

- **Explanation stability across calibration samples.** Whether repeated
  calibration draws produce stable feature-weight rankings. Calibrated
  interval guarantees (see `docs/foundations/concepts/calibrated_interval_semantics.md`)
  do not extend to rank stability or simultaneous feature comparisons; no
  code or documentation claims otherwise, and this entry does not change
  that.
- **Distribution-free coverage guarantees for feature-importance rankings.**
  An open research question, not an implemented or promised capability.
- **Multi-calibration across intersectional protected attributes.** Current
  support is Mondrian/conditional categorizer-based (ADR-039); formal
  multi-calibration guarantees are unproven and undesigned.
- **Higher-order feature interaction search.** `add_conjunctions()` already
  builds conjunctive rules iteratively up to a caller-supplied
  `max_rule_size` (`explanations/explanation.py`); open research is whether a
  computationally feasible search exists for interactions beyond what the
  current greedy pairwise-growth algorithm covers, and whether calibration
  guarantees survive such a search. This is a algorithmic-research question,
  not a documentation gap in the existing conjunction feature.

### D.4 Deferred architectural proposals

Deliberately not being implemented now. Each states what evidence would
justify reopening it.

- **External reject-strategy plugin contract.** ADR-029 Decision 2 explicitly
  adopted a lightweight internal registry inside `RejectOrchestrator` (B2),
  not a full external plugin chain (B3). ADR-029 §"Future Considerations"
  and the archived master plan's "Post-1.0 considerations" both flag
  strategy lifecycle hooks (`pre_apply_hook`, `pre_emit_hook`,
  `post_emit_hook`) and a fuller configuration surface as open, but still
  internal-registry scoped. Reopen only if: (a) ADR-029 is amended or
  superseded to authorize external strategies, (b) the internal strategy
  contract (task compatibility, result validation) is stable, and (c) a
  concrete external user need is evidenced — not merely that a registry
  exists.
- **Public ranking/selection plugin category.** No public extensibility point
  exists for feature ranking or selection today (see candidate D.2.3, which
  is consolidation, not extensibility). Reopen only with at least two
  credible independent implementations wanting a shared contract and a
  stable candidate schema.
- **Automatic, configuration-driven, or lazy plugin activation.** ADR-006's
  explicit-trust-action model (`CE_TRUST_PLUGIN`, `register_*(trusted=True)`,
  `mark_*_trusted(...)`) is retained; no evidence (issue, test failure, or
  user report) currently shows a material problem with explicit activation.
  Reopen only with a concrete proposal that separately addresses third-party
  code execution trust, deterministic ordering, namespace collisions,
  reproducibility, and failure behavior — not as a batch part of unrelated
  cleanup.
- **Reject visualization plugin.** ADR-029 Decision 4 chose no visualization
  integration (D1). Reopen only via a new ADR or plugin proposal explicitly
  scoped to reject visualization.
- **Caching/parallel on-by-default graduation.** Both remain explicit opt-in
  (`CE_CACHE`, `CE_PARALLEL`) per ADR-003/ADR-004. Reopen once the `1.0.1`
  telemetry regression sweep (committed milestone D.1) and at least one
  further maintenance cycle show stable, unsurprising telemetry with no
  fallback-rate regressions.

## E. Out of scope (see §C for detail)

Adaptive/online calibration, drift detection, checkpointing/rollback,
MLflow/KServe integration, enterprise governance telemetry and evidence
packs, manual-review/escalation operations, and deployment monitoring are not
OSS CE roadmap items. This repository does not track a destination for that
work.

## Release mechanics

`release.md` and the executable tooling (`Makefile` release targets,
`scripts/local_checks.py`) are authoritative for how a release actually
ships. This plan does not duplicate that sequence — see `release.md` for the
17-step maintainer sequence and `development/current-work/vX.Y.Z_plan.md` for
the current milestone's exact checklist.
