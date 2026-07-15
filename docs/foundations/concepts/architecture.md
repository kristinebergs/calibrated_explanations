# Architecture

`calibrated_explanations` has one calibrated runtime with several execution
paths and extension points. No single flowchart can represent all of them
without implying relationships that do not exist.

This page therefore uses separate architectural views. Each view answers one
question:

1. How does a user create and use a calibrated runtime?
2. Which components own the main runtime responsibilities?
3. What happens during ordinary prediction and explanation?
4. What changes when a reject policy is active?
5. How are plugins and runtime configuration resolved?
6. Which capabilities are optional or explicitly invoked downstream?

Solid arrows show calls, delegation, or produced outputs. Dashed arrows show
configuration, observation, or optional use. A component appearing in an
ownership diagram does not imply that every runtime call executes it.

## Public lifecycle

This view shows the recommended user-facing lifecycle. It does not show
internal orchestration.

```{mermaid}
flowchart LR
    MODEL["Predictive learner"]
    TRAIN["Training data"]
    CAL["Calibration data"]

    WRAP["WrapCalibratedExplainer<br/>recommended public facade"]
    RUNTIME["Calibrated runtime"]

    PREDICT["predict<br/>predict_proba"]
    FACTUAL["explain_factual"]
    ALTERNATIVE["explore_alternatives"]
    FAST["explain_fast"]

    MODEL --> WRAP
    TRAIN -->|"fit"| WRAP
    CAL -->|"calibrate"| WRAP
    WRAP --> RUNTIME

    RUNTIME --> PREDICT
    RUNTIME --> FACTUAL
    RUNTIME --> ALTERNATIVE
    RUNTIME --> FAST
```

`WrapCalibratedExplainer` coordinates the scikit-learn-style lifecycle:

1. `fit(...)` fits the predictive learner.
2. `calibrate(...)` validates the calibration data and constructs the
   calibrated runtime.
3. Prediction and explanation methods delegate to that runtime.

An already fitted learner can be wrapped and calibrated without fitting it
again.

## Core runtime ownership

This view shows component ownership and delegation. It is not an execution
sequence.

```{mermaid}
flowchart TB
    WRAP["WrapCalibratedExplainer<br/>public lifecycle facade"]

    CE["CalibratedExplainer<br/>calibration data, active calibrators,<br/>model metadata and runtime state"]

    PM["PluginManager<br/>plugin defaults, overrides, instances,<br/>fallback chains and orchestrator access"]

    PO["PredictionOrchestrator<br/>prediction and interval execution"]
    EO["ExplanationOrchestrator<br/>explanation execution"]
    RO["RejectOrchestrator<br/>reject decision and payload shaping"]

    WRAP -->|"delegates calibrated operations"| CE
    CE -->|"initializes and exposes"| PM

    PM --> PO
    PM --> EO
    PM --> RO

    PO -. "operates on shared runtime state" .-> CE
    EO -. "operates on shared runtime state" .-> CE
    RO -. "operates on shared runtime state" .-> CE
```

The responsibilities are deliberately separated:

| Component                 | Primary responsibility                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `WrapCalibratedExplainer` | Recommended public lifecycle: fit, calibrate, predict, explain, save, and load                                      |
| `CalibratedExplainer`     | Lower-level calibrated runtime and shared runtime state                                                             |
| `PluginManager`           | Single source of truth for plugin selection, overrides, fallback chains, instances, and orchestrator initialization |
| `PredictionOrchestrator`  | Calibrated prediction, interval execution, uncertainty output, and interval-calibrator lifecycle                    |
| `ExplanationOrchestrator` | Explanation request construction, plugin invocation, result validation, and explanation telemetry                   |
| `RejectOrchestrator`      | Reject learner lifecycle, reject decisions, policy filtering, and reject-result construction                        |

The three orchestrators are structurally available in the calibrated runtime.
The reject orchestrator is executed only when a reject operation or a
non-`NONE` reject policy requires it.

## Ordinary inference

This view shows prediction and explanation when no reject policy is active.
`RejectPolicy.NONE` follows these ordinary paths and does not create a reject
envelope.

```{mermaid}
flowchart LR
    subgraph PRED["Ordinary prediction"]
        PAPI["predict or predict_proba"]
        PO["PredictionOrchestrator"]
        IC["Active interval implementation<br/>resolved through PluginManager"]
        POUT["Prediction result<br/>with optional calibrated uncertainty"]

        PAPI --> PO
        PO --> IC
        IC --> POUT
    end

    subgraph EXP["Ordinary explanation"]
        EAPI["explain_factual<br/>explore_alternatives<br/>or explain_fast"]
        EO["ExplanationOrchestrator"]
        EP["Active explanation implementation<br/>resolved through PluginManager"]
        EOUT["Calibrated explanation objects"]

        EAPI --> EO
        EO --> EP
        EP --> EOUT
    end

    EP -. "requests calibrated prediction<br/>and interval evaluations" .-> PO
```

The prediction path returns prediction data directly. Depending on the method
and arguments, this can include calibrated probabilities, prediction
intervals, or uncertainty bounds.

The explanation path returns explanation-domain objects such as
`CalibratedExplanations` or `AlternativeExplanations`. Explanation
implementations use the calibrated runtime for the prediction and interval
evaluations required to construct feature rules.

Ordinary inference does not automatically:

* construct a reject result;
* serialize the explanation;
* build a `PlotSpec`; or
* invoke a renderer.

Those are separate paths described below.

## Reject-policy execution

This view applies only when the effective reject policy is not
`RejectPolicy.NONE`.

Reject execution is not a final wrapper around an already completed ordinary
prediction or explanation. The reject decision is calculated first. The
selected policy then determines which source rows are used to construct any
requested prediction or explanation payload.

```{mermaid}
flowchart TB
    CALL["Prediction or explanation call<br/>with a non-NONE reject policy"]

    RESOLVE["Resolve effective policy<br/>and reject configuration"]

    RO["RejectOrchestrator"]

    DECISION["Compute reject decision<br/>for the original input batch"]

    DA["RejectDecisionArtifact<br/>original-batch cardinality"]

    SELECT["Select payload source rows<br/><br/>FLAG: all rows<br/>ONLY_REJECTED: rejected rows<br/>ONLY_ACCEPTED: accepted rows"]

    PP["Optional prediction payload<br/>via PredictionOrchestrator"]

    XP["Optional explanation payload<br/>via the ordinary explanation path"]

    PA["RejectPayloadArtifact<br/>selected source indices and payloads"]

    META["Reject metadata<br/>policy, rates, provenance,<br/>fallback and degraded-mode state"]

    V2["RejectResultV2<br/>strict opt-in envelope"]

    LEGACY["RejectResult<br/>stable v1.0 public return"]

    CALL --> RESOLVE
    RESOLVE --> RO
    RO --> DECISION

    DECISION --> DA
    DECISION --> SELECT

    SELECT --> PA
    SELECT --> PP
    SELECT --> XP

    PP --> PA
    XP --> PA

    DA --> V2
    PA --> V2
    META --> V2

    V2 -->|"default compatibility conversion"| LEGACY
```

The reject result separates two concerns.

### Reject decision

The decision describes the complete original input batch. It includes the
rejected mask and reject diagnostics independently of how the payload is
filtered.

In the strict representation, these values are held by
`RejectDecisionArtifact`.

### Reject payload

The payload is shaped by the selected policy:

| Policy          | Payload rows       |
| --------------- | ------------------ |
| `FLAG`          | All source rows    |
| `ONLY_REJECTED` | Only rejected rows |
| `ONLY_ACCEPTED` | Only accepted rows |

`source_indices` preserve the mapping between payload rows and the original
input batch.

Prediction and explanation payloads are independently optional. For example,
an explanation call does not need to generate a separate prediction payload
merely to populate the reject result.

In the strict representation, payload information is held by
`RejectPayloadArtifact`.

### Public result types

`RejectResult` is the stable public reject return type for v1.0. It exposes:

* `prediction`;
* `explanation`;
* `rejected`;
* `policy`; and
* `metadata`.

`RejectResultV2` is the strict opt-in representation. It separates:

* `decision`;
* `payload`;
* `policy`;
* `metadata`; and
* `schema_version`.

The runtime constructs the strict representation and converts it to
`RejectResult` for the default compatibility surface.

`RejectPolicy.NONE` bypasses this complete path and returns the ordinary
prediction or explanation result.

## Plugin and configuration resolution

This view shows how implementations are selected. It does not show the
execution order of prediction or explanation algorithms.

```{mermaid}
flowchart LR
    subgraph SOURCES["Configuration sources"]
        CALLSITE["Call-site overrides"]
        ENV["Environment variables"]
        PYPROJECT["pyproject.toml"]
        DEFAULTS["Versioned defaults"]
    end

    CM["ConfigManager<br/>resolved configuration snapshot"]

    REG["Plugin registry<br/>metadata and discovery"]

    TRUST["Plugin trust policy<br/>trusted and denied identifiers"]

    PM["PluginManager<br/>selection, instances and fallback chains"]

    BUILTIN["Built-in plugins"]
    EXTERNAL["External plugins"]

    INTERVAL["Selected interval implementation"]
    EXPLANATION["Selected explanation implementation"]
    PLOT["Selected plot implementation"]

    CALLSITE --> CM
    ENV --> CM
    PYPROJECT --> CM
    DEFAULTS --> CM

    BUILTIN --> REG
    EXTERNAL --> REG
    TRUST --> REG

    CM --> PM
    REG --> PM

    PM --> INTERVAL
    PM --> EXPLANATION
    PM --> PLOT
```

Configuration precedence is:

1. call-site override;
2. environment variable;
3. `pyproject.toml`; and
4. versioned default.

`ConfigManager` provides the resolved configuration view.
`PluginManager` owns plugin-specific runtime decisions, including defaults,
overrides, cached instances, and fallback chains.

Prediction and explanation orchestrators do not define their own independent
plugin fallback policies. They delegate those decisions to `PluginManager`.

Plugin selection covers built-in implementations as well as trusted external
extensions. Trust and deny rules are applied before untrusted plugin code is
used.

## Optional and downstream capabilities

Performance services and output transformations are not mandatory stages in
ordinary inference.

```{mermaid}
flowchart TB
    RUNTIME["Prediction and explanation runtime"]

    CACHE["Cache"]
    PARALLEL["Parallel executor"]
    FILTER["Feature filtering"]
    OBS["Logging and telemetry"]

    EXPLANATIONS["Explanation-domain objects"]

    SERIALIZE["Explicit serialization"]
    SCHEMA["Explanation Schema v1<br/>portable JSON payload"]

    PLOTREQ["Explicit plot request"]
    PLOTSPEC["PlotSpec<br/>backend-independent representation"]
    RENDER["Renderer<br/>for example Matplotlib"]

    CACHE -. "optionally accelerates" .-> RUNTIME
    PARALLEL -. "optionally accelerates" .-> RUNTIME
    FILTER -. "optionally reduces explanation work" .-> RUNTIME
    RUNTIME -. "emits operational events" .-> OBS

    RUNTIME --> EXPLANATIONS

    EXPLANATIONS --> SERIALIZE
    SERIALIZE --> SCHEMA

    EXPLANATIONS --> PLOTREQ
    PLOTREQ --> PLOTSPEC
    PLOTSPEC --> RENDER
```

Cache, parallel execution, and feature filtering are opt-in performance
capabilities. They must not redefine the calibrated semantics of predictions
or explanations.

Logging and telemetry observe runtime behaviour, including plugin selection,
fallbacks, performance, and degraded modes.

Serialization is explicitly requested. Explanation Schema v1 serializes
explanation-domain objects and does not require a generic outer envelope.

Visualization is also explicitly requested. Explanation data is first
converted into a backend-independent `PlotSpec`; a renderer then converts that
specification into a concrete figure.

A reject result is a separate result contract. It is not the outer envelope
for Explanation Schema v1 and is not a required input to plotting.

## Output contracts

| Invocation                                                      | Primary result                                                              |
| --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `predict(...)` or `predict_proba(...)` without active rejection | Prediction result, optionally including calibrated uncertainty              |
| `explain_factual(...)` without active rejection                 | `CalibratedExplanations` containing factual explanations                    |
| `explore_alternatives(...)` without active rejection            | `AlternativeExplanations`                                                   |
| `explain_fast(...)` without active rejection                    | Fast explanation collection                                                 |
| Any supported call with an active non-`NONE` reject policy      | Stable `RejectResult`, or strict `RejectResultV2` when explicitly requested |
| Explicit explanation serialization                              | Explanation Schema v1 payload                                               |
| Explicit plotting                                               | `PlotSpec`, optionally followed by a rendered figure                        |

## Architectural invariants

The following rules define the intended architecture:

1. `WrapCalibratedExplainer` is the recommended public lifecycle facade.
2. `CalibratedExplainer` owns the calibrated runtime state.
3. `PluginManager` is the single source of truth for plugin defaults,
   overrides, fallback chains, and plugin instances.
4. Orchestrators coordinate operations but use state held by the calibrated
   runtime.
5. Ordinary prediction and explanation do not produce reject envelopes.
6. Reject orchestration runs only for explicit reject operations or an
   effective non-`NONE` reject policy.
7. Reject decisions are independent of policy-based payload filtering.
8. Reject payloads preserve their mapping to the original batch through source
   indices.
9. Prediction and explanation payloads inside reject results are optional.
10. Serialization and visualization are explicit downstream operations.
11. Cache, parallel execution, and feature filtering are optional and must
    preserve baseline semantics.
12. Fallback and degraded-mode behaviour must remain visible through warnings,
    logging, telemetry, or result metadata as defined by the relevant
    contract.

## Related documentation

* [Configure runtime behaviour](../how-to/configure_runtime.md)
* [Explanation structures](explanation_structures.md)
* [Explanation Schema v1](../../schema_v1.md)
* [Parameter reference](parameter-reference.md)
* [Error handling](error_handling.md)
