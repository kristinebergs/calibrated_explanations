# Optuna + MLflow Ablation Study Framework Implementation

**Date**: 2025-11-23  
**Status**: Complete  
**Location**: `evaluation/guards/ablation/`

## Executive Summary

An end-to-end ablation study framework has been implemented to systematically evaluate guard parameters (α, distance metric, cluster count) using **Optuna** for orchestration and **MLflow** for result tracking.

**Key capabilities**:
- Evaluates 36 parameter combinations (4 alpha × 3 distance × 3 clusters)
- Optional 20 random seeds per configuration (720 total trials)
- Optuna's TPE sampler for efficient exploration
- MLflow integration for result logging and comparison
- Comprehensive analysis tools (sensitivity analysis, ranking, export)
- Parallel execution support for local multi-core machines

## Why This Approach?

### Comparison with Pytest

| Aspect | Pytest | Optuna + MLflow |
|--------|--------|-----------------|
| **Design philosophy** | Testing (pass/fail) | Experimentation (measure/optimize) |
| **Parameter handling** | Implicit via fixtures | Explicit trial suggestions |
| **Result aggregation** | Plugin-based, custom | Built-in with MLflow |
| **Parallelization** | pytest-xdist (basic) | Optuna's n_jobs (flexible) |
| **Scalability** | Good for 50 tests | Excellent for 1000+ trials |
| **UI/Dashboard** | HTML reports (static) | MLflow UI (interactive) |

**Decision**: Optuna + MLflow chosen because ablation studies require:
1. Structured parameter space exploration (Optuna provides this)
2. Centralized result tracking (MLflow excels at this)
3. Analysis-focused workflows (both are optimized for this)

## Architecture

### Package Structure

```
evaluation/guards/ablation/
├── __init__.py                    # Public API
├── README.md                      # User guide
├── requirements_ablation.txt      # Dependencies
│
├── fixtures_adapter.py            # Reuses pytest fixtures
│   ├── PerturbationGuardConfig    # Parameter container
│   └── FixturesAdapter            # Caches datasets/models/explainers
│
├── ablation_executor.py           # Optuna orchestrator
│   └── AblationExecutor           # Main execution engine
│
├── metrics_collector.py           # Metrics computation
│   ├── AblationMetrics            # Dataclass for results
│   └── MetricsCollector           # Coverage & quality metrics
│
├── results_analyzer.py            # Post-hoc analysis
│   └── ResultsAnalyzer            # Comparison & reporting
│
└── example_ablation.py            # CLI example + documentation
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AblationExecutor.run(task_type)                         │
│    - Creates Optuna study with TPE sampler                 │
│    - Defines parameter space (alpha, distance, n_clusters) │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. For each trial: _evaluate_trial(trial)                  │
│    - Suggest parameters via trial.suggest_categorical()    │
│    - Create FixturesAdapter (cached)                       │
│    - Instantiate guarded explainer                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Compute metrics via MetricsCollector                     │
│    - Coverage metrics (acceptance, rejection)              │
│    - Quality metrics (validity, runtime)                   │
│    - Log to MLflow if available                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ResultsAnalyzer post-processing                         │
│    - Parameter sensitivity                                 │
│    - Best configurations                                   │
│    - Comparison tables                                     │
│    - Report generation & CSV export                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `FixturesAdapter` – Fixture Lifecycle Management

Bridges pytest fixture infrastructure to Optuna trials:

```python
adapter = FixturesAdapter(random_seed=42)

# Fixtures are created once and cached
data = adapter.get_binary_classification_data()
model = adapter.get_binary_classifier()
explainer = adapter.create_guarded_explainer(task_type, guard_config)
```

**Design decisions**:
- Session-scoped caching (dict-based, no pytest.fixture decorator)
- Reuses existing fixture logic from `conftest.py`
- Avoids repeated model training across trials

### 2. `AblationExecutor` – Optuna Orchestrator

Main entry point for running studies:

```python
executor = AblationExecutor(
    n_trials=36,
    n_seeds=20,
    storage_dir="./ablation_results",
    experiment_name="guard_parameter_ablation"
)
summary = executor.run(task_type="binary_classification", n_workers=8)
```

**Features**:
- TPESampler for intelligent parameter exploration
- SQLite backend for persistent study database
- MLflow integration for automatic logging
- Results saved as JSON and queryable Python objects

### 3. `MetricsCollector` – Metrics Computation

Standard metrics across all ablations:

| Metric | Type | Range | Interpretation |
|--------|------|-------|-----------------|
| `acceptance_rate` | Coverage | [0, 1] | Fraction of test instances accepted by guard |
| `rejection_rate` | Coverage | [0, 1] | Fraction of test instances rejected (should ≈ α) |
| `factual_validity_rate` | Quality | [0, 1] | Fraction of valid factual explanations |
| `factual_explanation_runtime` | Efficiency | seconds | Time to generate explanations |
| `alternative_explanations_runtime` | Efficiency | seconds | Time for alternative explorations |

### 4. `ResultsAnalyzer` – Post-Hoc Analysis

Analysis after trials complete:

```python
analyzer = ResultsAnalyzer("./ablation_results")

# Parameter sensitivity (main effects)
sensitivity = analyzer.get_parameter_sensitivity()

# Best configurations
top_5 = analyzer.get_best_configurations(top_k=5)

# Comparison tables
dist_comp = analyzer.compare_distance_metrics()
alpha_comp = analyzer.compare_alpha_values()

# Export for publication
analyzer.generate_report("report.txt")
analyzer.export_as_csv("results.csv")
```

## Usage Patterns

### Quick Start (36 trials, no seeds)

```bash
python evaluation/guards/ablation/example_ablation.py \
    --task binary_classification \
    --n-trials 36 \
    --n-workers 1
```

### Production Run (36 configs × 20 seeds = 720 trials)

```bash
python evaluation/guards/ablation/example_ablation.py \
    --task binary_classification \
    --n-trials 720 \
    --n-seeds 20 \
    --n-workers 8 \
    --timeout 3600 \
    --generate-report \
    --export-csv
```

### Analysis Only (from saved results)

```bash
python evaluation/guards/ablation/example_ablation.py \
    --task binary_classification \
    --analyze-only \
    --generate-report \
    --export-csv
```

## Integration with Release Plan

This framework supports the following research questions and gates from `RELEASE_PLAN_V1.md`:

### ORQ1 – Reliability Guarantees
- **Metric**: `acceptance_rate` should approximate `1 - α`
- **Gate**: Coverage guarantee verified across α ∈ [0.01, 0.05, 0.1, 0.2]

### ORQ2 – Uncertainty-Aware Robustness
- **Metric**: `factual_validity_rate` increases with better distance metrics
- **Gate**: Distance metric sensitivity analyzed via `analyzer.compare_distance_metrics()`

### ORQ4 – Generalizability & Parameter Behavior
- **Metric**: All metrics computed across task types (binary, multiclass, regression)
- **Gate**: Complete ablation grid (alpha × distance × clusters) evaluated

**Alignment document**: See `improvement_docs/ADR-006` on guard trust model and coverage testing.

## Technical Decisions

### Why Optuna vs. Ray Tune for This Scale?

For 36–720 trials on a local machine:
- **Optuna is simpler**: No remote infrastructure needed, built-in TPE sampler
- **Ray Tune is overkill**: Designed for 1000s of trials with distributed nodes
- **Hybrid approach**: MLflow + Optuna provides 90% of Ray's functionality locally

**If scale increases (10k+ trials)**: Migrate to Ray Tune with same metrics/analyzers.

### Why Dict-Based Caching vs. Fixtures?

Optuna trials don't integrate cleanly with pytest's fixture system:
- FixturesAdapter uses explicit `get_*` methods instead of fixtures
- Simpler lifecycle management without pytest plugin coupling
- Cache can be inspected/debugged easily

### Why Multiple Task Types?

Each task type (binary, multiclass, regression) has distinct:
- Model types (RandomForest classifier/regressor)
- Data generation (classification vs. regression)
- Evaluation metrics (class coverage vs. interval validity)

Running separate studies per task type allows for direct comparison of guard behavior.

## Extensibility

### Add Custom Metrics

```python
# In metrics_collector.py
class CustomMetricsCollector(MetricsCollector):
    @staticmethod
    def compute_perturbation_distance(explainer, X_test):
        # Your custom metric
        return {"avg_perturbation_distance": value}
```

Then use in `AblationExecutor._evaluate_trial()`.

### Use Different Samplers

```python
# In ablation_executor.py
sampler = optuna.samplers.GridSampler({
    "alpha": [0.01, 0.05, 0.1, 0.2],
    "distance": ["euclidean", "mahalanobis", "cosine"],
    "n_clusters": [5, 10, 20],
})
study = optuna.create_study(sampler=sampler)
```

### Integrate with Weights & Biases

```python
# In ablation_executor.py
import wandb

if self.mlflow_available:
    mlflow.log_metrics(all_metrics)
    
# Also log to W&B
wandb.log(all_metrics)
```

## Performance Considerations

### Estimated Runtime

| Configuration | Runtime (approx) |
|---------------|-----------------|
| 36 trials (1 seed each) | 15–30 minutes |
| 36 trials × 20 seeds | 5–10 hours |
| 36 trials × 20 seeds (8-core) | 1–2 hours |

Times depend on:
- Model training (dominant cost)
- Explanation generation (scales with test set size)
- Metrics computation

### Optimizations Applied

1. **Fixture caching**: Models trained once, reused across 36 trials
2. **Limited evaluation set**: `max_instances=50` in quality metrics (vs. full test set)
3. **Session-scope design**: No redundant data generation

### Future Optimizations

1. Model serialization (pickle) for zero-copy reuse
2. Batch explanation generation (vectorized where possible)
3. Early stopping via Optuna pruning (if time is primary objective)

## Validation

### Unit Tests Recommended

```python
# tests/integration/test_ablation_framework.py
def test_fixtures_adapter_caching():
    adapter = FixturesAdapter()
    data1 = adapter.get_binary_classification_data()
    data2 = adapter.get_binary_classification_data()
    assert data1 is data2  # Same object (cached)

def test_ablation_executor_runs_full_grid():
    executor = AblationExecutor(n_trials=36, n_seeds=1)
    summary = executor.run(task_type="binary_classification", n_workers=1)
    assert len(executor.results) == 36

def test_results_analyzer_loads_and_analyzes():
    analyzer = ResultsAnalyzer("./ablation_results")
    sensitivity = analyzer.get_parameter_sensitivity()
    assert "alpha" in sensitivity
    assert "distance" in sensitivity
    assert "n_clusters" in sensitivity
```

## Deployment Checklist

- [x] Core modules (fixtures_adapter, ablation_executor, metrics_collector, results_analyzer)
- [x] Example script with CLI
- [x] Requirements file
- [x] Comprehensive README
- [x] Docstrings in all functions
- [x] Error handling (try/except in trials, informative logging)
- [ ] Unit tests (recommended for CI/CD)
- [ ] Integration with CI/CD pipeline (future)

## Next Steps

### Phase 1 (Immediate)
1. Run quick smoke test: `example_ablation.py --n-trials 6 --n-workers 1`
2. Verify MLflow integration: `mlflow ui`
3. Review generated reports and CSV exports

### Phase 2 (After Validation)
1. Run full ablation: `example_ablation.py --n-trials 36 --n-workers 8`
2. Analyze parameter sensitivity
3. Document findings in paper/proposal

### Phase 3 (Enhancement)
1. Add custom metrics (e.g., explanation consistency across seeds)
2. Implement Optuna pruning for early stopping
3. Integrate with CI/CD (automated nightly ablations)

### Phase 4 (Scale-Up)
1. Migrate to Ray Tune if trials exceed 1000
2. Add Kubernetes support for multi-node execution
3. Integrate with W&B or TensorBoard for richer dashboards

## References

- **Optuna**: https://optuna.readthedocs.io/
- **MLflow**: https://mlflow.org/
- **Original pytest framework**: `evaluation/guards/conftest.py`, `evaluation/guards/pytest_plugin.py`
- **Release Plan**: `improvement_docs/RELEASE_PLAN_V1.md` (ADR-006 guard coverage testing)
- **ADRs**: `improvement_docs/adrs/` (ADR-004 parallel execution, ADR-005 schema)

## Contacts & Support

- **Framework author**: GitHub Copilot
- **Questions about guard implementation**: See `src/calibrated_explanations/core/explain/guards/regions.py`
- **Integration help**: Review `evaluation/guards/ablation/example_ablation.py`

---

**Last Updated**: 2025-11-23  
**Status**: Ready for testing and feedback
