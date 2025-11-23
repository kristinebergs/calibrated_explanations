# Ablation Study Framework Analysis & Recommendations

## Current Situation

You have a pytest-based ablation study in `evaluation/guards/` that evaluates a guarded explainer using parametrized tests. The setup includes:

- **Fixture-based model reuse**: Session-scoped fixtures for datasets, trained models, and calibrated predictors
- **Configuration objects**: `PerturbationGuardConfig` with `alpha`, `distance`, `n_clusters` parameters
- **Parameterized tests**: Using `@pytest.mark.parametrize` with Cartesian product of configs
- **Results collection**: Custom pytest plugin for collecting and serializing test results

**Limitation identified**: While pytest works, it's not optimized for ablation studies because:
1. Tests are designed for pass/fail outcomes, not measurement collection
2. Parameterization creates isolated test instances (less natural for shared state management)
3. Test discovery/reporting overhead not ideal for large experimental matrices
4. Limited built-in support for experimental workflows like hyperparameter tracking, result aggregation, and result comparison

---

## Recommended Frameworks for Ablation Studies

### 1. **Optuna** (Top Recommendation for Hyperparameter Ablation)
**Use case**: When you want to systematically explore a parameter space and potentially use Bayesian optimization to guide the search.

**Strengths**:
- Built for hyperparameter optimization and ablation studies
- Native Cartesian product support via `suggest_categorical`, `suggest_int`, `suggest_float`
- Automatic trial logging and visualization (built-in plotting)
- Per-trial metadata and intermediate value reporting
- Easy parallel trial execution with `ThreadPoolExecutor` or distributed samplers
- Result database/history for cross-experiment comparison

**Example approach**:
```python
import optuna

def objective(trial):
    alpha = trial.suggest_categorical('alpha', [0.01, 0.05, 0.1, 0.25])
    distance = trial.suggest_categorical('distance', ['euclidean', 'mahalanobis', 'cosine'])
    n_clusters = trial.suggest_categorical('n_clusters', [5, 10, 20])
    
    # Run your ablation
    result = evaluate_guard(alpha, distance, n_clusters)
    return result['coverage_metric']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=36)
```

**Integration cost**: Moderate. Add trial creation/reporting; reuse existing fixtures via direct function calls.

---

### 2. **Ray Tune** (Best for Distributed, Large-Scale Ablations)
**Use case**: When you need distributed execution, more sophisticated scheduling, and integration with monitoring tools.

**Strengths**:
- Distributed by default (local multi-process, remote Ray cluster, Kubernetes)
- Advanced schedulers (ASHA, PBT, Population Based Training)
- Built-in result logging, checkpointing, and visualization
- Population-based ablations (dynamic parameter updates mid-study)
- Integration with TensorBoard and Weights & Biases
- Graceful handling of long-running tasks

**Example approach**:
```python
from ray import tune

config = {
    'alpha': tune.grid_search([0.01, 0.05, 0.1, 0.25]),
    'distance': tune.grid_search(['euclidean', 'mahalanobis', 'cosine']),
    'n_clusters': tune.grid_search([5, 10, 20]),
}

tune.run(
    trainable=evaluate_guard,
    config=config,
    num_samples=1,
    checkpoint_dir='./ray_results',
)
```

**Integration cost**: Moderate-High. Requires refactoring to Ray's trainable class pattern, but excellent scalability.

---

### 3. **MLflow** (Best for Results Management & Tracking)
**Use case**: When your primary need is recording, comparing, and managing experiment results.

**Strengths**:
- Experiment tracking with rich metadata (parameters, metrics, artifacts, tags)
- Easy run comparison and filtering via UI
- Automatic logging of code version, environment, dependencies
- Parameter sweep support (no complex syntax needed)
- Direct integration with jupyter notebooks
- Results queryable via Python API for downstream analysis

**Example approach**:
```python
import mlflow

for alpha in [0.01, 0.05, 0.1, 0.25]:
    for distance in ['euclidean', 'mahalanobis', 'cosine']:
        for n_clusters in [5, 10, 20]:
            with mlflow.start_run():
                mlflow.log_params({'alpha': alpha, 'distance': distance, 'n_clusters': n_clusters})
                results = evaluate_guard(alpha, distance, n_clusters)
                mlflow.log_metrics(results)
```

**Integration cost**: Low. Minimal changes to your existing code; mainly add MLflow logging calls.

---

### 4. **Hydra** (Best for Configuration Management)
**Use case**: When you want a single, composable configuration system for experiments.

**Strengths**:
- Configuration as code (YAML-based or Python dataclasses)
- Automatic directory structure per run (logs, outputs, config snapshots)
- Composition and override system for sweep definitions
- Launcher plugins for distributed execution (e.g., Ray, Slurm)
- Easy multi-run with built-in Cartesian product via `--multirun`

**Example approach**:
```yaml
# config.yaml
experiment:
  alpha: 0.01
  distance: euclidean
  n_clusters: 5
```

```bash
python experiment.py --multirun \
  experiment.alpha=0.01,0.05,0.1,0.25 \
  experiment.distance=euclidean,mahalanobis,cosine \
  experiment.n_clusters=5,10,20
```

**Integration cost**: Moderate. Requires config structure setup; pairs well with MLflow or Ray.

---

### 5. **Pytest with Plugins** (Current Path, Enhanced)
**Use case**: If you want to stick with pytest but improve it for ablation workflows.

**Strengths**:
- Already familiar to your codebase
- Rich plugin ecosystem (pytest-benchmark, pytest-xdist, pytest-html)
- Fixtures naturally handle shared state
- Can combine with MLflow for results tracking

**Enhancement suggestions**:
- Add `pytest-benchmark` for reproducible timing measurements
- Use `pytest-xdist` for parallel execution
- Combine with MLflow logging in fixtures
- Use `pytest-html` for rich result reports

**Example enhancement**:
```python
@pytest.mark.parametrize("config", guard_config_grid)
def test_coverage(benchmark, config, fixtures):
    result = benchmark(evaluate_guard, config)
    mlflow.log_metric(f"coverage_{config}", result['coverage'])
```

**Integration cost**: Low. Minimal changes to existing structure.

---

## Recommendation Summary

| Framework | Best For | Ease of Integration | Scale | Recommendation |
|-----------|----------|---------------------|-------|-----------------|
| **Optuna** | Systematic exploration, HPO | Medium | Local/Distributed | ⭐ **Primary** for parameter ablation |
| **Ray Tune** | Distributed large-scale ablation | Medium-High | Distributed | ⭐ **Primary** if distributed needed |
| **MLflow** | Results management & comparison | Low | Any | ⭐ **Supplement** with any framework |
| **Hydra** | Configuration management | Medium | Any | ✅ Good complement to Optuna/Ray |
| **Enhanced Pytest** | Minimal refactor | Very Low | Local/Xdist | ✅ Good if staying with pytest |

---

## Recommended Approach (Hybrid)

### **Tier 1: Optuna + MLflow (Recommended)**

This combination gives you:
1. **Optuna** for clean, declarative ablation definitions
2. **MLflow** for centralized result tracking and comparison
3. **Reuse existing fixtures** by wrapping them in Optuna trials

**Benefits**:
- Clear ablation semantics (Optuna handles Cartesian product naturally)
- Results queryable for analysis (MLflow UI + Python API)
- Low friction: minimal code changes from current pytest approach
- Natural progression if you later add Bayesian optimization

**Estimated effort**: 2-3 hours to prototype, ~1 day to fully integrate.

---

### **Tier 2: Ray Tune (If Distributed/Scale Needed)**

If you anticipate running ablations on multiple datasets or need distributed execution:

1. **Ray Tune** for execution and scheduling
2. **MLflow** for result tracking
3. **Hydra** for configuration management

**Benefits**:
- Seamless scaling from laptop to cluster
- Advanced schedulers (e.g., ASHA for early stopping)
- Rich monitoring via TensorBoard

**Estimated effort**: 1-2 days for refactoring to Ray trainable pattern.

---

### **Tier 3: Enhanced Pytest (If Minimal Refactor Preferred)**

If you want to minimize changes:

1. Enhance current pytest with `pytest-benchmark` and `pytest-xdist`
2. Add MLflow logging to conftest and test functions
3. Use `pytest-html` for reporting

**Benefits**:
- No major structural changes
- Leverage existing fixtures and parametrization
- Still get result tracking and parallelization

**Estimated effort**: 1-2 hours.

---

## Next Steps

1. **Clarify ablation scale**: How many configurations? How many datasets? How long per run?
   - Small (< 100 trials): Optuna or enhanced pytest
   - Large (> 1000 trials): Ray Tune
   - Any scale with result comparison: Always add MLflow

2. **Choose integration path**:
   - Option A: Start with **Optuna + MLflow** (recommended)
   - Option B: Enhance **current pytest** with MLflow + benchmarking
   - Option C: Migrate to **Ray Tune** if distribution needed later

3. **Implement incrementally**:
   - Phase 1: Add MLflow to current pytest (1 hour)
   - Phase 2: Optionally refactor to Optuna (1 day)
   - Phase 3: Scale to Ray if needed (1-2 days)

---

## Questions to Finalize Recommendation

Before I implement a specific framework, please clarify:

1. **How many distinct parameter combinations** are you planning to ablate?
   - Example: 4 alphas × 3 distances × 3 cluster counts = 36 configurations

2. **How many runs/seeds** per configuration (for statistical robustness)?
   - Example: 20 random seeds per configuration = 720 total trials

3. **Average runtime per trial**?
   - Example: 5 seconds per trial → 1 hour total (sequential)

4. **Do you need distributed execution** or is local parallelization sufficient?

5. **What comparison/analysis** do you plan post-ablation?
   - Side-by-side metrics? Visualization? Statistical tests?

Once you clarify these, I can implement the most efficient framework for your use case.
