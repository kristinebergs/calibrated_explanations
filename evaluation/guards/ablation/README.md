# Guard Parameter Ablation Study Framework

A production-ready ablation study system using **Optuna** for parameter sweep orchestration and **MLflow** for results tracking and analysis.

## Overview

This framework enables systematic evaluation of the perturbation guard across different parameter configurations:

- **Alpha (α)**: Miscalibration level [0.01, 0.05, 0.1, 0.2]
- **Distance metric**: Nonconformity scoring metric [euclidean, mahalanobis, cosine]
- **Cluster count**: Feature-space stratification [5, 10, 20]

**Total configurations**: 4 × 3 × 3 = 36 parameter combinations  
**With 20 seeds per config**: 720 total trials

## Architecture

### Components

1. **`ablation_executor.py`** – Optuna orchestrator
   - Manages parameter space definition
   - Coordinates trial execution
   - Integrates with MLflow logging
   - Saves results to disk

2. **`fixtures_adapter.py`** – Fixture lifecycle manager
   - Reuses pytest fixtures for Optuna trials
   - Manages caching of datasets, models, explainers
   - Supports binary/multiclass classification, regression, and probabilistic regression

3. **`metrics_collector.py`** – Metrics computation
   - Coverage metrics (acceptance/rejection rates)
   - Explanation quality metrics (validity, runtime)
   - Aggregation across seeds

4. **`results_analyzer.py`** – Post-hoc analysis
   - Parameter sensitivity analysis
   - Comparison of distance metrics and alpha values
   - Report generation and CSV export

## Installation

### Prerequisites

```bash
pip install optuna mlflow
```

### Optional but Recommended

```bash
pip install pandas matplotlib seaborn  # For analysis and visualization
```

## Quick Start

### Basic Ablation Run

```python
from evaluation.guards.ablation import AblationExecutor

# Create executor
executor = AblationExecutor(
    n_trials=36,           # Number of parameter combinations
    n_seeds=20,            # Runs per configuration (optional for basic ablation)
    storage_dir="./ablation_results",
    experiment_name="guard_parameter_ablation"
)

# Run ablation
# Note: You can pass a single task string or a list of tasks
summaries = executor.run(
    task_type=["binary_classification", "regression"],
    n_workers=1,  # Set to CPU count for parallelization
    timeout=None
)

print(f"Best configuration for binary_classification: {executor.get_best_config('coverage')}")
```

## Key Features

- **Result Persistence**: New runs append to existing results in `ablation_details.json` instead of overwriting them. This allows you to pause and resume studies or stack multiple task evaluations.
- **Strict Evaluation**: The ablation executor automatically disables the legacy fallback mechanism (`CE_DISABLE_PLUGIN_FALLBACK=1`) to ensure that any guard failures are reported as errors rather than silently falling back to unguarded explanations.
- **Multi-Task Stacking**: You can pass a list of tasks to `run()`, and the executor will process them sequentially, accumulating all results in the same storage directory.

### Analyze Results

```python
from evaluation.guards.ablation import ResultsAnalyzer

analyzer = ResultsAnalyzer("./ablation_results")

# Generate report
report = analyzer.generate_report(output_file="ablation_report.txt")
print(report)

# Get parameter sensitivity
sensitivity = analyzer.get_parameter_sensitivity(metric="coverage")
print(f"Parameter sensitivity: {sensitivity}")

# Compare distance metrics
dist_comparison = analyzer.compare_distance_metrics()
print(f"Distance metrics: {dist_comparison}")

# Export to CSV
analyzer.export_as_csv("ablation_results.csv")
```

## Configuration Examples

### Minimal Smoke Test

```python
executor = AblationExecutor(
    n_trials=6,  # 2 alphas × 1 distance × 3 clusters
    n_seeds=1,
)
summary = executor.run(task_type="binary_classification")
```

### Full Ablation (36 configs × 20 seeds = 720 trials)

```python
executor = AblationExecutor(
    n_trials=720,  # All combinations
    n_seeds=20,
)
summary = executor.run(
    task_type="binary_classification",
    n_workers=8,  # Parallel execution on 8 CPU cores
    timeout=3600  # 1 hour timeout
)
```

### Custom Parameter Grid

To use a different parameter grid, modify `ablation_executor.py`:

```python
# In the _evaluate_trial method
alpha = trial.suggest_categorical("alpha", [0.05, 0.1, 0.15])  # Custom alphas
distance = trial.suggest_categorical("distance", ["euclidean", "mahalanobis"])  # Fewer metrics
n_clusters = trial.suggest_int("n_clusters", 3, 25, step=2)  # Range instead of discrete
```

## Results Structure

### Generated Files

```
ablation_results/
├── guard_ablation_binary_classification_20251123_143022.db  # Optuna study database
├── ablation_summary.json        # High-level results summary
├── ablation_details.json        # Detailed per-trial results
└── ablation_report.txt          # Human-readable report (if generated)
```

### Example Summary

```json
{
  "study_name": "guard_ablation_binary_classification_20251123_143022",
  "task_type": "binary_classification",
  "n_trials": 36,
  "n_complete_trials": 36,
  "best_trial_number": 17,
  "best_trial_value": 0.95,
  "best_params": {
    "alpha": 0.05,
    "distance": "mahalanobis",
    "n_clusters": 10
  },
  "aggregated_by_config": {
    "alpha_0.01_dist_euclidean_clusters_5": {
      "coverage": {"mean": 0.92, "std": 0.03, "min": 0.88, "max": 0.96},
      ...
    }
  }
}
```

## Metrics Collected

### Coverage Metrics
- **acceptance_rate**: Fraction of test instances accepted by guard
- **rejection_rate**: Fraction of test instances rejected by guard
- **coverage**: Estimate of true positive coverage

### Explanation Quality Metrics
- **factual_validity_rate**: Fraction of valid factual explanations
- **factual_explanation_runtime**: Time to generate factual explanations (seconds)
- **alternative_explanations_runtime**: Time for alternative explanations (seconds)

## Integration with MLflow

When MLflow is installed, all trials are automatically logged:

```bash
# View MLflow dashboard
mlflow ui --backend-store-uri "./ablation_results"
```

Dashboard shows:
- Parameter values per trial
- Metric values (coverage, validity, runtime)
- Comparative run analysis
- Parameter importance (if sufficient trials)

## Advanced Usage

### Custom Metric Functions

Extend `MetricsCollector` to add domain-specific metrics:

```python
class CustomMetricsCollector(MetricsCollector):
    @staticmethod
    def compute_custom_metric(explainer, X_test, y_test):
        # Your metric computation here
        return {"custom_metric": value}
```

Then use in `AblationExecutor._evaluate_trial()`.

### Resuming Interrupted Studies

Optuna automatically saves study progress. To resume:

```python
study = optuna.load_study(
    study_name="guard_ablation_binary_classification_20251123_143022",
    storage="sqlite:///ablation_results/guard_ablation_binary_classification_20251123_143022.db"
)
```

### Exporting Results for Publication

```python
analyzer = ResultsAnalyzer()

# Generate report
analyzer.generate_report("publication_report.txt")

# Export CSV for statistical analysis
analyzer.export_as_csv("ablation_results.csv")

# Get top configurations for paper
top_5 = analyzer.get_best_configurations(top_k=5)
for config in top_5:
    print(f"alpha={config['alpha']}, distance={config['distance']}, "
          f"n_clusters={config['n_clusters']}: {config['coverage']:.4f}")
```

## Troubleshooting

### MLflow Not Available

If MLflow is not installed, results are still saved to disk. Install with:

```bash
pip install mlflow
```

### Slow Trial Execution

If trials take minutes each:

1. **Reduce max_instances** in `MetricsCollector.compute_explanation_quality_metrics()`
2. **Use fewer seeds** initially (set `n_seeds=1` for quick iteration)
3. **Profile with**: `python -m cProfile -s cumtime your_script.py`

### Memory Issues with Large Parameter Grids

Use Optuna's pruning to stop unpromising trials early:

```python
sampler = TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
study = optuna.create_study(sampler=sampler, pruner=pruner)
```

## Next Steps

### Extend the Framework

1. **Add more task types**: Multiclass with imbalance
2. **Custom objectives**: Add constraint optimization (e.g., runtime < 1s)
3. **Visualization**: Generate plots of parameter sensitivity
4. **Hyperband scheduling**: Replace TPE with more efficient samplers

### Integration with Release Plan

This ablation framework supports:
- **ORQ1** (Reliability Guarantees): Measures coverage under different alpha values
- **ORQ2** (Uncertainty-Aware Robustness): Evaluates robustness across distance metrics
- **ORQ4** (Generalizability): Tests sensitivity to n_clusters across task types

See `improvement_docs/RELEASE_PLAN_V1.md` for alignment with guard evaluation gates.

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **MLflow Documentation**: https://mlflow.org/docs/
- **Guard Implementation**: `src/calibrated_explanations/core/explain/guards/regions.py`
- **Test Infrastructure**: `evaluation/guards/conftest.py`, `evaluation/guards/pytest_plugin.py`
