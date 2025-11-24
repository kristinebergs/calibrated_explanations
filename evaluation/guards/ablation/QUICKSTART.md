# Quick Start Guide: Guard Parameter Ablation

## Installation

1. **Install dependencies**:
```bash
cd evaluation/guards/ablation
pip install -r requirements_ablation.txt
```

2. **Verify installation**:
```bash
python -c "import optuna; import mlflow; print('✓ Ready to run ablations')"
```

## Run Your First Ablation (5 minutes)

### Option 1: Command Line

```bash
cd evaluation/guards/ablation

# Quick smoke test (6 trials, no parallelization)
python example_ablation.py \
    --task binary_classification \
    --n-trials 6 \
    --n-workers 1

# Or run multiple tasks in sequence (results are accumulated)
python example_ablation.py \
    --task binary_classification probabilistic_regression \
    --n-trials 6 \
    --n-workers 1

# View results
cat ../../../ablation_results/ablation_summary.json
```

### Option 2: Python Script

Create `my_ablation.py`:

```python
from evaluation.guards.ablation import AblationExecutor, ResultsAnalyzer

# Run ablation
executor = AblationExecutor(
    n_trials=6,  # Quick test: 2 alphas × 1 distance × 3 clusters
    n_seeds=1,
)
# Pass a list of tasks to run them sequentially
summaries = executor.run(
    task_type=["binary_classification", "regression"], 
    n_workers=1
)

print("✓ Ablation complete!")
# Results are saved to disk and appended to previous runs

# Analyze results
analyzer = ResultsAnalyzer()
sensitivity = analyzer.get_parameter_sensitivity()
print(f"\nParameter sensitivity:\n{sensitivity}")
```

Run:
```bash
python my_ablation.py
```

## View Results in MLflow UI

```bash
# Start MLflow dashboard
mlflow ui --backend-store-uri "./ablation_results"

# Open browser to http://localhost:5000
```

Then:
1. Select experiment: `guard_parameter_ablation`
2. Compare runs: View parameter values and metrics side-by-side
3. Download CSV: Export all trials for further analysis

## Full Ablation (36 trials, ~15 minutes)

```bash
python example_ablation.py \
    --task binary_classification \
    --n-trials 36 \
    --n-workers 1 \
    --generate-report \
    --export-csv

# View analysis
cat ./ablation_results/ablation_report.txt
```

## Full Ablation with All Seeds (720 trials, ~2-10 hours)

```bash
# Parallel on 8 cores
python example_ablation.py \
    --task binary_classification \
    --n-trials 720 \
    --n-seeds 20 \
    --n-workers 8 \
    --timeout 36000 \
    --generate-report \
    --export-csv

# Monitor progress in MLflow UI in another terminal
mlflow ui --backend-store-uri "./ablation_results"
```

## Common Tasks

### Get Best Configuration

```python
from evaluation.guards.ablation import ResultsAnalyzer

analyzer = ResultsAnalyzer()
best = analyzer.get_best_configurations(top_k=1)[0]
print(f"Best: alpha={best['alpha']}, distance={best['distance']}, "
      f"n_clusters={best['n_clusters']}")
```

### Compare Distance Metrics

```python
analyzer = ResultsAnalyzer()
comparison = analyzer.compare_distance_metrics()
for metric, stats in comparison.items():
    print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

### Generate Report

```bash
python example_ablation.py \
    --analyze-only \
    --generate-report
```

Then view:
```bash
cat ./ablation_results/ablation_report.txt
```

### Export to CSV for Further Analysis

```bash
python example_ablation.py \
    --analyze-only \
    --export-csv
```

Then in Python/R/Excel:
```python
import pandas as pd
df = pd.read_csv("./ablation_results/ablation_results.csv")
df.groupby("distance")["coverage"].agg(["mean", "std"])
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'optuna'"

**Solution**: Install optuna
```bash
pip install optuna mlflow
```

### Issue: Trials take too long (minutes per trial)

**Solution**: Reduce the evaluation set size

Edit `ablation_executor.py` line ~160:
```python
# Change from:
quality_metrics = MetricsCollector.compute_explanation_quality_metrics(
    explainer, X_test, y_test, max_instances=50
)

# To:
quality_metrics = MetricsCollector.compute_explanation_quality_metrics(
    explainer, X_test, y_test, max_instances=10  # Fewer instances
)
```

### Issue: Out of memory with parallel workers

**Solution**: Reduce number of workers
```bash
python example_ablation.py --n-workers 2  # Instead of 8
```

### Issue: Want to customize parameters

**Solution**: Edit `ablation_executor.py` in the `_evaluate_trial` method:

```python
# Current:
alpha = trial.suggest_categorical("alpha", [0.01, 0.05, 0.1, 0.2])

# Change to:
alpha = trial.suggest_categorical("alpha", [0.05, 0.1, 0.15])  # Your values
```

## Next Steps

1. **Read the full guide**: `evaluation/guards/ablation/README.md`
2. **Understand the architecture**: `improvement_docs/OPTUNA_MLFLOW_ABLATION_IMPLEMENTATION.md`
3. **Extend with custom metrics**: See "Extensibility" section in README
4. **Integrate with your paper**: Export CSV and use with matplotlib/seaborn

## File Structure

```
evaluation/guards/ablation/
├── README.md                 # Full documentation
├── example_ablation.py       # Entry point (read this first!)
├── ablation_executor.py      # Optuna orchestrator
├── fixtures_adapter.py       # Fixture management
├── metrics_collector.py      # Metrics computation
├── results_analyzer.py       # Analysis & reporting
└── requirements_ablation.txt # Dependencies
```

## One-Liner Quick Tests

```bash
# Just run and analyze (36 trials)
python example_ablation.py --task binary_classification --n-trials 36 --generate-report --export-csv

# Run all three task types
for task in binary_classification multiclass_classification regression; do
  python example_ablation.py --task $task --n-trials 36 --n-workers 1
done

# Compare results
ls -lh ./ablation_results/ablation_*.json
```

---

**Need help?** Check the example script: `python example_ablation.py --help`
