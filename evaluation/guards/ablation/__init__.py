"""
Optuna + MLflow Ablation Study Framework for Guard Parameter Evaluation.

This package provides a production-ready ablation study system for evaluating
the perturbation guard under varying parameter configurations (alpha, distance, n_clusters).

Key modules:
- ablation_executor.py: Main orchestrator for running parameter sweeps
- fixtures_adapter.py: Bridges pytest fixtures to Optuna trials
- metrics_collector.py: Collects and normalizes experiment metrics
- results_analyzer.py: Analyzes and compares ablation results

Example usage:
    from evaluation.guards.ablation import AblationExecutor
    
    executor = AblationExecutor(
        n_trials=36,
        n_seeds=20,
        storage_dir="./ablation_results"
    )
    executor.run(task_type="binary_classification")
"""

from .ablation_executor import AblationExecutor
from .fixtures_adapter import FixturesAdapter
from .metrics_collector import MetricsCollector
from .results_analyzer import ResultsAnalyzer

__all__ = [
    "AblationExecutor",
    "FixturesAdapter",
    "MetricsCollector",
    "ResultsAnalyzer",
]
