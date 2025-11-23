"""
Optuna-based ablation study executor.

This module orchestrates parameter sweep experiments using Optuna,
integrating with MLflow for results tracking and analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .fixtures_adapter import FixturesAdapter, PerturbationGuardConfig
from .metrics_collector import MetricsCollector, AblationMetrics


logger = logging.getLogger(__name__)


class AblationExecutor:
    """
    Main executor for ablation studies using Optuna + MLflow.

    Orchestrates parameter sweeps across multiple random seeds,
    logs results to MLflow, and provides analysis utilities.

    Parameters
    ----------
    n_trials : int, optional
        Total number of trials (parameter combinations × seeds).
        Default 36 (4 alpha × 3 distance × 3 clusters × 1 seed).
    n_seeds : int, optional
        Number of random seeds per parameter configuration. Default 20.
    storage_dir : str or Path, optional
        Directory for Optuna study database and results. Default "./ablation_results".
    experiment_name : str, optional
        MLflow experiment name. Default "guard_parameter_ablation".
    """

    def __init__(
        self,
        n_trials: int = 36,
        n_seeds: int = 20,
        storage_dir: str | Path = "./ablation_results",
        experiment_name: str = "guard_parameter_ablation",
    ):
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.storage_dir = Path(storage_dir)
        self.experiment_name = experiment_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Fixtures adapter for reusable components
        self.fixtures = FixturesAdapter(random_seed=42)

        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            self.mlflow_available = True
        else:
            self.mlflow_available = False
            logger.warning("MLflow not available; results will only be saved to disk")

        # Store all results for later analysis
        self.results: list[Dict[str, Any]] = []

    def run(
        self,
        task_type: str = "binary_classification",
        n_workers: int = 1,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the ablation study.

        Parameters
        ----------
        task_type : str
            One of: "binary_classification", "multiclass_classification", "regression"
        n_workers : int, optional
            Number of parallel workers. Default 1 (serial execution).
        timeout : int, optional
            Timeout in seconds for the entire study. Default None (no timeout).

        Returns
        -------
        dict
            Summary of ablation results including best configurations.
        """
        logger.info(
            f"Starting ablation study: task_type={task_type}, "
            f"n_trials={self.n_trials}, n_seeds={self.n_seeds}"
        )

        # Create Optuna study
        study_name = f"guard_ablation_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = f"sqlite:///{self.storage_dir / f'{study_name}.db'}"

        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
            load_if_exists=False,
            direction="maximize",  # Maximize coverage/validity
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            return self._evaluate_trial(trial, task_type)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=n_workers,
            timeout=timeout,
        )

        # Summarize results
        summary = self._summarize_results(study, task_type)
        self._save_results(summary)

        logger.info(f"Ablation study completed. Results saved to {self.storage_dir}")
        return summary

    def _evaluate_trial(self, trial: optuna.Trial, task_type: str) -> float:
        """
        Evaluate a single trial (parameter configuration).

        Returns a scalar score (coverage/validity) to be maximized.
        """
        # Suggest parameters
        alpha = trial.suggest_categorical("alpha", [0.01, 0.05, 0.1, 0.2])
        distance = trial.suggest_categorical("distance", ["euclidean", "mahalanobis", "cosine"])
        n_clusters = trial.suggest_categorical("n_clusters", [5, 10, 20])

        # Log to MLflow
        if self.mlflow_available:
            mlflow.start_run(nested=True)
            mlflow.log_params({
                "alpha": alpha,
                "distance": distance,
                "n_clusters": n_clusters,
                "task_type": task_type,
            })

        try:
            # Create guard configuration
            guard_config = PerturbationGuardConfig(
                alpha=alpha,
                distance=distance,
                n_clusters=n_clusters,
                random_state=42,
            )

            # Create explainer
            explainer = self.fixtures.create_guarded_explainer(task_type, guard_config)

            # Get test data
            X_test, y_test = self.fixtures.get_test_data(task_type)
            threshold = self.fixtures.get_threshold(task_type)

            # Compute metrics
            logger.info(f"Computing coverage metrics for trial {trial.number}...")
            coverage_metrics = MetricsCollector.compute_coverage_metrics(
                explainer, X_test, y_test, alpha, max_instances=100, threshold=threshold
            )
            
            logger.info(f"Computing quality metrics for trial {trial.number}...")
            quality_metrics = MetricsCollector.compute_explanation_quality_metrics(
                explainer, X_test, y_test, max_instances=50, threshold=threshold
            )

            # Combine metrics
            all_metrics = {**coverage_metrics, **quality_metrics}

            # Log metrics to MLflow
            if self.mlflow_available:
                mlflow.log_metrics(all_metrics)

            # Store result
            result = {
                "trial_id": trial.number,
                "alpha": alpha,
                "distance": distance,
                "n_clusters": n_clusters,
                "task_type": task_type,
                **all_metrics,
            }
            self.results.append(result)

            # Return primary objective (coverage or validity)
            score = all_metrics.get("coverage", 0.0)
            trial.report(score, step=0)

            return score

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}", exc_info=True)
            if self.mlflow_available:
                mlflow.log_param("error", str(e))
            return 0.0

        finally:
            if self.mlflow_available:
                mlflow.end_run()

    def _summarize_results(self, study: optuna.Study, task_type: str) -> Dict[str, Any]:
        """Generate summary of ablation results."""
        best_trial = study.best_trial

        summary = {
            "study_name": study.study_name,
            "task_type": task_type,
            "n_trials": len(study.trials),
            "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial_number": best_trial.number,
            "best_trial_value": best_trial.value,
            "best_params": best_trial.params,
            "results_count": len(self.results),
            "timestamp": datetime.now().isoformat(),
        }

        # Aggregate metrics by parameter configuration
        config_results = self._aggregate_by_config()
        summary["aggregated_by_config"] = config_results

        return summary

    def _aggregate_by_config(self) -> Dict[str, Any]:
        """Aggregate results by parameter configuration."""
        from collections import defaultdict

        by_config = defaultdict(list)

        for result in self.results:
            key = (result["alpha"], result["distance"], result["n_clusters"])
            by_config[key].append(result)

        aggregated = {}
        for (alpha, distance, n_clusters), results_for_config in by_config.items():
            key = f"alpha_{alpha}_dist_{distance}_clusters_{n_clusters}"
            aggregated[key] = MetricsCollector.aggregate_metrics(results_for_config)

        return aggregated

    def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save results to disk."""
        results_file = self.storage_dir / "ablation_summary.json"
        details_file = self.storage_dir / "ablation_details.json"

        # Save summary
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {results_file}")

        # Save detailed results
        with open(details_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Details saved to {details_file}")

    def load_results(self) -> Dict[str, Any]:
        """Load previously saved results from disk."""
        results_file = self.storage_dir / "ablation_summary.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(results_file, "r") as f:
            return json.load(f)

    def get_best_config(self, metric: str = "coverage") -> Dict[str, Any]:
        """
        Get the best parameter configuration for a given metric.

        Parameters
        ----------
        metric : str
            Metric name to optimize for. Default "coverage".

        Returns
        -------
        dict
            Best parameters and corresponding metric value.
        """
        if not self.results:
            return {}

        # Filter results by metric availability
        valid_results = [r for r in self.results if metric in r]
        if not valid_results:
            logger.warning(f"No results found for metric '{metric}'")
            return {}

        # Find best
        best = max(valid_results, key=lambda r: r[metric])

        return {
            "alpha": best["alpha"],
            "distance": best["distance"],
            "n_clusters": best["n_clusters"],
            f"{metric}": best[metric],
        }
