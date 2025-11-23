"""
Analysis utilities for ablation study results.

Provides functions for comparing configurations, visualizing results,
and generating publication-ready reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import numpy as np


class ResultsAnalyzer:
    """
    Analyze and compare ablation study results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ablation results.
    """

    def __init__(self, results_dir: str | Path = "./ablation_results"):
        self.results_dir = Path(results_dir)
        self.summary = None
        self.details = None
        self._load_results()

    def _load_results(self) -> None:
        """Load results from disk."""
        summary_file = self.results_dir / "ablation_summary.json"
        details_file = self.results_dir / "ablation_details.json"

        if summary_file.exists():
            with open(summary_file, "r") as f:
                self.summary = json.load(f)

        if details_file.exists():
            with open(details_file, "r") as f:
                self.details = json.load(f)

    def get_parameter_sensitivity(self, metric: str = "coverage") -> Dict[str, Dict[str, float]]:
        """
        Analyze sensitivity to each parameter.

        For each parameter (alpha, distance, n_clusters), compute
        mean metric value across all other configurations.

        Parameters
        ----------
        metric : str
            Metric name to analyze. Default "coverage".

        Returns
        -------
        dict
            Sensitivity analysis results.
        """
        if not self.details:
            return {}

        sensitivity = {"alpha": {}, "distance": {}, "n_clusters": {}}

        # Group by each parameter
        by_alpha = defaultdict(list)
        by_distance = defaultdict(list)
        by_clusters = defaultdict(list)

        for result in self.details:
            if metric in result:
                value = result[metric]
                by_alpha[result["alpha"]].append(value)
                by_distance[result["distance"]].append(value)
                by_clusters[result["n_clusters"]].append(value)

        # Compute mean for each parameter value
        for param_val, values in by_alpha.items():
            sensitivity["alpha"][param_val] = float(np.mean(values))

        for param_val, values in by_distance.items():
            sensitivity["distance"][param_val] = float(np.mean(values))

        for param_val, values in by_clusters.items():
            sensitivity["n_clusters"][param_val] = float(np.mean(values))

        return sensitivity

    def get_best_configurations(
        self,
        metric: str = "coverage",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k configurations by metric.

        Parameters
        ----------
        metric : str
            Metric to rank by. Default "coverage".
        top_k : int
            Number of top configurations to return. Default 5.

        Returns
        -------
        list[dict]
            Top-k configurations with their metrics.
        """
        if not self.details:
            return []

        # Filter and sort
        valid_results = [r for r in self.details if metric in r]
        sorted_results = sorted(valid_results, key=lambda r: r[metric], reverse=True)

        return sorted_results[:top_k]

    def get_worst_configurations(
        self,
        metric: str = "coverage",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get bottom-k configurations by metric."""
        if not self.details:
            return []

        valid_results = [r for r in self.details if metric in r]
        sorted_results = sorted(valid_results, key=lambda r: r[metric])

        return sorted_results[:top_k]

    def compare_distance_metrics(self, metric: str = "coverage") -> Dict[str, Dict[str, float]]:
        """
        Compare performance across distance metrics.

        Returns mean and std of metric for each distance metric.

        Parameters
        ----------
        metric : str
            Metric to compare. Default "coverage".

        Returns
        -------
        dict
            Comparison results (mean, std, count for each distance metric).
        """
        if not self.details:
            return {}

        by_distance = defaultdict(list)

        for result in self.details:
            if metric in result:
                by_distance[result["distance"]].append(result[metric])

        comparison = {}
        for distance, values in by_distance.items():
            comparison[distance] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        return comparison

    def compare_alpha_values(self, metric: str = "coverage") -> Dict[float, Dict[str, float]]:
        """
        Compare performance across alpha values.

        Returns mean and std of metric for each alpha.

        Parameters
        ----------
        metric : str
            Metric to compare. Default "coverage".

        Returns
        -------
        dict
            Comparison results (mean, std for each alpha).
        """
        if not self.details:
            return {}

        by_alpha = defaultdict(list)

        for result in self.details:
            if metric in result:
                by_alpha[result["alpha"]].append(result[metric])

        comparison = {}
        for alpha, values in by_alpha.items():
            comparison[alpha] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        return comparison

    def generate_report(self, output_file: Optional[str | Path] = None) -> str:
        """
        Generate a comprehensive text report of ablation results.

        Parameters
        ----------
        output_file : str or Path, optional
            File to save report. If None, only returns string.

        Returns
        -------
        str
            The report text.
        """
        lines = []

        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 80)
        lines.append("")

        if self.summary:
            lines.append("SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Study: {self.summary.get('study_name', 'N/A')}")
            lines.append(f"Task Type: {self.summary.get('task_type', 'N/A')}")
            lines.append(f"Total Trials: {self.summary.get('n_trials', 'N/A')}")
            lines.append(f"Complete Trials: {self.summary.get('n_complete_trials', 'N/A')}")
            lines.append(f"Timestamp: {self.summary.get('timestamp', 'N/A')}")
            lines.append("")

            lines.append("BEST CONFIGURATION")
            lines.append("-" * 40)
            lines.append(f"Trial: {self.summary.get('best_trial_number', 'N/A')}")
            lines.append(f"Value: {self.summary.get('best_trial_value', 'N/A'):.4f}")
            for param, value in self.summary.get("best_params", {}).items():
                lines.append(f"  {param}: {value}")
            lines.append("")

        # Parameter sensitivity
        lines.append("PARAMETER SENSITIVITY")
        lines.append("-" * 40)
        sensitivity = self.get_parameter_sensitivity()
        if sensitivity:
            for param, values in sensitivity.items():
                lines.append(f"\n{param}:")
                for value, metric in sorted(values.items()):
                    lines.append(f"  {value}: {metric:.4f}")
        lines.append("")

        # Distance metric comparison
        lines.append("DISTANCE METRIC COMPARISON")
        lines.append("-" * 40)
        dist_compare = self.compare_distance_metrics()
        for distance, stats in dist_compare.items():
            lines.append(f"{distance}:")
            for stat_name, stat_value in stats.items():
                lines.append(f"  {stat_name}: {stat_value:.4f}" if isinstance(stat_value, float) else f"  {stat_name}: {stat_value}")
        lines.append("")

        # Alpha comparison
        lines.append("ALPHA SENSITIVITY")
        lines.append("-" * 40)
        alpha_compare = self.compare_alpha_values()
        for alpha, stats in sorted(alpha_compare.items()):
            lines.append(f"alpha={alpha}:")
            for stat_name, stat_value in stats.items():
                lines.append(f"  {stat_name}: {stat_value:.4f}" if isinstance(stat_value, float) else f"  {stat_name}: {stat_value}")
        lines.append("")

        # Top configurations
        lines.append("TOP 5 CONFIGURATIONS")
        lines.append("-" * 40)
        top_configs = self.get_best_configurations(top_k=5)
        for i, config in enumerate(top_configs, 1):
            lines.append(
                f"{i}. alpha={config['alpha']}, distance={config['distance']}, "
                f"n_clusters={config['n_clusters']}: coverage={config.get('coverage', 'N/A'):.4f}"
            )
        lines.append("")

        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)

        return report_text

    def export_as_csv(self, output_file: str | Path) -> None:
        """
        Export detailed results as CSV for further analysis.

        Parameters
        ----------
        output_file : str or Path
            Output CSV file path.
        """
        if not self.details:
            raise ValueError("No details available to export")

        import csv

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Get all unique keys
        all_keys = set()
        for result in self.details:
            all_keys.update(result.keys())

        fieldnames = sorted(all_keys)

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.details:
                # Pad missing keys with empty strings
                row = {k: result.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"Results exported to {output_file}")
