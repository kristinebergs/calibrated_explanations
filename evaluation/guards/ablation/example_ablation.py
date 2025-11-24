#!/usr/bin/env python
"""
Example: Running a Guard Parameter Ablation Study

This script demonstrates how to use the Optuna + MLflow ablation framework
to systematically evaluate guard parameters across different configurations.

Usage:
    python example_ablation.py --task binary_classification --n-workers 1
    python example_ablation.py --task binary_classification --n-workers 8 --n-trials 36
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup path to import from parent package
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from evaluation.guards.ablation import AblationExecutor, ResultsAnalyzer


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run guard parameter ablation study with Optuna + MLflow"
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=["binary_classification"],
        help="Task type(s) to evaluate. One or more of: binary_classification, multiclass_classification, regression, probabilistic_regression",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=36,
        help="Number of trials (parameter combinations). Default 36 (4×3×3)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds per configuration. Default 1 (no averaging yet)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers. Default 1 (serial)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for entire study",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ablation_results",
        help="Directory for results. Default ./ablation_results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="guard_parameter_ablation",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip execution and only analyze existing results",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a text report after execution",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV",
    )

    args = parser.parse_args()
    setup_logging()

    logger = logging.getLogger(__name__)

    # =========================================================================
    # Execution
    # =========================================================================

    if not args.analyze_only:
        logger.info(f"Starting ablation study for task: {args.task}")
        logger.info(
            f"Configuration: n_trials={args.n_trials}, n_seeds={args.n_seeds}, "
            f"n_workers={args.n_workers}"
        )

        executor = AblationExecutor(
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            storage_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

        try:
            summaries = executor.run(
                task_type=args.task,
                n_workers=args.n_workers,
                timeout=args.timeout,
            )

            logger.info("Ablation study completed successfully")
            for task, summary in summaries.items():
                logger.info(f"\nTask: {task}")
                logger.info(f"Best trial: #{summary['best_trial_number']}")
                logger.info(f"Best params: {summary['best_params']}")
            logger.info(f"\nResults saved to: {args.output_dir}")

        except KeyboardInterrupt:
            logger.info("Ablation study interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during ablation: {e}", exc_info=True)
            sys.exit(1)

    # =========================================================================
    # Analysis
    # =========================================================================

    logger.info("Loading results for analysis...")
    analyzer = ResultsAnalyzer(args.output_dir)

    # Parameter sensitivity
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    sensitivity = analyzer.get_parameter_sensitivity(metric="coverage")
    for param, values in sensitivity.items():
        logger.info(f"\n{param}:")
        for val, metric in sorted(values.items()):
            logger.info(f"  {val}: {metric:.4f}")

    # Top configurations
    logger.info("\n" + "=" * 60)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 60)
    top_configs = analyzer.get_best_configurations(top_k=5)
    for i, config in enumerate(top_configs, 1):
        logger.info(
            f"{i}. alpha={config['alpha']}, distance={config['distance']}, "
            f"n_clusters={config['n_clusters']}: coverage={config.get('coverage', 'N/A'):.4f}"
        )

    # Distance metric comparison
    logger.info("\n" + "=" * 60)
    logger.info("DISTANCE METRIC COMPARISON")
    logger.info("=" * 60)
    dist_comp = analyzer.compare_distance_metrics()
    for distance, stats in dist_comp.items():
        logger.info(f"{distance}:")
        logger.info(
            f"  mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
            f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
        )

    # Alpha comparison
    logger.info("\n" + "=" * 60)
    logger.info("ALPHA SENSITIVITY")
    logger.info("=" * 60)
    alpha_comp = analyzer.compare_alpha_values()
    for alpha, stats in sorted(alpha_comp.items()):
        logger.info(f"alpha={alpha}:")
        logger.info(
            f"  mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
            f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
        )

    # Generate report if requested
    if args.generate_report:
        report_file = Path(args.output_dir) / "ablation_report.txt"
        logger.info(f"\nGenerating report: {report_file}")
        analyzer.generate_report(output_file=report_file)
        logger.info(f"Report saved to {report_file}")

    # Export CSV if requested
    if args.export_csv:
        csv_file = Path(args.output_dir) / "ablation_results.csv"
        logger.info(f"Exporting to CSV: {csv_file}")
        analyzer.export_as_csv(csv_file)
        logger.info(f"CSV exported to {csv_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Ablation analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
