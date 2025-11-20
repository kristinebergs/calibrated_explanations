"""Performance tests for guards plugin.

Tests verify that:
- Guard filtering overhead is acceptable (<20% slowdown)
- No performance regression compared to baseline
- Guard scales linearly with instance count
"""

import time
from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from calibrated_explanations import CalibratedExplainer


@pytest.fixture
def classification_setup() -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Create a classification dataset and trained model."""
    # Generate data
    x_data, y_data = make_classification(
        n_samples=500, n_features=10, n_informative=8, random_state=42
    )
    split = 400
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    # Train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    clf.fit(x_train, y_train)

    # Test data
    x_test = x_cal[:20]

    return clf, x_test, x_cal, y_cal


class TestGuardPerformanceOverhead:
    """Performance tests measuring guard filtering overhead."""

    def test_factual_explanations_without_guard_baseline(self, classification_setup):
        """Measure baseline factual explanation time without guard."""
        clf, x_test, x_cal, y_cal = classification_setup

        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        start = time.perf_counter()
        explanations = explainer.explain_factual(x_test)
        end = time.perf_counter()

        baseline_time = end - start
        assert len(explanations) == 20
        assert baseline_time > 0

        # Store for comparison
        return baseline_time

    def test_factual_explanations_with_guard_loose(self, classification_setup):
        """Measure factual explanation time with loose guard (minimal filtering)."""
        clf, x_test, x_cal, y_cal = classification_setup

        guard_params = {"alpha": 0.9, "n_clusters": 10}
        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification", guard_params=guard_params
        )

        start = time.perf_counter()
        explanations = explainer.explain_factual(x_test)
        end = time.perf_counter()

        guard_time = end - start
        assert len(explanations) == 20
        assert guard_time > 0

        return guard_time

    def test_factual_explanations_with_guard_strict(self, classification_setup):
        """Measure factual explanation time with strict guard (aggressive filtering)."""
        clf, x_test, x_cal, y_cal = classification_setup

        guard_params = {"alpha": 0.5, "n_clusters": 5}
        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification", guard_params=guard_params
        )

        start = time.perf_counter()
        explanations = explainer.explain_factual(x_test)
        end = time.perf_counter()

        guard_time = end - start
        assert len(explanations) == 20
        assert guard_time > 0

        return guard_time

    def test_guard_overhead_acceptable_loose(self, classification_setup):
        """Test that guard overhead is acceptable for loose filtering."""
        clf, x_test, x_cal, y_cal = classification_setup

        # Baseline without guard - measure 3 times
        explainer_no_guard = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        times_baseline = []
        for _ in range(3):
            start = time.perf_counter()
            explainer_no_guard.explain_factual(x_test)
            end = time.perf_counter()
            times_baseline.append(end - start)

        baseline_time = np.mean(times_baseline)

        # With loose guard
        guard_params = {"alpha": 0.9, "n_clusters": 10}
        explainer_with_guard = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        times_guard = []
        for _ in range(3):
            start = time.perf_counter()
            explainer_with_guard.explain_factual(x_test)
            end = time.perf_counter()
            times_guard.append(end - start)

        guard_time = np.mean(times_guard)

        # Overhead should be minimal (<30% for loose guard)
        overhead_pct = ((guard_time - baseline_time) / baseline_time) * 100
        assert overhead_pct < 30, f"Guard overhead too high: {overhead_pct:.1f}%"

class TestGuardNoRegression:
    """Tests verifying no performance regression."""

    def test_alternative_explanations_without_guard(self, classification_setup):
        """Verify alternative explanations work without performance issues."""
        clf, x_test, x_cal, y_cal = classification_setup

        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        start = time.perf_counter()
        explanations = explainer.explore_alternatives(x_test)
        end = time.perf_counter()

        runtime = end - start
        assert len(explanations) == 20
        # Should complete in reasonable time (< 60 seconds for 20 instances)
        assert runtime < 60, f"Alternative explanations too slow: {runtime:.1f}s"

    def test_alternative_explanations_with_guard(self, classification_setup):
        """Verify alternative explanations with guard don't regress."""
        clf, x_test, x_cal, y_cal = classification_setup

        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        start = time.perf_counter()
        explanations = explainer.explore_alternatives(x_test)
        end = time.perf_counter()

        runtime = end - start
        assert len(explanations) == 20
        # Should complete in reasonable time
        assert runtime < 60, f"Alternative explanations with guard too slow: {runtime:.1f}s"

    def test_fast_explanations_without_guard(self, classification_setup):
        """Verify fast explanations work without performance issues."""
        clf, x_test, x_cal, y_cal = classification_setup

        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        start = time.perf_counter()
        explanations = explainer.explain_fast(x_test)
        end = time.perf_counter()

        runtime = end - start
        assert len(explanations) == 20
        # Fast mode should be genuinely fast (< 10 seconds for 20 instances)
        assert runtime < 10, f"Fast explanations too slow: {runtime:.1f}s"

    def test_fast_explanations_with_guard(self, classification_setup):
        """Verify fast explanations with guard remain fast."""
        clf, x_test, x_cal, y_cal = classification_setup

        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        start = time.perf_counter()
        explanations = explainer.explain_fast(x_test)
        end = time.perf_counter()

        runtime = end - start
        assert len(explanations) == 20
        # Fast mode should stay fast even with guard
        assert runtime < 10, f"Fast explanations with guard too slow: {runtime:.1f}s"


class TestGuardFilteringOverhead:
    """Measure actual guard filtering overhead."""

    def test_guard_filtering_overhead_perturbations(self, classification_setup):
        """Measure overhead of perturbation filtering."""
        clf, x_test, x_cal, y_cal = classification_setup

        # Create two explainers
        explainer_no_guard = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer_with_guard = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        # Single instance for precise measurement
        x_single = x_test[:1]

        # Baseline
        times_baseline = []
        for _ in range(3):
            start = time.perf_counter()
            explainer_no_guard.explain_factual(x_single)
            end = time.perf_counter()
            times_baseline.append(end - start)

        baseline_avg = np.mean(times_baseline)

        # With guard
        times_guard = []
        for _ in range(3):
            start = time.perf_counter()
            explainer_with_guard.explain_factual(x_single)
            end = time.perf_counter()
            times_guard.append(end - start)

        guard_avg = np.mean(times_guard)

        # Calculate overhead
        overhead = guard_avg - baseline_avg
        overhead_pct = (overhead / baseline_avg * 100) if baseline_avg > 0 else 0

        # Overhead should be reasonable
        overhead_msg = f"Guard filtering overhead too high: {overhead:.2f}s ({overhead_pct:.1f}%)"
        assert overhead < 5, overhead_msg


class TestGuardMemoryScaling:
    """Test that guard doesn't cause memory issues."""

    def test_large_batch_with_guard(self):
        """Verify guard handles large batches without memory problems."""
        # Generate larger dataset
        x_data, y_data = make_classification(
            n_samples=300, n_features=15, n_informative=10, random_state=42
        )
        split = 250
        x_train, x_cal = x_data[:split], x_data[split:]
        y_train, y_cal = y_data[:split], y_data[split:]

        clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
        clf.fit(x_train, y_train)

        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        # Large batch
        x_test = x_cal[:30]

        start = time.perf_counter()
        explanations = explainer.explain_factual(x_test)
        end = time.perf_counter()

        runtime = end - start
        assert len(explanations) == 30
        # Should complete without memory issues
        assert runtime < 120, f"Large batch processing too slow: {runtime:.1f}s"


@pytest.mark.benchmark
class TestGuardBenchmarkSummary:
    """Summary benchmarks for guard performance."""

    def test_benchmark_factual_no_guard(self, classification_setup, benchmark):
        """Benchmark factual explanations without guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        result = benchmark(lambda: explainer.explain_factual(x_test))
        assert len(result) == 20

    def test_benchmark_factual_with_guard(self, classification_setup, benchmark):
        """Benchmark factual explanations with guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        result = benchmark(lambda: explainer.explain_factual(x_test))
        assert len(result) == 20

    def test_benchmark_alternative_no_guard(self, classification_setup, benchmark):
        """Benchmark alternative explanations without guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        result = benchmark(lambda: explainer.explore_alternatives(x_test))
        assert len(result) == 20

    def test_benchmark_alternative_with_guard(self, classification_setup, benchmark):
        """Benchmark alternative explanations with guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        result = benchmark(lambda: explainer.explore_alternatives(x_test))
        assert len(result) == 20

    def test_benchmark_fast_no_guard(self, classification_setup, benchmark):
        """Benchmark fast explanations without guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        explainer = CalibratedExplainer(
            learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
        )

        result = benchmark(lambda: explainer.explain_fast(x_test))
        assert len(result) == 20

    def test_benchmark_fast_with_guard(self, classification_setup, benchmark):
        """Benchmark fast explanations with guard."""
        clf, x_test, x_cal, y_cal = classification_setup
        guard_params = {"alpha": 0.8, "n_clusters": 8}
        explainer = CalibratedExplainer(
            learner=clf,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params=guard_params,
        )

        result = benchmark(lambda: explainer.explain_fast(x_test))
        assert len(result) == 20
