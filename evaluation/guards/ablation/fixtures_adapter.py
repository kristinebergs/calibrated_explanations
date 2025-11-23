"""
Adapter to reuse pytest fixtures with Optuna trials.

This module provides a bridge between the pytest fixture infrastructure
(in conftest.py) and Optuna's trial-based execution model. It manages
fixture lifecycle and state across trials.
"""

from __future__ import annotations

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import numpy as np

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle


class PerturbationGuardConfig:
    """Configuration for guard parameters."""

    def __init__(
        self,
        alpha: float,
        distance: str,
        n_clusters: int,
        random_state: int | None = None,
    ):
        self.alpha = alpha
        self.distance = distance
        self.n_clusters = n_clusters
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f"PerturbationGuardConfig(alpha={self.alpha}, distance={self.distance}, "
            f"n_clusters={self.n_clusters})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "alpha": self.alpha,
            "distance": self.distance,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
        }


class FixturesAdapter:
    """
    Manages fixture lifecycle for Optuna trials.

    This adapter lazily creates and caches session-scoped fixtures
    (datasets, models, explainers) to avoid redundant computation.

    Attributes
    ----------
    random_seed : int
        Global random seed for reproducibility
    _data_cache : dict
        Cached datasets (binary_classification, multiclass_classification, regression)
    _models_cache : dict
        Cached trained models (binary_classifier, multiclass_classifier, regression_model)
    _explainers_cache : dict
        Cached baseline explainers (binary_explainer_baseline, etc.)
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the fixtures adapter."""
        self.random_seed = random_seed
        self._data_cache: Dict[str, Dict[str, Any]] = {}
        self._models_cache: Dict[str, Any] = {}
        self._explainers_cache: Dict[str, WrapCalibratedExplainer] = {}

    # =========================================================================
    # Data Fixtures
    # =========================================================================

    def get_binary_classification_data(self) -> Dict[str, Any]:
        """Get or create binary classification data."""
        if "binary_classification" not in self._data_cache:
            self._data_cache["binary_classification"] = self._create_binary_classification_data()
        return self._data_cache["binary_classification"]

    def get_multiclass_classification_data(self) -> Dict[str, Any]:
        """Get or create multiclass classification data."""
        if "multiclass_classification" not in self._data_cache:
            self._data_cache["multiclass_classification"] = self._create_multiclass_classification_data()
        return self._data_cache["multiclass_classification"]

    def get_regression_data(self) -> Dict[str, Any]:
        """Get or create regression data."""
        if "regression" not in self._data_cache:
            self._data_cache["regression"] = self._create_regression_data()
        return self._data_cache["regression"]

    def _create_binary_classification_data(self) -> Dict[str, Any]:
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=10_000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.05,
            random_state=self.random_seed,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_seed
        )

        X_proper, X_cal, y_proper, y_cal = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=self.random_seed
        )

        feature_names = [f"f{i}" for i in range(X.shape[1])]

        return {
            "X_proper": X_proper,
            "y_proper": y_proper,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    def _create_multiclass_classification_data(self) -> Dict[str, Any]:
        """Create multiclass classification dataset."""
        X, y = make_classification(
            n_samples=10_000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_classes=4,
            n_clusters_per_class=1,
            class_sep=1.5,
            flip_y=0.05,
            random_state=self.random_seed,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_seed
        )

        X_proper, X_cal, y_proper, y_cal = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=self.random_seed
        )

        feature_names = [f"f{i}" for i in range(X.shape[1])]

        return {
            "X_proper": X_proper,
            "y_proper": y_proper,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    def _create_regression_data(self) -> Dict[str, Any]:
        """Create regression dataset."""
        X, y = make_regression(
            n_samples=10_000,
            n_features=20,
            n_informative=10,
            noise=10.0,
            random_state=self.random_seed,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        X_proper, X_cal, y_proper, y_cal = train_test_split(
            X_train, y_train, test_size=0.25, random_state=self.random_seed
        )

        thresholds = np.quantile(y_train, [0.25, 0.5, 0.75])

        feature_names = [f"f{i}" for i in range(X.shape[1])]

        return {
            "X_proper": X_proper,
            "y_proper": y_proper,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_test,
            "y_test": y_test,
            "thresholds": thresholds,
            "feature_names": feature_names,
        }

    # =========================================================================
    # Model Fixtures
    # =========================================================================

    def get_binary_classifier(self) -> RandomForestClassifier:
        """Get or create binary classifier."""
        if "binary_classifier" not in self._models_cache:
            data = self.get_binary_classification_data()
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_seed,
                n_jobs=1,
            )
            model.fit(data["X_proper"], data["y_proper"])
            self._models_cache["binary_classifier"] = model
        return self._models_cache["binary_classifier"]

    def get_multiclass_classifier(self) -> RandomForestClassifier:
        """Get or create multiclass classifier."""
        if "multiclass_classifier" not in self._models_cache:
            data = self.get_multiclass_classification_data()
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_seed,
                n_jobs=1,
            )
            model.fit(data["X_proper"], data["y_proper"])
            self._models_cache["multiclass_classifier"] = model
        return self._models_cache["multiclass_classifier"]

    def get_regression_model(self) -> RandomForestRegressor:
        """Get or create regression model."""
        if "regression_model" not in self._models_cache:
            data = self.get_regression_data()
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_seed,
                n_jobs=1,
            )
            model.fit(data["X_proper"], data["y_proper"])
            self._models_cache["regression_model"] = model
        return self._models_cache["regression_model"]

    # =========================================================================
    # Explainer Factories
    # =========================================================================

    def create_guarded_explainer(
        self,
        task_type: str,
        guard_config: PerturbationGuardConfig,
    ) -> WrapCalibratedExplainer:
        """
        Create a guarded explainer for the specified task type.

        Parameters
        ----------
        task_type : str
            One of: "binary_classification", "multiclass_classification", "regression"
        guard_config : PerturbationGuardConfig
            Guard parameters (alpha, distance, n_clusters)

        Returns
        -------
        WrapCalibratedExplainer
            A new explainer instance with guard initialized.
        """
        if task_type == "binary_classification":
            return self._create_guarded_explainer_binary(guard_config)
        elif task_type == "multiclass_classification":
            return self._create_guarded_explainer_multiclass(guard_config)
        elif task_type == "regression":
            return self._create_guarded_explainer_regression(guard_config)
        elif task_type == "probabilistic_regression":
            return self._create_guarded_explainer_regression(guard_config)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def _create_guarded_explainer_binary(
        self, guard_config: PerturbationGuardConfig
    ) -> WrapCalibratedExplainer:
        """Create guarded explainer for binary classification."""
        model = self.get_binary_classifier()
        data = self.get_binary_classification_data()

        explainer = WrapCalibratedExplainer(model)
        explainer.fit(data["X_proper"], data["y_proper"])
        explainer.calibrate(
            data["X_cal"],
            data["y_cal"],
            feature_names=data["feature_names"],
            guard_params={
                "alpha": guard_config.alpha,
                "n_clusters": guard_config.n_clusters,
                "nonconformity_metric": guard_config.distance,
                "random_state": guard_config.random_state,
            },
        )
        return explainer

    def _create_guarded_explainer_multiclass(
        self, guard_config: PerturbationGuardConfig
    ) -> WrapCalibratedExplainer:
        """Create guarded explainer for multiclass classification."""
        model = self.get_multiclass_classifier()
        data = self.get_multiclass_classification_data()

        explainer = WrapCalibratedExplainer(model)
        explainer.fit(data["X_proper"], data["y_proper"])
        explainer.calibrate(
            data["X_cal"],
            data["y_cal"],
            feature_names=data["feature_names"],
            guard_params={
                "alpha": guard_config.alpha,
                "n_clusters": guard_config.n_clusters,
                "nonconformity_metric": guard_config.distance,
                "random_state": guard_config.random_state,
            },
        )
        return explainer

    def _create_guarded_explainer_regression(
        self, guard_config: PerturbationGuardConfig
    ) -> WrapCalibratedExplainer:
        """Create guarded explainer for regression."""
        model = self.get_regression_model()
        data = self.get_regression_data()

        explainer = WrapCalibratedExplainer(model)
        explainer.fit(data["X_proper"], data["y_proper"])
        explainer.calibrate(
            data["X_cal"],
            data["y_cal"],
            feature_names=data["feature_names"],
            guard_params={
                "alpha": guard_config.alpha,
                "n_clusters": guard_config.n_clusters,
                "nonconformity_metric": guard_config.distance,
                "random_state": guard_config.random_state,
            },
        )
        return explainer

    def get_test_data(self, task_type: str) -> tuple:
        """Get test data and labels for a task type."""
        if task_type == "binary_classification":
            data = self.get_binary_classification_data()
        elif task_type == "multiclass_classification":
            data = self.get_multiclass_classification_data()
        elif task_type == "regression":
            data = self.get_regression_data()
        elif task_type == "probabilistic_regression":
            data = self.get_regression_data()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return data["X_test"], data["y_test"]

    def get_threshold(self, task_type: str) -> float | None:
        """Get threshold for probabilistic tasks."""
        if task_type == "probabilistic_regression":
            data = self.get_regression_data()
            # Use median threshold
            return data["thresholds"][1]
        return None
