"""Tests for :mod:`calibrated_explanations.explanations.explanations`."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import List, Sequence
import sys

import numpy as np
import pytest

from calibrated_explanations.explanations import (
    AlternativeExplanations,
    CalibratedExplanations,
)
from calibrated_explanations.explanations.explanations import (
    FrozenCalibratedExplainer,
    MultiClassCalibratedExplanations,
)
from calibrated_explanations.utils.exceptions import ConfigurationError
from tests.helpers.dataset_utils import make_binary_dataset
from tests.helpers.explainer_utils import initiate_explainer
from tests.helpers.model_utils import get_classification_model


def install_fake_matplotlib_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal matplotlib.colors stub for non-viz test environments."""
    fake_matplotlib = ModuleType("matplotlib")
    fake_colors = ModuleType("matplotlib.colors")
    fake_colors.BASE_COLORS = {"b": (0, 0, 1), "g": (0, 0.5, 0), "r": (1, 0, 0)}
    fake_matplotlib.colors = fake_colors

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", fake_colors)


@dataclass
class DummyOriginalExplainer:
    """Lightweight stand-in for :class:`CalibratedExplainer`."""

    feature_names: Sequence[str]
    class_labels: dict[int, str]
    mode: str = "classification"

    def __post_init__(self) -> None:
        self.x_cal = np.array([[0.1, 0.2]])
        self.y_cal = np.array([0.25, 0.75])
        self.num_features = len(self.feature_names)
        self.categorical_features: List[int] = []
        self.categorical_labels: dict[int, Sequence[str]] = {}
        self.feature_values: List[Sequence[str]] = [() for _ in self.feature_names]
        self.sample_percentiles = (5.0, 95.0)
        self.is_multiclass = True
        self.discretizer = object()
        self.rule_boundaries: List[float] = []
        self.learner = "dummy-learner"
        self.difficulty_estimator = "dummy-difficulty"
        from calibrated_explanations.plugins.manager import PluginManager

        self.plugin_manager = PluginManager(self)

        class PredOrch:
            def predict_internal(self, _x, **_kw):
                return (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))

        self.prediction_orchestrator = PredOrch()

    def predict(self, data):  # pragma: no cover - not used directly
        return np.asarray(data)

    def discretize(self, x):
        return x

    def preload_lime(self):
        """Return a minimal structure compatible with :meth:`CalibratedExplanations.as_lime`."""

        lime_exp = SimpleNamespace(
            intercept={1: 0.0},
            local_pred=None,
            predict_proba=[0.0, 1.0],
            local_exp={1: [None] * self.num_features},
            domain_mapper=SimpleNamespace(
                discretized_feature_names=None,
                feature_values=None,
            ),
        )
        return None, lime_exp

    def preload_shap(self):
        """Return a minimal structure compatible with :meth:`CalibratedExplanations.as_shap`."""

        shap_exp = SimpleNamespace(
            base_values=np.array([0.0]),
            values=np.zeros((1, self.num_features)),
            data=None,
        )
        return None, shap_exp


class DummyExplanation:
    """Minimal explanation object exercising list-handling logic."""

    def __init__(
        self,
        index: int,
        predict: float,
        *,
        probabilities=None,
        interval=(None, None),
    ) -> None:
        self.index = index
        self.predict = predict
        self.prediction_interval = interval
        self.prediction = {
            "predict": float(predict),
            "low": float(predict) - 0.1,
            "high": float(predict) + 0.1,
        }
        weight = np.array([predict, predict / 2])
        self.feature_weights = {"predict": weight}
        self.feature_predict = {"predict": weight}
        self.prediction_probabilities = probabilities
        self.x_test = np.array([predict, predict + 1])
        self.plot_calls: list[tuple] = []
        self.conjunction_calls: list[str] = []

    # -- Helpers used by :mod:`CalibratedExplanations` --
    def remove_conjunctions(self) -> None:
        self.conjunction_calls.append("remove")

    def add_conjunctions(self, *_args, **_kwargs) -> None:
        self.conjunction_calls.append("add")

    def reset(self) -> None:
        self.conjunction_calls.append("reset")

    def plot(self, **kwargs) -> None:
        self.plot_calls.append(tuple(sorted(kwargs.items())))

    def rank_features(self, _weights, num_to_show=None):  # pylint: disable=unused-argument
        size = len(self.feature_weights["predict"])
        return list(range(size if num_to_show is None else num_to_show))

    def define_conditions(self):
        return [f"feature_{i}" for i in range(len(self.feature_weights["predict"]))]

    def get_rules(self):
        return {"rule": [0, 1]}

    def super_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("super")
        return self

    def semi_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("semi")
        return self

    def counter_explanations(
        self, only_ensured=False, include_potential=True, copy=True
    ) -> DummyExplanation:
        self.conjunction_calls.append("counter")
        return self

    def ensured_explanations(self, include_potential=True, copy=True) -> DummyExplanation:
        self.conjunction_calls.append("ensured")
        return self

    def filter_rule_sizes(self, *, rule_sizes=None, size_range=None, copy=True) -> DummyExplanation:
        self.conjunction_calls.append(("filter_rule_sizes", rule_sizes, size_range, copy))
        return self


@pytest.fixture()
def collection() -> CalibratedExplanations:
    explainer = DummyOriginalExplainer(feature_names=("f0", "f1"), class_labels={0: "no", 1: "yes"})
    x = np.arange(6, dtype=float).reshape(3, 2)
    thresholds = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
    bins = ["bin0", "bin1", "bin2"]
    coll = CalibratedExplanations(explainer, x, thresholds, bins)
    coll.explanations = [
        DummyExplanation(0, 0.1, probabilities=np.array([0.9, 0.1]), interval=(0.0, 0.2)),
        DummyExplanation(1, 0.2, probabilities=np.array([0.8, 0.2]), interval=(0.1, 0.3)),
        DummyExplanation(2, 0.3, probabilities=np.array([0.7, 0.3]), interval=(0.2, 0.4)),
    ]
    return coll


def test_collection_filter_rule_sizes_delegates(collection: CalibratedExplanations) -> None:
    filtered = collection.filter_rule_sizes(rule_sizes=1, copy=True)
    for exp in filtered.explanations:
        assert ("filter_rule_sizes", 1, None, True) in exp.conjunction_calls


def test_one_sided_confidence_logic(collection: CalibratedExplanations) -> None:
    collection.low_high_percentiles = (5.0, np.inf)
    assert collection.is_one_sided
    assert collection.get_confidence() == 95.0
    collection.low_high_percentiles = (np.inf, 90.0)
    assert collection.get_confidence() == 90.0
    collection.low_high_percentiles = (10.0, 90.0)
    assert collection.get_confidence() == 80.0


def test_indexing_validates_bounds_and_type(collection: CalibratedExplanations) -> None:
    from calibrated_explanations.utils.exceptions import ValidationError

    assert collection[1] is collection.explanations[1]
    with pytest.raises(ValidationError):
        _ = collection["1"]  # type: ignore[index]
    with pytest.raises((ValidationError, IndexError)):
        _ = collection[len(collection.x_test)]


def test_alternative_explanation_proxies(collection: CalibratedExplanations) -> None:
    alt = AlternativeExplanations(
        collection.calibrated_explainer.explainer,
        collection.x_test,
        collection.y_threshold,
        collection.bins,
    )
    alt.explanations = collection.explanations
    alt.super_explanations()
    alt.semi_explanations()
    alt.counter_explanations()
    alt.ensured_explanations()
    calls = [exp.conjunction_calls for exp in collection.explanations]
    assert all({"super", "semi", "counter", "ensured"}.issubset(set(call)) for call in calls)


def test_multiclass_getitem_string_label_nonzero_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    class FakeFactual:
        def get_class_labels(self):
            return {2: "two", 5: "five"}

    monkeypatch.setattr(explanations_module, "FactualExplanation", FakeFactual)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0]])
    exp_two = FakeFactual()
    exp_five = FakeFactual()
    coll = MultiClassCalibratedExplanations(
        explainer, x, bins=None, num_classes=2, explanations=[{2: exp_two, 5: exp_five}]
    )

    assert coll[(0, "five")] is coll.explanations[0][5]
    with pytest.raises(KeyError, match="Unknown class label"):
        _ = coll[(0, "unknown")]


def test_multiclass_getitem_int_returns_single_instance_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    class FakeFactual:
        def get_class_labels(self):
            return {2: "two", 5: "five"}

    monkeypatch.setattr(explanations_module, "FactualExplanation", FakeFactual)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0]])
    exp_two = FakeFactual()
    exp_five = FakeFactual()
    coll = MultiClassCalibratedExplanations(
        explainer,
        x,
        bins=None,
        num_classes=2,
        explanations=[{2: exp_two, 5: exp_five}],
    )

    sliced = coll[0]

    assert isinstance(sliced, MultiClassCalibratedExplanations)
    assert len(sliced) == 1
    assert len(sliced.explanations) == 1
    assert set(sliced.explanations[0].keys()) == {2, 5}


def test_multiclass_interface_parity_slice_list_and_get_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    class FakeFactual:
        def get_class_labels(self):
            return {2: "two", 5: "five"}

    monkeypatch.setattr(explanations_module, "FactualExplanation", FakeFactual)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0], [2.0], [3.0]])
    coll = MultiClassCalibratedExplanations(
        explainer,
        x,
        bins=None,
        num_classes=2,
        explanations=[
            {2: FakeFactual(), 5: FakeFactual()},
            {2: FakeFactual(), 5: FakeFactual()},
            {2: FakeFactual(), 5: FakeFactual()},
        ],
    )

    sliced = coll[:2]
    subset = coll[[0, 2]]
    masked = coll[np.array([True, False, True])]

    assert isinstance(sliced, MultiClassCalibratedExplanations)
    assert isinstance(subset, MultiClassCalibratedExplanations)
    assert isinstance(masked, MultiClassCalibratedExplanations)
    assert len(sliced) == 2
    assert len(subset) == 2
    assert len(masked) == 2
    assert coll.X_test.shape == coll.x_test.shape
    assert coll[(0, 2)] is coll.explanations[0][2]


def test_should_raise_configuration_error_when_multiclass_to_narrative_uses_format_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    class FakeFactual:
        def get_class_labels(self):
            return {2: "two", 5: "five"}

    monkeypatch.setattr(explanations_module, "FactualExplanation", FakeFactual)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0]])
    coll = MultiClassCalibratedExplanations(
        explainer,
        x,
        bins=None,
        num_classes=2,
        explanations=[{2: FakeFactual(), 5: FakeFactual()}],
    )

    with pytest.raises(ConfigurationError, match="Unsupported narrative keyword arguments"):
        coll.to_narrative(format="short")


def test_multiclass_plot_factual_dispatches_dict_path_for_nonzero_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    install_fake_matplotlib_colors(monkeypatch)

    class FakeFactual:
        def __init__(self, klass: int):
            self.prediction = {"predict": 0.6, "low": 0.5, "high": 0.7, "classes": klass}

        def get_class_labels(self):
            return {2: "two", 5: "five"}

        def _check_preconditions(self):
            return None

        def rank_features(self, feature_weights=None, width=None, num_to_show=None):
            return [0]

        def get_rules(self):
            return {
                "base_predict": [0.6],
                "base_predict_low": [0.5],
                "base_predict_high": [0.7],
                "predict": [0.62],
                "predict_low": [0.52],
                "predict_high": [0.72],
                "weight": [0.1],
                "weight_low": [0.05],
                "weight_high": [0.15],
                "value": ["v"],
                "rule": ["r"],
                "feature": ["f"],
                "feature_value": ["fv"],
                "is_conjunctive": [False],
                "classes": [2],
            }

    monkeypatch.setattr(explanations_module, "FactualExplanation", FakeFactual)

    calls = {"count": 0}

    def _record(*_args, **_kwargs):
        calls["count"] += 1

    monkeypatch.setattr(explanations_module, "_plot_probabilistic_dict", _record)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0]])
    coll = MultiClassCalibratedExplanations(
        explainer,
        x,
        bins=None,
        num_classes=2,
        explanations=[{2: FakeFactual(2), 5: FakeFactual(5)}],
    )

    coll.plot_factual(show=False)

    assert calls["count"] == 1


def test_multiclass_plot_alternative_dispatches_dict_path_for_nonzero_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrated_explanations.explanations import explanations as explanations_module

    install_fake_matplotlib_colors(monkeypatch)

    class FakeAlternative:
        def __init__(self, klass: int):
            self.prediction = {"predict": 0.6, "low": 0.5, "high": 0.7, "classes": klass}

        def _check_preconditions(self):
            return None

        def _rank_features(self, feature_weights=None, width=None, num_to_show=None):
            return [0]

        def _get_rules(self):
            return {
                "base_predict": [0.6],
                "base_predict_low": [0.5],
                "base_predict_high": [0.7],
                "predict": [0.62],
                "predict_low": [0.52],
                "predict_high": [0.72],
                "weight": [0.1],
                "weight_low": [0.05],
                "weight_high": [0.15],
                "value": ["v"],
                "rule": ["r"],
                "feature": ["f"],
                "feature_value": ["fv"],
                "is_conjunctive": [False],
                "classes": [2],
            }

    monkeypatch.setattr(explanations_module, "AlternativeExplanation", FakeAlternative)

    calls = {"count": 0}

    def _record(*_args, **_kwargs):
        calls["count"] += 1

    monkeypatch.setattr(explanations_module, "_plot_alternative_dict", _record)

    explainer = DummyOriginalExplainer(feature_names=("f0",), class_labels={2: "two", 5: "five"})
    x = np.array([[1.0]])
    coll = MultiClassCalibratedExplanations(
        explainer,
        x,
        bins=None,
        num_classes=2,
        explanations=[{2: FakeAlternative(2), 5: FakeAlternative(5)}],
    )
    monkeypatch.setattr(coll, "sort_factuals_by_rule", lambda factuals: factuals)

    coll.plot_alternative(show=False)

    assert calls["count"] == 1


def _build_fitted_explainer():
    """Build a real, fitted CalibratedExplainer for frozen-snapshot isolation tests."""
    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        _x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    return initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )


class TestFrozenCalibratedExplainerIsolation:
    """Regression tests for pre-v4 finding S4-H5: frozen snapshots must not share
    live mutable state with the explainer they were created from."""

    def test_reset_on_frozen_copy_does_not_clear_live_plugin_identifiers(self):
        live = _build_fitted_explainer()
        live.predict(live.x_cal[:1])
        live.plugin_manager.explanation_plugin_identifiers["factual"] = "sentinel-plugin"

        frozen = FrozenCalibratedExplainer(live)
        frozen.explainer.reset()

        assert (
            live.plugin_manager.explanation_plugin_identifiers.get("factual") == "sentinel-plugin"
        )

    def test_mutating_frozen_plugin_manager_does_not_affect_live(self):
        live = _build_fitted_explainer()
        frozen = FrozenCalibratedExplainer(live)

        assert frozen.explainer.plugin_manager is not live.plugin_manager
        frozen.explainer.plugin_manager.explanation_plugin_overrides["factual"] = "mutated"

        assert live.plugin_manager.explanation_plugin_overrides.get("factual") != "mutated"

    def test_mutating_frozen_interval_learner_does_not_affect_live(self):
        live = _build_fitted_explainer()
        frozen = FrozenCalibratedExplainer(live)

        assert frozen.explainer.interval_learner is not live.interval_learner

    def test_mutating_frozen_rng_does_not_affect_live(self):
        live = _build_fitted_explainer()
        frozen = FrozenCalibratedExplainer(live)

        assert frozen.explainer.rng is not live.rng
        live_state = live.rng.bit_generator.state
        frozen.explainer.rng.standard_normal()

        assert live.rng.bit_generator.state == live_state

    def test_mutating_frozen_learner_does_not_affect_live(self):
        live = _build_fitted_explainer()
        frozen = FrozenCalibratedExplainer(live)

        assert frozen.explainer.learner is not live.learner


class TestCalibratedExplainerDeepcopyVisibility:
    """Regression tests for pre-v4 finding S4-H5: deepcopy failures must be
    visible, not silently suppressed to a shared original instance."""

    def test_deepcopy_falls_back_visibly_for_unpicklable_attribute(self, enable_fallbacks):
        import threading

        live = _build_fitted_explainer()
        # Locks cannot be deep-copied; attach one under an attribute name that
        # is not part of the deliberately-shared runtime-helper set.
        live.unpicklable_marker = threading.Lock()

        with pytest.warns(UserWarning, match="deep-?copy"):
            frozen = FrozenCalibratedExplainer(live)

        # The visibly-flagged attribute is shared (fallback), but the rest of
        # the isolation guarantees still hold.
        assert frozen.explainer.plugin_manager is not live.plugin_manager
