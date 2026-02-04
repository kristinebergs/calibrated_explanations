import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.conformal_guard import ConformalGuard, ConformalGuardConfig
from calibrated_explanations.plugins.conformal_guard import ConformalGuardPlugin
from calibrated_explanations.plugins.explanations import ExplanationContext


def _build_context(explainer: CalibratedExplainer) -> ExplanationContext:
    return ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=explainer.feature_names,
        categorical_features=explainer.categorical_features,
        categorical_labels=explainer.categorical_labels or {},
        discretizer=explainer.discretizer,
        helper_handles={},
        predict_bridge=None,
        interval_settings={},
        plot_settings={},
    )


def test_tree_conformal_is_deterministic():
    x_cal = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y_cal = np.array([0, 0, 1, 1, 1])
    cfg = ConformalGuardConfig(precompute=True, candidate_grid=8, n_leaf_quantiles=8)
    guard = ConformalGuard(
        mode="factual",
        task="classification",
        x_cal=x_cal,
        y_cal=y_cal,
        categorical_features=set(),
        cfg=cfg,
    )
    inst = x_cal[0]
    first = guard.conforming_ranges_for_instance(inst)
    second = guard.conforming_ranges_for_instance(inst)
    assert first == second


def test_tree_conformal_parallel_determinism():
    x_cal = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y_cal = np.array([0, 0, 1, 1, 1])
    cfg = ConformalGuardConfig(precompute=True, candidate_grid=8, n_leaf_quantiles=8)
    guard = ConformalGuard(
        mode="factual",
        task="classification",
        x_cal=x_cal,
        y_cal=y_cal,
        categorical_features=set(),
        cfg=cfg,
    )
    inst = x_cal[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: guard.conforming_ranges_for_instance(inst), range(4)))
    assert all(results[0] == result for result in results[1:])


def test_tree_conformal_metadata_is_json_serializable():
    x_cal = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y_cal = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression().fit(x_cal, y_cal)
    explainer = CalibratedExplainer(lr, x_cal, y_cal, mode="classification")
    cfg = ConformalGuardConfig(precompute=True, candidate_grid=8, n_leaf_quantiles=8)
    plugin = ConformalGuardPlugin()
    plugin.initialize(_build_context(explainer))
    metadata = plugin.build_metadata(explainer, x_cal[:1], cfg)
    json.dumps(metadata)


def test_tree_conformal_is_value_conforming():
    x_cal = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y_cal = np.array([0, 0, 1, 1, 1])
    cfg = ConformalGuardConfig(precompute=True, candidate_grid=8, n_leaf_quantiles=8)
    guard = ConformalGuard(
        mode="factual",
        task="classification",
        x_cal=x_cal,
        y_cal=y_cal,
        categorical_features=set(),
        cfg=cfg,
    )
    inst = np.array([0.2])
    guard.conforming_ranges_for_instance(inst)
    assert guard.is_value_conforming(inst, 0) in (True, False)


def test_tree_conformal_warning_on_small_calib(caplog):
    x_cal = np.array([[0.0], [0.1]])
    y_cal = np.array([0, 1])
    cfg = ConformalGuardConfig(precompute=True, min_calib_samples=5)
    guard = ConformalGuard(
        mode="factual",
        task="classification",
        x_cal=x_cal,
        y_cal=y_cal,
        categorical_features=set(),
        cfg=cfg,
    )
    with caplog.at_level(logging.WARNING):
        guard.fit(precompute_per_feature=True)
    assert any("Insufficient calibration rows" in rec.message for rec in caplog.records)


def test_tree_conformal_explain_integration_suffix_and_exclusion():
    x_cal = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y_cal = np.array([0, 0, 1, 1, 1])
    lr = LogisticRegression().fit(x_cal, y_cal)

    explainer = CalibratedExplainer(
        lr,
        x_cal,
        y_cal,
        mode="classification",
        conformal_guard={"alpha": 0.0, "precompute": True, "use_for_perturbation": True},
    )
    explanations = explainer.explain_factual(x_cal[:1])
    explanation = explanations[0]
    explanation.define_conditions()
    assert any("conforms_to" in condition for condition in explanation.conditions if condition)

    explainer_strict = CalibratedExplainer(
        lr,
        x_cal,
        y_cal,
        mode="classification",
        conformal_guard={"alpha": 1.0, "precompute": True},
    )
    strict_explanations = explainer_strict.explain_factual(x_cal[:1])
    strict_explanation = strict_explanations[0]
    strict_explanation.define_conditions()
    ignored = strict_explanation.ignored_features_for_instance()
    assert 0 in ignored
