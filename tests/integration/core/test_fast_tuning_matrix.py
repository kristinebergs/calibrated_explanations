import copyreg
import pickle
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core import CalibratedExplainer
from tests.helpers.dataset_utils import read_csv_cached
from tests.helpers.model_utils import get_regression_model

pytestmark = pytest.mark.integration


_FAST_NOISE_CONFIGS = (
    {
        "noise_type": "uniform",
        "scale_factor": 3,
        "severity": 0.2,
        "sample_percentiles": [10, 50, 90],
        "seed": 123,
    },
    {
        "noise_type": "gaussian",
        "scale_factor": 7,
        "severity": 1.2,
        "sample_percentiles": [10, 50, 90],
        "seed": 123,
    },
)
_SAMPLE_PERCENTILE_CONFIGS = (
    {
        "noise_type": "uniform",
        "scale_factor": 5,
        "severity": 1.0,
        "sample_percentiles": [10, 50, 90],
        "seed": 123,
    },
    {
        "noise_type": "uniform",
        "scale_factor": 5,
        "severity": 1.0,
        "sample_percentiles": [5, 50, 95],
        "seed": 123,
    },
)


def _make_reduced_regression_case(max_rows: int = 200):
    data_dir = Path(__file__).resolve().parents[2] / "data" / "reg"
    ds = read_csv_cached(str(data_dir / "abalone.txt"))
    x = ds.drop("REGRESSION", axis=1).values[:max_rows, :]
    y = ds["REGRESSION"].values[:max_rows]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    feature_names = ds.drop("REGRESSION", axis=1).columns
    categorical_features = [i for i in range(x.shape[1]) if len(np.unique(x[:, i])) < 10]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2, random_state=42)
    x_prop_train, x_cal, y_prop_train, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=min(100, len(x_train) - 2),
        random_state=42,
    )
    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        categorical_features,
        feature_names,
    )


def _threshold(y_test):
    return float(np.median(y_test))


def _noise_snapshot(metadata):
    noise_config = metadata["fast"]["noise_config"]
    return {
        "noise_type": noise_config["noise_type"],
        "scale_factor": noise_config["scale_factor"],
        "severity": noise_config["severity"],
        "seed": noise_config["seed"],
    }


def _assert_fast_runtime_markers(explainer, config, calibration_rows):
    assert explainer.plugin_manager.interval_plugin_identifiers["fast"] == "core.interval.fast"
    assert explainer.plugin_manager.telemetry_interval_sources["fast"] == "core.interval.fast"
    assert _noise_snapshot(explainer.plugin_manager.interval_context_metadata) == {
        key: config[key] for key in ("noise_type", "scale_factor", "severity", "seed")
    }
    assert explainer.scaled_x_cal.shape == (
        calibration_rows * config["scale_factor"],
        explainer.num_features,
    )
    assert explainer.fast_x_cal.shape == explainer.scaled_x_cal.shape
    assert explainer.scaled_y_cal.shape == (calibration_rows * config["scale_factor"],)


def _wrapper_from_config(config):
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        categorical_features,
        feature_names,
    ) = _make_reduced_regression_case()
    wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=25, random_state=7))
    wrapper.fit(x_prop_train, y_prop_train)
    wrapper.calibrate(
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="regression",
        fast=True,
        **config,
    )
    return wrapper, x_cal, y_cal, x_test, y_test


def _core_from_config(config):
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        categorical_features,
        feature_names,
    ) = _make_reduced_regression_case()
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="regression",
        fast=True,
        **config,
    )
    return explainer, x_cal, x_test, y_test


def test_fast_tuning_knobs_should_change_public_fast_behavior_across_wrapper_core_load_and_recalibration(
    tmp_path: Path,
):
    wrapper_payloads = []
    core_payloads = []
    for index, config in enumerate(_FAST_NOISE_CONFIGS):
        wrapper, fresh_x_cal, fresh_y_cal, x_test, y_test = _wrapper_from_config(config)
        fresh_fast = wrapper.explain_fast(x_test[:1])
        _assert_fast_runtime_markers(wrapper.explainer, config, len(fresh_x_cal))
        wrapper_payloads.append(np.asarray(fresh_fast[0].feature_weights["predict"], dtype=float))

        state_dir = tmp_path / f"fast-state-{index}"
        wrapper.save_state(state_dir)
        restored = WrapCalibratedExplainer.load_state(state_dir)
        restored_fast = restored.explain_fast(x_test[:1])
        _assert_fast_runtime_markers(restored.explainer, config, len(fresh_x_cal))
        np.testing.assert_allclose(
            restored_fast[0].feature_weights["predict"],
            fresh_fast[0].feature_weights["predict"],
        )

        recal_x = fresh_x_cal[:80]
        recal_y = fresh_y_cal[:80]
        wrapper.calibrate(recal_x, recal_y, fast=True, mode="regression", **config)
        recal_fast = wrapper.explain_fast(x_test[:1])
        _assert_fast_runtime_markers(wrapper.explainer, config, len(recal_x))
        assert np.isfinite(np.linalg.norm(recal_fast[0].feature_weights["predict"]))

        core, core_x_cal, core_x_test, _core_y_test = _core_from_config(config)
        core_fast = core.explain_fast(core_x_test[:1])
        _assert_fast_runtime_markers(core, config, len(core_x_cal))
        core_payloads.append(np.asarray(core_fast[0].feature_weights["predict"], dtype=float))

    assert not np.allclose(wrapper_payloads[0], wrapper_payloads[1])
    assert not np.allclose(core_payloads[0], core_payloads[1])


def test_sample_percentiles_should_change_explanation_surface_without_needing_exact_float_baselines():
    wrapper_vectors = []
    core_vectors = []
    for config in _SAMPLE_PERCENTILE_CONFIGS:
        wrapper, _, _, x_test, _ = _wrapper_from_config(config)
        factual = wrapper.explain_factual(x_test[:1])
        wrapper_vectors.append(np.asarray(factual[0].feature_weights["predict"], dtype=float))

        core, _, core_x_test, _ = _core_from_config(config)
        factual_core = core.explain_factual(core_x_test[:2])
        core_vectors.append(
            np.concatenate(
                [np.asarray(item.feature_weights["predict"], dtype=float) for item in factual_core]
            )
        )

    assert not np.allclose(wrapper_vectors[0], wrapper_vectors[1])
    assert not np.allclose(core_vectors[0], core_vectors[1])


def test_ce_owned_pickle_boundaries_should_work_without_a_process_wide_mappingproxy_reducer():
    wrapper, x_cal, _, x_test, y_test = _wrapper_from_config(_FAST_NOISE_CONFIGS[0])
    explanations = wrapper.explain_fast(x_test[:1])
    core, _, core_x_test, _core_y_test = _core_from_config(_FAST_NOISE_CONFIGS[0])
    core.explain_fast(core_x_test[:1])

    copyreg.dispatch_table.pop(MappingProxyType, None)
    with pytest.raises(TypeError, match="mappingproxy"):
        pickle.dumps(MappingProxyType({"demo": 1}))

    wrapper_payload = pickle.dumps(wrapper)
    restored_wrapper = pickle.loads(wrapper_payload)
    _assert_fast_runtime_markers(restored_wrapper.explainer, _FAST_NOISE_CONFIGS[0], len(x_cal))

    explanation_payload = pickle.dumps(explanations)
    restored_explanations = pickle.loads(explanation_payload)
    assert len(restored_explanations) == len(explanations)

    core_payload = pickle.dumps(core)
    restored_core = pickle.loads(core_payload)
    restored_fast = restored_core.explain_fast(core_x_test[:1])
    assert np.isfinite(np.linalg.norm(restored_fast[0].feature_weights["predict"]))
