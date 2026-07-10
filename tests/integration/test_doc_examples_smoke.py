from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer

pytestmark = pytest.mark.integration

matplotlib.use("Agg")

_README_PATH = Path("README.md")
_QUICK_API_PATH = Path("docs/get-started/quick_api.md")


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def _build_classification_context() -> dict[str, object]:
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        stratify=data.target,
        random_state=42,
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=42,
    )
    explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=42))
    return {
        "WrapCalibratedExplainer": WrapCalibratedExplainer,
        "RandomForestClassifier": RandomForestClassifier,
        "d": data,
        "explainer": explainer,
        "feature_names": data.feature_names,
        "gender_col_index": 0,
        "np": np,
        "X_cal": x_cal,
        "X_pr": x_proper,
        "X_proper": x_proper,
        "X_query": x_test[:3],
        "X_sample": x_test[:3],
        "X_te": x_test,
        "X_tr": x_train,
        "load_breast_cancer": load_breast_cancer,
        "train_test_split": train_test_split,
        "x_cal": x_cal,
        "x_proper": x_proper,
        "y_cal": y_cal,
        "y_pr": y_proper,
        "y_proper": y_proper,
        "y_te": y_test,
        "y_tr": y_train,
    }


def _build_regression_context() -> dict[str, object]:
    x, y = make_regression(n_samples=240, n_features=6, noise=0.5, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=42,
    )
    explainer = WrapCalibratedExplainer(RandomForestRegressor(random_state=42))
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal)
    return {
        "explainer": explainer,
        "X_sample": x_test[:3],
        "low_high_percentiles": (5, 95),
        "np": np,
        "threshold": 120.0,
        "x_cal": x_cal,
        "x_proper": x_proper,
        "y_cal": y_cal,
        "y_proper": y_proper,
    }


def test_readme_python_examples_execute_without_error() -> None:
    readme_blocks = _python_blocks(_README_PATH)
    assert len(readme_blocks) == 3

    quickstart_globals = _build_classification_context()
    exec(readme_blocks[0], quickstart_globals)
    assert "exp" in quickstart_globals

    snippet_globals = _build_classification_context()
    snippet_globals["explainer"].fit(snippet_globals["X_proper"], snippet_globals["y_proper"])
    snippet_globals["explainer"].calibrate(
        snippet_globals["X_cal"],
        snippet_globals["y_cal"],
        feature_names=snippet_globals["feature_names"],
    )
    exec(readme_blocks[1], snippet_globals)
    exec(readme_blocks[2], snippet_globals)


def test_quick_api_python_examples_execute_without_error() -> None:
    quick_api_blocks = _python_blocks(_QUICK_API_PATH)
    assert len(quick_api_blocks) == 4

    classification_globals = _build_classification_context()
    exec(quick_api_blocks[1], classification_globals)
    exec(quick_api_blocks[0], classification_globals)

    regression_globals = _build_regression_context()
    exec(quick_api_blocks[2], regression_globals)
    exec(quick_api_blocks[3], regression_globals)
