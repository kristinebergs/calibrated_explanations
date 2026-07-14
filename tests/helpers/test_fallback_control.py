from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path

import pytest

from tests.helpers.fallback_control import assert_no_fallbacks_triggered


REPO_ROOT = Path(__file__).resolve().parents[2]
WARNING_POLICY_PATH = REPO_ROOT / "scripts" / "quality" / "check_warning_policy.py"


def _load_warning_policy_module():
    spec = importlib.util.spec_from_file_location("check_warning_policy", WARNING_POLICY_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_should_detect_logger_only_fallback_records() -> None:
    logger = logging.getLogger("calibrated_explanations.test")
    with (
        pytest.raises(AssertionError, match="Unexpected fallback log records detected"),
        assert_no_fallbacks_triggered(),
    ):
        logger.warning("Execution plugin failed for mode 'factual'; falling back to legacy")


def test_should_still_detect_warning_based_fallbacks() -> None:
    with (
        pytest.raises(AssertionError, match="Unexpected fallback warnings detected"),
        assert_no_fallbacks_triggered(),
    ):
        warnings.warn("Test fallback warning", UserWarning, stacklevel=2)


def test_should_flag_missing_warning_for_user_visible_registry_site() -> None:
    module = _load_warning_policy_module()
    warn_sites = []
    log_sites = [
        module.LogSite(
            rel_path="core/wrap_explainer.py",
            line=1,
            level="INFO",
            message_snippet="runtime fallback engaged",
            context="from_config",
        )
    ]
    registry = (
        module.FallbackSiteSpec(
            site_id="missing_warning",
            rel_path="core/wrap_explainer.py",
            context="from_config",
            message_pattern=r"runtime fallback engaged",
            disposition="user_visible",
            required_warning=True,
            required_log_level="INFO",
            reason="user-visible fallback should have both signals",
        ),
    )

    results = module.evaluate_fallback_registry(warn_sites, log_sites, registry)

    assert results[0].status == "fail"
    assert "missing UserWarning signal" in results[0].violations


def test_should_allow_exempt_registry_site_when_required_log_exists() -> None:
    module = _load_warning_policy_module()
    warn_sites = []
    log_sites = [
        module.LogSite(
            rel_path="core/calibrated_explainer.py",
            line=1,
            level="WARNING",
            message_snippet="feature filter enforcement skipped due to setup failure",
            context="_enforce_feature_filter_plugin_preferences",
        )
    ]
    registry = (
        module.FallbackSiteSpec(
            site_id="feature_filter_exempt",
            rel_path="core/calibrated_explainer.py",
            context="_enforce_feature_filter_plugin_preferences",
            message_pattern=r"feature filter enforcement skipped",
            disposition="exempt",
            required_log_level="WARNING",
            reason="recorded exemption",
        ),
    )

    results = module.evaluate_fallback_registry(warn_sites, log_sites, registry)

    assert results[0].status == "pass"
