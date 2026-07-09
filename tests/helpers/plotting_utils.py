"""Shared helpers for plotting tests."""

from __future__ import annotations

import calibrated_explanations.plotting as plotting


def reset_plotting_config_manager() -> None:
    """Reset plotting config manager state between tests."""
    plotting.reset_plotting_config_manager()
