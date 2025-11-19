"""Explain executor system for calibrated explanations.

This package provides a plugin-based architecture for explain execution strategies:
- Sequential: single-threaded feature-by-feature processing
- Feature-parallel: parallel processing across features
- Instance-parallel: parallel processing across instances

The plugin system replaces branching logic in CalibratedExplainer.explain,
providing clean separation between orchestration and execution strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BaseExplainExecutor
from ._shared import ExplainConfig, ExplainRequest, ExplainResponse
from .orchestrator import ExplanationOrchestrator
from .parallel_feature import FeatureParallelExplainExecutor
from .parallel_instance import InstanceParallelExplainExecutor
from .sequential import SequentialExplainExecutor

if TYPE_CHECKING:
    # The CalibratedExplainer type is intentionally omitted here; importing it
    # only for TYPE_CHECKING when not used would trigger unused-import linting.
    pass


__all__ = [
    "BaseExplainExecutor",
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "ExplanationOrchestrator",
    "FeatureParallelExplainExecutor",
    "GuardOrchestratorPlugin",
    "InstanceParallelExplainExecutor",
    "SequentialExplainExecutor",
]
