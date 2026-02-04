"""Conformal guard plugin helper for explanation pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.conformal_guard import ConformalGuard, ConformalGuardConfig
from ..core.explain._helpers import validate_and_prepare_input
from .explanations import ExplanationContext


@dataclass
class ConformalGuardPlugin:
    """Helper that builds conformal metadata using the PredictBridge."""

    _context: ExplanationContext | None = None
    _guard_cache: dict[tuple[int, int], ConformalGuard] = field(default_factory=dict)

    def initialize(self, context: ExplanationContext) -> None:
        self._context = context

    def _guard_key(
        self, explainer: Any, cfg: ConformalGuardConfig, bins: Any | None
    ) -> tuple[int, int]:
        return (
            id(explainer),
            hash((cfg, bins is not None)),
        )

    def build_metadata(
        self,
        explainer: Any,
        x: Any,
        cfg: ConformalGuardConfig,
        *,
        bins: Any | None = None,
    ) -> list[dict[int, Any]]:
        if self._context is None:
            raise RuntimeError("ConformalGuardPlugin must be initialized before use")
        x_prepared = validate_and_prepare_input(explainer, x)
        key = self._guard_key(explainer, cfg, bins)
        guard = self._guard_cache.get(key)
        if guard is None:
            guard = ConformalGuard(
                mode=self._context.mode,
                task=self._context.task,
                x_cal=np.asarray(explainer.x_cal),
                y_cal=np.asarray(explainer.y_cal) if getattr(explainer, "y_cal", None) is not None else None,
                categorical_features=set(explainer.categorical_features),
                cfg=cfg,
                bins=bins,
            )
            if cfg.precompute:
                guard.fit(precompute_per_feature=True)
            self._guard_cache[key] = guard
            setattr(explainer, "_conformal_guard", guard)
        conformal_metadata = []
        for instance in x_prepared:
            info = guard.conforming_ranges_for_instance(instance)
            meta: dict[int, Any] = {}
            for f_idx in range(explainer.num_features):
                entry = info.get(f_idx, {})
                intervals = [(float(lo), float(hi)) for lo, hi in entry.get("intervals", [])]
                candidates = [
                    float(val) for val in np.asarray(entry.get("candidate_points", [])).reshape(-1)
                ]
                values = list(entry.get("values", [])) if entry.get("values", []) is not None else []
                meta[f_idx] = {
                    "intervals": intervals,
                    "interval": entry.get("interval"),
                    "candidate_points": candidates,
                    "values": values,
                    "tree_used": bool(entry.get("tree_used", False)),
                    "fallback": bool(entry.get("fallback", False)),
                    "error": entry.get("error"),
                }
            conformal_metadata.append(meta)
        return conformal_metadata


__all__ = ["ConformalGuardPlugin", "ConformalGuardConfig"]
