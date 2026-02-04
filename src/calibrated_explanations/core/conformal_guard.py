"""Tree-Conformal Plausibility Filter: fast, deterministic per-instance plausibility."""

from __future__ import annotations

import hashlib
import logging
import threading
from bisect import bisect_right
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

ConformingInterval = Tuple[float, float]
ConformingInfo = Dict[str, Any]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformalGuardConfig:
    """Config for the Tree-Conformal Plausibility Filter."""

    alpha: float = 0.1
    prop_fraction: float = 0.8
    max_depth: int = 8
    min_samples_leaf: int = 50
    n_leaf_quantiles: int = 64
    candidate_grid: int = 64
    use_interaction_gate: bool = True
    use_for_perturbation: bool = False
    precompute: bool = True
    min_calib_samples: int = 30
    cache_ttl_seconds: Optional[int] = 600
    seed: int = 42
    verbose: bool = False

    @classmethod
    def from_user(cls, payload: Any, *, seed: int = 42) -> "ConformalGuardConfig":
        """Normalize user input into a config instance."""
        if isinstance(payload, cls):
            return payload
        if payload is True:
            return cls(seed=seed)
        if payload is False or payload is None:
            raise TypeError("conformal_guard must be truthy or a configuration")
        if isinstance(payload, dict):
            cfg = dict(payload)
            cfg.setdefault("seed", seed)
            return cls(**cfg)
        raise TypeError("conformal_guard must be bool or dict or ConformalGuardConfig")


class TreeConformalGuard:
    """Fast conformal guard using per-feature decision trees + optional interaction tree."""

    def __init__(
        self,
        *,
        mode: str,
        task: str,
        x_cal: np.ndarray,
        y_cal: Optional[np.ndarray],
        categorical_features: Sequence[int],
        cfg: ConformalGuardConfig,
        bins: Any | None = None,
    ) -> None:
        """Initialize the tree-based conformal guard.

        The tree guard does not require prediction calls.
        """
        self.mode = mode
        self.task = task
        self.bins = bins
        self.x_cal = np.asarray(x_cal)
        self.y_cal = None if y_cal is None else np.asarray(y_cal)
        if self.x_cal.ndim == 1:
            self.x_cal = self.x_cal.reshape(-1, 1)
        self.num_features = int(self.x_cal.shape[1])
        self.categorical_features: Set[int] = set(categorical_features or ())
        self.cfg = cfg
        self._rng = check_random_state(self.cfg.seed)

        self._lock = threading.RLock()
        self._propX: Optional[np.ndarray] = None
        self._calX: Optional[np.ndarray] = None
        self._feat_models: Dict[int, Dict[str, Any]] = {}
        self._calib_scores: Dict[int, np.ndarray] = {}
        self._validity_tree: Optional[DecisionTreeClassifier] = None
        self._validity_calib_scores: Optional[np.ndarray] = None
        self._cache: Dict[str, Tuple[float, Dict[int, ConformingInfo]]] = {}

        if self.cfg.precompute:
            self.fit(precompute_per_feature=True)

    def fit(self, precompute_per_feature: bool = True) -> None:
        """Fit per-feature trees and calibration score distributions."""
        with self._lock:
            if self._propX is not None and self._calX is not None:
                return

            if self.x_cal.shape[0] < max(2 * self.cfg.min_calib_samples, 10):
                _LOG.warning(
                    "Insufficient calibration rows for conformal guard: %d",
                    self.x_cal.shape[0],
                )

            propX, calX = train_test_split(
                self.x_cal,
                train_size=float(self.cfg.prop_fraction),
                random_state=self.cfg.seed,
                shuffle=True,
            )
            self._propX = np.asarray(propX)
            self._calX = np.asarray(calX)

            if precompute_per_feature:
                for f in range(self.num_features):
                    try:
                        self._fit_feature(f)
                    except Exception:
                        _LOG.exception(
                            "Failed to fit feature guard for feature=%d; continuing", f
                        )

            if self.cfg.use_interaction_gate:
                try:
                    self._fit_validity_gate()
                except Exception:
                    _LOG.exception("Validity gate fit failed; continuing without it")
                    self._validity_tree = None
                    self._validity_calib_scores = None

    def _fit_feature(self, feature_idx: int) -> None:
        with self._lock:
            if feature_idx in self._feat_models:
                return
            X_prop = self._propX
            X_cal = self._calX
            col_prop = X_prop[:, feature_idx]
            col_cal = X_cal[:, feature_idx]

            model_entry: Dict[str, Any] = {"feature_idx": feature_idx, "tree_type": None}

            X_minus_prop = np.delete(X_prop, feature_idx, axis=1)
            X_minus_cal = np.delete(X_cal, feature_idx, axis=1)

            if feature_idx in self.categorical_features:
                clf = DecisionTreeClassifier(
                    max_depth=self.cfg.max_depth,
                    min_samples_leaf=max(1, int(self.cfg.min_samples_leaf)),
                    random_state=self.cfg.seed,
                )
                clf.fit(X_minus_prop, col_prop)
                leaves = clf.apply(X_minus_prop)
                leaf_counts: Dict[int, Dict[Any, int]] = {}
                leaf_sizes: Dict[int, int] = {}
                for leaf, val in zip(leaves, col_prop):
                    leaf_sizes.setdefault(leaf, 0)
                    leaf_counts.setdefault(leaf, {})
                    leaf_counts[leaf][val] = leaf_counts[leaf].get(val, 0) + 1
                    leaf_sizes[leaf] += 1

                cal_leaves = clf.apply(X_minus_cal)
                alpha = 1.0
                s_vals = []
                Vj = len(np.unique(col_prop))
                for leaf, y in zip(cal_leaves, col_cal):
                    counts = leaf_counts.get(leaf, {})
                    n_leaf = leaf_sizes.get(leaf, 0)
                    p_hat = (counts.get(y, 0) + alpha) / (n_leaf + alpha * max(1, Vj))
                    s_vals.append(1.0 - float(p_hat))

                model_entry.update(
                    {
                        "tree_type": "categorical",
                        "model": clf,
                        "leaf_counts": leaf_counts,
                        "leaf_sizes": leaf_sizes,
                        "Vj": Vj,
                        "alpha": alpha,
                        "calib_scores": np.asarray(s_vals, dtype=float),
                        "unique_values": np.unique(col_prop).tolist(),
                    }
                )
                self._calib_scores[feature_idx] = np.sort(
                    np.asarray(s_vals, dtype=float)
                )[::-1]
                self._feat_models[feature_idx] = model_entry
                return

            reg = DecisionTreeRegressor(
                max_depth=self.cfg.max_depth,
                min_samples_leaf=max(1, int(self.cfg.min_samples_leaf)),
                random_state=self.cfg.seed,
            )
            reg.fit(X_minus_prop, col_prop)
            leaves = reg.apply(X_minus_prop)
            leaf_samples: Dict[int, np.ndarray] = {}
            for leaf, v in zip(leaves, col_prop):
                leaf_samples.setdefault(leaf, []).append(float(v))
            for k, arr in leaf_samples.items():
                leaf_samples[k] = np.sort(np.asarray(arr, dtype=float))

            cal_leaves = reg.apply(X_minus_cal)
            s_vals = []
            for leaf, val in zip(cal_leaves, col_cal):
                samples = leaf_samples.get(leaf, np.array([], dtype=float))
                if samples.size == 0:
                    samples = np.sort(np.asarray(self.x_cal[:, feature_idx], dtype=float))
                pos = bisect_right(samples, float(val))
                F = pos / samples.size if samples.size else 0.0
                t = 2.0 * min(F, 1.0 - F) if samples.size else 0.0
                s_vals.append(1.0 - t)

            model_entry.update(
                {
                    "tree_type": "numeric",
                    "model": reg,
                    "leaf_samples": leaf_samples,
                    "n_leaf_quantiles": int(self.cfg.n_leaf_quantiles),
                    "calib_scores": np.asarray(s_vals, dtype=float),
                }
            )
            self._calib_scores[feature_idx] = np.sort(
                np.asarray(s_vals, dtype=float)
            )[::-1]
            self._feat_models[feature_idx] = model_entry

    def _fit_validity_gate(self) -> None:
        X_prop = self._propX
        if X_prop.shape[0] < max(50, 2 * self.cfg.min_calib_samples):
            _LOG.info("Not enough rows to fit interaction gate")
            self._validity_tree = None
            self._validity_calib_scores = None
            return

        rng = check_random_state(self.cfg.seed)
        n = X_prop.shape[0]
        fake = X_prop.copy()
        for col_idx in range(self.num_features):
            marg = self.x_cal[:, col_idx]
            fake[:, col_idx] = rng.choice(marg, size=n, replace=True)

        X_train = np.vstack([X_prop, fake])
        y_train = np.hstack([np.ones(n), np.zeros(n)])

        tree = DecisionTreeClassifier(
            max_depth=max(2, int(self.cfg.max_depth / 2)),
            min_samples_leaf=max(1, int(self.cfg.min_samples_leaf)),
            random_state=self.cfg.seed,
        )
        tree.fit(X_train, y_train)

        calX = self._calX
        if calX is None or calX.shape[0] == 0:
            self._validity_tree = None
            self._validity_calib_scores = None
            return

        probs = tree.predict_proba(calX)
        idx_real = 1 if tree.classes_[1] == 1 else int(np.where(tree.classes_ == 1)[0][0])
        p_real = probs[:, idx_real] if probs.ndim == 2 else probs
        s_cap = 1.0 - p_real
        self._validity_tree = tree
        self._validity_calib_scores = np.sort(np.asarray(s_cap, dtype=float))[::-1]

    def instance_p_value(self, instance: Sequence[Any]) -> float:
        """Conformal p-value of instance joint plausibility under the validity gate."""
        if not self.cfg.use_interaction_gate:
            return 1.0
        if self._validity_tree is None or self._validity_calib_scores is None:
            return 0.0
        x = np.asarray(instance).reshape(1, -1)
        probs = self._validity_tree.predict_proba(x)
        idx_real = 1 if self._validity_tree.classes_[1] == 1 else int(
            np.where(self._validity_tree.classes_ == 1)[0][0]
        )
        p_real = float(probs[0, idx_real])
        s = 1.0 - p_real
        return self._p_value_from_calib_array(self._validity_calib_scores, s)

    def is_instance_conforming(self, instance: Sequence[Any]) -> bool:
        """True iff instance passes the joint validity gate at alpha."""
        if not self.cfg.use_interaction_gate:
            return True
        return self.instance_p_value(instance) > float(self.cfg.alpha)

    def _instance_key(self, instance: Sequence[Any]) -> str:
        arr = np.asarray(instance, dtype=object).ravel().tolist()
        normalized = [v.item() if hasattr(v, "item") else v for v in arr]
        digest = hashlib.blake2b(repr(normalized).encode("utf-8"), digest_size=8)
        return digest.hexdigest()

    def conforming_ranges_for_instance(self, instance: Sequence[Any]) -> Dict[int, ConformingInfo]:
        inst = np.asarray(instance, dtype=object).ravel()
        key = self._instance_key(inst)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                ts, val = entry
                if self.cfg.cache_ttl_seconds is None or (time() - ts) <= self.cfg.cache_ttl_seconds:
                    return val
                self._cache.pop(key, None)

        self.fit(precompute_per_feature=self.cfg.precompute)

        meta_all: Dict[int, ConformingInfo] = {}
        for f in range(self.num_features):
            try:
                meta = self._compute_feature_meta_for_instance(inst, f)
            except Exception as exc:
                _LOG.exception("Error computing conformal ranges for feature=%d: %s", f, exc)
                meta = {
                    "intervals": [],
                    "candidate_points": [],
                    "values": [],
                    "tree_used": False,
                    "fallback": True,
                    "error": str(exc),
                    "interval": None,
                }
            meta_all[f] = meta

        with self._lock:
            self._cache[key] = (time(), meta_all)
        return meta_all

    def _compute_feature_meta_for_instance(self, inst: np.ndarray, f_idx: int) -> ConformingInfo:
        inst = np.asarray(inst).ravel()
        fallback = False
        tree_used = False
        error = None
        intervals: list[ConformingInterval] = []
        candidate_points: list[float] = []
        values: list[Any] = []
        interval_for_observed = None

        model_entry = self._feat_models.get(f_idx)
        if model_entry is None:
            col = np.asarray(self.x_cal[:, f_idx], dtype=float)
            unique = np.unique(col)
            candidate_points = np.sort(unique).tolist()[: self.cfg.candidate_grid]
            fallback = True
            return {
                "intervals": [],
                "candidate_points": candidate_points,
                "values": [],
                "tree_used": False,
                "fallback": True,
                "error": None,
                "interval": None,
            }

        tree_used = True
        if model_entry["tree_type"] == "categorical":
            tree: DecisionTreeClassifier = model_entry["model"]
            leaf = tree.apply(np.delete(inst.reshape(1, -1), f_idx, axis=1))[0]
            leaf_counts: Dict[int, Dict[Any, int]] = model_entry["leaf_counts"]
            leaf_sizes: Dict[int, int] = model_entry["leaf_sizes"]
            counts = leaf_counts.get(leaf, {})
            n_leaf = leaf_sizes.get(leaf, 0)
            Vj = model_entry["Vj"]
            alpha = model_entry["alpha"]
            local_values = sorted(counts.keys())
            if len(local_values) == 0:
                local_values = model_entry.get("unique_values", [])
                fallback = True
            calib_scores = self._calib_scores.get(f_idx, np.array([], dtype=float))
            accepted: list[Any] = []
            for v in local_values:
                p_hat = (counts.get(v, 0) + alpha) / (n_leaf + alpha * max(1, Vj))
                s = 1.0 - float(p_hat)
                pval = self._p_value_from_calib_array(calib_scores, s)
                if pval > float(self.cfg.alpha):
                    if self.cfg.use_interaction_gate:
                        inst2 = np.array(inst, copy=True)
                        inst2[f_idx] = v
                        if not self.is_instance_conforming(inst2):
                            continue
                    accepted.append(v)
            values = accepted
            return {
                "intervals": [],
                "candidate_points": [],
                "values": values,
                "tree_used": True,
                "fallback": fallback,
                "error": error,
                "interval": None,
            }

        reg: DecisionTreeRegressor = model_entry["model"]
        leaf = reg.apply(np.delete(inst.reshape(1, -1), f_idx, axis=1))[0]
        leaf_samples: Dict[int, np.ndarray] = model_entry["leaf_samples"]
        samples = leaf_samples.get(leaf, np.asarray([]))
        if samples.size == 0:
            samples = np.sort(np.asarray(self.x_cal[:, f_idx], dtype=float))
            fallback = True

        n_points = min(int(self.cfg.candidate_grid), max(4, int(self.cfg.n_leaf_quantiles)))
        if samples.size <= n_points:
            pts = np.unique(samples)
        else:
            qs = np.linspace(0.0, 1.0, n_points, endpoint=True)
            pts = np.quantile(samples, qs)
        pts = np.unique(np.asarray(pts, dtype=float))
        obs = float(inst[f_idx])
        if not np.any(np.isclose(pts, obs)):
            pts = np.sort(np.concatenate(([obs], pts)))
        calib_scores = self._calib_scores.get(f_idx, np.array([], dtype=float))
        accepted_mask = []
        for v in pts:
            pos = bisect_right(samples, float(v))
            F = pos / samples.size if samples.size else 0.0
            t = 2.0 * min(F, 1.0 - F) if samples.size else 0.0
            s = 1.0 - t
            pval = self._p_value_from_calib_array(calib_scores, s)
            ok = pval > float(self.cfg.alpha)
            if ok and self.cfg.use_interaction_gate:
                inst2 = np.array(inst, copy=True)
                inst2[f_idx] = float(v)
                ok = self.is_instance_conforming(inst2)
            accepted_mask.append(ok)
        accepted_mask = np.asarray(accepted_mask, dtype=bool)
        candidate_points = pts.tolist()
        i = 0
        while i < len(pts):
            if not accepted_mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(pts) and accepted_mask[j + 1]:
                j += 1
            intervals.append((float(pts[i]), float(pts[j])))
            i = j + 1
        for lo, hi in intervals:
            if lo <= obs <= hi:
                interval_for_observed = (lo, hi)
                break
        return {
            "intervals": intervals,
            "candidate_points": [float(v) for v in candidate_points],
            "values": [],
            "tree_used": tree_used,
            "fallback": fallback,
            "error": error,
            "interval": interval_for_observed,
        }

    def _p_value_from_calib_array(self, calib_array: np.ndarray, s_val: float) -> float:
        if calib_array is None or calib_array.size == 0:
            return 0.0
        asc = calib_array[::-1]
        pos = np.searchsorted(asc, s_val, side="left")
        count_ge = asc.size - pos
        p = (1.0 + float(count_ge)) / (asc.size + 1.0)
        return float(p)

    def sample_perturbations(
        self, instance: Sequence[Any], feature_idx: int, n_samples: int
    ) -> np.ndarray:
        meta = self.conforming_ranges_for_instance(instance).get(feature_idx, {})
        if feature_idx in self.categorical_features:
            vals = meta.get("values", []) or []
            if not vals:
                return np.array([])
            if self.cfg.use_interaction_gate:
                filtered = []
                inst = np.asarray(instance, dtype=object).ravel()
                for v in vals:
                    inst2 = np.array(inst, copy=True)
                    inst2[feature_idx] = v
                    if self.is_instance_conforming(inst2):
                        filtered.append(v)
                vals = filtered
                if not vals:
                    return np.array([])
            rng = check_random_state(self._deterministic_seed(instance, feature_idx))
            size = min(int(n_samples), len(vals))
            return np.asarray(rng.choice(vals, size=size, replace=False))
        candidates = np.asarray(meta.get("candidate_points", []), dtype=float).ravel()
        intervals = meta.get("intervals", [])
        if candidates.size == 0 or not intervals:
            return np.array([])
        mask = np.zeros_like(candidates, dtype=bool)
        for lo, hi in intervals:
            mask |= (candidates >= lo) & (candidates <= hi)
        eligible = candidates[mask]
        if eligible.size == 0:
            return np.array([])
        if self.cfg.use_interaction_gate:
            inst = np.asarray(instance, dtype=float).ravel()
            keep = []
            for v in eligible:
                inst2 = np.array(inst, copy=True)
                inst2[feature_idx] = float(v)
                if self.is_instance_conforming(inst2):
                    keep.append(float(v))
            eligible = np.asarray(keep, dtype=float)
            if eligible.size == 0:
                return np.array([])
        rng = check_random_state(self._deterministic_seed(instance, feature_idx))
        size = min(int(n_samples), eligible.size)
        if size == eligible.size:
            return eligible
        idxs = rng.choice(np.arange(eligible.size), size=size, replace=False)
        return eligible[idxs]

    def is_value_conforming(self, instance: Sequence[Any], feature_idx: int) -> bool:
        meta = self.conforming_ranges_for_instance(instance).get(feature_idx, {})
        if feature_idx in self.categorical_features:
            allowed = set(meta.get("values", []))
            return instance[feature_idx] in allowed if allowed else False
        intervals = meta.get("intervals", [])
        if not intervals:
            return False
        val = float(instance[feature_idx])
        for lo, hi in intervals:
            if lo <= val <= hi:
                return True
        return False

    def _deterministic_seed(self, instance: Sequence[Any], feature_idx: int) -> int:
        key = f"{self._instance_key(instance)}:{feature_idx}:{self.cfg.seed}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], "little") ^ int(self.cfg.seed)


ConformalGuard = TreeConformalGuard

__all__ = ["ConformalGuard", "ConformalGuardConfig", "TreeConformalGuard"]
