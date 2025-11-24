"""Confidence-modulated conformal regions for perturbation guarding.

This module implements ConformalRegionOracle, which provides in-distribution
guarantees for perturbed instances during explanation generation.

Based on conformal prediction theory (Vovk, 1999) and calibrated confidence
modulation, the oracle:
- Clusters data in feature space to capture heteroscedasticity
- Computes Mahalanobis distance-based nonconformity scores
- Derives conformal radii with calibration guarantees
- Modulates acceptance by model confidence (interval width)

No thresholds or categorical contexts required.
"""

import logging

import numpy as np
from scipy.linalg import pinvh
from sklearn.cluster import KMeans
from sklearn.utils import check_array

logger = logging.getLogger(__name__)


class ConformalRegionOracle:
    """Confidence-modulated conformal regions for filtering out-of-distribution perturbations.

    Uses conformal prediction with clustering-based nonconformity to provide finite-sample
    coverage guarantees on in-distribution perturbations.

    Parameters
    ----------
    alpha : float, default=0.1
        Miscalibration level; coverage guarantee is 1 - alpha.
        Typical values: 0.01 (99%), 0.05 (95%), 0.1 (90%), 0.2 (80%).

    n_clusters : int, default=5
        Number of clusters for feature-space stratification.
        Rule of thumb: sqrt(n_samples / 10).

    prop_size : float, default=0.75
        Proportion of training data to use for proper set (the rest for calibration).
        Must be in (0, 1). Inductive conformal prediction splits data internally.

    random_state : int or None, default=None
        Random seed for reproducibility (clustering, data splitting).

    nonconformity_metric : str, default="mahalanobis"
        Distance metric for nonconformity scoring in conformal prediction.
        Options:
            - "euclidean" (L2 norm): General-purpose, isotropic
            - "mahalanobis" (Mahalanobis distance): Accounts for covariance structure
            - "cosine" (Angle-based): For high-dimensional/embedding data;
              falls back to Euclidean for score computation (Mahalanobis incompatible)

    Attributes
    ----------
    _fitted : bool
        Whether the oracle has been fitted.
    _cluster_centers : np.ndarray, shape (n_clusters, n_features)
        Cluster centers from KMeans.
    _cluster_covs : list of np.ndarray
        Per-cluster covariance matrices.
    _conformal_radii : np.ndarray, shape (n_clusters,)
        Conformal radius per cluster.
    _width_min, _width_max : float
        Range of interval widths in training data (for normalization).
    _global_mins, _global_maxs : np.ndarray
        Feature-wise min/max of training data (for edge case handling).
    """

    def __init__(
        self,
        alpha=0.1,
        n_clusters=5,
        prop_size=0.75,
        random_state=None,
        nonconformity_metric="mahalanobis",
        enforcement: bool = True,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if not 0 < prop_size <= 1:
            raise ValueError(f"prop_size must be in (0, 1], got {prop_size}")
        
        # Validate nonconformity_metric
        valid_metrics = {"euclidean", "mahalanobis", "cosine"}
        if nonconformity_metric not in valid_metrics:
            raise ValueError(
                f"nonconformity_metric must be one of {valid_metrics}, "
                f"got '{nonconformity_metric}'"
            )
        
        self.alpha = alpha
        self.n_clusters = n_clusters
        # Bounds for width-based modulation (clamped)
        self._modulation_min = 0.5
        self._modulation_max = 2.0
        self.prop_size = prop_size
        self.random_state = random_state
        self._nonconformity_metric = nonconformity_metric
        self.enforcement = enforcement

        self._fitted = False
        self._cluster_centers = None
        self._cluster_covs = None
        self._cluster_radii = None
        self._width_min = None
        self._width_max = None
        self._global_mins = None
        self._global_maxs = None
        self._kmeans = None
        self._n_features_base = None  # Base feature dimensionality (before augmentation)
        # Cached calibration diagnostics (populated by fit())
        self._cal_scores = None
        # Cached calibration widths (for normalized conformal regression)
        self._cal_widths = None
        # Per-cluster normalized quantiles (q_norm) such that effective radius
        # at test = q_norm[cluster] * width_test
        self._cluster_norm_quantiles = None
        # Small epsilon to stabilize division by width
        self._eps_width = 1e-12
        self._cal_nearest = None

    # noqa: ARG002, ARG001
    def fit(self, x, y, interval_learner, x_cal=None, y_cal=None):
        """Fit the conformal region oracle using normalized conformal regression.

        Performs inductive conformal prediction with confidence modulation:

        1. Split x into proper (75%) and calibration (25%) sets
        2. Cluster the proper set in the augmented feature space [x || pred]
           where pred comes from interval_learner.predict(x_proper, uq_interval=True)
        3. Compute per-cluster covariance and Mahalanobis distances in augmented space
        4. Extract uncertainty intervals from interval_learner for calibration set
        5. Normalize nonconformity scores by interval width: s_norm = s_raw / width
        6. Compute (1 - alpha) quantile on normalized scores: q_norm
        7. Store q_norm per cluster for dynamic radius modulation at test time
        8. Record width statistics (min/max) for confidence modulation diagnostics

        **Key Innovation: Normalized Conformal Regression (NCR)**

        This oracle implements NCR for confidence-aware perturbation filtering:
        - At test time, effective radius scales with prediction confidence
        - Wider intervals (low confidence) → larger acceptance regions
        - Narrower intervals (high confidence) → smaller acceptance regions
        - Formula: r_eff(cluster, width_test) = q_norm(cluster) * width_test

        This approach ensures that the conformal guarantee (coverage 1 - α) is
        maintained while adapting the radius to the model's confidence.

        **Mandatory Requirement**: interval_learner must provide calibrated
        predictions with uncertainty intervals. The oracle uses:
        - Predictions to augment feature space for clustering
        - Interval widths to normalize nonconformity scores
        - Coverage guarantee depends on interval_learner being well-calibrated

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training instances used to define conformal regions in augmented space.

        y : array-like, shape (n_samples,)
            Training targets. Not used directly but kept for interface consistency.

        interval_learner : fitted calibrator
            Fitted interval learner (e.g., from CalibratedExplainer).
            **MUST** support: `interval_learner.predict(x, uq_interval=True)`
            **MUST** return: `(predictions, (lower_bounds, upper_bounds))`
            - predictions: shape (n_samples,), calibrated predictions
            - lower_bounds: shape (n_samples,), lower uncertainty bounds
            - upper_bounds: shape (n_samples,), upper uncertainty bounds
            Width = upper - lower is used to normalize conformal scores.

        x_cal : array-like, optional
            Calibration instances (for external split). If None, split x internally.

        y_cal : array-like, optional
            Calibration targets. Used only if x_cal is provided.

        Returns
        -------
        self
            The fitted oracle instance (enables chaining).

        Raises
        ------
        ValueError
            If interval_learner is None (mandatory for NCR).
            If training data is too small (need at least 2 * n_clusters samples).
            If interval_learner.predict(uq_interval=True) fails or returns invalid format.

        Notes
        -----
        The oracle stores:
        - _cluster_centers: cluster centers in augmented space (n_clusters, n_features + 1)
        - _cluster_covs: per-cluster covariance matrices in augmented space
        - _cluster_norm_quantiles: normalized quantiles for radius modulation
        - _cal_widths: interval widths on calibration set
        - _width_min, _width_max: range of widths (for diagnostics)

        During accept(), the oracle expects both x_new and its calibrated
        prediction (pred, (lower, upper)) to compute the effective radius
        using the stored normalized quantiles.
        """
        x_arr = check_array(x, accept_sparse=False, ensure_2d=True)
        _ = np.asarray(y)

        # Require an interval learner: the oracle relies on interval widths
        # for normalized conformal regression and confidence modulation.
        if interval_learner is None:
            raise ValueError("interval_learner must be provided; None is not allowed")

        # Validate interval_learner supports required interface
        try:
            # Test predict signature with small sample
            test_sample = x_arr[: min(2, len(x_arr))]
            test_result = interval_learner.predict(test_sample, uq_interval=True)
            # Verify return format: (predictions, (lower, upper))
            if not isinstance(test_result, tuple) or len(test_result) != 2:
                raise ValueError(
                    f"interval_learner.predict() must return "
                    f"(predictions, (lower, upper)), got {type(test_result)}"
                )
            _, bounds = test_result
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"interval_learner.predict() second element must be "
                    f"(lower, upper) tuple, got {type(bounds)}"
                )
        except TypeError as exc:
            if "uq_interval" in str(exc):
                raise ValueError(
                    f"interval_learner.predict() does not support uq_interval parameter. "
                    f"Error: {exc}"
                ) from exc
            raise ValueError(
                f"interval_learner.predict() signature incompatible. Error: {exc}"
            ) from exc
        except ValueError:
            raise  # Re-raise our own ValueError checks
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "interval_learner validation encountered unexpected error; "
                "confidence modulation may not work correctly. Error: %s",
                exc,
            )

        # Ensure input array is float type
        x_arr = np.asarray(x_arr, dtype=float)

        if len(x_arr) < 2 * self.n_clusters:
            raise ValueError(
                f"Training set too small: {len(x_arr)} samples, but "
                f"need at least {2 * self.n_clusters} samples "
                f"for {self.n_clusters} clusters with proper/calib split."
            )

        n_features = x_arr.shape[1]
        n_samples = x_arr.shape[0]

        # Store global feature bounds
        self._global_mins = np.min(x_arr, axis=0)
        self._global_maxs = np.max(x_arr, axis=0)

        # ICP: Split into proper and calibration sets
        rng = np.random.RandomState(self.random_state)
        if x_cal is not None and y_cal is not None:
            x_cal = np.asarray(x_cal, dtype=float)
            x_proper = x_arr
            n_samples += x_cal.shape[0]
        else:
            indices = rng.permutation(n_samples)
            n_proper = max(1, int(self.prop_size * n_samples))
            prop_indices = indices[:n_proper]
            cal_indices = indices[n_proper:]

            x_proper = x_arr[prop_indices]
            x_cal = x_arr[cal_indices]

        if len(x_cal) == 0:
            raise ValueError("Calibration set is empty; increase training data or reduce prop_size")

        # --- Augment feature space with predictions: [x || calibrated_prediction] ---
        # Extract calibrated predictions for proper set
        try:
            preds_proper, (_lower_proper, _upper_proper) = interval_learner.predict(
                x_proper, uq_interval=True
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Failed to extract calibrated predictions for proper set: %s. "
                "interval_learner must provide uq_interval=True output.",
                exc,
            )
            raise ValueError(
                f"interval_learner.predict(x, uq_interval=True) failed: {exc}"
            ) from exc

        # Handle case where predictions are returned as string class labels
        # (e.g., from CalibratedExplainer with class_labels mapping)
        preds_proper_arr = np.asarray(preds_proper)
        if preds_proper_arr.dtype.kind in ("U", "O", "S"):  # Unicode, object, or bytes string
            # Try to convert string labels to numeric
            try:
                preds_proper = np.asarray(preds_proper, dtype=float)
            except (ValueError, TypeError):
                # If strings can't be converted to float, use ordinal encoding
                unique_labels = np.unique(preds_proper_arr)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                preds_proper = np.asarray([label_map[p] for p in preds_proper_arr], dtype=float)

        # Concatenate features with predictions for augmented space
        # Shape: (n_proper, n_features + 1)
        # Ensure preds_proper is float to avoid dtype issues with np.column_stack
        preds_proper = np.asarray(preds_proper, dtype=float).ravel()
        x_proper_augmented = np.column_stack([x_proper, preds_proper])
        # Force float dtype to ensure numerical operations work
        x_proper_augmented = np.asarray(x_proper_augmented, dtype=float)
        self._n_features_base = n_features  # Store base feature count for later

        # Cluster on augmented space [x || pred]
        n_clusters_actual = min(self.n_clusters, len(x_proper))
        self._kmeans = KMeans(
            n_clusters=n_clusters_actual,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(x_proper_augmented)
        self._cluster_centers = self._kmeans.cluster_centers_

        # Compute per-cluster covariance on augmented space only when needed
        # for Mahalanobis-based nonconformity. This avoids unnecessary O(d^3)
        # work and memory use for Euclidean/Cosine metrics.
        self._cluster_covs = []
        if self._nonconformity_metric == "mahalanobis":
            for k in range(n_clusters_actual):
                mask = self._kmeans.labels_ == k
                if np.sum(mask) > 1:
                    cov = np.cov(x_proper_augmented[mask].T)
                    # Handle 1D covariance
                    if cov.ndim == 0:
                        cov = np.array([[cov]])
                    elif cov.ndim == 1:
                        cov = np.diag(cov)
                else:
                    # Single point in cluster; use identity in augmented space
                    cov = np.eye(x_proper_augmented.shape[1])
                self._cluster_covs.append(cov)
        else:
            # For non-Mahalanobis metrics, covariance matrices are not used.
            self._cluster_covs = None

        # Compute nonconformity scores on calibration set
        # Extract calibrated predictions for calibration set
        try:
            preds_cal, (_lower_cal, _upper_cal) = interval_learner.predict(x_cal, uq_interval=True)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Failed to extract calibrated predictions for calibration set: %s. "
                "interval_learner must provide uq_interval=True output.",
                exc,
            )
            raise ValueError(
                f"interval_learner.predict(x_cal, uq_interval=True) failed: {exc}"
            ) from exc

        # Handle case where predictions are returned as string class labels
        preds_cal_arr = np.asarray(preds_cal)
        if preds_cal_arr.dtype.kind in ("U", "O", "S"):  # Unicode, object, or bytes string
            # Try to convert string labels to numeric
            try:
                preds_cal = np.asarray(preds_cal, dtype=float)
            except (ValueError, TypeError):
                # If strings can't be converted to float, use ordinal encoding
                unique_labels = np.unique(preds_cal_arr)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                preds_cal = np.asarray([label_map[p] for p in preds_cal_arr], dtype=float)

        # Compute nonconformity scores in augmented space [x || pred]
        cal_scores = self._compute_nonconformity_scores(x_cal, preds_cal)

        # Cache calibration scores and assignments so callers can recompute
        # radii for new alpha values without a full refit.
        try:
            # Augment x_cal with predictions to match cluster_centers dimensionality
            x_cal_augmented = np.array([
                np.concatenate([x_cal[i], preds_cal[i].ravel()])
                for i in range(len(x_cal))
            ])
            x_cal_augmented = np.asarray(x_cal_augmented, dtype=float)
            dists = np.linalg.norm(
                x_cal_augmented[:, None, :] - self._cluster_centers[None, :, :], axis=2
            )
            nearest = np.argmin(dists, axis=1)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if self.enforcement:
                raise
            logger.debug("Fallback to zero-nearest cluster assignment: %s", exc)
            nearest = np.zeros(len(cal_scores), dtype=int)

        # store cached calibration diagnostics for later recomputation
        self._cal_scores = np.asarray(cal_scores)
        self._cal_nearest = np.asarray(nearest)

        # --- Normalized conformal regression: use interval width as difficulty ---
        # Compute calibration widths for x_cal (interval_learner is required)
        # Extract predictions and interval bounds
        if interval_learner is not None:
            try:
                preds_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
                if preds_cal is not None and len(preds_cal) == len(x_cal):
                    # Direct array subtraction: upper and lower are already arrays
                    widths_cal = upper - lower
                    if widths_cal.ndim == 0:
                        widths_cal = np.full(len(cal_scores), float(widths_cal))
                else:
                    widths_cal = np.ones(len(cal_scores))
            except Exception as exc:  # pylint: disable=broad-except
                if self.enforcement:
                    raise
                logger.debug("Width extraction failed; falling back to unit widths: %s", exc)
                widths_cal = np.ones(len(cal_scores))
        else:
            if self.enforcement:
                raise ValueError("interval_learner must be provided for modulation")
            widths_cal = np.ones(len(cal_scores))

        # Cache widths for later recomputation in set_alpha
        self._cal_widths = widths_cal

        # Compute normalized calibration scores and store per-cluster normalized quantile
        try:
            s_cal_norm = self._cal_scores / (widths_cal + self._eps_width)
            global_norm_q = float(np.quantile(s_cal_norm, 1.0 - self.alpha))
            self._cluster_norm_quantiles = np.full(n_clusters_actual, global_norm_q)
        except Exception as exc:  # pylint: disable=broad-except
            if self.enforcement:
                raise
            logger.debug("Normalization failed; disabling modulation fallback: %s", exc)
            self._cluster_norm_quantiles = None

        # Compute conformal radius as (1 - alpha) quantile (global by default)
        quantile_idx = int(np.ceil((1 - self.alpha) * len(cal_scores)))
        quantile_idx = min(quantile_idx, len(cal_scores) - 1)
        global_radius = np.sort(cal_scores)[quantile_idx]
        self._cluster_radii = np.full(n_clusters_actual, global_radius)

        # Report calibration diagnostics to help debugging (counts per cluster)
        # Use cached nearest assignments (already computed in augmented space above)
        if len(self._cal_nearest) > 0:
            counts = np.bincount(self._cal_nearest, minlength=n_clusters_actual)
        else:
            counts = np.zeros(n_clusters_actual, dtype=int)

        # Summary in two shorter log lines to respect line-length
        logger.info(
            "Guard fit summary: alpha=%s, n_cal=%d, quantile_idx=%d",
            self.alpha,
            len(cal_scores),
            quantile_idx,
        )
        logger.info(
            "cluster_cal_counts=%s, cluster_radii_sample=%s",
            counts.tolist(),
            np.sort(cal_scores)[max(0, quantile_idx - 1) : quantile_idx + 1].tolist(),
        )

        # Warn about clusters with very few calibration points (estimates unreliable)
        for k, count in enumerate(counts.tolist()):
            if count < 2:
                logger.warning(
                    "Cluster %d has only %d calibration points; radius estimate may be unreliable",
                    k,
                    count,
                )

        # Record width statistics for confidence modulation
        # Extract predictions and interval bounds from full training set
        try:
            prediction_full, (lower_full, upper_full) = interval_learner.predict(
                x, uq_interval=True
            )
            if prediction_full is not None and len(prediction_full) > 0:
                # Direct array subtraction: upper_full and lower_full are already arrays
                widths = upper_full - lower_full
                self._width_min = float(np.min(widths))
                self._width_max = float(np.max(widths))
                if self._width_max < self._width_min:
                    # Safety check: should not happen, but swap if it does
                    self._width_min, self._width_max = self._width_max, self._width_min
            else:
                self._width_min = 0.0
                self._width_max = 1.0
        except Exception as exc:  # pylint: disable=broad-except
            if self.enforcement:
                raise
            logger.warning(
                "Could not compute width statistics from interval_learner: %s",
                exc,
            )
            self._width_min = 0.0
            self._width_max = 1.0

        # Inform whether normalized conformal regression (NCR) is active
        # NCR enables confidence-modulated perturbation filtering
        if self._width_max <= self._width_min:
            logger.warning(
                "Normalized Conformal Regression (NCR) disabled: "
                "width_min=%s, width_max=%s. "
                "Confidence modulation will not be active. "
                "Check interval_learner output.",
                self._width_min,
                self._width_max,
            )
        else:
            # Check if interval widths are nearly uniform (limited NCR benefit)
            width_range = self._width_max - self._width_min
            width_relative_range = width_range / max(abs(self._width_max), 1e-10)
            
            try:
                has_quantiles = self._cluster_norm_quantiles is not None
                if has_quantiles:
                    q_norm_min = float(np.min(self._cluster_norm_quantiles))
                    q_norm_max = float(np.max(self._cluster_norm_quantiles))
                    q_norm_med = float(np.median(self._cluster_norm_quantiles))
                    logger.info(
                        "Normalized Conformal Regression (NCR) active. "
                        "Effective radius will scale with prediction interval width. "
                        "q_norm range: [%.6f, %.6f], median: %.6f. "
                        "width range: [%.6f, %.6f]",
                        q_norm_min,
                        q_norm_max,
                        q_norm_med,
                        self._width_min,
                        self._width_max,
                    )
                    
                    # Inform users if widths are nearly uniform (NCR won't be effective)
                    if width_relative_range < 0.01:  # Less than 1% variation
                        logger.info(
                            "Prediction intervals are nearly uniform (variation < 1%%). "
                            "NCR radius modulation will be minimal. "
                            "Consider configuring a difficulty estimator for adaptive intervals."
                        )
                else:
                    logger.info(
                        "Normalized Conformal Regression (NCR) initialized. "
                        "width range: [%.6f, %.6f]. "
                        "Effective radius will scale with prediction confidence.",
                        self._width_min,
                        self._width_max,
                    )
            except Exception:  # pylint: disable=broad-except
                logger.info(
                    "Normalized Conformal Regression (NCR) active. " "width_min=%s, width_max=%s",
                    self._width_min,
                    self._width_max,
                )

        self._fitted = True
        return self

    def set_alpha(
        self,
        alpha: float,
        *,
        per_cluster: bool = False,
        min_cluster_samples: int = 5,
    ) -> None:
        """Set a new alpha and recompute conformal radii from cached calibration scores.

        If per_cluster is True, attempt to compute a (1 - alpha) quantile per
        cluster using cached calibration-to-cluster assignments. Clusters with
        fewer than ``min_cluster_samples`` calibration points will fall back to
        the global quantile to avoid unstable estimates.

        Raises
        ------
        RuntimeError
            If the oracle has not been fitted or calibration scores are not cached.
        ValueError
            If alpha is not in (0, 1).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not getattr(self, "_fitted", False):
            raise RuntimeError("ConformalRegionOracle not fitted; set alpha after fit")
        if (
            not hasattr(self, "_cal_scores")
            or self._cal_scores is None
            or len(self._cal_scores) == 0
        ):
            raise RuntimeError("No cached calibration scores available; refit required")

        self.alpha = alpha

        # compute normalized global quantile (normalized conformal regression)
        # If calibration widths are cached, compute quantiles on normalized scores
        if hasattr(self, "_cal_widths") and self._cal_widths is not None:
            try:
                if len(self._cal_widths) == len(self._cal_scores):
                    s_cal_norm = self._cal_scores / (self._cal_widths + self._eps_width)
                    global_norm_q = float(np.quantile(s_cal_norm, 1.0 - self.alpha))
                else:
                    global_norm_q = None
            except Exception:  # pylint: disable=broad-except
                global_norm_q = None
        else:
            global_norm_q = None

        if not per_cluster:
            # If normalized quantile available, store it as per-cluster norm quantile
            if global_norm_q is not None:
                self._cluster_norm_quantiles = np.full(len(self._cluster_centers), global_norm_q)
            # Fallback: recompute unnormalized radii (legacy behavior)
            global_quantile = float(np.quantile(self._cal_scores, 1.0 - self.alpha))
            self._cluster_radii = np.full(len(self._cluster_centers), global_quantile)
            return

        # Per-cluster normalized quantiles with fallback for small clusters
        n_clusters_actual = len(self._cluster_centers)
        if global_norm_q is None:
            # Fallback to computing raw per-cluster radii
            radii = np.empty(n_clusters_actual, dtype=float)
            for k in range(n_clusters_actual):
                mask = self._cal_nearest == k
                scores_k = self._cal_scores[mask]
                if len(scores_k) >= min_cluster_samples:
                    radii[k] = float(np.quantile(scores_k, 1.0 - self.alpha))
                else:
                    radii[k] = float(np.quantile(self._cal_scores, 1.0 - self.alpha))

            self._cluster_radii = radii
            return

        # Compute per-cluster normalized quantiles
        norm_qs = np.empty(n_clusters_actual, dtype=float)
        for k in range(n_clusters_actual):
            mask = self._cal_nearest == k
            scores_k = self._cal_scores[mask]
            widths_k = self._cal_widths[mask]
            if (
                len(scores_k) >= min_cluster_samples
                and len(widths_k) == len(scores_k)
                and np.any(widths_k >= 0)
            ):
                try:
                    norm_scores_k = scores_k / (widths_k + self._eps_width)
                    norm_qs[k] = float(np.quantile(norm_scores_k, 1.0 - self.alpha))
                except Exception:  # pylint: disable=broad-except
                    norm_qs[k] = global_norm_q
            else:
                norm_qs[k] = global_norm_q

        self._cluster_norm_quantiles = norm_qs

    def accept(self, x_new, calibrated_prediction=None):
        """Check if perturbation is within conformal region.

        Computes Mahalanobis distance to nearest cluster center in the augmented
        feature space [x || pred] and checks against the modulated conformal radius.

        Parameters
        ----------
        x_new : array-like, shape (n_features,)
            Candidate perturbation instance (base features only).

        calibrated_prediction : tuple of (float, tuple) or None
            **MANDATORY** Calibrated prediction for the original instance.
            Format: (pred_value, (lower_bound, upper_bound))
            where pred_value is the prediction value and (lower_bound, upper_bound)
            are the uncertainty bounds from interval_learner.predict(uq_interval=True).
            Used to compute confidence-based radius modulation:
            - Extract width = upper - lower
            - Effective radius = q_norm * width (confidence-modulated)
            - Compare Mahalanobis distance to effective radius

        Returns
        -------
        bool
            True if x_new is within conformal region (accepted), False otherwise.

        Raises
        ------
        RuntimeError
            If oracle has not been fitted yet.
        ValueError
            If calibrated_prediction is None or malformed.

        Notes
        -----
        **Normalized Conformal Regression Implementation**

        1. Augment x_new with its prediction value: x_aug = [x_new || pred_value]
        2. Find nearest cluster center in augmented space
        3. Compute Mahalanobis distance: d = sqrt((x_aug - μ)^T Σ^-1 (x_aug - μ))
        4. Compute effective radius using normalized conformal regression:
           - If normalized quantiles available (from fit):
             r_eff = q_norm(cluster) * (upper - lower)
           - Otherwise use base radius (legacy fallback)
        5. Accept if: d ≤ r_eff

        **Coverage Guarantee**

        If interval_learner provides well-calibrated intervals, the conformal
        coverage guarantee (1 - α) is maintained:
        - At most α fraction of true in-distribution points will be rejected
        - Wider prediction intervals (uncertain regions) get larger radii
        - Narrower intervals (certain regions) get smaller radii

        **Usage Example**

        ```python
        # After fit(x_train, y_train, interval_learner):
        x_test = x_test_set[0]
        pred, (low, high) = interval_learner.predict(
            x_test.reshape(1, -1), uq_interval=True
        )
        calibrated_pred = (pred[0], (low[0], high[0]))
        is_accepted = oracle.accept(x_test, calibrated_prediction=calibrated_pred)
        ```
        """
        if not self._fitted:
            raise RuntimeError("ConformalRegionOracle not fitted. Call fit() first.")

        if calibrated_prediction is None:
            raise ValueError(
                "calibrated_prediction is mandatory. Format: (pred_value, (lower, upper))"
            )

        x_arr = check_array(x_new, accept_sparse=False, ensure_2d=False)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        x_point = x_arr[0]  # Take first row if multiple

        # Extract prediction value from calibrated_prediction
        try:
            pred_value, (_lower, _upper) = calibrated_prediction
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"calibrated_prediction must be (pred_value, (lower, upper)). Got {e}"
            ) from e

        # Augment feature space: [x || calibrated_prediction]
        x_point_augmented = np.concatenate([x_point, [pred_value]])

        # Find nearest cluster center in augmented space (always Euclidean)
        nearest_cluster_idx = self._find_nearest_cluster(x_point_augmented)

        # Compute nonconformity distance using selected metric
        mu_center = self._cluster_centers[nearest_cluster_idx]
        mahal_dist = self._compute_single_nonconformity_score(
            x_point_augmented, mu_center, nearest_cluster_idx
        )

        # Base conformal radius (legacy fallback)
        base_radius = self._cluster_radii[nearest_cluster_idx]

        # If we have normalized quantiles from calibration, use normalized
        # conformal regression: r_eff = q_norm(cluster) * width_test
        r_eff = base_radius
        if self._cluster_norm_quantiles is not None:
            try:
                _pred_value, (lower, upper) = calibrated_prediction
                width = float(upper - lower)
                width_safe = max(width, self._eps_width)
                q_norm = float(self._cluster_norm_quantiles[nearest_cluster_idx])
                r_eff = q_norm * width_safe
            except Exception as exc:  # pylint: disable=broad-except
                if self.enforcement:
                    raise
                logger.debug("Width modulation failed; using base radius: %s", exc)
                r_eff = base_radius
        elif self.enforcement:
            raise RuntimeError("Confidence modulation unavailable; calibrated prediction required")

        return mahal_dist <= r_eff

    def accept_batch(self, x_new_batch, calibrated_predictions):
        """Check multiple perturbations at once.

        Parameters
        ----------
        x_new_batch : array-like, shape (n_samples, n_features)
            Candidate perturbations.

        calibrated_predictions : list of tuples
            Calibrated predictions for each instance.
            Format: list of (pred_value, (lower, upper)) tuples.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Boolean array indicating acceptance.

        Raises
        ------
        ValueError
            If calibrated_predictions is None or not provided.
        """
        if calibrated_predictions is None:
            raise ValueError(
                "calibrated_predictions is mandatory. Expected list of "
                "(pred_value, (lower, upper)) tuples."
            )

        x_arr = check_array(x_new_batch, accept_sparse=False, ensure_2d=True)
        results = np.zeros(len(x_arr), dtype=bool)

        if len(calibrated_predictions) != len(x_arr):
            raise ValueError(
                f"calibrated_predictions length ({len(calibrated_predictions)}) "
                f"must match x_new_batch length ({len(x_arr)})."
            )

        for i, (x_new, cal_pred) in enumerate(zip(x_arr, calibrated_predictions)):
            if cal_pred is None:
                raise ValueError(
                    f"calibrated_prediction at index {i} is None. All predictions "
                    "must be provided."
                )
            results[i] = self.accept(x_new, cal_pred)

        return results

    def intervals(self, x_orig, calibrated_prediction=None):
        """Compute per-feature intervals for valid perturbations.

        For each feature j, computes the allowed interval [L_j, U_j] such that
        if all other features stay at x_orig, a perturbation in [L_j, U_j] for
        feature j remains within the conformal region.

        Uses confidence-modulated effective radius computed from the normalized
        quantiles and interval width (normalized conformal regression).

        Parameters
        ----------
        x_orig : array-like, shape (n_features,)
            Original instance (center point for intervals).

        calibrated_prediction : tuple or None
            **MANDATORY** Calibrated prediction for computing radius modulation.
            Format: (pred_value, (lower, upper))
            Required to compute effective radius = q_norm * (upper - lower).

        Returns
        -------
        list of list of tuple
            intervals[j] = list of (low, high) tuples defining allowed intervals
            for feature j, clipped to global bounds.
            If a feature is at the boundary of the conformal region, its
            interval may be empty (empty list).

        Raises
        ------
        RuntimeError
            If oracle has not been fitted yet.
        ValueError
            If calibrated_prediction is None or malformed.

        Notes
        -----
        **Computation Details**

        1. Augment x_orig with its prediction value: x_aug = [x_orig || pred_value]
        2. Find nearest cluster center in augmented space
        3. Compute effective radius: r_eff = q_norm(cluster) * (upper - lower)
        4. For each feature j:
           - Compute sum of squared Mahalanobis distances from other features
           - Solve for maximum allowed perturbation: delta_j
           - Return interval: [x_orig[j] - delta_j, x_orig[j] + delta_j]
           - Clip to global training bounds

        **Use Case**

        Intervals are useful for:
        - Generating valid perturbations within conformal bounds
        - Visualizing the acceptance region
        - Debugging specific feature contributions to acceptance
        """
        if not self._fitted:
            raise RuntimeError("ConformalRegionOracle not fitted. Call fit() first.")

        if calibrated_prediction is None:
            raise ValueError(
                "calibrated_prediction is mandatory. Format: (pred_value, (lower, upper))"
            )

        x_point = check_array(x_orig, accept_sparse=False, ensure_2d=False).ravel()
        n_features = len(x_point)

        # Extract prediction value from calibrated_prediction
        try:
            pred_value, (_lower, _upper) = calibrated_prediction
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"calibrated_prediction must be (pred_value, (lower, upper)). Got {exc}"
            ) from exc

        # Augment feature space: [x || calibrated_prediction]
        x_point_augmented = np.concatenate([x_point, [pred_value]])

        # Find nearest cluster in augmented space (always Euclidean)
        nearest_cluster_idx = self._find_nearest_cluster(x_point_augmented)

        mu_center = self._cluster_centers[nearest_cluster_idx]
        cov = self._cluster_covs[nearest_cluster_idx]
        base_radius = self._cluster_radii[nearest_cluster_idx]

        # Compute effective radius using normalized conformal regression if
        # normalized quantiles are available; otherwise fallback to base radius
        r_eff = base_radius
        if self._cluster_norm_quantiles is not None:
            try:
                _pred_value, (lower, upper) = calibrated_prediction
                width = float(upper - lower)
                width_safe = max(width, self._eps_width)
                q_norm = float(self._cluster_norm_quantiles[nearest_cluster_idx])
                r_eff = q_norm * width_safe
            except Exception as exc:  # pylint: disable=broad-except
                if self.enforcement:
                    raise
                logger.debug("Width modulation failed for intervals; using base radius: %s", exc)
                r_eff = base_radius
        elif self.enforcement:
            raise RuntimeError("Confidence modulation unavailable; calibrated prediction required")

        # Compute effective radius squared
        r_eff_sq = r_eff**2

        # Regularize covariance
        cov_reg = cov + 1e-6 * np.eye(cov.shape[0])

        try:
            cov_inv = pinvh(cov_reg)
        except Exception as exc:  # pylint: disable=broad-except
            if self.enforcement:
                raise
            logger.warning("Covariance inversion failed: %s. Using identity.", exc)
            # Use identity matrix of augmented space size
            cov_inv = np.eye(n_features + 1)

        # Extract standard deviations from augmented covariance
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-6))

        intervals = []
        # Compute intervals only for base features (exclude augmented prediction dimension)
        for j in range(n_features):
            # Compute sum of squared Mahalanobis distances from other features
            s_j = 0.0
            # Sum over all features and the augmented prediction dimension
            for i in range(n_features + 1):
                if i != j:
                    s_j += ((x_point_augmented[i] - mu_center[i]) ** 2) * cov_inv[i, i]

            # Remaining budget for feature j
            budget_j = r_eff_sq - s_j

            if budget_j < 0:
                # Feature is at boundary; no interval
                interval_j = []
            else:
                # Compute allowed interval
                delta_j = np.sqrt(budget_j) * stds[j]
                low = x_point[j] - delta_j
                high = x_point[j] + delta_j

                # Clip to global bounds (original feature space bounds)
                if self._global_mins is not None:
                    low = np.maximum(low, self._global_mins[j])
                if self._global_maxs is not None:
                    high = np.minimum(high, self._global_maxs[j])

                interval_j = [(low, high)] if low <= high else []

            intervals.append(interval_j)

        return intervals

    # =====================================================================
    # Private helpers
    # =====================================================================

    def _compute_distances(self, points, center):
        """Compute Euclidean distances from points to center.
        
        Parameters
        ----------
        points : np.ndarray, shape (n_points, n_features)
            Points to measure distances from.
        center : np.ndarray, shape (n_features,)
            Reference center point.
        
        Returns
        -------
        np.ndarray, shape (n_points,)
            Euclidean distance from each point to center.
        
        Notes
        -----
        Clustering always uses Euclidean distance (sklearn KMeans limitation).
        The nonconformity_metric affects only the nonconformity score computation,
        not cluster selection.
        """
        return np.linalg.norm(points - center, axis=1)

    def _find_nearest_cluster(self, point_augmented):
        """Find the index of nearest cluster center for augmented point.
        
        Parameters
        ----------
        point_augmented : np.ndarray, shape (n_features + 1,)
            Augmented point [x || prediction_value] in cluster space.
        
        Returns
        -------
        int
            Index of nearest cluster center.
        """
        distances = self._compute_distances(self._cluster_centers, point_augmented)
        return int(np.argmin(distances))

    def _compute_single_nonconformity_score(self, x_augmented, mu_center, cluster_idx):
        """Compute nonconformity score for a single instance using selected metric.

        Parameters
        ----------
        x_augmented : np.ndarray, shape (n_features + 1,)
            Augmented point [x || calibrated_prediction] in cluster space.
        mu_center : np.ndarray, shape (n_features + 1,)
            Center (mean) of the nearest cluster in augmented space.
        cluster_idx : int
            Index of the nearest cluster.

        Returns
        -------
        float
            Nonconformity score (distance from point to cluster center).

        Raises
        ------
        ValueError
            If metric is not recognized or computation fails with enforcement enabled.
        """
        if self._nonconformity_metric == "euclidean":
            return float(np.linalg.norm(x_augmented - mu_center))

        elif self._nonconformity_metric == "mahalanobis":
            cov = self._cluster_covs[cluster_idx]
            try:
                cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
                cov_inv = pinvh(cov_reg)
                mahal_dist = np.sqrt(
                    np.dot(
                        (x_augmented - mu_center),
                        np.dot(cov_inv, (x_augmented - mu_center).T),
                    )
                )
                return float(mahal_dist)
            except (ValueError, np.linalg.LinAlgError) as exc:
                if self.enforcement:
                    raise
                logger.debug("Mahalanobis fallback to Euclidean: %s", exc)
                return float(np.linalg.norm(x_augmented - mu_center))

        elif self._nonconformity_metric == "cosine":
            try:
                norm_aug = np.linalg.norm(x_augmented)
                norm_center = np.linalg.norm(mu_center)
                if norm_aug == 0 or norm_center == 0:
                    if self.enforcement:
                        raise ZeroDivisionError("Zero norm encountered in cosine distance")
                    logger.debug("Cosine fallback to Euclidean due to zero norm")
                    return float(np.linalg.norm(x_augmented - mu_center))
                cosine_sim = np.dot(x_augmented, mu_center) / (norm_aug * norm_center)
                # Clamp to [-1, 1] to handle numerical errors
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                return float(1.0 - cosine_sim)
            except (ValueError, ZeroDivisionError) as exc:
                if self.enforcement:
                    raise
                logger.debug("Cosine fallback to Euclidean: %s", exc)
                return float(np.linalg.norm(x_augmented - mu_center))

        else:
            raise ValueError(
                f"Unknown nonconformity_metric: {self._nonconformity_metric}. "
                f"Supported metrics: euclidean, mahalanobis, cosine"
            )

    def _compute_nonconformity_scores(self, x_arr, calibrated_predictions):
        """Compute nonconformity scores for instances in augmented feature space.

        Computes the Mahalanobis distance from each instance (augmented with its
        calibrated prediction) to its nearest cluster center, in the augmented space.

        Parameters
        ----------
        x_arr : np.ndarray, shape (n_samples, n_features)
            Feature instances (base features only; predictions will be concatenated).

        calibrated_predictions : array-like, shape (n_samples,)
            Calibrated prediction values for each instance. These are the values to
            concatenate with features to form the augmented space.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Nonconformity scores using the selected distance metric.
        """
        scores = []
        for idx, x_point in enumerate(x_arr):
            pred_value = calibrated_predictions[idx]

            # Augment feature space: [x || calibrated_prediction]
            x_augmented = np.concatenate([x_point, [pred_value]])

            # Find nearest cluster in augmented space (always Euclidean)
            nearest_idx = self._find_nearest_cluster(x_augmented)

            # Nonconformity distance using selected metric
            mu_center = self._cluster_centers[nearest_idx]
            score = self._compute_single_nonconformity_score(
                x_augmented, mu_center, nearest_idx
            )
            scores.append(score)

        return np.array(scores)

    def _compute_effective_radius(self, base_radius, calibrated_prediction):
        """Compute effective radius using normalized conformal regression.

        If normalized calibration quantiles are available (computed during
        fit), this method returns r_eff = q_norm_global * width_test where
        q_norm_global is a robust aggregate (median) of per-cluster normalized
        quantiles. If normalized quantiles are not available, falls back to
        returning the provided base_radius.

        Parameters
        ----------
        base_radius : float
            Legacy base radius (unused when normalized quantiles available).

        calibrated_prediction : tuple or None
            (pred_value, (lower, upper)) or None.

        Returns
        -------
        float
            Effective radius.
        """
        if calibrated_prediction is None or self._cluster_norm_quantiles is None:
            if self.enforcement:
                raise RuntimeError(
                    "Confidence modulation requires calibrated_prediction and quantiles"
                )
            return base_radius

        try:
            _, (lower, upper) = calibrated_prediction
            width = float(upper - lower)
            width_safe = max(width, self._eps_width)
        except Exception as exc:  # pylint: disable=broad-except
            if self.enforcement:
                raise
            logger.debug("Effective radius modulation failed; using base radius: %s", exc)
            return base_radius

        # Use a central value (median) of the per-cluster normalized quantiles
        try:
            q_norm_global = float(np.median(self._cluster_norm_quantiles))
        except Exception as exc:  # pylint: disable=broad-except
            if self.enforcement:
                raise
            logger.debug("Quantile aggregation failed; using base radius: %s", exc)
            return base_radius

        r_eff = q_norm_global * width_safe

        logger.debug(
            "_compute_effective_radius(ncrm): q_norm=%s, width=%s, r_eff=%s",
            q_norm_global,
            width_safe,
            r_eff,
        )

        return r_eff
