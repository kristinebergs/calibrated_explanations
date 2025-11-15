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

    ncm_method : str, default="mahalanobis"
        Nonconformity measure method.
        Options: "mahalanobis" (Mahalanobis distance to cluster center)

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
        ncm_method="mahalanobis",
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if not 0 < prop_size <= 1:
            raise ValueError(f"prop_size must be in (0, 1], got {prop_size}")
        self.alpha = alpha
        self.n_clusters = n_clusters
        # Bounds for width-based modulation (clamped)
        self._modulation_min = 0.5
        self._modulation_max = 2.0
        self.prop_size = prop_size
        self.random_state = random_state
        self.ncm_method = ncm_method

        self._fitted = False
        self._cluster_centers = None
        self._cluster_covs = None
        self._cluster_radii = None
        self._width_min = None
        self._width_max = None
        self._global_mins = None
        self._global_maxs = None
        self._kmeans = None
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
        """Fit the conformal region oracle.

        Performs inductive conformal prediction:
        1. Split x into proper (75%) and calibration (25%) sets
        2. Cluster the proper set in feature space
        3. Compute per-cluster covariance and Mahalanobis distances on proper set
        4. Compute conformal radii on calibration set
        5. Record width statistics for confidence modulation

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training instances. Used to define conformal regions.

        y : array-like, shape (n_samples,)
            Training targets. Not used directly but kept for interface consistency.

        interval_learner : fitted calibrator
            Fitted interval learner (e.g., from CalibratedExplainer).
            Provides (L, U) intervals for confidence modulation.
            Currently not used; will be called during accept().

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If training data is too small or malformed.
        """
        x_arr = check_array(x, accept_sparse=False, ensure_2d=True)
        _ = np.asarray(y)

        # Require an interval learner: the oracle relies on interval widths
        # for normalized conformal regression and confidence modulation.
        if interval_learner is None:
            raise ValueError("interval_learner must be provided; None is not allowed")

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

        # Cluster on proper set
        n_clusters_actual = min(self.n_clusters, len(x_proper))
        self._kmeans = KMeans(
            n_clusters=n_clusters_actual,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(x_proper)
        self._cluster_centers = self._kmeans.cluster_centers_

        # Compute per-cluster covariance on proper set
        self._cluster_covs = []
        for k in range(n_clusters_actual):
            mask = self._kmeans.labels_ == k
            if np.sum(mask) > 1:
                cov = np.cov(x_proper[mask].T)
                # Handle 1D covariance
                if cov.ndim == 0:
                    cov = np.array([[cov]])
                elif cov.ndim == 1:
                    cov = np.diag(cov)
            else:
                # Single point in cluster; use identity
                cov = np.eye(n_features)
            self._cluster_covs.append(cov)

        # Compute nonconformity scores on calibration set
        cal_scores = self._compute_nonconformity_scores(x_cal)

        # Cache calibration scores and assignments so callers can recompute
        # radii for new alpha values without a full refit.
        try:
            # assign calibration points to nearest clusters
            dists = np.linalg.norm(x_cal[:, None, :] - self._cluster_centers[None, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)
        except Exception:  # pylint: disable=broad-except
            nearest = np.zeros(len(cal_scores), dtype=int)

        # store cached calibration diagnostics for later recomputation
        self._cal_scores = np.asarray(cal_scores)
        self._cal_nearest = np.asarray(nearest)

        # --- Normalized conformal regression: use interval width as difficulty ---
        # Compute calibration widths for x_cal (if interval_learner provided)
        if interval_learner is not None:
            try:
                intervals_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
                if intervals_cal is not None and len(intervals_cal) == len(x_cal):
                    widths_cal = np.array([upper - lower for lower, upper in intervals_cal])
                else:
                    widths_cal = np.ones(len(cal_scores))
            except Exception:  # pylint: disable=broad-except
                widths_cal = np.ones(len(cal_scores))
        else:
            widths_cal = np.ones(len(cal_scores))

        # Cache widths for later recomputation in set_alpha
        self._cal_widths = widths_cal

        # Compute normalized calibration scores and store per-cluster normalized quantile
        try:
            s_cal_norm = self._cal_scores / (widths_cal + self._eps_width)
            global_norm_q = float(np.quantile(s_cal_norm, 1.0 - self.alpha))
            self._cluster_norm_quantiles = np.full(n_clusters_actual, global_norm_q)
        except Exception:  # pylint: disable=broad-except
            # Fallback: if normalization fails, leave cluster_norm_quantiles unset
            self._cluster_norm_quantiles = None

        # Compute conformal radius as (1 - alpha) quantile (global by default)
        quantile_idx = int(np.ceil((1 - self.alpha) * len(cal_scores)))
        quantile_idx = min(quantile_idx, len(cal_scores) - 1)
        global_radius = np.sort(cal_scores)[quantile_idx]
        self._cluster_radii = np.full(n_clusters_actual, global_radius)

        try:
            # Report calibration diagnostics to help debugging (counts per cluster)
            # Assign calibration points to nearest clusters
            if len(x_cal) > 0 and self._cluster_centers is not None:
                # distances shape: (n_cal, n_clusters)
                dists = np.linalg.norm(
                    x_cal[:, None, :] - self._cluster_centers[None, :, :], axis=2
                )
                nearest = np.argmin(dists, axis=1)
                counts = np.bincount(nearest, minlength=n_clusters_actual)
            else:
                counts = np.zeros(n_clusters_actual, dtype=int)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Could not compute per-cluster calibration counts")
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
        if interval_learner is not None:
            try:
                intervals = interval_learner.predict(x)
                if intervals is not None and len(intervals) > 0:
                    widths = np.array([upper - lower for lower, upper in intervals])
                    self._width_min = np.min(widths)
                    self._width_max = np.max(widths)
                    if self._width_max < self._width_min:
                        # Safety check: should not happen, but swap if it does
                        self._width_min, self._width_max = self._width_max, self._width_min
                else:
                    self._width_min = 0.0
                    self._width_max = 1.0
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Could not compute width statistics from interval_learner: %s",
                    exc,
                )
                self._width_min = 0.0
                self._width_max = 1.0
        else:
            self._width_min = 0.0
            self._width_max = 1.0

        # Inform whether confidence modulation will be active
        if self._width_max <= self._width_min:
            logger.info(
                "Confidence modulation disabled: width_min=%s, width_max=%s",
                self._width_min,
                self._width_max,
            )
        else:
            logger.info(
                "Confidence modulation active: width_min=%s, width_max=%s",
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

        Computes Mahalanobis distance to nearest cluster center and checks
        against modulated conformal radius.

        Parameters
        ----------
        x_new : array-like, shape (n_features,)
            Candidate perturbation instance.

        calibrated_prediction : tuple of (float, tuple) or None, optional
            Calibrated prediction for the original instance.
            Format: (pred_value, (lower_bound, upper_bound))
            If provided, used to compute confidence-based radius modulation.

        Returns
        -------
        bool
            True if x_new is within conformal region, False otherwise.

        Raises
        ------
        NotFittedError
            If oracle has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("ConformalRegionOracle not fitted. Call fit() first.")

        x_arr = check_array(x_new, accept_sparse=False, ensure_2d=False)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        x_point = x_arr[0]  # Take first row if multiple

        # Find nearest cluster center
        distances_to_centers = np.linalg.norm(self._cluster_centers - x_point, axis=1)
        nearest_cluster_idx = np.argmin(distances_to_centers)

        # Compute Mahalanobis distance to nearest cluster center
        mu_center = self._cluster_centers[nearest_cluster_idx]
        cov = self._cluster_covs[nearest_cluster_idx]

        try:
            # Regularize covariance if needed (for numerical stability)
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            cov_inv = pinvh(cov_reg)
            mahal_dist = np.sqrt(
                np.dot(
                    (x_point - mu_center),
                    np.dot(cov_inv, (x_point - mu_center).T),
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Mahalanobis distance computation failed: %s. Using Euclidean.",
                exc,
            )
            mahal_dist = np.linalg.norm(x_point - mu_center)

        # Base conformal radius (legacy fallback)
        base_radius = self._cluster_radii[nearest_cluster_idx]

        # If we have normalized quantiles from calibration, use normalized
        # conformal regression: r_eff = q_norm(cluster) * width_test
        r_eff = base_radius
        if calibrated_prediction is not None and self._cluster_norm_quantiles is not None:
            try:
                _, (lower, upper) = calibrated_prediction
                width = float(upper - lower)
                width_safe = max(width, self._eps_width)
                q_norm = float(self._cluster_norm_quantiles[nearest_cluster_idx])
                r_eff = q_norm * width_safe
            except Exception:  # pylint: disable=broad-except
                # Fall back to legacy base radius
                r_eff = base_radius

        return mahal_dist <= r_eff

    def accept_batch(self, x_new_batch, calibrated_predictions=None):
        """Check multiple perturbations at once.

        Parameters
        ----------
        x_new_batch : array-like, shape (n_samples, n_features)
            Candidate perturbations.

        calibrated_predictions : list of tuples or None, optional
            Calibrated predictions for each instance.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Boolean array indicating acceptance.
        """
        x_arr = check_array(x_new_batch, accept_sparse=False, ensure_2d=True)
        results = np.zeros(len(x_arr), dtype=bool)

        if calibrated_predictions is None:
            calibrated_predictions = [None] * len(x_arr)

        for i, (x_new, cal_pred) in enumerate(zip(x_arr, calibrated_predictions)):
            results[i] = self.accept(x_new, cal_pred)

        return results

    def intervals(self, x_orig, calibrated_prediction=None):
        """Compute per-feature intervals for valid perturbations.

        For each feature j, computes the allowed interval [L_j, U_j] such that
        if all other features stay at x_orig, a perturbation in [L_j, U_j] for
        feature j remains within the conformal region.

        Parameters
        ----------
        x_orig : array-like, shape (n_features,)
            Original instance (center point for intervals).

        calibrated_prediction : tuple or None, optional
            Calibrated prediction for computing modulation.

        Returns
        -------
        list of list of tuple
            intervals[j] = list of (low, high) tuples defining allowed intervals
            for feature j, clipped to global bounds.
        """
        if not self._fitted:
            raise RuntimeError("ConformalRegionOracle not fitted. Call fit() first.")

        x_point = check_array(x_orig, accept_sparse=False, ensure_2d=False).ravel()
        n_features = len(x_point)

        # Find nearest cluster
        distances_to_centers = np.linalg.norm(self._cluster_centers - x_point, axis=1)
        nearest_cluster_idx = np.argmin(distances_to_centers)

        mu_center = self._cluster_centers[nearest_cluster_idx]
        cov = self._cluster_covs[nearest_cluster_idx]
        base_radius = self._cluster_radii[nearest_cluster_idx]

        # Compute effective radius using normalized conformal regression if
        # normalized quantiles are available; otherwise fallback to base radius
        r_eff = base_radius
        if calibrated_prediction is not None and self._cluster_norm_quantiles is not None:
            try:
                _, (lower, upper) = calibrated_prediction
                width = float(upper - lower)
                width_safe = max(width, self._eps_width)
                q_norm = float(self._cluster_norm_quantiles[nearest_cluster_idx])
                r_eff = q_norm * width_safe
            except Exception:  # pylint: disable=broad-except
                r_eff = base_radius

        # Compute effective radius squared
        r_eff_sq = r_eff**2

        # Regularize covariance
        cov_reg = cov + 1e-6 * np.eye(cov.shape[0])

        try:
            cov_inv = pinvh(cov_reg)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Covariance inversion failed: %s. Using identity.", exc)
            cov_inv = np.eye(n_features)

        # Extract standard deviations
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-6))

        intervals = []
        for j in range(n_features):
            # Compute sum of squared Mahalanobis distances from other features
            s_j = 0.0
            for i in range(n_features):
                if i != j:
                    s_j += ((x_point[i] - mu_center[i]) ** 2) * cov_inv[i, i]

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

                # Clip to global bounds
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

    def _compute_nonconformity_scores(self, x_arr):
        """Compute nonconformity scores for instances.

        For each instance, compute Mahalanobis distance to its nearest
        cluster center.

        Parameters
        ----------
        x_arr : np.ndarray, shape (n_samples, n_features)
            Instances.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Nonconformity scores (distances).
        """
        scores = []
        for x_point in x_arr:
            # Find nearest cluster
            distances = np.linalg.norm(self._cluster_centers - x_point, axis=1)
            nearest_idx = np.argmin(distances)

            # Mahalanobis distance to nearest cluster
            mu_center = self._cluster_centers[nearest_idx]
            cov = self._cluster_covs[nearest_idx]

            try:
                cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
                cov_inv = pinvh(cov_reg)
                mahal_dist = np.sqrt(
                    np.dot(
                        (x_point - mu_center),
                        np.dot(cov_inv, (x_point - mu_center).T),
                    )
                )
            except Exception:  # pylint: disable=broad-except
                mahal_dist = np.linalg.norm(x_point - mu_center)

            scores.append(mahal_dist)

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
            return base_radius

        try:
            _, (lower, upper) = calibrated_prediction
            width = float(upper - lower)
            width_safe = max(width, self._eps_width)
        except Exception:  # pylint: disable=broad-except
            return base_radius

        # Use a central value (median) of the per-cluster normalized quantiles
        try:
            q_norm_global = float(np.median(self._cluster_norm_quantiles))
        except Exception:  # pylint: disable=broad-except
            return base_radius

        r_eff = q_norm_global * width_safe

        logger.debug(
            "_compute_effective_radius(ncrm): q_norm=%s, width=%s, r_eff=%s",
            q_norm_global,
            width_safe,
            r_eff,
        )

        return r_eff
