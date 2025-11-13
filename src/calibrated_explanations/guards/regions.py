"""Conformal Region Oracle for perturbation guards."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.utils import check_array

from .intervals import union_intervals


class ConformalRegionOracle:
    """Conformal Region Oracle for filtering out-of-distribution perturbations.

    This guard uses Inductive Conformal Prediction (ICP) when prop_size < 1.0,
    splitting the training data into a proper set (for model training) and a
    calibration set (for nonconformity score computation). When prop_size = 1.0,
    it falls back to the original split-conformal approach.

    Supports two nonconformity measures:
    - 'mahalanobis': Mahalanobis distance to nearest cluster center
    - 'knn': Sum of distances to k nearest neighbors

    When epsilon is provided, uses p-value based acceptance with significance level epsilon.
    """

    def __init__(
        self,
        alpha=0.1,
        mode="clf",
        threshold=None,
        n_clusters=5,
        covariance="diag",
        random_state=None,
        use_martingale=False,
        e_gamma=10.0,
        e_knn=30,
        e_neigh=500,
        prop_size=0.5,
        ncm_method="mahalanobis",
        k=5,
        epsilon=None,
    ):
        self.alpha = alpha
        self.mode = mode
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.covariance = covariance
        self.random_state = random_state
        self.use_martingale = use_martingale
        self.e_gamma = e_gamma
        self.e_knn = e_knn
        self.e_neigh = e_neigh
        self.prop_size = prop_size
        self.ncm_method = ncm_method
        self.k = k
        self.epsilon = epsilon

        self._fitted = False
        self._clusters = {}
        self._radii = {}
        self._trees = {}
        self._variances = {}
        self._martingale = None
        # Global feature-wise bounds computed at fit-time (min, max)
        self._global_mins = None
        self._global_maxs = None
        # For ICP: calibration scores for p-value computation
        self._cal_scores = {}
        # For k-NN: nearest neighbors models
        self._nn_models = {}

    def fit(self, xs, ys, x_cal=None, y_cal=None, prop_size=None):
        """Build label-conditional cluster regions and calibrate radii.

        When prop_size < 1.0, uses Inductive Conformal Prediction by splitting
        data into proper set (for training) and calibration set (for scores).
        When prop_size = 1.0 or None, uses the original split-conformal approach.

        Parameters
        ----------
        xs : array-like
            Training instances.
        ys : array-like
            Training labels.
        x_cal : array-like, optional
            Calibration instances for ICP. If provided, used instead of splitting xs.
        y_cal : array-like, optional
            Calibration labels for ICP. If provided, used instead of splitting ys.
        prop_size : float, optional
            Proportion of data to use for proper training set. If None, uses self.prop_size.
            Should be in (0, 1] for ICP, or 1.0 for original behavior. If x_cal and y_cal
            are provided, this parameter is ignored.
        """
        x = check_array(xs)
        y = np.asarray(ys)
        if x_cal is not None and y_cal is not None:
            x_prop = x
            y_prop = y
            x_calib = check_array(x_cal)
            y_calib = np.asarray(y_cal)
        else:
            # Use provided prop_size or default to self.prop_size
            prop_size = prop_size if prop_size is not None else self.prop_size

            # Split data into proper and calibration sets for ICP
            n_samples = len(x)
            n_prop = int(prop_size * n_samples)
            indices = np.random.RandomState(self.random_state).permutation(n_samples)
            prop_indices = indices[:n_prop]
            calib_indices = indices[n_prop:]

            x_prop = x[prop_indices]
            y_prop = y[prop_indices]
            x_calib = x[calib_indices]
            y_calib = y[calib_indices]

        # Store global per-feature bounds to clip intervals later (G-05)
        try:
            self._global_mins = np.min(x_prop, axis=0)
            self._global_maxs = np.max(x_prop, axis=0)
        except Exception:
            self._global_mins = None
            self._global_maxs = None

        if self.mode == "clf":
            labels = np.unique(y_prop)
        elif self.mode == "reg":
            if self.threshold is None:
                raise ValueError("Threshold must be provided for regression mode")
            labels = np.array([0, 1])
            y_prop = (y_prop >= self.threshold).astype(int)
            y_calib = (y_calib >= self.threshold).astype(int)
        else:
            raise ValueError("Mode must be 'clf' or 'reg'")

        for label in labels:
            # Train on proper set
            mask_prop = y_prop == label
            x_label_prop = x_prop[mask_prop]

            if len(x_label_prop) < self.n_clusters:
                # Not enough data, skip or use all as one cluster
                continue

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            clusters = kmeans.fit_predict(x_label_prop)
            centers = kmeans.cluster_centers_

            # Compute variances per cluster on proper set
            variances = []
            for cluster_idx in range(self.n_clusters):
                cluster_data = x_label_prop[clusters == cluster_idx]
                if len(cluster_data) > 1:
                    var = np.var(cluster_data, axis=0, ddof=1)
                else:
                    var = np.zeros(x_prop.shape[1])
                variances.append(var)
            variances = np.array(variances)

            # Compute nonconformity scores on calibration set
            mask_calib = y_calib == label
            x_label_calib = x_calib[mask_calib]

            if len(x_label_calib) == 0:
                # No calibration data for this label, use a large radius
                radius = np.inf
                cal_scores = np.array([])
            else:
                if self.ncm_method == "mahalanobis":
                    # Mahalanobis distance to nearest cluster
                    scores = []
                    for xi in x_label_calib:
                        # Find nearest cluster
                        tree_temp = KDTree(centers)
                        dist, idx = tree_temp.query(xi.reshape(1, -1), k=1)
                        cluster_idx = idx[0][0]
                        center = centers[cluster_idx]
                        var = variances[cluster_idx]
                        mahal = np.sum(((xi - center) ** 2) / (var + 1e-8))
                        scores.append(mahal)
                    cal_scores = np.array(scores)
                    radius = np.quantile(cal_scores, 1 - self.alpha)

                elif self.ncm_method == "knn":
                    # k-NN distance sum
                    from sklearn.neighbors import NearestNeighbors
                    nn_model = NearestNeighbors(n_neighbors=self.k)
                    nn_model.fit(x_label_prop)
                    distances, _ = nn_model.kneighbors(x_label_calib)
                    cal_scores = np.sum(distances, axis=1)
                    radius = np.quantile(cal_scores, 1 - self.alpha)
                    self._nn_models[label] = nn_model
                else:
                    raise ValueError("ncm_method must be 'mahalanobis' or 'knn'")

            self._clusters[label] = centers
            self._radii[label] = radius
            self._variances[label] = variances
            self._trees[label] = KDTree(centers)
            self._cal_scores[label] = cal_scores

        if self.use_martingale:
            from .martingale import MartingaleETest

            self._martingale = MartingaleETest(
                k=self.e_knn, n_neighbors=self.e_neigh, gamma=self.e_gamma
            )
            self._martingale.fit(x_prop)

        self._fitted = True
        return self

    def label_context(self, x_instance, *, clf_predict_proba=None, reg_predict=None):
        """Return the label-conditional context for x_instance."""
        if self.mode == "clf":
            if clf_predict_proba is None:
                raise ValueError("clf_predict_proba required for classification")
            proba = clf_predict_proba(x_instance.reshape(1, -1))
            return np.argmax(proba)
        elif self.mode == "reg":
            if reg_predict is None:
                raise ValueError("reg_predict required for regression")
            pred = reg_predict(x_instance.reshape(1, -1))
            return int(pred >= self.threshold)
        else:
            raise ValueError("Invalid mode")

    def intervals(self, x_instance, label_ctx):
        """Return per-feature allowed 1D intervals for x_instance under label_ctx."""
        if not self._fitted:
            raise ValueError("Guard not fitted")

        if label_ctx not in self._clusters:
            # No data for this label, return empty intervals
            return [[] for _ in range(len(x_instance))]

        centers = self._clusters[label_ctx]
        variances = self._variances[label_ctx]
        radius = self._radii[label_ctx]

        intervals = []

        # Precompute per-cluster denominators (var + eps) for numerical stability
        eps = 1e-8
        n_features = len(x_instance)

        # For each cluster, compute S_total = sum_i ((x_i - mu_i)^2 / denom_i)
        # We'll reuse S_total and denom for each feature to avoid O(d^2) work.
        per_cluster_s = []
        per_cluster_centers = []
        per_cluster_denoms = []
        for c in range(len(centers)):
            center = centers[c]
            var = variances[c]
            denom = var + eps
            # Use numpy vectorized operations
            s_total = float(np.sum(((x_instance - center) ** 2) / denom))
            per_cluster_s.append(s_total)
            per_cluster_centers.append(center)
            per_cluster_denoms.append(denom)

        # For each feature, compute intervals by subtracting the j-th contribution
        for j in range(n_features):
            abs_intervals = []
            for c, center in enumerate(per_cluster_centers):
                denom = per_cluster_denoms[c]
                mu_j = center[j]
                # s = S_total - contribution from feature j
                s = per_cluster_s[c] - (((x_instance[j] - mu_j) ** 2) / denom[j])

                if radius < s:
                    continue  # No interval from this cluster

                # sigma_j^2 is denom[j]
                d = (radius - s) * denom[j]
                if d < 0:
                    continue

                delta = np.sqrt(d)
                low = mu_j - delta
                high = mu_j + delta

                # Intersect with known global feature domain (if available)
                if self._global_mins is not None and self._global_maxs is not None:
                    low = max(low, float(self._global_mins[j]))
                    high = min(high, float(self._global_maxs[j]))
                    if low > high:
                        continue

                abs_intervals.append((low, high))

            # Merge overlapping absolute intervals to reduce sampling overhead (G-06)
            merged = union_intervals(abs_intervals) if abs_intervals else []

            # Convert merged absolute intervals into relative intervals around x_instance[j]
            feature_intervals = [
                (low - x_instance[j], high - x_instance[j]) for (low, high) in merged
            ]
            intervals.append(feature_intervals)

        return intervals

    def accept(self, x_prime, label_ctx):
        """Return True if x_prime is inside a calibrated region; apply the e-test if enabled.

        Uses the configured nonconformity measure (mahalanobis or knn) to compute
        the score for x_prime and compares against the calibrated radius.

        If epsilon is set, additionally requires p-value >= epsilon for acceptance.

        The method returns a boolean indicating whether the provided point
        ``x_prime`` lies inside the calibrated region for ``label_ctx``. If an
        e-test (martingale) has been attached to the oracle it is consulted and
        may cause the point to be rejected.
        """
        if not self._fitted:
            return True  # If not fitted, accept

        if label_ctx not in self._clusters:
            return False

        centers = self._clusters[label_ctx]
        variances = self._variances[label_ctx]
        radius = self._radii[label_ctx]

        if self.ncm_method == "mahalanobis":
            # Find nearest cluster
            tree = self._trees[label_ctx]
            _, idx = tree.query(x_prime.reshape(1, -1), k=1)
            cluster_idx = idx[0][0]
            center = centers[cluster_idx]
            var = variances[cluster_idx]
            score = np.sum(((x_prime - center) ** 2) / (var + 1e-8))
        elif self.ncm_method == "knn":
            # Compute k-NN distance sum
            nn_model = self._nn_models[label_ctx]
            distances, _ = nn_model.kneighbors(x_prime.reshape(1, -1))
            score = np.sum(distances)
        else:
            raise ValueError("ncm_method must be 'mahalanobis' or 'knn'")

        # Check against radius
        if score > radius:
            return False

        # If epsilon is provided, use p-value based acceptance
        if self.epsilon is not None:
            p_val = self.pvalue(x_prime, label_ctx)
            if p_val < self.epsilon:
                return False

        # If a martingale e-test has been attached (either created in `fit`
        # or injected manually in tests), consult it. This preserves the
        # previous behavior where presence of `_martingale` governs whether
        # the e-test is applied. The `use_martingale` flag controls whether
        # `fit()` creates the `_martingale` by default, but we allow manual
        # injection for tests and advanced usage.
        if self._martingale is not None:
            if self._martingale.reject(x_prime):
                return False

        return True

    def pvalue(self, x_candidate, label_ctx):
        """Compute the p-value of x_candidate given calibration scores for label_ctx."""
        if not self._fitted:
            raise ValueError("Guard not fitted")

        if label_ctx not in self._cal_scores:
            return 0.0  # No calibration data, reject

        cal_scores = self._cal_scores[label_ctx]
        if len(cal_scores) == 0:
            return 0.0

        # Compute the score for the candidate
        if self.ncm_method == "mahalanobis":
            centers = self._clusters[label_ctx]
            variances = self._variances[label_ctx]
            tree = self._trees[label_ctx]
            _, idx = tree.query(x_candidate.reshape(1, -1), k=1)
            cluster_idx = idx[0][0]
            center = centers[cluster_idx]
            var = variances[cluster_idx]
            candidate_score = np.sum(((x_candidate - center) ** 2) / (var + 1e-8))
        elif self.ncm_method == "knn":
            nn_model = self._nn_models[label_ctx]
            distances, _ = nn_model.kneighbors(x_candidate.reshape(1, -1))
            candidate_score = np.sum(distances)
        else:
            raise ValueError("ncm_method must be 'mahalanobis' or 'knn'")

        # Compute p-value: (number of calibration scores >= candidate_score + 1) / (n_cal + 1)
        n_cal = len(cal_scores)
        rank = np.sum(cal_scores >= candidate_score) + 1
        return rank / (n_cal + 1)
