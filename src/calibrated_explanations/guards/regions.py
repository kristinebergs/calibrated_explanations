"""Conformal Region Oracle for perturbation guards."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.utils import check_array

from .intervals import union_intervals


class ConformalRegionOracle:
    """Conformal Region Oracle for filtering out-of-distribution perturbations."""

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

        self._fitted = False
        self._clusters = {}
        self._radii = {}
        self._trees = {}
        self._variances = {}
        self._martingale = None
        # Global feature-wise bounds computed at fit-time (min, max)
        self._global_mins = None
        self._global_maxs = None

    def fit(self, xs, ys):
        """Build label-conditional cluster regions and calibrate radii."""
        x = check_array(xs)
        y = np.asarray(ys)

        # Store global per-feature bounds to clip intervals later (G-05)
        try:
            self._global_mins = np.min(x, axis=0)
            self._global_maxs = np.max(x, axis=0)
        except Exception:
            self._global_mins = None
            self._global_maxs = None

        if self.mode == "clf":
            labels = np.unique(y)
        elif self.mode == "reg":
            if self.threshold is None:
                raise ValueError("Threshold must be provided for regression mode")
            labels = np.array([0, 1])
            y = (y >= self.threshold).astype(int)
        else:
            raise ValueError("Mode must be 'clf' or 'reg'")

        for label in labels:
            mask = y == label
            x_label = x[mask]

            if len(x_label) < self.n_clusters:
                # Not enough data, skip or use all as one cluster
                continue

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            clusters = kmeans.fit_predict(x_label)
            centers = kmeans.cluster_centers_

            # Compute variances per cluster
            variances = []
            for c in range(self.n_clusters):
                cluster_data = x_label[clusters == c]
                if len(cluster_data) > 1:
                    var = np.var(cluster_data, axis=0, ddof=1)
                else:
                    var = np.zeros(x.shape[1])
                variances.append(var)
            variances = np.array(variances)

            # Compute nonconformity scores
            scores = []
            for i, xi in enumerate(x_label):
                c = clusters[i]
                center = centers[c]
                var = variances[c]
                mahal = np.sum(((xi - center) ** 2) / (var + 1e-8))
                scores.append(mahal)

            scores = np.array(scores)

            # Split conformal calibration
            n = len(scores)
            cal_size = n // 2
            cal_scores = scores[:cal_size]
            radius = np.quantile(cal_scores, 1 - self.alpha)

            self._clusters[label] = centers
            self._radii[label] = radius
            self._variances[label] = variances
            self._trees[label] = KDTree(centers)

        if self.use_martingale:
            from .martingale import MartingaleETest

            self._martingale = MartingaleETest(
                k=self.e_knn, n_neighbors=self.e_neigh, gamma=self.e_gamma
            )
            self._martingale.fit(x)

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

        # Find nearest cluster
        tree = self._trees[label_ctx]
        dist, idx = tree.query(x_prime.reshape(1, -1), k=1)
        c = idx[0][0]
        center = centers[c]
        var = variances[c]

        mahal = np.sum(((x_prime - center) ** 2) / (var + 1e-8))
        if mahal > radius:
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
