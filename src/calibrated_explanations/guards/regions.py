"""Conformal Region Oracle for perturbation guards."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from typing import Optional, List, Tuple, Union
from sklearn.utils import check_array


class ConformalRegionOracle:
    """Conformal Region Oracle for filtering out-of-distribution perturbations."""

    def __init__(self, alpha=0.1, mode="clf", threshold=None,
                 n_clusters=5, covariance="diag", random_state=None,
                 use_martingale=False, e_gamma=10.0, e_knn=30, e_neigh=500):
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

    def fit(self, X, y):
        """Build label-conditional cluster regions and calibrate radii."""
        X = check_array(X)
        y = np.asarray(y)

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
            X_label = X[mask]

            if len(X_label) < self.n_clusters:
                # Not enough data, skip or use all as one cluster
                continue

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            clusters = kmeans.fit_predict(X_label)
            centers = kmeans.cluster_centers_

            # Compute variances per cluster
            variances = []
            for c in range(self.n_clusters):
                cluster_data = X_label[clusters == c]
                if len(cluster_data) > 1:
                    var = np.var(cluster_data, axis=0, ddof=1)
                else:
                    var = np.zeros(X.shape[1])
                variances.append(var)
            variances = np.array(variances)

            # Compute nonconformity scores
            scores = []
            for i, x in enumerate(X_label):
                c = clusters[i]
                center = centers[c]
                var = variances[c]
                mahal = np.sum(((x - center) ** 2) / (var + 1e-8))
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
            self._martingale = MartingaleETest(k=self.e_knn, n_neighbors=self.e_neigh, gamma=self.e_gamma)
            self._martingale.fit(X)

        self._fitted = True
        return self

    def label_context(self, x, *, clf_predict_proba=None, reg_predict=None):
        """Return the label-conditional context for x."""
        if self.mode == "clf":
            if clf_predict_proba is None:
                raise ValueError("clf_predict_proba required for classification")
            proba = clf_predict_proba(x.reshape(1, -1))
            return np.argmax(proba)
        elif self.mode == "reg":
            if reg_predict is None:
                raise ValueError("reg_predict required for regression")
            pred = reg_predict(x.reshape(1, -1))
            return int(pred >= self.threshold)
        else:
            raise ValueError("Invalid mode")

    def intervals(self, x, label_ctx):
        """Return per-feature allowed 1D intervals for x under label_ctx."""
        if not self._fitted:
            raise ValueError("Guard not fitted")

        if label_ctx not in self._clusters:
            # No data for this label, return empty intervals
            return [[] for _ in range(len(x))]

        centers = self._clusters[label_ctx]
        variances = self._variances[label_ctx]
        radius = self._radii[label_ctx]

        intervals = []
        for j in range(len(x)):
            feature_intervals = []
            for c in range(len(centers)):
                center = centers[c]
                var = variances[c]
                mu_j = center[j]
                sigma_j = np.sqrt(var[j] + 1e-8)

                # Compute S = sum_{i!=j} ((x_i - mu_i)^2 / sigma_i^2)
                S = 0
                for i in range(len(x)):
                    if i != j:
                        mu_i = center[i]
                        sigma_i = np.sqrt(var[i] + 1e-8)
                        S += ((x[i] - mu_i) ** 2) / (sigma_i ** 2 + 1e-8)

                if S > radius:
                    continue  # No interval from this cluster

                D = (radius - S) * (sigma_j ** 2)
                if D < 0:
                    continue

                delta = np.sqrt(D)
                low = mu_j - delta
                high = mu_j + delta

                # Convert to relative intervals around x[j]
                rel_low = low - x[j]
                rel_high = high - x[j]

                feature_intervals.append((rel_low, rel_high))

            # Union intervals (simplified: just collect all)
            intervals.append(feature_intervals)

        return intervals

    def accept(self, x_prime, label_ctx):
        """True if x_prime is inside a calibrated region; applies e-test if enabled."""
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

        # TODO: implement martingale if use_martingale
        if self._martingale is not None and self._martingale.reject(x_prime):
            return False
        return True