"""Martingale e-test for perturbation guards."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class MartingaleETest:
    """Martingale e-test using k-NN distances."""

    def __init__(self, k: int = 30, n_neighbors: int = 500, gamma: float = 10.0):
        self.k = k
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self._nn = None
        self._distances = None

    def fit(self, X):
        """Fit the nearest neighbors on training data."""
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self._nn.fit(X)
        # Precompute distances for efficiency, but for now, we'll compute on the fly
        return self

    def e_value(self, x):
        """Compute e-value for instance x."""
        if self._nn is None:
            raise ValueError("Not fitted")

        # Find k nearest neighbors
        distances, _ = self._nn.kneighbors(x.reshape(1, -1), n_neighbors=self.k)
        # Use the distance to the k-th neighbor as the test statistic
        test_stat = distances[0][self.k - 1]

        # Simple e-value: exponential decay
        e_val = np.exp(-test_stat)
        return e_val

    def reject(self, x):
        """Reject if e-value exceeds gamma."""
        return self.e_value(x) > self.gamma