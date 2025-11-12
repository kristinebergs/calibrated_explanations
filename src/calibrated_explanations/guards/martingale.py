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
        # Baseline k-th neighbor distance computed on training data. Used to
        # center e-values so that in-distribution points produce e≈1 and
        # outliers produce e>1 (monotone in distance).
        self._baseline = None

    def fit(self, x_train):
        """Fit the nearest neighbors on training data."""
        x = np.asarray(x_train)
        n_samples = x.shape[0]

        # Ensure sensible neighbor counts relative to training size
        n_neighbors_eff = min(self.n_neighbors, max(1, n_samples))
        # Ensure k < n_neighbors_eff (we will use the k-th neighbor excluding self)
        self.k = max(1, min(self.k, n_neighbors_eff - 1))

        self._nn = NearestNeighbors(n_neighbors=n_neighbors_eff, metric="euclidean")
        self._nn.fit(x)

        # Compute baseline: median of the k-th neighbor distances within training set
        # We request k+1 neighbors so that the 0-th neighbor (self) is included and
        # the k-th index corresponds to the k-th nearest neighbor excluding self.
        try:
            distances, _ = self._nn.kneighbors(x, n_neighbors=self.k + 1)
            # distances shape: (n_samples, k+1) ; take column k
            kth_dists = distances[:, self.k]
            self._baseline = float(np.median(kth_dists))
        except Exception:
            # Fallback: if kneighbors fails for any reason, set baseline to 0
            self._baseline = 0.0

        return self

    def e_value(self, x_instance):
        """Compute e-value for instance x_instance."""
        if self._nn is None:
            raise ValueError("Not fitted")

        # Find k nearest neighbors
        distances, _ = self._nn.kneighbors(x_instance.reshape(1, -1), n_neighbors=self.k)
        # Use the distance to the k-th neighbor as the test statistic
        test_stat = float(distances[0][-1])

        # Map distances to e-values so that in-distribution points produce
        # e≈1 and larger distances produce e>1. We center on the baseline
        # computed at fit-time and use an exponential transform:
        #   e = exp(test_stat - baseline)
        # This is monotone in distance and makes the default threshold
        # semantics (reject when e >= gamma) meaningful.
        if self._baseline is None:
            # If baseline missing, behave conservatively and return 1.0
            return 1.0

        # Clip exponent to avoid overflow in extreme cases
        exponent = test_stat - self._baseline
        exponent = float(np.clip(exponent, -100.0, 100.0))
        e_val = float(np.exp(exponent))
        return e_val

    def reject(self, x_instance):
        """Reject if e-value exceeds gamma."""
        # Reject when e_value is at least the configured gamma. We use >= to
        # make the threshold inclusive and consistent with typical e-value
        # testing semantics where e>=threshold indicates evidence.
        return self.e_value(x_instance) >= self.gamma


class EMartingale:
    """Minimal e-martingale accumulator using log-space accumulation.

    The class tracks the sum of log e-values so the current e-martingale is
    numerically stable: current_value() == exp(sum(log e_i)). Zero e-values
    are handled by representing the log-sum as -inf (current value 0.0).
    """

    def __init__(self):
        # store sum of log e-values; start at 0.0 (log(1))
        self._log_sum = 0.0
        self._updates = 0

    def reset(self):
        """Reset the internal state to the neutral element (1.0)."""
        self._log_sum = 0.0
        self._updates = 0

    def update(self, e_value: float):
        """Update the e-martingale with a single numeric e-value.

        Passing 0.0 will set the internal value to zero (log-sum becomes
        -inf); non-zero e-values are multiplied into the running product in
        log-space.
        """
        if not isinstance(e_value, (int, float)):
            raise TypeError("e_value must be numeric")

        # handle zero separately to avoid log(0) exceptions
        if e_value == 0.0:
            # represent zero by setting log-sum to -inf
            self._log_sum = float("-inf")
        else:
            # if we've previously set -inf, it stays -inf (0 * anything = 0)
            if np.isneginf(self._log_sum):
                # nothing to do, stays -inf
                pass
            else:
                self._log_sum += float(np.log(e_value))

        self._updates += 1

    def update_from_test(self, x_instance, e_test: MartingaleETest):
        """Compute an e-value from an e-test for x_instance and update.

        This is a small convenience wrapper so callers don't need to call
        the e-test and update separately.
        """
        e_val = e_test.e_value(x_instance)
        self.update(e_val)
        return e_val

    def current_value(self) -> float:
        """Return the current e-martingale value (product of e-values).

        Returns 0.0 when the internal log-sum is -inf.
        """
        if np.isneginf(self._log_sum):
            return 0.0
        return float(np.exp(self._log_sum))

    @property
    def n_updates(self) -> int:
        """Return the number of updates applied to the e-martingale."""
        return self._updates

