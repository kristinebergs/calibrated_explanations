Place the new logic in an optional submodule. Wire it through a single choke point where CE generates and accepts perturbations. Keep default off. No changes to the fast track.

# Where to integrate

* Package layout

  * `src/calibrated_explanations/guards/`

    * `__init__.py`
    * `regions.py`  → ConformalRegionOracle (CRO)
    * `intervals.py` → 1D line–region intersection utilities
    * `martingale.py` → optional e-test
    * `conjunctions.py` → validate combined rules
* Touch points in core

  * `src/calibrated_explanations/core/calibrated_explainer.py`
    Add an optional `guard` slot to the base explainer so all core explainers can use the same filter path.
  * The method that emits one-feature perturbations (wherever CE currently generates candidates per feature). If this is in a helper, add the filter there. If it’s in the explainer classes, add a tiny adapter in the base class and keep subclasses unchanged.
  * Conjunction builder (only if you use `add_conjunctions`): post-validate the final combined point via the same guard.

# Minimal API change

* New, optional kwargs in the main explainer(s) and/or wrapper(s):

  * `guard=None`
    Either a fitted guard object or a factory spec. Default `None` preserves current behavior.
  * `guard_params=None`
    If `guard` is a string (e.g., `"conformal_regions"`), use this dict to configure.
* New public class:

  * `calibrated_explanations.guards.ConformalRegionOracle`

No other public API changes. No changes to the fast explainer.

# Guard interfaces

```python
class ConformalRegionOracle:
    def __init__(self, alpha=0.1, mode="clf", threshold=None,
                 n_clusters=5, covariance="diag", random_state=None,
                 use_martingale=False, e_gamma=10.0, e_knn=30, e_neigh=500):
        ...

    def fit(self, X, y):
        """Build label-conditional cluster regions and calibrate radii."""
        return self

    def label_context(self, x, *, clf_predict_proba=None, reg_predict=None):
        """Return the label-conditional context for x:
        - clf: argmax class
        - thresholded regression: 1{y>=tau}."""
        ...

    def intervals(self, x, label_ctx):
        """Return per-feature allowed 1D intervals for x under label_ctx."""
        # list of (low, high) for each feature; possibly multiple intervals per feature
        ...

    def accept(self, x_prime, label_ctx):
        """True if x_prime is inside a calibrated region; applies e-test if enabled."""
        ...
```

# Hook points and control flow

1. **Training time**

   * If `guard` is provided and not fitted:

     * During explainer `.fit(X, y, ...)`, call `guard.fit(X, y)`.
     * Cache on the explainer: `self._guard = guard`.
   * If `guard` is `None`, do nothing.

2. **Explanation time**

   * For test instance `x`:

     * `label_ctx = guard.label_context(x, clf_predict_proba=self._predict_proba, reg_predict=self._predict_reg)`
     * When CE proposes a candidate change for feature `j`:

       * If guard exists, compute 1D admissible intervals for `j`:

         * `allowed = guard.intervals(x, label_ctx)[j]`
         * Reject any candidate `x_j'` outside `allowed`. If `allowed` empty, mark feature `j` as inadmissible near `x`.
     * Continue with existing CE logic on the filtered candidate set.

3. **Conjunctions (nice-to-have)**

   * After CE selects a conjunction and materializes the combined perturbed point `x_conj`, call `guard.accept(x_conj, label_ctx)`. If false, discard this conjunction and continue search.

# Concrete code edits

* `src/calibrated_explanations/core/calibrated_explainer.py`

  * Add to `__init__`: `guard: Optional[BaseGuard]=None, guard_params: Optional[dict]=None`
  * In `fit(...)`: if `self.guard` is a class or string, instantiate then `fit(X, y)`.
  * Add two tiny helpers:

    ```python
    def _label_ctx(self, x):
        if self.guard is None: return None
        return self.guard.label_context(x,
               clf_predict_proba=getattr(self, "_predict_proba", None),
               reg_predict=getattr(self, "_predict_reg", None))

    def _accept(self, x_prime, label_ctx):
        return True if self.guard is None else self.guard.accept(x_prime, label_ctx)
    ```
  * Wrap the existing perturbation producer with an inlined filter:

    ```python
    # Pseudocode inside the loop over feature j
    if self.guard is not None:
        allowed_intervals = self.guard.intervals(x, label_ctx)[j]
        candidates = [v for v in candidates if _in_intervals(v, allowed_intervals, x_j)]
    ```
  * If conjunctions are built in this file, add:

    ```python
    if self.guard is not None and not self.guard.accept(x_conj, label_ctx):
        continue  # skip OOD conjunction
    ```

* `src/calibrated_explanations/guards/regions.py`

  * Implement CRO:

    * Fit: per-label clustering (e.g., KMeans or GMM-lite). Store centers `μ_{y,k}`, scatters `Σ_{y,k}` (diagonal by default), split-conformal radii `r_{y,k}(α)`.
    * Efficient nearest-center index (sklearn KDTree or faiss if available).
    * `label_context`: predicted class for classification; event `y>=τ` for thresholded regression.
    * `intervals`: for feature `j`, solve the quadratic inequality for each cluster and return the union of valid 1D intervals along the axis through `x`.

      * Diagonal case closed form:

        * Let `S = Σ_{i≠j} ((x_i-μ_i)^2 / σ_i^2)`.
        * Require `S <= r^2`. If not, no interval from this cluster.
        * Then `t ∈ [ −Δ ± sqrt( (r^2 − S) * σ_j^2 ) ]` where `Δ = (x_j − μ_j)`.
        * Convert to absolute values for `x_j' = x_j + t`.
      * Union intervals over clusters covering `x`’s label context. Optionally merge overlapping intervals.
    * `accept`: check membership in any calibrated ball. If `use_martingale`, also compute a local k-NN e-value and reject if it exceeds `e_gamma`.

* `src/calibrated_explanations/guards/martingale.py`

  * Lightweight k-NN distance score on a local neighborhood indexed at fit. Maintain per-label neighbor pools. Implement an anytime-valid e-test thresholded by `e_gamma`.

* `src/calibrated_explanations/guards/intervals.py`

  * Utility for stable numeric interval union and membership.

* `src/calibrated_explanations/guards/conjunctions.py`

  * Optionally expose `validate_conjunction(x_conj, guard, label_ctx)` used by core.

# Step-by-step implementation guide

1. **Create the subpackage**

   * Add `src/calibrated_explanations/guards/` with the four modules above and `__init__.py` exporting `ConformalRegionOracle`.

2. **Add a base type for guards** (internal)

   * In `guards/__init__.py`:

     ```python
     class BaseGuard(Protocol):
         def fit(self, X, y): ...
         def label_context(self, x, **kwargs): ...
         def intervals(self, x, label_ctx): ...
         def accept(self, x_prime, label_ctx): ...
     ```
   * Make `ConformalRegionOracle` implement it.

3. **Implement CRO.fit**

   * Split data by label context `y_ctx`:

     * For classification: `y_ctx = y`.
     * For thresholded regression: `y_ctx = 1{y>=τ}`.
   * For each `y_ctx`:

     * Fit `K` clusters on `X[y_ctx]`:

       * Default: KMeans; store per-cluster empirical variances for diagonal `Σ`.
     * Compute nonconformity scores = squared Mahalanobis to nearest cluster.
     * Split conformal: hold-out calibration set per `y_ctx` to set radii `r_{y,k}(α)` as the `1-α` quantile of scores per cluster or globally per label (choose one and keep consistent).
     * Build nearest-center index and persist structures.

4. **Implement CRO.label_context**

   * Accept injected `clf_predict_proba` or `reg_predict` from explainer.
   * Return `argmax` class for classification or `1{reg_predict(x)>=τ}`.

5. **Implement CRO.intervals**

   * For each feature `j`, per nearest cluster under `label_ctx`:

     * Compute constant term `S` and discriminant `D = (r^2 − S) * σ_j^2`.
     * If `D < 0`: no interval from this cluster.
     * Else interval in absolute coordinate:
       `[ μ_j − sqrt(D) , μ_j + sqrt(D) ]` ∩ feature domain.
       Convert to deltas relative to `x_j` if you prefer the CE perturbation parameterization.
   * Union intervals across clusters. Return a list of intervals per feature as Python tuples.

6. **Implement CRO.accept**

   * Fast check: nearest cluster distance squared ≤ `r^2`.
   * If enabled: compute local k-NN e-value on stored neighbor pool; reject if `e > e_gamma`.

7. **Wire into `calibrated_explainer.py`**

   * Add `guard` plumbing as shown.
   * Identify the method that constructs per-feature candidate values. Before adding a candidate, call `_in_intervals` or `_accept`.
   * For `add_conjunctions`, add the post-check.

8. **Keep fast path untouched**

   * Do not import or call `guards` in fast explainer classes. The optional guard only lives in the core explainer and explainer.

9. **Docs**

   * Add `docs/guards.md`: motivation, guarantees, usage.
   * Example:

     ```python
     from calibrated_explanations import CalibratedExplanation
     from calibrated_explanations.guards import ConformalRegionOracle

     guard = ConformalRegionOracle(alpha=0.1, mode="clf", n_clusters=5)
     expl = CalibratedExplanation(model, guard=guard).fit(X_train, y_train)
     fx = expl.explain(x_test)
     ```
   * Note: defaults keep guard off. Later release flips default.

10. **Tests**

* Unit: interval math on synthetic diagonal Gaussians. Accept/reject correctness vs analytic ground truth.
* Integration: run CE with and without guard on small tabular sets. Assert fewer OOD samples used for rules, identical public API, tolerable overhead.
* Conjunctions: ensure invalid combinations get rejected.

11. **Performance**

* Cache per-instance `S` constants across features: reuse `S_total = Σ_i ((x_i-μ_i)^2/σ_i^2)`. For feature `j`, `S = S_total − ((x_j-μ_j)^2/σ_j^2)`.
* Keep `Σ` diagonal by default. Expose full covariance as experimental.
* Limit clusters per label to small `K` and pick nearest one for intervals unless you need unions.

12. **Release plan**

* Minor release with guard optional.
* Deprecation note: in two releases, `guard` will default to `ConformalRegionOracle(alpha=0.1)` unless `guard="none"`.
* Benchmark overhead target ≤ 20% at explanation time.

# Backward compatibility

* With `guard=None`, the code path is identical. No behavior changes. No new dependencies beyond numpy/scikit-learn.

# Notes on edge cases

* If a feature has no admissible interval, mark it “inadmissible near x” and skip it. Preserve CE’s result format and uncertainty fields.
* Continuous vs categorical features:

  * For one-hot categoricals, compute acceptance on the full vector; or simpler, whitelist only observed categories in the same label context cluster.
* Feature bounds:

  * Intersect intervals with known feature ranges before sampling candidate values.

This plan adds a single optional dependency path, localizes risk to one hook, and gives you a clean switch to make it default later without breaking users.
