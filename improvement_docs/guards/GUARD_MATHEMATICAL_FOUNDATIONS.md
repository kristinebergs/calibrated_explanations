# Mathematical Foundations: Conformal Prediction & Calibrated Guarantees

**Date:** November 13, 2025  
**Scope:** Formal theory underlying the confidence-modulated conformal guard  
**Audience:** Researchers, advanced practitioners, implementers

---

## Introduction

This document provides rigorous mathematical foundations for the `ConformalRegionOracle` guard, connecting:

1. **Conformal Prediction Theory** — provides finite-sample validity guarantees
2. **Calibrated Predictions** — provide confidence-based modulation
3. **Feature-Space Clustering** — captures heteroscedasticity
4. **Composition** — how these elements combine to provide in-distribution guarantees

---

## Part 1: Foundational Definitions

### 1.1 Exchangeability and i.i.d. Assumption

**Definition (Exchangeability):**
A finite sequence of random variables $(Z_1, \ldots, Z_n)$ is exchangeable if its joint distribution is invariant under permutations.

$$\Pr[(Z_1, \ldots, Z_n) = (z_1, \ldots, z_n)] = \Pr[(Z_{\sigma(1)}, \ldots, Z_{\sigma(n)}) = (z_1, \ldots, z_n)]$$

for any permutation $\sigma$.

**Relationship to i.i.d.:**
- i.i.d. implies exchangeability
- Exchangeability does NOT imply independence
- Key advantage: conformal methods rely ONLY on exchangeability, not on distributional assumptions

### 1.2 Nonconformity Measures

**Definition (Nonconformity Measure):**
A nonconformity measure $A$ is a real-valued function that quantifies how "atypical" a point is:

$$A : \mathcal{Z} \to \mathbb{R}$$

where $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$ is the input-output space.

**Properties:**
- Symmetry: swapping arguments should not fundamentally change the measure
- Higher values = more atypical
- Can be defined on features only ($A(x)$) or on feature-target pairs ($A(x, y)$)

**Common choices:**

1. **Prediction error (Regression):**
   $$A(x, y) = |y - \hat{f}(x)|$$

2. **Distance to center (Feature-based, this work):**
   $$A(x) = d(x, \mu_k)$$
   where $d$ is a distance metric (Euclidean, Mahalanobis, etc.)

3. **k-Nearest Neighbor distance:**
   $$A(x) = \sum_{i=1}^k d(x, x^{(i)}_{\text{NN}})$$

### 1.3 Conformal Prediction Sets

**Definition (Prediction Set):**
A prediction set $\Gamma(x)$ is a set-valued function that maps inputs to subsets of the output space:

$$\Gamma : \mathcal{X} \to 2^{\mathcal{Y}}$$

**Goal:** Construct $\Gamma$ such that:
$$\Pr[y \in \Gamma(x)] \geq 1 - \alpha$$

for all new test points, with only exchangeability assumption.

---

## Part 2: Conformal Prediction Theory

### 2.1 Fundamental Theorem (Vanilla Conformal Prediction)

**Theorem (Vovk, Vapnik, 1999):**

Let $(Z_1, \ldots, Z_n, Z_{n+1}) = \{(x_i, y_i)\}_{i=1}^{n+1}$ be an exchangeable sequence.

Let $A$ be a nonconformity measure and define:

$$A_i = A(x_i, y_i) \quad \text{for } i = 1, \ldots, n$$

$$r = \lceil (n+1)(1-\alpha) \rceil / n$$

Let $r_{\text{quant}} = \text{quantile}_{r}(\{A_1, \ldots, A_n\})$ be the $r$-th largest nonconformity score.

Then the prediction set:
$$\Gamma(x_{n+1}) = \{y : A(x_{n+1}, y) \leq r_{\text{quant}}\}$$

satisfies:
$$\Pr[y_{n+1} \in \Gamma(x_{n+1})] \geq 1 - \alpha$$

**Key insight:** This holds for ANY exchangeable sequence, without assuming any parametric model!

### 2.2 Proof Sketch

**Why does it work?**

1. **Exchangeability:** The joint distribution of $(A_1, \ldots, A_n, A_{n+1})$ is invariant to permutations.

2. **Rank argument:** Under exchangeability, $A_{n+1}$ is equally likely to be any rank among $\{A_1, \ldots, A_n, A_{n+1}\}$ (ranks 1 to $n+1$).

3. **Coverage:** The probability that $A_{n+1}$ is among the top $r \cdot n$ largest values is at least $1 - \alpha$.

4. **Conclusion:** With probability $\geq 1 - \alpha$, $A_{n+1} \leq r_{\text{quant}}$, so $y_{n+1} \in \Gamma(x_{n+1})$.

### 2.3 Distribution-Free Validity

**Key property:**

The guarantee does NOT depend on:
- Dimensionality of $\mathcal{X}$
- Whether $\mathcal{Y}$ is continuous or discrete
- The specific form of the model $\hat{f}$
- Any assumed model class

**Only depends on:**
- Exchangeability
- Size of calibration set $n$
- Significance level $\alpha$

This is remarkable: we get worst-case guarantees without ANY distributional assumptions.

---

## Part 3: Inductive Conformal Prediction (ICP)

### 3.1 The ICP Framework

**Motivation:** Vanilla CP uses all training data for both model and calibration, wasting data.

**ICP Solution:** Split into two sets:
- Proper set: $\mathcal{Z}_{\text{prop}}$ (train model)
- Calibration set: $\mathcal{Z}_{\text{cal}}$ (compute nonconformity)

### 3.2 ICP Algorithm

```
Input: Training data Z = {(x_i, y_i)}_{i=1}^n, significance level α

1. Split: Partition Z into Z_prop (proportion p) and Z_cal (proportion 1-p)

2. Train: Fit predictor f on Z_prop

3. Calibrate: Compute nonconformity scores on Z_cal
   A_i = A(x_i, y_i) for all (x_i, y_i) ∈ Z_cal

4. Threshold: Compute (1-α) quantile
   r = quantile_{1-α}(A_1, ..., A_{n_cal})

5. Predict: For new x, output prediction set
   Γ(x) = {y : A(x, y) ≤ r}
```

### 3.3 Validity Guarantee for ICP

**Theorem (ICP Validity):**

The prediction set $\Gamma$ computed via ICP satisfies:

$$\Pr[y \in \Gamma(x)] \geq (1 - \alpha) \left(1 - \frac{1}{n_{\text{cal}}+1}\right)$$

where $n_{\text{cal}}$ is the size of the calibration set.

**Note:** The factor $(1 - 1/(n_{\text{cal}}+1))$ accounts for the fact that we're using finite calibration set. As $n_{\text{cal}} \to \infty$, this approaches 1.

---

## Part 4: Clustering and Heteroscedasticity

### 4.1 Mondrian Conformal Prediction (Stratified by Features)

**Motivation:** In some regions, the nonconformity distribution may differ (heteroscedasticity).

**Example:** In regression, predicting house prices:
- Cheap houses ($y < $100K) might have prediction error $\sim N(0, 10K)$
- Expensive houses ($y > $1M) might have prediction error $\sim N(0, 200K)$

A global conformal radius would be either too strict for expensive or too lenient for cheap.

**Mondrian CP Solution:** Compute separate radii for different regions (mondrians).

### 4.2 Clustering-Based Mondrian

In our implementation, we partition feature space into clusters $\{C_1, \ldots, C_K\}$ using K-means.

**For each cluster $C_k$:**
- Compute nonconformity scores: $\{A(x_i) : x_i \in C_k \cap \mathcal{Z}_{\text{cal}}\}$
- Radius: $r_k = \text{quantile}_{1-\alpha}(\text{scores}_k)$

**For a new point $x_{\text{new}}$ in region $C_k$:**
- Use radius $r_k$ specific to that cluster

### 4.3 Validity with Clustering

**Theorem (Mondrian ICP Validity):**

If we partition the data into $K$ mondrians $\{M_1, \ldots, M_K\}$ and compute separate radii per mondrian:

$$r_k = \text{quantile}_{1-\alpha}(\{A_i : i \in M_k \cap \mathcal{Z}_{\text{cal}}\})$$

Then for any new point $x \in M_k$:

$$\Pr[y \in \Gamma_k(x) | x \in M_k] \geq 1 - \alpha$$

with coverage conditional on the mondrian assignment.

**Key insight:** Clustering improves efficiency by adapting radii to local structure, while preserving coverage.

---

## Part 5: Distance Metrics and Mahalanobis Distance

### 5.1 Euclidean vs. Mahalanobis

**Euclidean distance:**
$$d_E(x, \mu) = \sqrt{\sum_{i=1}^d (x_i - \mu_i)^2}$$

**Problem:** Assumes all features have equal variance; ignores covariance.

**Mahalanobis distance:**
$$d_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

where $\Sigma$ is the sample covariance matrix.

**Advantage:** Accounts for feature variance and correlation.

### 5.2 Covariance Estimation

**Sample covariance (unbiased):**
$$\Sigma = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T$$

**Issues with small samples:**
- Can be ill-conditioned (non-invertible)
- High-dimensional regime ($d > n$) is problematic

**Regularization (Ledoit-Wolf):**
$$\Sigma_{\text{shrink}} = (1 - \alpha) \Sigma_{\text{sample}} + \alpha I$$

where $\alpha$ is a shrinkage parameter (typically data-driven).

### 5.3 Diagonal Covariance Assumption

In our implementation, we use **diagonal covariance** (features assumed uncorrelated):

$$\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$$

**Mahalanobis distance becomes:**
$$d_M(x, \mu) = \sqrt{\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{\sigma_i^2}}$$

**Advantages:**
- Avoids inversion of large matrix
- More stable with finite samples
- Still captures per-feature variance

**Limitation:**
- Ignores feature correlations
- Okay for most practical scenarios; can extend if needed

---

## Part 6: Confidence-Modulated Acceptance

### 6.1 The Modulation Principle

**Standard conformal:** Accept all points within radius $r$.

**Modulated conformal:** Accept points within radius $r_{\text{eff}}(w)$ where $w$ is a confidence measure.

### 6.2 Formal Framework

**Let:**
- $w(x)$ = interval width (confidence measure) for point $x$
- $r(x) = r_{\text{eff}}(w(x))$ = confidence-modulated radius

**Modulation function:**
$$r_{\text{eff}}(w) = r_{\text{base}} \cdot f(w)$$

where $f : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ is the modulation function.

**Requirement:** $f$ must be **monotone non-decreasing** in $w$.

$$w_1 \leq w_2 \implies f(w_1) \leq f(w_2)$$

### 6.3 Validity Theorem for Modulated Acceptance

**Theorem (Modulated Conformal Coverage):**

If $f$ is monotone non-decreasing and $f(w) \geq 1$ for all $w$, then the modulated acceptance criterion:

$$A(x) \leq r_{\text{base}} \cdot f(w(x))$$

provides coverage **at least as good as** the unmodulated criterion:

$$\Pr[A(x) \leq r_{\text{eff}}(w(x))] \geq \Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$$

**Proof:**

1. By conformal theorem: $\Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$

2. Since $f(w) \geq 1$: $r_{\text{eff}}(w) = r_{\text{base}} \cdot f(w) \geq r_{\text{base}}$

3. If $A(x) \leq r_{\text{base}}$, then $A(x) \leq r_{\text{eff}}(w)$

4. Therefore: $\Pr[A(x) \leq r_{\text{eff}}(w)] \geq \Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$ ∎

**Key insight:** Modulation by non-decreasing function **preserves or improves coverage**.

### 6.4 Interpretation

- **High confidence** ($w$ small): $f(w)$ small, so $r_{\text{eff}}$ small, radius **strict**
- **Low confidence** ($w$ large): $f(w)$ large, so $r_{\text{eff}}$ large, radius **lenient**

This is **conservative**: when the model is uncertain, we don't reject perturbations.

---

## Part 7: Default Modulation Function

### 7.1 Linear Modulation

**Formula:**
$$f(w) = 1 + (1 - c(w)) \cdot \lambda$$

where:
- $c(w) = 1 - \frac{w - w_{\min}}{w_{\max} - w_{\min}}$ is normalized confidence
- $\lambda \geq 0$ is the relaxation factor

**Behavior:**
- $c(w) = 1$ (minimum $w$): $f(w) = 1.0$ (use base radius)
- $c(w) = 0$ (maximum $w$): $f(w) = 1 + \lambda$ (relax radius by factor $1+\lambda$)

**Properties:**
- Monotone non-decreasing in $w$ ✓
- $f(w) \geq 1$ ✓
- Simple interpretation ✓

### 7.2 Alternative Modulation Functions

**Exponential modulation:**
$$f(w) = e^{(1 - c(w)) \lambda}$$

**Properties:**
- Smooth, differentiable
- Less aggressive relaxation
- Could use for sensitivity analysis

**Sigmoid modulation:**
$$f(w) = 1 + \frac{\lambda}{1 + e^{k(c(w) - 0.5)}}$$

**Properties:**
- S-shaped curve
- Smooth transition
- More complex hyperparameter tuning

**Recommendation:** Linear modulation (default) is simplest and most interpretable.

---

## Part 8: Interval Width as Confidence Proxy

### 8.1 Calibrated Intervals from VennAbers (Classification)

**VennAbers output for classification:**
$$\text{predict}(x) \to (p_{\text{pred}}, [p_L, p_U])$$

where $[p_L, p_U]$ is a **calibrated confidence interval** for the predicted probability.

**Interpretation:**
- With probability $\geq 1 - \alpha_{\text{VA}}$ (VennAbers' calibration level), the true predicted probability $p^* \in [p_L, p_U]$

**Width as confidence:**
$$w = p_U - p_L$$

- Narrow interval → model is confident in its probability estimate
- Wide interval → model is uncertain

### 8.2 Calibrated Intervals from CPD (Regression)

**CPD (Conformal Predictive Distribution) output:**
$$\text{predict}(x) \to (\hat{y}, [y_L, y_U])$$

where $[y_L, y_U]$ is a **conformal prediction interval**.

**Interpretation:**
- With probability $\geq 1 - \alpha_{\text{CPD}}$ (conformal level), the true value $y^* \in [y_L, y_U]$

**Width as confidence:**
$$w = y_U - y_L$$

- Narrow interval → model is confident in its prediction
- Wide interval → model is uncertain

### 8.3 Advantages of Using Interval Width

1. **Calibration-aware:** Directly from calibrated predictors
2. **Interpretable:** Corresponds to model's stated confidence
3. **Adaptive:** Per-instance, not global
4. **No additional computation:** Intervals already computed

---

## Part 9: Composition of Guarantees

### 9.1 Full Guard Guarantee

**Theorem (Guard Coverage):**

Let:
- $r_k$ = conformal radius for cluster $k$ (from ICP)
- $w(x)$ = interval width for instance $x$ (from calibrated predictor)
- $r_{\text{eff}}(w) = r_k \cdot f(w)$ = modulated radius
- $A(x') = d_M(x', \mu_k)$ = Mahalanobis distance to cluster center

The guard acceptance criterion:
$$A(x') \leq r_{\text{eff}}(w(x))$$

provides coverage guarantee:
$$\Pr[x' \text{ is in-distribution}] \geq 1 - \alpha$$

where "in-distribution" means: $x'$ comes from same cluster as $x$ with similar confidence.

### 9.2 Assumptions

**Required for validity:**

1. **Exchangeability:** Training and test data are exchangeable
   - Holds if: same distribution, i.i.d., no covariate shift
   - Violated if: test comes from different distribution

2. **Calibration quality:** Interval learner provides valid intervals
   - Holds if: VennAbers/CPD properly calibrated
   - Not violated if: poorly calibrated (just loose bounds)

3. **Clustering choice:** K-means captures meaningful structure
   - Helps if: features have local structure
   - Doesn't hurt if: doesn't capture structure (just uses global radius)

**Robustness:**
- Violations of (2) or (3) degrade performance but don't break guarantee
- Violations of (1) can break guarantee (distribution shift)

### 9.3 Coverage vs. Tightness Trade-off

**Coverage guarantee:** At least $1 - \alpha$ of perturbations accepted

**Tightness:** Depending on:
- Model quality ($R^2$, accuracy)
- Calibration quality (how tight are intervals?)
- Clustering fit (does K-means capture structure?)
- Modulation factor $\lambda$

**Observation:** We get **validity** (coverage $\geq 1 - \alpha$) but **not optimality** (radius might be loose).

---

## Part 10: Practical Guarantees for Explanations

### 10.1 What the Guard Guarantees

**IF a perturbation is ACCEPTED by the guard, then:**

1. **Distance property:** It's within Mahalanobis distance $r_{\text{eff}}$ to a cluster center
2. **Cluster property:** The cluster was trained on data from similar feature distribution
3. **Confidence property:** The original instance had confidence level $w$

**THEN with probability $\geq 1 - \alpha$:**
- $x'$ comes from the region where training data was present
- The model's confidence at $x'$ should be similar to confidence at $x$

### 10.2 What the Guard Does NOT Guarantee

**The guard does NOT guarantee:**

1. ✗ That the explanation rules are correct (that's up to the explainer)
2. ✗ That the feature importance rankings are true causes
3. ✗ That rejected perturbations are truly out-of-distribution (maybe just rare)
4. ✗ That the model behaves the same at $x'$ as at $x$ (only that we stayed in-distribution)

### 10.3 How Guard Helps Explanations

**The guard ensures:**

1. ✓ Perturbations are reasonably similar to training data
2. ✓ Explanations are based on realistic counterfactuals
3. ✓ Extracted rules apply to "normal" instances, not extreme perturbations
4. ✓ Calibrated by model confidence (tighter for confident predictions)

---

## Part 11: Failure Modes and Mitigations

### 11.1 Distribution Shift

**Failure:** Test set comes from different distribution than training

**Detection:**
- OOB rate $\gg \alpha$ (guard rejects too much)
- Model performance degrades on test set

**Mitigation:**
- Increase $\alpha$ (tolerate more)
- Retrain guard on representative calibration set
- Use domain adaptation techniques

### 11.2 Poor Clustering

**Failure:** K-means doesn't capture meaningful structure

**Detection:**
- Cluster sizes very imbalanced
- Conformal radius per cluster varies wildly

**Mitigation:**
- Increase $n_{\text{clusters}}$
- Use different clustering (spectral, DBSCAN)
- Check if problem is truly high-dimensional

### 11.3 Heteroscedasticity Mismatch

**Failure:** Model confidence is not correlated with actual difficulty

**Detection:**
- Modulation factor has extreme values
- OOB rate independent of $\lambda$

**Mitigation:**
- Check interval learner quality (is it well-calibrated?)
- Use different confidence measure (e.g., entropy for classification)
- Increase $\alpha$ if model is unreliable

---

## Part 12: Extensions and Future Work

### 12.1 Adaptive Conformal Prediction

**Future:** Compute different $\alpha$ per instance based on difficulty estimator.

$$\alpha(x) = \alpha_0 + \text{difficulty}(x) \cdot c$$

**Advantage:** Tighter regions for easy instances, looser for hard.

### 12.2 Regression-Specific Nonconformity

**Future:** Use residual-based nonconformity instead of distance.

$$A(x) = |\text{residual}| = |y - \hat{f}(x)|$$

**Advantage:** Directly measures prediction error, not feature distance.

### 12.3 Multi-Level Modulation

**Future:** Modulate based on multiple factors:
- Prediction confidence $w_1$
- Model uncertainty estimate $w_2$
- Sample difficulty $w_3$

$$f(w_1, w_2, w_3) = f_1(w_1) \cdot f_2(w_2) \cdot f_3(w_3)$$

---

## Conclusion

The guard's mathematical foundations rest on three pillars:

1. **Conformal Prediction:** Provides distribution-free validity guarantees
2. **Clustering:** Captures heteroscedasticity in feature space
3. **Modulation:** Adapts to model confidence

Together, they provide a principled mechanism to filter in-distribution perturbations during explanation, with formal guarantees on coverage.

**Key properties:**
- ✓ No distributional assumptions (only exchangeability)
- ✓ Finite-sample coverage guarantee (exact for any $n$)
- ✓ Adaptive per-instance (via confidence modulation)
- ✓ Interpretable (Mahalanobis distance + conformal)

---

**Document Version:** 1.0  
**Status:** Final  
**Next:** Refer to `GUARD_DESIGN_CONFIDENCE_MODULATION.md` for implementation details
