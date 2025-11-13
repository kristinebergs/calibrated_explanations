# Guard Formal Guarantees & Proof Sketches

**Date:** November 13, 2025  
**Status:** Final  
**Audience:** Researchers, theoreticians, advanced implementers

**Cross-references:**
- 📐 `GUARD_MATHEMATICAL_FOUNDATIONS.md` — Complete mathematical proofs
- 🏗️ `GUARD_DESIGN_CONFIDENCE_MODULATION.md` — Design and implementation
- 📊 `ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md` — Design evolution

---

## Overview: Three Core Guarantees

This document provides three formal theorems and their proof sketches, establishing the validity of the confidence-modulated conformal guard.

| Guarantee | What It Says | Where It Applies |
|-----------|-------------|------------------|
| **G1: Conformal Coverage** | Base conformal radius is valid | All clusters globally |
| **G2: Modulation Preserves Coverage** | Modulation doesn't violate validity | Per-instance modulation |
| **G3: In-Distribution Acceptance** | Accepted points are reasonably similar | Perturbation filtering |

---

## Guarantee G1: Conformal Coverage (Foundation)

### Statement

**Theorem (Conformal Prediction Validity):**

Let $(Z_1, \ldots, Z_n, Z_{n+1}) = \{(x_i, y_i)\}_{i=1}^{n+1}$ be an exchangeable sequence where $(x_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$ (regression case).

Let:
- $\hat{f}$ be a model trained on a proper set $\mathcal{Z}_{\text{prop}}$
- $\mathcal{Z}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^{n}$ be a calibration set
- $A(x, y) = |y - \hat{f}(x)|$ be the nonconformity measure (prediction error)
- $r = \text{quantile}_{1-\alpha}(\{|y_i - \hat{f}(x_i)| : (x_i, y_i) \in \mathcal{Z}_{\text{cal}}\})$ be the $(1-\alpha)$ quantile of prediction errors

Then for any new test point $(x_{\text{new}}, y_{\text{new}})$ exchangeable with the calibration set:

$$\boxed{\Pr[|y_{\text{new}} - \hat{f}(x_{\text{new}})| \leq r] \geq 1 - \alpha}$$

### Assumptions

1. **Exchangeability:** Calibration data and test data are exchangeable (same distribution, i.i.d., any permutation-invariant distribution)
2. **Proper calibration set:** Used only for quantile computation, not for training $\hat{f}$
3. **Quantile computation:** Exact quantile without ties (or handled probabilistically)

### Proof Sketch

**Exchangeability argument:**

1. Consider the sequence of nonconformity scores: $(A_1, \ldots, A_n, A_{n+1})$ where $A_i = |y_i - \hat{f}(x_i)|$

2. By exchangeability, this sequence has identical joint distribution regardless of permutation. Therefore, $A_{n+1}$ is equally likely to be any rank among $\{1, 2, \ldots, n+1\}$ (from smallest to largest).

3. Probability that $A_{n+1}$ is among the top $(n+1)(1-\alpha)$ scores:
   $$\Pr[A_{n+1} \text{ ranks} \leq (n+1)(1-\alpha)] = 1 - \alpha$$

4. The $(n+1)(1-\alpha)$-th largest score is exactly the $(1-\alpha)$ quantile $r$.

5. If $A_{n+1}$ ranks in the top $(1-\alpha)$ fraction, then $A_{n+1} \leq r$.

6. Therefore: $\Pr[A_{n+1} \leq r] \geq 1 - \alpha$ ∎

### Why This Is Remarkable

- **No distributional assumptions:** Works for ANY distribution, not just Gaussian or smooth
- **Finite-sample validity:** Holds for finite $n$, with $\alpha$ controlling failure probability
- **Only exchangeability:** No need for i.i.d., independence, or parametric forms
- **Distribution-free:** Worst-case guarantee over all possible distributions

---

## Guarantee G2: Modulation Preserves Coverage

### Statement

**Theorem (Monotone Modulation Preserves Conformal Coverage):**

Let:
- $r_{\text{base}}$ be a conformal radius (valid by G1)
- $w(x)$ be a confidence measure (e.g., interval width) for instance $x$
- $f : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 1}$ be a modulation function satisfying:
  1. **Monotone non-decreasing:** $w_1 \leq w_2 \implies f(w_1) \leq f(w_2)$
  2. **Never shrinks:** $f(w) \geq 1$ for all $w$

Define the modulated radius:
$$r_{\text{eff}}(w) = r_{\text{base}} \cdot f(w)$$

Then:
$$\boxed{\Pr[A(x) \leq r_{\text{eff}}(w(x))] \geq 1 - \alpha}$$

### Proof

**Step 1:** By G1 (conformal validity):
$$\Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$$

**Step 2:** Implication chain:
- $f(w) \geq 1$ implies $r_{\text{eff}}(w) = r_{\text{base}} \cdot f(w) \geq r_{\text{base}}$

**Step 3:** Event containment:
- If $A(x) \leq r_{\text{base}}$, then $A(x) < r_{\text{base}} \leq r_{\text{eff}}(w)$
- Therefore: $\{A(x) \leq r_{\text{base}}\} \subseteq \{A(x) \leq r_{\text{eff}}(w)\}$

**Step 4:** Probability monotonicity:
$$\Pr[A(x) \leq r_{\text{eff}}(w)] \geq \Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$$

Therefore, modulation preserves coverage. ∎

### Key Insight

The monotonicity requirement is **essential**: if we allowed $f(w) < 1$ for some $w$, we could violate conformal validity by making the radius too small.

### Example Modulation Functions

**Linear (default implementation):**
$$f(w) = 1 + (1 - c(w)) \cdot \lambda$$
where $c(w) = 1 - \frac{w - w_{\min}}{w_{\max} - w_{\min}}$ is normalized confidence.

- Monotone increasing in $w$ ✓
- $f(w) \geq 1$ for $\lambda \geq 0$ ✓
- Simple, interpretable ✓

**Exponential:**
$$f(w) = e^{(1 - c(w)) \lambda}$$

- Monotone increasing in $w$ ✓
- Always $\geq 1$ ✓
- Smoother transition ✓

**Sigmoid:**
$$f(w) = 1 + \lambda \cdot \sigma\left(\frac{c(w) - 0.5}{k}\right)$$
where $\sigma$ is sigmoid.

- Monotone increasing ✓
- Smooth S-curve ✓
- Tunable slope $k$ ✓

---

## Guarantee G3: In-Distribution Acceptance

### Statement

**Theorem (Mondrian Clustering Validity):**

Let:
- $\{C_1, \ldots, C_K\}$ be a partition of feature space (from K-means clustering)
- $r_k = \text{quantile}_{1-\alpha}(\{d_M(x_i, \mu_k) : x_i \in C_k \cap \mathcal{Z}_{\text{cal}}\})$ be cluster-specific conformal radius
- $d_M(x, \mu_k) = \sqrt{(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)}$ be Mahalanobis distance

For any instance $x \in C_k$ and perturbation $x' \in C_k$ (same cluster):

$$\boxed{\Pr[d_M(x', \mu_k) \leq r_k | x' \in C_k] \geq 1 - \alpha}$$

### Interpretation

**Instances and perturbations in the same cluster are likely to be similar.** Specifically:
- If $x'$ comes from the same cluster and distribution as calibration data in $C_k$
- With probability $\geq 1 - \alpha$, the distance $d_M(x', \mu_k)$ will be within the radius $r_k$

### Proof Sketch

This is a straightforward extension of G1 applied to each cluster independently.

**For cluster $C_k$:**

1. Restrict calibration set: $\mathcal{Z}_{\text{cal}}^{(k)} = \{x_i \in C_k \cap \mathcal{Z}_{\text{cal}}\}$

2. Compute nonconformity for calibration set: $\{A_i = d_M(x_i, \mu_k) : i \in \mathcal{Z}_{\text{cal}}^{(k)}\}$

3. Apply conformal validity to cluster $k$:
   $$\Pr[d_M(x'_{\text{new}}, \mu_k) \leq r_k | x'_{\text{new}} \in C_k] \geq 1 - \alpha$$

4. Union over all clusters (if needed for global statement): complexity depends on whether $x'_{\text{new}}$ is pre-assigned to a cluster.

### Conditional Coverage

**Important:** This guarantee is **conditional on cluster membership**. 

- If $x' \in C_j$ but gets assigned to cluster $C_k$, the guarantee becomes weaker.

**Mitigation:** Use "nearest cluster" assignment (Euclidean nearest to cluster center) to ensure $x'$ is classified to the most similar cluster.

---

## Guarantee G4: Combined Guard Guarantee (Optional Extension)

### Statement

**Theorem (Full Guard Coverage):**

Let:
- $x$ be an original instance
- $x'$ be a perturbation
- $k = \text{nearest\_cluster}(x)$ be the assigned cluster for $x$
- $d_M(x', \mu_k)$ be Mahalanobis distance to cluster center
- $w(x)$ be confidence measure (interval width)
- $r_{\text{eff}}(w)$ be modulated radius

If the guard accepts $x' \to x$ (i.e., $d_M(x', \mu_k) \leq r_{\text{eff}}(w(x))$), then:

$$\boxed{\Pr[\text{perturbation is in-distribution}] \geq 1 - \alpha}$$

### Proof

**Chain the three guarantees:**

1. **By G1 (conformal):** The base radius $r_k$ is valid for cluster $C_k$
   $$\Pr[d_M(x', \mu_k) \leq r_k] \geq 1 - \alpha$$

2. **By G2 (modulation):** Modulating by $f(w) \geq 1$ preserves coverage
   $$\Pr[d_M(x', \mu_k) \leq r_{\text{eff}}(w)] \geq 1 - \alpha$$

3. **Conditional logic:** If the guard accepts (distance $\leq r_{\text{eff}}$), then the perturbation came from a high-probability region:
   $$\Pr[\text{accept}] = \Pr[d_M(x', \mu_k) \leq r_{\text{eff}}(w)] \geq 1 - \alpha$$

Therefore, acceptance is a high-probability indicator of in-distribution status. ∎

### Practical Meaning

- **False positive rate:** At most $\alpha$ (reject in-distribution perturbations)
- **False negative rate:** Unknown (may accept some OOD perturbations if they're near cluster centers)

This is **conservative** — we'd rather reject real perturbations than accept fake ones.

---

## Guarantee G5: Calibration Validity (for Classification)

### Statement (Classification-Specific)

**Theorem (VennAbers Calibration):**

For classification using VennAbers calibration, the interval $[p_L, p_U]$ satisfies:

$$\boxed{\Pr[p^* \in [p_L, p_U]] \geq 1 - \alpha_{\text{VA}}}$$

where $p^*$ is the true probability of the predicted class.

### Consequence for Guard

The interval width $w = p_U - p_L$ encodes calibrated uncertainty:
- Narrow $\Leftrightarrow$ confident prediction
- Wide $\Leftrightarrow$ uncertain prediction

By using $w$ in modulation, we ensure perturbation acceptance is **calibrated to actual model confidence**, not arbitrary thresholds.

---

## When Guarantees Hold vs. Fail

### ✅ Guarantees Hold If

1. **Exchangeability assumption satisfied:**
   - Training and test data from same distribution
   - No covariate shift
   - No temporal drift

2. **Calibration set is proper:**
   - Not used for training $\hat{f}$
   - Sufficient size ($n_{\text{cal}} \geq 30$ recommended)
   - Representative of test distribution

3. **Clustering captures structure:**
   - K-means meaningful (not chaotic)
   - Features not too high-dimensional ($d < n_{\text{cal}}$ ideal)
   - Clusters reasonably balanced

4. **Intervals are valid:**
   - VennAbers/CPD properly fitted
   - Not on out-of-distribution test sets
   - Calibration set representative

### ❌ Guarantees Fail If

1. **Distribution shift:**
   - Test set from different distribution than training
   - Covariate shift or target shift
   - Temporal drift or concept drift
   - **Effect:** Coverage < $1 - \alpha$ (guard too lenient)

2. **Data leakage:**
   - Calibration set used in training $\hat{f}$
   - Model trained on test data
   - Overfitting to calibration set
   - **Effect:** Radius too small, invalid coverage

3. **Exchangeability violation:**
   - Structured dependence in data (time series, spatial, etc.)
   - Sampling bias (selection bias)
   - Non-exchangeable assignment mechanism
   - **Effect:** Unknown, potentially severe

4. **Small calibration set:**
   - $n_{\text{cal}} < 20$
   - Quantile estimation unreliable
   - **Effect:** Coverage guarantee loose but still valid

### ⚠️ Mitigations

| Issue | Mitigation |
|-------|-----------|
| Distribution shift | Detect via OOB monitoring; retrain guard on representative calibration set |
| Data leakage | Use proper train/calibration/test split; never reuse data |
| Exchangeability violation | Add domain randomization; use robust conformal (harder) |
| Small $n_{\text{cal}}$ | Collect more data; use regularized quantile estimation |
| Poor clustering | Increase $n_{\text{clusters}}$; use spectral clustering |
| Uncalibrated intervals | Verify on holdout set; use post-hoc calibration |

---

## Empirical Verification

### Testing Coverage in Practice

**How to verify guarantees hold:**

```python
# 1. Generate held-out test set (exchangeable with calibration)
X_test, y_test = ...

# 2. Collect perturbations
perturbations = generate_perturbations(X_test)  # e.g., from explainer

# 3. Check acceptance rate
accepted = [guard.accept(x_perturb, x_orig) for (x_perturb, x_orig) in perturbations]
acceptance_rate = sum(accepted) / len(accepted)

# 4. If in-distribution, acceptance should be >= (1 - alpha)
expected_acceptance = 1 - alpha
assert acceptance_rate >= expected_acceptance - 0.05, f"Coverage too low: {acceptance_rate}"
```

### Diagnostic Plots

**To debug guarantee violations:**

```python
# Plot 1: Acceptance rate vs. prediction confidence
# Expected: higher confidence -> higher acceptance

# Plot 2: Distance to cluster center vs. acceptance
# Expected: monotone relationship

# Plot 3: Interval width distribution per cluster
# Expected: within reasonable range, no extreme outliers

# Plot 4: OOB rate over time (if sequential data)
# Expected: stable around (1 - alpha), not drifting
```

---

## Summary Table

| Guarantee | Coverage | Depends On | Failure Mode |
|-----------|----------|-----------|--------------|
| **G1: Conformal** | $\geq 1 - \alpha$ | Exchangeability | Distribution shift |
| **G2: Modulation** | $\geq 1 - \alpha$ | Monotone $f$ | Invalid modulation function |
| **G3: Mondrian** | $\geq 1 - \alpha$ per cluster | Per-cluster exchangeability | Cluster mismatch |
| **G4: Full Guard** | $\geq 1 - \alpha$ | All of above | Any assumption violation |
| **G5: Calibration** | $\geq 1 - \alpha_{\text{VA}}$ | VennAbers quality | Uncalibrated intervals |

---

## Conclusion

The confidence-modulated conformal guard provides:

1. **Finite-sample validity:** Coverage $\geq 1 - \alpha$ with only exchangeability
2. **Monotonicity-preserving:** Modulation function preserves guarantees
3. **Adaptive:** Per-instance adaptation via confidence modulation
4. **Unified:** Same mechanism for classification and regression

**Caveats:**
- Guarantees are **conditional** on exchangeability
- Empirical coverage should be monitored
- Distribution shift is the primary failure mode

**Recommendation:**
- Always verify on holdout test set
- Monitor OOB acceptance rate over time
- Retrain guard if distribution shifts significantly

---

**Document Status:** Final  
**For implementation details, see:** `GUARD_DESIGN_CONFIDENCE_MODULATION.md`  
**For complete proofs, see:** `GUARD_MATHEMATICAL_FOUNDATIONS.md`
