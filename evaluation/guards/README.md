# Guard Overarching Resarch Questions

Below are **3 overarching research questions** that capture the full intent of the study. Each can be operationalized via the detailed RQs previously listed, but they stand alone as clear, high-level goals suitable for a paper, proposal, or introduction.

These are tightly framed around:
**(i) guarantees, (ii) uncertainty-aware robustness, (iii) trustworthiness of explanations**, and **(iv) generality across tasks**.

---

# **Overarching Research Questions**

## **ORQ1 — Reliability Guarantees:**

**To what extent does the guard mechanism improve the reliability of predictions and explanations by providing finite-sample coverage guarantees for in-distribution test instances, compared to the legacy (unguarded) implementation?**

This captures:

* calibration and coverage effects,
* avoidance of overconfident predictions,
* improved adherence to theoretical guarantees across classification and regression,
* sensitivity of guarantees to α.

---

## **ORQ2 — Uncertainty-Aware Robustness:**

**How effectively does the guard detect, reject, or otherwise mitigate explanations and predictions arising from out-of-distribution or high-uncertainty situations (both aleatoric and epistemic), thereby avoiding misleading or invalid outputs?**

This captures:

* robustness against synthetic perturbations,
* epistemic vs. aleatoric regimes,
* suppression of misleading explanations,
* local validity of perturbation-based rules,
* effects of distance metrics and clustering on boundary behaviour.

---

## **ORQ3 — Trustworthiness of Local Explanations:**

**Does the guard mechanism produce more trustworthy local explanations—factual, counterfactual, and alternative—by filtering out perturbations that are incompatible with the calibration distribution, and how does this compare to explanations generated without the guard?**

This captures:

* explanation fidelity under uncertainty,
* stability and validity of rule conditions,
* reduction of explanation failure modes,
* interpretability and consistency across uncertainty regimes.

---

## **ORQ4 — Generalizability Across Tasks and Parameters:**

**Are the benefits of the guard mechanism consistent across different learning settings (classification, regression, probabilistic regression), and how do core parameters (e.g., α, distance metrics, number of clusters) influence the trade-off between coverage, rejection rates, and explanation quality?**

This captures:

* task-general validity,
* the ablation study on α,
* metric/clustering sensitivity,
* practical deployment envelope.

---

# Summary of Framing

* **ORQ1** focuses on *guarantees* – the key technical promise of the guard.
* **ORQ2** focuses on *uncertainty-aware robustness* – why the guard exists.
* **ORQ3** focuses on *trustworthiness of explanations* – the XAI contribution.
* **ORQ4** focuses on *generality and parameter behaviour* – the broad usefulness and interpretability of hyperparameters.

Together, these 3–4 overarching RQs cover all aspects of your evaluation and allow the detailed sub-RQs (coverage, uncertainty types, α sensitivity, metrics/clustering) to unfold naturally in the methods and results sections.


# Guard Evaluation Description

Below is the **fully integrated, polished, end-to-end Experimental Protocol section**, with the split of Section 3.1 into:

* **3.1 Test-Set Coverage of the Guard**
* **3.2 Guarded Neighbourhood and Perturbation-Level Validity**

The section is coherent, publication-ready, and follows a clear methodological narrative.
You can drop it directly into a journal/conference manuscript.

---

# Experimental Protocol

This section details the experimental design used to evaluate the perturbation-guard mechanism implemented in *Calibrated Explanations*. The goal is to determine when and why the guarded implementation succeeds where the legacy (unguarded) variant fails. The experiments address **ORQ1–ORQ4** by assessing (i) finite-sample guarantees of the guard, (ii) misleading-explanation suppression, (iii) robustness across uncertainty regimes, and (iv) sensitivity to α, distance metrics, and clustering.

Evaluations are performed for **binary classification**, **multi-class classification**, **regression**, and **probabilistic regression**, ensuring coverage across all modelling paradigms supported by the library. In addition to synthetic data, real-world datasets are used to verify the guard’s distribution-free guarantees empirically.

---

## 1. Data Generating Processes

### 1.1 Synthetic Settings

Synthetic datasets allow full control over generative mechanisms, ground-truth explanations, and uncertainty structure. For each experiment we define:

* **Feature distributions**: multivariate Gaussian, Gaussian mixtures, or correlated covariates.
* **Target mechanisms**:

  * *Classification*: logistic model with nonlinear interactions.
  * *Regression*: nonlinear function with noise variable ε.
  * *Probabilistic regression*: functions with heteroscedastic conditional variance Var[Y|X].
* **Uncertainty regimes**:

  * *High aleatoric*: increasing noise ε.
  * *High epistemic*: sparsity regions, covariate shift pockets, or feature-space holes.

Synthetic experiments support effectively unlimited sample sizes and exact ground-truth comparison.

### 1.2 Real Datasets

To demonstrate finite-sample validity in practice, we employ several publicly available tabular datasets (UCI, OpenML). Each is divided into:

* training set,
* calibration set,
* test set.

All experiments use 20 random seeds to assess variability.

---

## 2. Models, Explanation Variants, and Reuse of Trained Models

### 2.1 Predictive Models

We evaluate a range of supervised models:

* gradient boosted trees,
* random forests,
* logistic regression or ridge regression,
* neural architectures (MLPs or LSTMs depending on dataset).

### 2.2 Explanation Variants

For each dataset and test instance we compute:

1. **Legacy Calibrated Explanations** (no guard), and
2. **Guarded Calibrated Explanations**, in which perturbations (\tilde x) are filtered by the normalized nonconformity score:
   [
   s(x) = \frac{\ell(x)}{w(x)+\varepsilon},
   ]
   where (\ell(x)) is distance to the closest calibration cluster and (w(x)) is the Venn–Abers or CPS interval width.

Perturbations satisfying (s(x) \le \tau_\alpha = Q_{1-\alpha}(S)) are accepted; others are rejected.

### 2.3 Reuse of Trained Models via Fixture-Like Infrastructure

To avoid repeated training and ensure comparability across conditions, we organize experiments using a fixture-based structure, akin to pytest:

* A **dataset fixture** yields `(X_train, y_train, X_cal, y_cal, X_test, y_test)`.
* A **model fixture** takes the training split and returns a trained predictive model.
* A **calibration fixture** constructs Venn–Abers or CPS calibration layers from the calibration split.
* A **guard fixture** constructs the guard (clusters, distances, scores) once per calibration object.

This makes the following reusable:

* the same *trained model* across:

  * legacy vs. guarded CE,
  * regression vs. probabilistic regression,
  * multiple α values, distances, and cluster counts;
* the same *calibrated predictor* across all explanation experiments;
* the same *guard object* across test-set coverage evaluation, perturbation-level evaluation, and ablations.

This ensures that differences across experimental conditions arise solely from guard behaviour and explanation logic.

---

## 3. Evaluation Dimensions

### 3.1 Test-Set Coverage of the Guard (ORQ1)

This part evaluates whether the guard satisfies its finite-sample acceptance guarantee when applied directly to *real, unperturbed* test instances.

For each dataset and seed:

1. Train a predictive model (fixture).
2. Construct calibrated predictor (VA/CPS) (fixture).
3. Fit guard on the calibration set:

   * compute (\ell(x_i)), (w(x_i)),
   * compute scores (s_i = \ell(x_i)/(w(x_i)+\varepsilon)),
   * compute threshold (\tau_\alpha = Q_{1-\alpha}({s_i})).
4. For each test instance (x_0):

   * compute (s(x_0)) and determine if (x_0 \in \mathcal{A}*\alpha = {x : s(x) \le \tau*\alpha}).

We compute:

* **Acceptance rate**
  [
  \widehat{A}*\alpha = \frac{1}{|\mathcal{X}*{\text{test}}|}\sum_{x_0 \in \mathcal{X}*{\text{test}}}
  \mathbf{1}{x_0 \in \mathcal{A}*\alpha},
  ]
* **Rejection rate** (\widehat{R}*\alpha = 1 - \widehat{A}*\alpha),

and compare (\widehat{R}*\alpha) to the nominal α.
Under exchangeability, Theorem 1 predicts (\widehat{R}*\alpha \approx \alpha).
This verifies the **coverage guarantee** independently from perturbations or explanations.

We repeat this for:

* binary classification,
* multi-class classification,
* regression (CPS),
* probabilistic regression (threshold queries with CPS).

---

### 3.2 Guarded Neighbourhood and Perturbation-Level Validity (ORQ1, ORQ2)

This part evaluates **how the guard shapes the local neighbourhood used for explanations** and whether it eliminates misleading perturbations that would be included by the legacy method.

For each test instance (x_0):

1. Reuse the same calibrated predictor and guard as in Section 3.1.
2. Generate perturbations (\tilde x) used for:

   * factual,
   * counterfactual,
   * semi-factual,
   * super-factual,
   * alternative explanations.
3. For each perturbation:

   * determine guard acceptance ((\tilde x \in \mathcal{A}_\alpha)),
   * determine ground-truth validity (V(x_0, \tilde x)):

     * *classification*: does the true class behaviour under the DGP agree with the allowed uncertainty band at (x_0)?
     * *regression*: does the true f(x) satisfy a neighbourhood-consistency condition?
     * *probabilistic regression*: does the threshold-crossing indicator respect the CPS uncertainty?
4. Compute:

   * **Guarded neighbourhood validity**
     [
     \widehat{V}*\alpha =
     \frac{
     \sum*{x_0,\tilde x} \mathbf{1}{\tilde x\in\mathcal{A}*\alpha} V(x_0,\tilde x)
     }{
     \sum*{x_0,\tilde x} \mathbf{1}{\tilde x\in\mathcal{A}_\alpha}
     },
     ]
   * **Legacy explanation validity** (\widehat{V}_{\text{legacy}}) (no guard),
   * **Misleading-explanation suppression**
     [
     \Delta V = \widehat{V}*\alpha - \widehat{V}*{\text{legacy}}.
     ]

We also record:

* false acceptances (invalid perturbations accepted by the guard),
* false rejections (valid perturbations rejected),
* proportion of legacy perturbations rejected by the guard due to inconsistency.

This answers ORQ2 by exposing how the guard removes perturbations that would otherwise produce misleading feature weights, incorrect counterfactuals, or overconfident decisions.

---

## 4. Metrics (Task-Specific)

### 4.1 Classification (Binary)

* Expected Calibration Error (ECE)
* Brier score
* Test-set guard rejection (\widehat{R}_\alpha)
* Explanation correctness (sign-of-effect vs. ground truth)
* Misleading Explanation Rate (MER)
* Guard neighbourhood validity (\widehat{V}_\alpha)

### 4.2 Multi-Class Classification

* Per-class coverage
* Multi-class ECE
* Top-k explanation stability
* Class-normalized MER
* Neighbourhood validity as above

### 4.3 Regression

* CPS interval coverage
* Interval width analysis
* Explanation correctness (partial derivative sign; contribution error)
* MER and perturbation validity
* Test-set guard coverage (\widehat{A}_\alpha)

### 4.4 Probabilistic Regression

* Threshold-crossing calibration
* Threshold interval coverage
* Neighbourhood validity and MER
* Explanation stability under uncertainty

---

## 5. Ablation Studies (ORQ3–ORQ4)

### 5.1 Sensitivity to α

Sweep α ∈ {0.01, 0.05, 0.10, 0.25}.
Record:

* test-set rejection (\widehat{R}_\alpha) vs. α,
* neighbourhood validity (\widehat{V}_\alpha) vs. α,
* MER vs. α,
* explanation stability and perturbation count vs. α.

### 5.2 Distance Metrics

Evaluate Euclidean, Mahalanobis, Manhattan, learned metrics.
Measure:

* guard coverage stability,
* OOD sensitivity,
* neighbourhood-width changes,
* computation time.

### 5.3 Number of Clusters

Evaluate k ∈ {1,2,4,8,16}.
Measure:

* intra-cluster variance,
* test-set rejection (\widehat{R}_\alpha),
* perturbation-level validity (\widehat{V}_\alpha),
* explanation stability.

---

## 6. Uncertainty-Normalized Neighbourhood Experiments

These experiments demonstrate the critical property of the guard:

**because the nonconformity score is normalized by the predictive interval width, the admissible neighbourhood expands or contracts based on uncertainty.**

### 6.1 Aleatoric Uncertainty

For increasing noise σ:

* measure predictive interval widths (w(x)),
* compute acceptance radii (r_\alpha(x)) induced by the guard,
* verify:
  [
  \sigma_1 < \sigma_2 \Rightarrow r_\alpha(x|\sigma_1) < r_\alpha(x|\sigma_2),
  ]
  i.e., higher noise → wider admissible neighbourhoods.

### 6.2 Epistemic Uncertainty

In sparse regions:

* CPS/VA intervals widen,
* guard becomes more permissive,
* accepted perturbation maps visually track epistemic pockets,
* we verify that admissibility correlates with model uncertainty, not density alone.

Visualizations:

* acceptance heatmaps,
* uncertainty-normalized neighbourhood contours,
* guard acceptance vs. interval width.

---

## 7. Failure Mode Analysis

We analyze:

* legacy acceptance of invalid perturbations,
* guard suppression of those perturbations,
* cases where both fail
  (e.g., true uncertainty too high),
* cases where guard is overly conservative
  (useful for interpreting α).

Qualitative examples include:

* sign-flip feature weights,
* incorrect counterfactual direction,
* composite rules producing false alternatives.

---

## 8. Statistical Testing and Reporting

We use:

* McNemar tests for paired MER comparisons,
* bootstrap CIs for explanation fidelity,
* KS tests for distribution-shift detection,
* Clopper–Pearson CIs for guard coverage (\widehat{A}_\alpha).

Plots include:

* rejection vs. α curves,
* MER vs. α,
* guard-vs-legacy neighbourhood validity,
* acceptance heatmaps vs. uncertainty.

---

## 9. Reproducibility

* the **perturbation_guard** branch is used for all guard implementations,
* fixtures reproduce identical models across runs,
* synthetic datasets and scripts released publicly,
* real datasets publicly accessible,
* results reproducible from a single experiment driver.


# Guard Empirical Coverage Test Infrastructure

## Overview

This directory contains a comprehensive pytest-based test infrastructure for validating perturbation guard empirical coverage guarantees across multiple task types and configurations.

### Key Features

- **Session-scoped fixtures**: Models, calibrators, and explainers are trained once per session and reused
- **Guard ablation grid**: Cartesian product of alpha, distance metrics, and cluster counts
- **Multi-task support**: Binary classification, multiclass classification, standard regression, and probabilistic regression
- **Baseline comparisons**: Every test compares guarded vs. non-guarded explainers
- **Finite-sample tolerance**: Empirical coverage checks allow 5% tolerance for finite-sample effects

## Architecture

### Fixture Hierarchy

```
conftest.py
├── Global Configuration (session-scoped)
│   ├── random_seed
│   ├── alpha_grid: [0.01, 0.05, 0.1, 0.2]
│   ├── distance_metrics: ["euclidean", "mahalanobis", "cosine"]
│   └── cluster_counts: [5, 10, 20]
│
├── Synthetic Data (session-scoped)
│   ├── binary_classification_data
│   ├── multiclass_classification_data
│   └── regression_data
│
├── Base Models (session-scoped)
│   ├── binary_classifier (RandomForestClassifier)
│   ├── multiclass_classifier (RandomForestClassifier)
│   └── regression_model (RandomForestRegressor)
│
├── Baseline Explainers (session-scoped, no guard)
│   ├── binary_explainer_baseline
│   ├── multiclass_explainer_baseline
│   └── regression_explainer_baseline
│
└── Guard Configuration & Factory Fixtures
    ├── guard_config_grid: Cartesian product (36 configs by default)
    ├── guard_config_minimal: Representative subset (2 configs)
    ├── binary_explainer_guarded_factory: Creates guarded binary explainers on demand
    ├── multiclass_explainer_guarded_factory: Creates guarded multiclass explainers
    └── regression_explainer_guarded_factory: Creates guarded regression explainers
```

### Guard Configuration

The `PerturbationGuardConfig` class encapsulates guard parameters:

```python
config = PerturbationGuardConfig(
    alpha=0.05,                      # 95% coverage target
    distance="mahalanobis",          # Mahalanobis distance metric
    n_clusters=10,                   # 10 clusters in feature space
    random_state=42,                 # Reproducibility
)

guard = make_guard(config)  # Returns ConformalRegionOracle instance
```

## Test Files

### `test_binary_guard_coverage.py`

Tests empirical coverage for binary classification with guard.

**Key Tests:**

- `test_should_maintain_coverage_with_guard_factual_explanations`: Validates (1 - alpha) coverage on factual rules
- `test_should_maintain_coverage_with_guard_alternative_explanations`: Validates alternative rule coverage
- `test_should_show_guard_effect_over_alpha_sweep`: Demonstrates filtering effect across alpha values
- `test_should_batch_explain_factual_with_guard`: Smoke test for batch operations
- `test_should_reject_invalid_alpha_in_guard_config`: Validates parameter constraints

**Parametrization:**

```python
@pytest.mark.parametrize("config_idx", range(6))
def test_should_maintain_coverage_with_guard_factual_explanations(config_idx, ...):
    """Tests 6 different guard configurations from the full grid."""
```

### `test_multiclass_guard_coverage.py`

Tests empirical coverage for multiclass classification with guard.

**Key Tests:**

- `test_should_maintain_coverage_multiclass_factual`: Validates factual explanations for all classes
- `test_should_maintain_coverage_multiclass_alternatives`: Validates alternative explanations
- `test_should_distinguish_classes_in_explanations`: Verifies different classes produce different rules
- `test_should_handle_guard_across_all_classes`: Guard works uniformly across classes
- `test_should_not_raise_with_guard_on_multiclass`: Smoke test for robustness

### `test_regression_guard_coverage.py`

Tests empirical coverage for standard and probabilistic regression with guard.

**Key Tests (Standard Regression):**

- `test_should_maintain_coverage_regression_factual_intervals`: Validates interval coverage
- `test_should_maintain_coverage_regression_alternatives`: Validates alternative rules
- `test_should_show_guard_reduces_interval_width_variability`: Demonstrates confidence modulation
- `test_should_batch_explain_factual_regression_with_guard`: Batch operations work

**Key Tests (Probabilistic Regression):**

- `test_should_support_threshold_parameter_with_guard`: Threshold parameter works
- `test_should_maintain_coverage_probabilistic_with_guard`: Coverage maintained for thresholded events

## Running the Tests

### Run All Guard Coverage Tests

```bash
pytest evaluation/guards/ -v
```

### Run Binary Classification Tests Only

```bash
pytest evaluation/guards/test_binary_guard_coverage.py -v
```

### Run With Minimal Config (Fast Iteration)

```bash
pytest evaluation/guards/ -v -k "minimal"
```

### Run With Full Guard Grid (Comprehensive Ablation)

```bash
pytest evaluation/guards/test_binary_guard_coverage.py::TestBinaryClassificationGuardCoverage::test_should_maintain_coverage_with_guard_factual_explanations -v
```

### Collect Coverage Report

```bash
pytest evaluation/guards/ --cov=src/calibrated_explanations --cov-report=html
```

## Fixture Usage Patterns

### Pattern 1: Single Configuration Test

Test a specific guard configuration against the baseline:

```python
def test_my_guard_property(
    guard_config_minimal,
    binary_explainer_guarded_factory,
    binary_explainer_baseline,
    binary_classification_data,
):
    cfg = guard_config_minimal[0]
    explainer_guarded = binary_explainer_guarded_factory(cfg)
    explainer_baseline = binary_explainer_baseline
    
    X_test = binary_classification_data["X_test"]
    
    # Compare explanations
    guarded_explanations = explainer_guarded.explain_factual(X_test)
    baseline_explanations = explainer_baseline.explain_factual(X_test)
    
    # Assert properties
    assert len(guarded_explanations) == len(X_test)
```

### Pattern 2: Parametrized Grid Sweep

Test across the full guard configuration grid:

```python
@pytest.mark.parametrize("config_idx", range(12))  # Test 12 configs
def test_property_across_grid(
    config_idx,
    guard_config_grid,
    binary_explainer_guarded_factory,
    binary_classification_data,
):
    cfg = guard_config_grid[config_idx]
    explainer = binary_explainer_guarded_factory(cfg)
    
    # Test with this configuration
    explanations = explainer.explain_factual(binary_classification_data["X_test"])
    assert len(explanations) > 0
```

### Pattern 3: Reuse Baseline Across Tests

Multiple tests can use the same baseline explainer:

```python
def test_baseline_consistency_1(binary_explainer_baseline, binary_classification_data):
    explanations_1 = binary_explainer_baseline.explain_factual(
        binary_classification_data["X_test"][:5]
    )
    assert len(explanations_1) == 5

def test_baseline_consistency_2(binary_explainer_baseline, binary_classification_data):
    explanations_2 = binary_explainer_baseline.explain_factual(
        binary_classification_data["X_test"][5:10]
    )
    assert len(explanations_2) == 5
    
    # Both use the same calibrated model; results are deterministic
```

## Coverage Guarantees

### Empirical Coverage Check

The tests validate the fundamental conformal prediction guarantee:

$$\mathbb{P}(\text{true label} \in [\ell, u]) \geq 1 - \alpha$$

**Tolerance:** 5% finite-sample tolerance is applied:

```python
target_coverage = 1.0 - cfg.alpha
tolerance = 0.05

assert empirical_coverage >= target_coverage - tolerance
```

For example:
- α = 0.1 (90% coverage target): empirical coverage ≥ 85%
- α = 0.05 (95% coverage target): empirical coverage ≥ 90%
- α = 0.01 (99% coverage target): empirical coverage ≥ 94%

### Per-Cluster Coverage

Advanced tests can validate coverage per cluster or per class:

```python
# Coverage per cluster (requires guard instrumentation)
for cluster_id in range(n_clusters):
    cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
    cluster_coverage = sum(covered[i] for i in cluster_indices) / len(cluster_indices)
    assert cluster_coverage >= target_coverage - tolerance
```

## Extending the Test Suite

### Adding a New Guard Configuration Ablation

1. Update `conftest.py` fixtures:

```python
@pytest.fixture(scope="session")
def nonconformity_metrics() -> list[str]:
    """New dimension for ablation."""
    return ["euclidean", "manhattan", "chebyshev"]
```

2. Extend `guard_config_grid` to include new dimension.

3. Add test cases using the new parameterization.

### Adding a New Task Type

1. Create new data fixture in `conftest.py`:

```python
@pytest.fixture(scope="session")
def time_series_data(random_seed):
    """Generate synthetic time series data."""
    # ... implementation
    return {
        "X_proper": ...,
        "y_proper": ...,
        # ...
    }
```

2. Create base model fixture:

```python
@pytest.fixture(scope="session")
def time_series_regressor(random_seed, time_series_data):
    """Train time series regressor."""
    # ...
    return model
```

3. Create explainer factory:

```python
@pytest.fixture(scope="session")
def time_series_explainer_guarded_factory(time_series_regressor, time_series_data):
    def _factory(cfg):
        # ... same pattern as binary/multiclass/regression
    return _factory
```

4. Add test file `test_time_series_guard_coverage.py` following existing patterns.

## Performance Notes

- **Total Runtime:** ~5–10 minutes for full ablation (36 configs × 3 task types)
- **Memory:** ~500 MB for session-scoped fixtures (shared models + data)
- **Parallelization:** Tests can be parallelized with `pytest-xdist`:

```bash
pytest evaluation/guards/ -n auto  # Use all CPU cores
```

## References

- **ConformalRegionOracle**: `src/calibrated_explanations/core/explain/guards/regions.py`
- **Guard Plugin Architecture**: `src/calibrated_explanations/plugins/guards.py`
- **WrapCalibratedExplainer API**: See legacy API contract at `improvement_docs/legacy_user_api_contract.md`
- **ADR-004 (Parallel Backend)**: Async execution frameworks for tests
- **Release Plan**: `improvement_docs/RELEASE_PLAN_V1.md` (ADR-006 trust model, guard coverage testing)

## Future Work

1. **Online Coverage Monitoring**: Stream test results to telemetry backend
2. **Per-Cluster Diagnostics**: Instrument guard to emit per-cluster coverage metrics
3. **Adaptive Alpha Selection**: Test dynamic alpha adjustment based on instance difficulty
4. **Cross-Task Transfer**: Validate guard generalization across related tasks
5. **Real Data Validation**: Mirror test suite on public benchmark datasets (UCI, Kaggle)
